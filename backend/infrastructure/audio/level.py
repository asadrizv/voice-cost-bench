from __future__ import annotations

import numpy as np
import numpy.typing as npt

from backend.domain.value_objects.audio import AudioChunk

AUDIBLE_DBFS = -45.0
"""EnergyVad's minimum speech threshold, read from here by the VAD itself rather than
written twice. Nothing EnergyVad calls speech is inaudible here, so the harness's end of
caller speech is never earlier than the endpointer's."""
SILENCE_DBFS = -120.0
WINDOW_S = 0.02


def unit_samples(data: bytes) -> npt.NDArray[np.float32]:
    """PCM16 bytes as floats in [-1, 1)."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def dbfs(samples: npt.NDArray[np.float32]) -> float:
    """RMS level of unit-scaled samples, floored at SILENCE_DBFS so digital silence reads
    as a number rather than -inf."""
    if not samples.size:
        return SILENCE_DBFS
    rms = float(np.sqrt(np.mean(samples * samples)))
    return 20 * float(np.log10(max(rms, 1e-6)))


def first_audible_s(chunk: AudioChunk) -> float | None:
    """Offset of the first 20 ms window whose RMS exceeds AUDIBLE_DBFS, or None if silent."""
    samples = unit_samples(chunk.data)
    window = chunk.format.byte_count(WINDOW_S) // 2
    for start in range(0, samples.size, window):
        if dbfs(samples[start : start + window]) > AUDIBLE_DBFS:
            return start / window * WINDOW_S
    return None
