from __future__ import annotations

import numpy as np

from backend.domain.value_objects.audio import AudioChunk

AUDIBLE_DBFS = -45.0
"""EnergyVad's minimum speech threshold. Nothing EnergyVad calls speech is inaudible here,
so the harness's end of caller speech is never earlier than the endpointer's."""
WINDOW_S = 0.02


def first_audible_s(chunk: AudioChunk) -> float | None:
    """Offset of the first 20 ms window whose RMS exceeds AUDIBLE_DBFS, or None if silent."""
    samples = np.frombuffer(chunk.data, dtype="<i2").astype(np.float32) / 32768.0
    window = chunk.format.byte_count(WINDOW_S) // 2
    for start in range(0, samples.size, window):
        part = samples[start : start + window]
        rms = float(np.sqrt(np.mean(part * part)))
        if 20 * np.log10(max(rms, 1e-6)) > AUDIBLE_DBFS:
            return start / window * WINDOW_S
    return None
