from __future__ import annotations

import numpy as np

from backend.domain.value_objects.audio import AudioChunk


class EnergyVad:
    """RMS energy against an adaptive noise floor, with onset smoothing only.

    No hangover on purpose: the endpointer stamps speech_end from the last frame this
    reports as speech, so any hangover would silently shave that much off every measured
    perceived_delay. Gaps between words are the endpointer's job.

    Adequate behind the browser's echo cancellation and noise suppression, which is where
    the demo runs. For open-mic telephony, swap in a model VAD behind the same port.
    """

    def __init__(
        self,
        margin_db: float = 12.0,
        min_threshold_dbfs: float = -45.0,
        onset_ms: float = 60.0,
        onset_reset_ms: float = 120.0,
    ) -> None:
        self._margin = margin_db
        self._min_threshold = min_threshold_dbfs
        self._onset_ms = onset_ms
        self._onset_reset_ms = onset_reset_ms
        self._noise_floor = -60.0
        self._voiced_ms = 0.0
        self._silent_ms = 0.0

    def is_speech(self, chunk: AudioChunk) -> bool:
        if chunk.is_empty():
            return False
        samples = np.frombuffer(chunk.data, dtype="<i2").astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        level = 20 * np.log10(max(rms, 1e-6))
        threshold = max(self._noise_floor + self._margin, self._min_threshold)
        frame_ms = chunk.duration_seconds * 1000

        if level > threshold:
            self._voiced_ms += frame_ms
            self._silent_ms = 0.0
            return self._voiced_ms >= self._onset_ms

        # Track the floor quickly downward and slowly upward, so speech can't drag it up.
        rate = 0.2 if level < self._noise_floor else 0.02
        self._noise_floor += (level - self._noise_floor) * rate
        self._silent_ms += frame_ms
        if self._silent_ms >= self._onset_reset_ms:
            self._voiced_ms = 0.0
        return False
