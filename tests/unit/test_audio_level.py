"""The one speech floor the VAD and the harness both read.

`PacedAudioOutput.caller_observed_ms` times from the harness's first audible frame to the
agent's; it is only honest while nothing `EnergyVad` reports as speech is inaudible to
`first_audible_s`. Retuning either threshold on its own would move every published
caller-observed figure with no test to notice, so the invariant is asserted here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.domain.value_objects.audio import PCM16_16K_MONO, AudioChunk
from backend.infrastructure.audio.level import AUDIBLE_DBFS, dbfs, first_audible_s, unit_samples
from backend.infrastructure.vad.energy_vad import EnergyVad


def tone(level_dbfs: float, seconds: float = 0.02) -> AudioChunk:
    """A 220 Hz sine at the requested RMS level."""
    samples = np.arange(int(PCM16_16K_MONO.sample_rate * seconds))
    amplitude = 10 ** (level_dbfs / 20) * math.sqrt(2)
    wave = amplitude * np.sin(2 * np.pi * 220 * samples / PCM16_16K_MONO.sample_rate)
    return AudioChunk((wave * 32767).astype("<i2").tobytes(), PCM16_16K_MONO)


def test_a_tone_reads_back_at_the_level_it_was_made_at() -> None:
    assert dbfs(unit_samples(tone(-30.0).data)) == pytest.approx(-30.0, abs=0.1)
    assert dbfs(np.zeros(0, dtype=np.float32)) == -120.0


@pytest.mark.parametrize("level_dbfs", [AUDIBLE_DBFS + 1, -40.0, -30.0, -12.0])
def test_nothing_the_vad_calls_speech_is_inaudible_to_the_harness(level_dbfs: float) -> None:
    vad = EnergyVad()
    chunk = tone(level_dbfs)
    assert any(vad.is_speech(chunk) for _ in range(5))
    assert first_audible_s(chunk) == 0.0


def test_below_that_floor_neither_of_them_hears_anything() -> None:
    vad = EnergyVad()
    chunk = tone(AUDIBLE_DBFS - 1)
    assert not any(vad.is_speech(chunk) for _ in range(20))
    assert first_audible_s(chunk) is None
