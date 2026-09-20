"""The real Smart Turn v3.2 ONNX model on fixture audio. Skipped wherever the weights are
not in the Hugging Face cache, so CI never downloads 8.7 MB to run the suite."""

from __future__ import annotations

import hashlib
import statistics
import time
from pathlib import Path

import numpy as np
import pytest

from backend.domain.value_objects.audio import PCM16_16K_MONO
from backend.infrastructure.audio.wav import read_wav
from backend.infrastructure.endpointing import smart_turn

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "audio"
COMPLETE = FIXTURES / "intake_en" / "02.wav"
GERMAN = FIXTURES / "intake_de" / "00.wav"
TARGET_INFERENCE_MS = 25.0

needs_audio = pytest.mark.skipif(
    not COMPLETE.is_file(), reason="no fixture audio: run `make fixtures`"
)


def cached_model() -> smart_turn.SmartTurnModel:
    """Skips only when the weights are missing; a model that fails to load is a failure."""
    try:
        smart_turn.weights(offline=True)
    except Exception as exc:  # noqa: BLE001 - any lookup failure means "not downloaded here"
        pytest.skip(f"Smart Turn weights are not cached: {exc}")
    return smart_turn.load_model(offline=True)


def test_the_pinned_weights_are_the_published_int8_cpu_model() -> None:
    """The digest of pipecat-ai/smart-turn-v3 `smart-turn-v3.2-cpu.onnx` at the pinned
    revision, read from Hugging Face on 19 Sep 2026."""
    try:
        path = smart_turn.weights(offline=True)
    except Exception as exc:  # noqa: BLE001 - any lookup failure means "not downloaded here"
        pytest.skip(f"Smart Turn weights are not cached: {exc}")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert digest == "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"


def pcm16(path: Path, fraction: float = 1.0) -> bytes:
    audio, fmt = read_wav(path)
    assert fmt == PCM16_16K_MONO
    return audio[: int(len(audio) * fraction) // 2 * 2]


@needs_audio
def test_the_model_tells_a_finished_utterance_from_one_cut_mid_sentence() -> None:
    model = cached_model()
    assert model(pcm16(COMPLETE)) > 0.5
    assert model(pcm16(COMPLETE, fraction=0.6)) < 0.5
    assert model(pcm16(GERMAN)) > 0.5


@needs_audio
def test_a_decision_costs_less_than_the_latency_it_saves() -> None:
    model = cached_model()
    audio = pcm16(COMPLETE)
    model(audio)  # first call warms ONNX Runtime's arenas
    timings = []
    for _ in range(30):
        started = time.perf_counter()
        model(audio)
        timings.append((time.perf_counter() - started) * 1000)
    p50 = statistics.median(timings)
    p95 = sorted(timings)[int(0.95 * len(timings))]
    print(f"Smart Turn v3.2 inference: p50 {p50:.1f} ms, p95 {p95:.1f} ms over 30 decisions")
    assert p50 < TARGET_INFERENCE_MS


@needs_audio
def test_the_model_reads_any_length_of_audio_as_its_last_eight_seconds() -> None:
    model = cached_model()
    short = pcm16(COMPLETE)
    assert model(short) == pytest.approx(model(bytes(smart_turn.WINDOW_BYTES - len(short)) + short))
    assert 0.0 <= model(pcm16(COMPLETE) * 5) <= 1.0


@needs_audio
def test_features_match_the_whisper_extractor_the_model_was_trained_with() -> None:
    transformers = pytest.importorskip("transformers")
    audio = np.frombuffer(pcm16(COMPLETE), dtype=np.int16).astype(np.float32) / 32768
    audio = np.pad(audio, (smart_turn.WINDOW_SAMPLES - len(audio), 0))
    extractor = transformers.WhisperFeatureExtractor(chunk_length=8)
    expected = extractor(
        audio,
        sampling_rate=smart_turn.SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=smart_turn.WINDOW_SAMPLES,
        truncation=True,
        do_normalize=True,
    ).input_features

    assert smart_turn.log_mel_features(audio) == pytest.approx(expected, abs=1e-4)


@needs_audio
def test_the_function_the_container_injects_asks_the_real_model() -> None:
    """The container passes `probability`, not the model object: a live call found that
    wiring broken while every stub-backed test passed, and the detector's fallback hid it
    behind the silence ceiling."""
    cached_model()
    assert smart_turn.probability(pcm16(COMPLETE)) > 0.5
