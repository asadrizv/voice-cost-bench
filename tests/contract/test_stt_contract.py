"""SttPort contract, run against both implementations: Deepgram (against a server speaking
its protocol from fixtures) and Whisper (against the real gpu/whisper_service code with the
model swapped for a stub)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

import numpy as np
import pytest

from backend.application.ports.stt_port import FlushSignal, SttPort, TranscriptEvent
from backend.domain.value_objects.audio import PCM16_24K_MONO, AudioChunk
from backend.infrastructure.audio.wav import frames, read_wav
from backend.infrastructure.stt.deepgram_stt import DeepgramStt
from backend.infrastructure.stt.whisper_stt import WhisperStt
from gpu.whisper_service.app import create_app, selected_transcriber
from tests.integration.servers import FakeDeepgram, run_asgi

AUDIO = Path(__file__).resolve().parents[2] / "fixtures" / "audio" / "intake_en" / "00.wav"
pytestmark = pytest.mark.skipif(not AUDIO.exists(), reason="no fixture audio: run `make fixtures`")


class StubTranscriber:
    def __init__(self) -> None:
        self.calls: list[tuple[float, str]] = []

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        self.calls.append((audio.size / 16000, language))
        return "Hi, I need to speak to someone about my landlord." if audio.size else ""


@asynccontextmanager
async def deepgram() -> AsyncIterator[SttPort]:
    server = FakeDeepgram()
    async with server.running() as url:
        yield DeepgramStt("dg-test-key", url=url)


@asynccontextmanager
async def whisper() -> AsyncIterator[SttPort]:
    async with run_asgi(create_app(StubTranscriber, workers=2, interim_interval_s=0.0)) as host:
        yield WhisperStt(f"ws://{host}/v1/stream")


IMPLEMENTATIONS: dict[str, Callable[[], AbstractAsyncContextManager[SttPort]]] = {
    "deepgram": deepgram,
    "whisper": whisper,
}


@pytest.fixture(params=list(IMPLEMENTATIONS))
async def stt(request: pytest.FixtureRequest) -> AsyncIterator[SttPort]:
    async with IMPLEMENTATIONS[request.param]() as impl:
        yield impl


async def speech_then(*extra: FlushSignal) -> AsyncIterator[AudioChunk | FlushSignal]:
    pcm, fmt = read_wav(AUDIO)
    for chunk in frames(pcm, fmt):
        yield chunk
        await asyncio.sleep(0.001)
    for item in extra:
        # A caller pauses before the endpointer commits, and a flush cancels any interim
        # still in flight, so flushing the instant audio ends would race the interim.
        await asyncio.sleep(0.3)
        yield item
        await asyncio.sleep(0.3)


async def collect(stream: AsyncIterator[TranscriptEvent]) -> list[TranscriptEvent]:
    return [event async for event in stream]


async def test_flush_is_answered_once_with_the_utterance(stt: SttPort) -> None:
    events = await asyncio.wait_for(collect(stt.stream(speech_then(FlushSignal()), "en")), 10)
    flushed = [e for e in events if e.flushed]
    assert len(flushed) == 1
    assert flushed[0].is_final
    finals = " ".join(e.text for e in events if e.is_final)
    assert "landlord" in finals


async def test_interim_results_arrive_before_the_final(stt: SttPort) -> None:
    events = await asyncio.wait_for(collect(stt.stream(speech_then(FlushSignal()), "en")), 10)
    first_final = next(i for i, e in enumerate(events) if e.is_final)
    assert any(not e.is_final and e.text for e in events[:first_final])


async def test_flush_without_speech_is_still_answered(stt: SttPort) -> None:
    async def only_flush() -> AsyncIterator[AudioChunk | FlushSignal]:
        yield AudioChunk(b"\x00\x00" * 320)
        yield FlushSignal()
        await asyncio.sleep(1.0)

    events = await asyncio.wait_for(collect(stt.stream(only_flush(), "en")), 10)
    assert [e.flushed for e in events if e.flushed] == [True]


async def test_stream_ends_when_audio_ends(stt: SttPort) -> None:
    async def nothing() -> AsyncIterator[AudioChunk | FlushSignal]:
        return
        yield  # pragma: no cover

    assert await asyncio.wait_for(collect(stt.stream(nothing(), "en")), 5) == []


async def test_rejects_audio_in_the_wrong_format(stt: SttPort) -> None:
    async def wrong() -> AsyncIterator[AudioChunk | FlushSignal]:
        yield AudioChunk(b"\x00\x00" * 480, PCM16_24K_MONO)

    with pytest.raises(ValueError, match="16 kHz"):
        await asyncio.wait_for(collect(stt.stream(wrong(), "en")), 5)


async def test_deepgram_request_is_configured_for_our_endpointing() -> None:
    server = FakeDeepgram()
    async with server.running() as url:
        await collect(DeepgramStt("dg-key", model="nova-3", url=url).stream(speech_then(), "de"))
    path = server.paths[0]
    assert "endpointing=false" in path and "language=de" in path and "model=nova-3" in path
    assert "sample_rate=16000" in path and "encoding=linear16" in path
    assert server.auth == ["Token dg-key"]
    assert server.closed_cleanly


async def test_deepgram_silence_on_finalize_is_covered_by_timeout() -> None:
    server = FakeDeepgram(silent_on_finalize=True)
    async with server.running() as url:
        stt = DeepgramStt("k", url=url)
        stt.flush_timeout_s = 0.2
        events = await collect(stt.stream(speech_then(FlushSignal()), "en"))
    assert [e for e in events if e.flushed] == [TranscriptEvent("", True, flushed=True)]


async def test_whisper_service_drops_leading_silence_and_passes_language() -> None:
    stub = StubTranscriber()

    async def silence_then_speech() -> AsyncIterator[AudioChunk | FlushSignal]:
        for _ in range(100):  # 2 s of silence while the agent was talking
            yield AudioChunk(b"\x00\x00" * 320)
        async for item in speech_then(FlushSignal()):
            yield item

    async with run_asgi(create_app(lambda: stub, workers=1, interim_interval_s=10)) as host:
        await collect(WhisperStt(f"ws://{host}/v1/stream").stream(silence_then_speech(), "de"))
    final_seconds, language = stub.calls[-1]
    pcm, fmt = read_wav(AUDIO)
    assert final_seconds <= fmt.duration_seconds(len(pcm)) + 0.05
    assert language == "de"


@pytest.mark.parametrize(("setting", "engine"), [(None, "faster-whisper"), ("mlx", "mlx")])
def test_the_environment_says_which_engine_the_service_loads(
    monkeypatch: pytest.MonkeyPatch, setting: str | None, engine: str
) -> None:
    monkeypatch.delenv("WHISPER_BACKEND", raising=False)
    if setting is not None:
        monkeypatch.setenv("WHISPER_BACKEND", setting)
    assert selected_transcriber().engine == engine


def test_an_engine_this_service_cannot_run_is_refused_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WHISPER_BACKEND", "parakeet")
    with pytest.raises(RuntimeError, match="parakeet"):
        selected_transcriber()


def test_whisper_drops_low_confidence_segments() -> None:
    from gpu.whisper_service.app import confident_text

    assert confident_text([(" what", -2.76), (" you", -2.76)]) == ""
    assert confident_text([(" My name is Anna Weber.", -0.26)]) == "My name is Anna Weber."
    assert confident_text([(" Thank you.", -0.44), (" um", -1.8)]) == "Thank you."
