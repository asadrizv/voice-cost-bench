"""SttPort contract, run against every implementation: Deepgram (against a server speaking
its protocol from fixtures) and both engines of the STT service (real gpu/whisper_service
code with the model swapped for a stub)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import pytest

from backend.application.ports.stt_port import FlushSignal, SttPort, TranscriptEvent
from backend.domain.value_objects.audio import PCM16_24K_MONO, AudioChunk
from backend.infrastructure.audio.wav import frames, read_wav
from backend.infrastructure.stt.deepgram_stt import DeepgramStt
from backend.infrastructure.stt.whisper_stt import WhisperStt
from gpu.whisper_service.app import (
    TranscriptionSession,
    VoxtralTranscriber,
    create_app,
    selected_transcriber,
)
from tests.integration.servers import FakeDeepgram, run_asgi

AUDIO = Path(__file__).resolve().parents[2] / "fixtures" / "audio" / "intake_en" / "00.wav"
pytestmark = pytest.mark.skipif(not AUDIO.exists(), reason="no fixture audio: run `make fixtures`")

UTTERANCE = "Hi, I need to speak to someone about my landlord."


class StubTranscriber:
    def __init__(self) -> None:
        self.calls: list[tuple[float, str]] = []

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        self.calls.append((audio.size / 16000, language))
        return UTTERANCE if audio.size else ""


class StubSession:
    """Decodes one more word of the utterance per advance, as a streaming model does."""

    def __init__(self) -> None:
        self.fed_seconds = 0.0
        self._words = 0

    def feed(self, audio: np.ndarray) -> None:
        self.fed_seconds += audio.size / 16000

    def advance(self) -> str:
        self._words += 1
        return " ".join(UTTERANCE.split()[: self._words])

    def finish(self) -> str:
        return UTTERANCE


class StubStreamingTranscriber:
    engine, model = VoxtralTranscriber.engine, VoxtralTranscriber.model
    speech_floor_dbfs = VoxtralTranscriber.speech_floor_dbfs

    def __init__(self) -> None:
        self.sessions: list[StubSession] = []

    def session(self) -> TranscriptionSession:
        self.sessions.append(StubSession())
        return self.sessions[-1]


@asynccontextmanager
async def deepgram() -> AsyncIterator[SttPort]:
    server = FakeDeepgram()
    async with server.running() as url:
        yield DeepgramStt("dg-test-key", url=url)


@asynccontextmanager
async def whisper() -> AsyncIterator[SttPort]:
    async with run_asgi(create_app(StubTranscriber, workers=2, interim_interval_s=0.0)) as host:
        yield WhisperStt(f"ws://{host}/v1/stream")


@asynccontextmanager
async def voxtral() -> AsyncIterator[SttPort]:
    app = create_app(StubStreamingTranscriber, workers=2, interim_interval_s=0.0)
    async with run_asgi(app) as host:
        yield WhisperStt(f"ws://{host}/v1/stream")


IMPLEMENTATIONS: dict[str, Callable[[], AbstractAsyncContextManager[SttPort]]] = {
    "deepgram": deepgram,
    "whisper": whisper,
    "voxtral": voxtral,
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


async def test_a_silent_caller_is_flushed_without_whisper_being_asked() -> None:
    """Whisper answers silence with an invented "Thank you.", and the caller is silent for
    as long as the agent talks, so the buffer's gate has to keep it away from the model."""
    stub = StubTranscriber()

    async def silence_then_flush() -> AsyncIterator[AudioChunk | FlushSignal]:
        for _ in range(100):  # 2 s of silence while the agent was talking
            yield AudioChunk(b"\x00\x00" * 320)
        yield FlushSignal()
        await asyncio.sleep(0.5)

    async with run_asgi(create_app(lambda: stub, workers=1, interim_interval_s=0.0)) as host:
        warmed = len(stub.calls)
        events = await collect(
            WhisperStt(f"ws://{host}/v1/stream").stream(silence_then_flush(), "de")
        )

    assert [(e.text, e.flushed) for e in events] == [("", True)]
    assert stub.calls[warmed:] == []


async def test_a_silent_caller_is_flushed_without_the_streaming_model_being_opened() -> None:
    """Voxtral invents nothing on silence, so it needs no confidence filter; what it does
    need is to be kept off the silence, which costs it real decoding time."""
    transcriber = StubStreamingTranscriber()

    async def silence_then_flush() -> AsyncIterator[AudioChunk | FlushSignal]:
        for _ in range(100):  # 2 s of silence while the agent was talking
            yield AudioChunk(b"\x00\x00" * 320)
        yield FlushSignal()
        await asyncio.sleep(0.5)

    async with run_asgi(create_app(lambda: transcriber, interim_interval_s=0.0)) as host:
        warmed = len(transcriber.sessions)
        stt = WhisperStt(f"ws://{host}/v1/stream")
        events = await collect(stt.stream(silence_then_flush(), "de"))

    assert [(e.text, e.flushed) for e in events] == [("", True)]
    assert transcriber.sessions[warmed:] == []


class SlowStubSession(StubSession):
    """Blocks inside the model until released, and records any second piece of work that
    gets in while it is there."""

    def __init__(self, release: threading.Event) -> None:
        super().__init__()
        self._release = release
        self.inside = 0
        self.overlapped = False

    def _enter(self) -> None:
        self.inside += 1
        self.overlapped = self.overlapped or self.inside > 1

    def advance(self) -> str:
        self._enter()
        self._release.wait(5)
        self.inside -= 1
        return super().advance()

    def finish(self) -> str:
        self._enter()
        self.inside -= 1
        return super().finish()


async def test_a_flush_waits_for_the_interim_already_inside_the_streaming_model() -> None:
    """mlx-audio's session decodes on one thread and keeps its state between steps, so a
    final that started while an interim was still running would corrupt the utterance."""
    release = threading.Event()

    class BlockingTranscriber(StubStreamingTranscriber):
        def session(self) -> TranscriptionSession:
            self.sessions.append(SlowStubSession(release))
            return self.sessions[-1]

    transcriber = BlockingTranscriber()

    async def speech_then_flush() -> AsyncIterator[AudioChunk | FlushSignal]:
        async for item in speech_then():
            yield item
        yield FlushSignal()
        asyncio.get_running_loop().call_later(0.3, release.set)
        await asyncio.sleep(1.0)

    async with run_asgi(create_app(lambda: transcriber, workers=3, interim_interval_s=0.0)) as host:
        stt = WhisperStt(f"ws://{host}/v1/stream")
        events = await asyncio.wait_for(collect(stt.stream(speech_then_flush(), "en")), 15)

    assert [e.text for e in events if e.flushed] == [UTTERANCE]
    assert not any(s.overlapped for s in transcriber.sessions)


def speech_peaking_at(dbfs: float) -> bytes:
    """The fixture utterance rescaled so its loudest 20 ms frame sits at `dbfs`, which is
    what the service's gate looks at. Rescaled rather than attenuated by a fixed factor
    because `make fixtures` re-renders the audio and its level moves with the renderer."""
    pcm, fmt = read_wav(AUDIO)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    frame = fmt.byte_count(0.020) // 2
    loudest = max(
        np.sqrt(np.mean(samples[i : i + frame] ** 2)) for i in range(0, samples.size - frame, frame)
    )
    return (samples * (10 ** (dbfs / 20) / loudest) * 32768.0).astype("<i2").tobytes()


@pytest.mark.parametrize(("dbfs", "sessions"), [(-58.0, 1), (-65.0, 0)])
async def test_the_voxtral_gate_opens_at_the_quietest_speech_it_was_measured_to_read(
    dbfs: float, sessions: int
) -> None:
    """Voxtral reads speech far below Whisper's -45 dBFS gate -- down to a loudest frame
    of about -58 dBFS -- and returns nothing below about -62, so its own gate has to sit
    between the two or quiet callers are dropped before the model ever sees them."""
    transcriber = StubStreamingTranscriber()
    _, fmt = read_wav(AUDIO)

    async def quiet_speech() -> AsyncIterator[AudioChunk | FlushSignal]:
        for chunk in frames(speech_peaking_at(dbfs), fmt):
            yield chunk
        await asyncio.sleep(0.3)
        yield FlushSignal()
        await asyncio.sleep(0.3)

    async with run_asgi(create_app(lambda: transcriber, interim_interval_s=0.0)) as host:
        warmed = len(transcriber.sessions)
        stt = WhisperStt(f"ws://{host}/v1/stream")
        events = await collect(stt.stream(quiet_speech(), "en"))

    assert [e.text for e in events if e.flushed] == [UTTERANCE if sessions else ""]
    assert len(transcriber.sessions[warmed:]) == sessions


async def test_each_utterance_is_decoded_by_a_session_of_its_own() -> None:
    """mlx-audio's session is spent once closed, so the flush that finalises an utterance
    has to be the end of it."""
    transcriber = StubStreamingTranscriber()

    async with run_asgi(create_app(lambda: transcriber, interim_interval_s=0.0)) as host:
        warmed = len(transcriber.sessions)
        stt = WhisperStt(f"ws://{host}/v1/stream")
        events = await collect(stt.stream(speech_then(FlushSignal(), FlushSignal()), "en"))

    assert [e.text for e in events if e.flushed] == [UTTERANCE, ""]
    assert len(transcriber.sessions[warmed:]) == 1


async def test_the_service_reports_the_streaming_engine_it_loaded() -> None:
    async with (
        run_asgi(create_app(StubStreamingTranscriber)) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        info = (await client.get("/v1/info")).json()

    assert info["engines"] == [{"id": "voxtral", "model": VoxtralTranscriber.model}]


@pytest.mark.parametrize(
    ("setting", "engine"), [(None, "faster-whisper"), ("mlx", "mlx"), ("voxtral", "voxtral")]
)
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
    assert confident_text([(" Yes.", -0.44), (" Tuesday.", -1.0)]) == "Yes. Tuesday."
