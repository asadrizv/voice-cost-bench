"""SttPort contract, run against every implementation: Deepgram (against a server speaking
its protocol from fixtures), both Apple Silicon engines of the STT service (real
gpu/whisper_service code with the model swapped for a stub) and its CUDA engine (real
backend code against a server speaking vLLM's realtime protocol)."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi import FastAPI, WebSocket

from backend.application.ports.stt_port import FlushSignal, SttPort, TranscriptEvent
from backend.domain.value_objects.audio import PCM16_24K_MONO, AudioChunk
from backend.infrastructure.audio.wav import frames, read_wav
from backend.infrastructure.stt.deepgram_stt import DeepgramStt
from backend.infrastructure.stt.whisper_stt import WhisperStt
from gpu.whisper_service import app as whisper_service
from gpu.whisper_service.app import (
    WARMUP_PCM,
    TranscriptionSession,
    VllmVoxtralTranscriber,
    VoxtralTranscriber,
    create_app,
    selected_transcriber,
    to_samples,
)
from tests.integration.servers import FakeDeepgram, FakeVllmRealtime, run_asgi

AUDIO = Path(__file__).resolve().parents[2] / "fixtures" / "audio" / "intake_en" / "00.wav"
pytestmark = pytest.mark.skipif(not AUDIO.exists(), reason="no fixture audio: run `make fixtures`")

UTTERANCE = "Hi, I need to speak to someone about my landlord."
FLUSH_TIMEOUT_S = 0.4
"""What the CUDA engine waits for a realtime server's final answer in these tests, under
WhisperStt.flush_timeout_s so the service answers a flush before the client gives up on it."""


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
    engine, backend = VoxtralTranscriber.engine, VoxtralTranscriber.backend
    model = VoxtralTranscriber.model
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


@asynccontextmanager
async def realtime_service(server: FakeVllmRealtime) -> AsyncIterator[SttPort]:
    """The STT service running its CUDA engine, whose model is a vLLM server away."""
    async with run_asgi(server.app()) as vllm:
        transcriber = VllmVoxtralTranscriber(
            url=f"ws://{vllm}/v1/realtime", flush_timeout_s=FLUSH_TIMEOUT_S
        )
        async with run_asgi(create_app(lambda: transcriber, interim_interval_s=0.0)) as host:
            yield WhisperStt(f"ws://{host}/v1/stream")


@asynccontextmanager
async def voxtral_vllm() -> AsyncIterator[SttPort]:
    async with realtime_service(FakeVllmRealtime(text=UTTERANCE)) as stt:
        yield stt


IMPLEMENTATIONS: dict[str, Callable[[], AbstractAsyncContextManager[SttPort]]] = {
    "deepgram": deepgram,
    "whisper": whisper,
    "voxtral": voxtral,
    "voxtral-vllm": voxtral_vllm,
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


async def test_the_cuda_engine_is_reported_as_voxtral_by_the_runtime_that_served_it() -> None:
    """The engine is the same one the Apple Silicon backend runs, so /transparency names
    the same catalogue entry; the model says which runtime produced the numbers, which the
    catalogue's version -- fixed configuration -- cannot."""
    async with run_asgi(FakeVllmRealtime(text=UTTERANCE).app()) as vllm:
        transcriber = VllmVoxtralTranscriber(
            url=f"ws://{vllm}/v1/realtime",
            model="mistralai/Voxtral-Mini-4B-Realtime-2602",
            flush_timeout_s=FLUSH_TIMEOUT_S,
        )
        async with (
            run_asgi(create_app(lambda: transcriber)) as host,
            httpx.AsyncClient(base_url=f"http://{host}") as client,
        ):
            info = (await client.get("/v1/info")).json()

    assert info["engines"] == [
        {"id": "voxtral", "model": "mistralai/Voxtral-Mini-4B-Realtime-2602 (vLLM realtime)"}
    ]


async def test_the_cuda_engine_sends_the_realtime_server_the_model_and_the_callers_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wire format is vLLM's: the model is named before any commit, and the audio is
    base64 PCM16 at 16 kHz, so the samples the server decodes are the ones the caller sent.
    They survive a round trip through float32, which the service's own protocol is in, to
    within the last bit."""
    monkeypatch.delenv("VOXTRAL_VLLM_MODEL", raising=False)
    server = FakeVllmRealtime(text=UTTERANCE)
    pcm, _ = read_wav(AUDIO)

    async with realtime_service(server) as stt:
        warmed = len(server.audio)
        await collect(stt.stream(speech_then(FlushSignal()), "en"))

    # The repo id vLLM's own Voxtral recipe serves, and the default this service asks for.
    assert server.models == ["mistralai/Voxtral-Mini-4B-Realtime-2602"] * 2  # warm-up, then call
    assert server.generations == 2
    decoded = np.frombuffer(server.audio[warmed:], dtype="<i2").astype(int)
    spoken = np.frombuffer(pcm, dtype="<i2").astype(int)
    assert decoded.size == spoken.size
    assert np.abs(decoded - spoken).max() <= 1


async def test_the_flush_is_answered_with_the_servers_final_text_not_the_running_deltas() -> None:
    """transcription.done carries the generation's own text, which a model is free to have
    revised; the deltas are a running transcript for interims, not the answer to a flush."""
    corrected = "Hi, I need to speak to someone about my landlord, Mr Weber."

    async with realtime_service(FakeVllmRealtime(text=UTTERANCE, corrected=corrected)) as stt:
        stt.flush_timeout_s = 5.0
        events = await asyncio.wait_for(collect(stt.stream(speech_then(FlushSignal()), "en")), 10)

    assert [e.text for e in events if e.flushed] == [corrected]
    interims = [e.text for e in events if not e.is_final]
    assert max(len(text.split()) for text in interims) > 1  # deltas accumulate


async def test_a_flush_waits_for_the_answer_the_realtime_server_is_still_generating() -> None:
    """The server answers a commit when its generation ends, not when the audio does, and
    what it then sends is the final -- so the wait is the point of the flush timeout."""
    corrected = "Landlord dispute, Mrs Weber speaking."
    server = FakeVllmRealtime(text=UTTERANCE, corrected=corrected, answer_delay_s=0.2)

    async with realtime_service(server) as stt:
        stt.flush_timeout_s = 5.0
        events = await asyncio.wait_for(collect(stt.stream(speech_then(FlushSignal()), "en")), 10)

    assert [e.text for e in events if e.flushed] == [corrected]


async def test_a_realtime_server_that_answers_nothing_still_answers_the_flush_once() -> None:
    """A realtime server has no equivalent of an empty transcript: it simply says nothing
    until it has something. The caller is still owed exactly one answer to the flush."""
    async with realtime_service(FakeVllmRealtime(silent=True)) as stt:
        stt.flush_timeout_s = 5.0  # the service's own answer, not the client's fallback
        events = await asyncio.wait_for(collect(stt.stream(speech_then(FlushSignal()), "en")), 10)

    assert [(e.text, e.flushed) for e in events] == [("", True)]


async def test_a_realtime_server_that_fails_mid_utterance_does_not_answer_the_flush() -> None:
    """A server that dies mid-utterance has transcribed only part of what was said, and a
    final carrying that part is indistinguishable from a caller who stopped there."""
    events: list[TranscriptEvent] = []
    async with realtime_service(FakeVllmRealtime(text=UTTERANCE, fail_after_chunks=2)) as stt:
        stt.flush_timeout_s = 5.0
        with pytest.raises(Exception, match="close frame"):  # the service broke the socket
            async for event in stt.stream(speech_then(FlushSignal()), "en"):
                events.append(event)

    assert [e for e in events if e.flushed or e.is_final] == []


async def test_a_server_that_does_not_open_a_realtime_session_fails_the_utterance() -> None:
    """Something else answering on that port -- another vLLM endpoint, a proxy -- would
    otherwise take the audio and never transcribe it."""
    other = FastAPI()

    @other.websocket("/v1/realtime")
    async def greet(ws: WebSocket) -> None:
        await ws.accept()
        await ws.send_json({"type": "error", "error": "the model does not exist"})
        await asyncio.sleep(1.0)

    async with run_asgi(other) as host:
        session = VllmVoxtralTranscriber(
            url=f"ws://{host}/v1/realtime", flush_timeout_s=FLUSH_TIMEOUT_S
        ).session()
        session.feed(to_samples(WARMUP_PCM))
        with pytest.raises(RuntimeError, match="the model does not exist"):
            await asyncio.to_thread(session.finish)


async def test_a_realtime_server_that_accepts_and_says_nothing_gives_the_model_thread_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A socket that is open but never greets would otherwise hold the one model thread for
    as long as it stays open, and with it every call on this service."""
    monkeypatch.setattr(whisper_service, "VLLM_OPEN_TIMEOUT_S", 0.3)
    mute = FastAPI()

    @mute.websocket("/v1/realtime")
    async def accept_only(ws: WebSocket) -> None:
        await ws.accept()
        await asyncio.sleep(1.0)

    async with run_asgi(mute) as host:
        session = VllmVoxtralTranscriber(url=f"ws://{host}/v1/realtime").session()
        session.feed(to_samples(WARMUP_PCM))
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.to_thread(session.finish), 3)
        assert time.monotonic() - started < 2  # the session's own timeout, not this one's


async def test_the_deep_health_check_names_what_the_realtime_server_said() -> None:
    """The failure is a server away, so the operator needs the server's own words here; a
    bare 503 would send them to the wrong logs."""
    server = FakeVllmRealtime(text=UTTERANCE, fail_after_chunks=2)  # warm-up passes, this does not

    async with run_asgi(server.app()) as vllm:
        transcriber = VllmVoxtralTranscriber(
            url=f"ws://{vllm}/v1/realtime", flush_timeout_s=FLUSH_TIMEOUT_S
        )
        async with (
            run_asgi(create_app(lambda: transcriber)) as host,
            httpx.AsyncClient(base_url=f"http://{host}") as client,
        ):
            health = await client.get("/health/deep")

    assert health.status_code == 503
    assert "engine died" in health.json()["error"]


def test_a_realtime_server_that_never_finishes_the_handshake_gives_the_model_thread_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A socket that accepts and leaves the upgrade unanswered is the other way a wedged
    server takes the one model thread with it."""
    monkeypatch.setattr(whisper_service, "VLLM_OPEN_TIMEOUT_S", 0.3)
    with socket.socket() as listening:
        listening.bind(("127.0.0.1", 0))
        listening.listen(1)
        port = listening.getsockname()[1]
        session = VllmVoxtralTranscriber(url=f"ws://127.0.0.1:{port}/v1/realtime").session()
        session.feed(to_samples(WARMUP_PCM))
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            session.finish()

    assert time.monotonic() - started < 2  # the session's own timeout, not the suite's


def test_a_realtime_server_that_is_not_there_fails_the_utterance_rather_than_emptying_it() -> None:
    """The service's warm-up runs this path, so a GPU node whose vLLM server is missing
    refuses to start instead of answering every flush with silence."""
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        unused_port = taken.getsockname()[1]

    session = VllmVoxtralTranscriber(url=f"ws://127.0.0.1:{unused_port}/v1/realtime").session()
    session.feed(to_samples(WARMUP_PCM))
    with pytest.raises(OSError):
        session.finish()


@pytest.mark.parametrize(
    ("setting", "engine"),
    [
        (None, "faster-whisper"),
        ("mlx", "mlx"),
        ("voxtral", "voxtral"),
        ("voxtral-vllm", "voxtral"),
    ],
)
def test_the_environment_says_which_engine_the_service_loads(
    monkeypatch: pytest.MonkeyPatch, setting: str | None, engine: str
) -> None:
    monkeypatch.delenv("WHISPER_BACKEND", raising=False)
    if setting is not None:
        monkeypatch.setenv("WHISPER_BACKEND", setting)
    chosen = selected_transcriber()
    assert (chosen.backend, chosen.engine) == (setting or "faster-whisper", engine)


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
