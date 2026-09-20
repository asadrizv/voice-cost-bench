"""Streaming STT over WebSocket, backed by Whisper or Voxtral.

Protocol (per call, one socket):
  -> binary frames: PCM16 LE, 16 kHz, mono
  -> {"type": "flush"}   finalise the current utterance
  -> {"type": "close"}
  <- {"type": "transcript", "text": str, "is_final": bool, "flushed": bool}

GET /v1/info -> {"engines": [{"id": str, "model": str}]}, what the loaded engine runs.

Whisper isn't a streaming model. Interims come from re-transcribing the growing utterance
buffer, and only when a worker is idle, so they never delay another call's final. Voxtral
decodes while the caller is still speaking instead, so an interim only reads off what its
session has already produced. Either way a flush is answered exactly once, empty text
included, and one engine's shape never reaches the socket.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CollectorRegistry, Gauge, Histogram, generate_latest
from websockets.sync.client import ClientConnection, connect

if TYPE_CHECKING:
    from mlx_audio.stt.streaming import StreamingSession

log = logging.getLogger("whisper_service")

SAMPLE_RATE = 16_000
MAX_UTTERANCE_S = 30.0  # Whisper's window; older audio is dropped, not silently truncated
VOICED_DBFS = -45.0
INTERIM_AFTER_S = 0.5

Work = Callable[[], str]
"""A unit of model work, run on the service's pool rather than the event loop."""

WARMUP_PCM = (
    ((0.1 * np.sin(np.linspace(0, 440 * 2 * np.pi, SAMPLE_RATE // 2))) * 32767)
    .astype("<i2")
    .tobytes()
)
"""Half a second of tone, loud enough to pass any engine's speech gate, so warm-up and the
deep health check reach the model by the same path a caller's audio does."""


MIN_AVG_LOGPROB = -1.0
"""Segments below this are dropped. Measured on large-v3-turbo: real speech scores about
-0.3, while the "what you" it invents from a breath scores about -2.8. Its no-speech
probability reads 0.0 for pure hiss, so it can't be used as the filter."""


def confident_text(segments: list[tuple[str, float]]) -> str:
    return " ".join(t.strip() for t, logprob in segments if logprob >= MIN_AVG_LOGPROB).strip()


def to_samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def carries_speech(samples: np.ndarray, floor_dbfs: float) -> bool:
    rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
    return 20 * np.log10(max(rms, 1e-6)) >= floor_dbfs


class Transcriber(Protocol):
    backend: str
    """The id WHISPER_BACKEND names to load this one. It is the engine id where the engine
    has a single runtime here, and `<engine>-<runtime>` where it has more than one."""
    engine: str
    """The engine's id in config/components.yaml, where /transparency names it from."""
    model: str

    def transcribe(self, audio: np.ndarray, language: str) -> str: ...


class FasterWhisperTranscriber:
    engine = backend = "faster-whisper"
    model = os.environ.get("WHISPER_MODEL", "large-v3-turbo")

    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model,
            device=os.environ.get("WHISPER_DEVICE", "cuda"),
            compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "float16"),
        )

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language=language,
            beam_size=1,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        return confident_text([(s.text, s.avg_logprob) for s in segments])


class MlxWhisperTranscriber:
    """Apple Silicon: same Whisper weights on the Mac's GPU via MLX. For the local demo;
    its latency says nothing about the L40S benchmark."""

    engine = backend = "mlx"
    model = os.environ.get("WHISPER_MLX_REPO", "mlx-community/whisper-large-v3-turbo")

    def __init__(self) -> None:
        import mlx_whisper

        self._transcribe = mlx_whisper.transcribe

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        result = self._transcribe(
            audio,
            path_or_hf_repo=self.model,
            language=language,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return confident_text([(s["text"], s["avg_logprob"]) for s in result["segments"]])


class TranscriptionSession(Protocol):
    """One utterance being decoded as it is spoken. `feed` is called from the event loop
    and must not block; `advance` and `finish` do the model's work and the service runs
    them one at a time on one thread. Nothing is called after `finish`."""

    def feed(self, audio: np.ndarray) -> None: ...

    def advance(self) -> str:
        """Everything decoded from the audio fed so far."""
        ...

    def finish(self) -> str:
        """End of the utterance: the text the caller is answered with."""
        ...


@runtime_checkable
class StreamingTranscriber(Protocol):
    backend: str
    engine: str
    model: str
    speech_floor_dbfs: float
    """Below this the service keeps audio off the model; see VoxtralTranscriber."""

    def session(self) -> TranscriptionSession: ...


VOXTRAL_DELAY_MS = 480
"""How much audio the encoder gathers before the decoder emits. Mistral's recommended
setting and the sweet spot between latency and word error rate; must be a multiple of 80 ms.
https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602#recommended-settings"""

VOXTRAL_DECODE_TOKENS = 16
"""Tokens per step before the model thread is handed back, so a flush queued behind an
interim waits for one step rather than a whole utterance."""


class VoxtralTranscriber:
    """Apple Silicon: Voxtral Mini 4B Realtime through mlx-audio, the only runtime that
    streams it without CUDA (the vLLM realtime backend is #32). It identifies the spoken
    language itself, so the socket's `language` is not passed on."""

    engine = backend = "voxtral"
    model = os.environ.get("VOXTRAL_MLX_REPO", "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit")

    speech_floor_dbfs = -60.0
    """Voxtral needs no confidence filter but it does need keeping off silence, and the
    threshold cannot be Whisper's. Measured on this quantisation, 16 kHz fixture audio:

    - it invents nothing, returning "" for digital silence, for white noise from -90 to
      -30 dBFS, for a 440 Hz tone and for low-passed room rumble. Whisper's avg-logprob
      filter has no counterpart here because there is nothing to filter;
    - it reads speech far below Whisper's -45 dBFS gate: fixture speech attenuated to a
      loudest 20 ms frame of -51 dBFS still transcribes word for word, and -58 dBFS
      partially, so Whisper's gate would drop quiet callers Voxtral hears;
    - below about -62 dBFS it returns "" for attenuated speech too;
    - a session fed nothing costs real time -- 15 s of silence took 9.8 s of the one MLX
      thread -- and a caller is silent for as long as the agent is talking.

    So the guard is the gate alone, at -60 dBFS: under everything Voxtral can still read,
    over the floor where it stops reading anyway."""

    def __init__(self) -> None:
        from mlx_audio.stt.utils import load

        self._model = load(self.model)

    def session(self) -> TranscriptionSession:
        return VoxtralSession(
            self._model.create_streaming_session(
                temperature=0.0, transcription_delay_ms=VOXTRAL_DELAY_MS
            )
        )


class VoxtralSession:
    """Accumulates mlx-audio's append-only deltas. Its session is spent once closed, so a
    flush ends this one and the next utterance opens another."""

    def __init__(self, session: StreamingSession) -> None:
        self._session = session
        self._text = ""

    def feed(self, audio: np.ndarray) -> None:
        self._session.feed(audio)

    def advance(self) -> str:
        while not self._session.done:
            deltas = self._session.step(max_decode_tokens=VOXTRAL_DECODE_TOKENS)
            if not deltas:
                break  # decoding has caught up with the audio fed so far
            self._text += "".join(deltas)
        return self._text.strip()

    def finish(self) -> str:
        self._session.close()
        while not self._session.done:
            self._text += "".join(self._session.step(max_decode_tokens=VOXTRAL_DECODE_TOKENS))
        return self._text.strip()


VLLM_OPEN_TIMEOUT_S = 10.0
VLLM_FLUSH_TIMEOUT_S = float(os.environ.get("VOXTRAL_VLLM_FLUSH_TIMEOUT_S", "3.0"))
"""How long a flush waits for the realtime server's transcription.done before answering
with the deltas it has. Without it a server that goes quiet would hold the caller's turn
open until the client's own flush timeout; with it the caller is always answered."""


class VllmVoxtralTranscriber:
    """CUDA: the same Voxtral realtime weights, served by a vLLM process on the GPU and
    reached over its realtime WebSocket, so nothing here imports CUDA or holds the card.
    vLLM needs >= 16 GiB for it and mistral_common >= 1.9; see gpu/README.md.

    The reported model names the runtime as well as the weights: which runtime served a
    benchmark run is not otherwise recoverable from provenance, since the catalogue's
    version is fixed configuration and the two runtimes share an engine id."""

    engine = "voxtral"
    backend = "voxtral-vllm"
    speech_floor_dbfs = VoxtralTranscriber.speech_floor_dbfs
    """The gate measured on the 4-bit MLX quantisation, reused unmeasured for bf16 on CUDA.
    It sits under the quietest speech that quantisation was measured to read, so it errs
    towards spending the GPU rather than dropping a quiet caller."""

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        flush_timeout_s: float = VLLM_FLUSH_TIMEOUT_S,
    ) -> None:
        self.served_model = model or os.environ.get(
            "VOXTRAL_VLLM_MODEL", "mistralai/Voxtral-Mini-4B-Realtime-2602"
        )
        self.model = f"{self.served_model} (vLLM realtime)"
        self._url = url or os.environ.get("VOXTRAL_VLLM_URL", "ws://127.0.0.1:8003/v1/realtime")
        self._flush_timeout_s = flush_timeout_s

    def session(self) -> TranscriptionSession:
        return VllmRealtimeSession(self._url, self.served_model, self._flush_timeout_s)


class VllmRealtimeSession:
    """One utterance on one connection to vLLM's realtime endpoint, whose protocol is:
    session.created on connect, session.update names the model, a commit starts a
    generation, input_audio_buffer.append carries base64 PCM16 at 16 kHz, transcription
    .delta streams the text, and a commit with final=true ends the generation with
    transcription.done.
    https://github.com/vllm-project/vllm/blob/main/docs/serving/online_serving/speech_to_text.md

    `feed` runs on the event loop, so it only buffers; the socket is opened, written and
    read on the service's model thread in `advance` and `finish`. A failed utterance is
    remembered: a flush must not answer with the text from before the server broke, which
    the caller cannot tell from a caller who stopped speaking there."""

    def __init__(self, url: str, model: str, flush_timeout_s: float) -> None:
        self._url = url
        self._model = model
        self._flush_timeout_s = flush_timeout_s
        self._pending: list[bytes] = []
        self._text = ""
        self._ws: ClientConnection | None = None
        self._open = contextlib.ExitStack()
        self._failure: Exception | None = None

    def feed(self, audio: np.ndarray) -> None:
        self._pending.append(to_pcm16(audio))

    def advance(self) -> str:
        self._run(lambda: self._drain_until(time.monotonic()))
        return self._text.strip()

    def finish(self) -> str:
        try:
            self._run(self._commit)
        finally:
            self._close()
        return self._text.strip()

    def _commit(self) -> None:
        self._send({"type": "input_audio_buffer.commit", "final": True})
        self._drain_until(time.monotonic() + self._flush_timeout_s)

    def _run(self, step: Callable[[], None]) -> None:
        if self._failure is not None:
            raise self._failure
        try:
            self._send_audio()
            step()
        except Exception as exc:
            self._failure = exc
            self._close()
            raise

    def _connection(self) -> ClientConnection:
        if self._ws is None:
            ws = self._open.enter_context(connect(self._url, open_timeout=VLLM_OPEN_TIMEOUT_S))
            self._ws = ws
            greeting = json.loads(ws.recv(timeout=VLLM_OPEN_TIMEOUT_S))
            if greeting.get("type") != "session.created":
                raise RuntimeError(f"vLLM realtime greeted with {greeting}")
            # The server refuses a commit until a session.update has named a model it
            # serves, which is also what tells a mismatched deployment from a silent one.
            self._send({"type": "session.update", "model": self._model})
            self._send({"type": "input_audio_buffer.commit"})
        return self._ws

    def _send(self, event: dict[str, object]) -> None:
        self._connection().send(json.dumps(event))

    def _send_audio(self) -> None:
        pending, self._pending = self._pending, []
        for pcm in pending:
            self._send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode(),
                }
            )

    def _drain_until(self, deadline: float) -> None:
        ws = self._connection()
        while True:
            try:
                message = ws.recv(timeout=max(deadline - time.monotonic(), 0.0))
            except TimeoutError:
                return
            event = json.loads(message)
            kind = event.get("type")
            if kind == "transcription.delta":
                self._text += str(event.get("delta", ""))
            elif kind == "transcription.done":
                self._text = str(event.get("text", self._text))
                return
            elif kind == "error":
                raise RuntimeError(f"vLLM realtime: {event.get('error')}")

    def _close(self) -> None:
        self._ws = None
        with contextlib.suppress(Exception):
            self._open.close()


TRANSCRIBERS: tuple[type[Transcriber] | type[StreamingTranscriber], ...] = (
    FasterWhisperTranscriber,
    MlxWhisperTranscriber,
    VoxtralTranscriber,
    VllmVoxtralTranscriber,
)
"""Every backend this service can run; the API's catalogue must name each engine behind
them. Two backends may share an engine id, one runtime each."""


def selected_transcriber() -> type[Transcriber] | type[StreamingTranscriber]:
    """The backend WHISPER_BACKEND names, refused here rather than quietly replaced: a typo
    would otherwise put a different engine behind a benchmark run and its provenance."""
    backend = os.environ.get("WHISPER_BACKEND", FasterWhisperTranscriber.backend)
    chosen = next((t for t in TRANSCRIBERS if t.backend == backend), None)
    if chosen is None:
        raise RuntimeError(
            f"WHISPER_BACKEND names {backend!r}; this service runs "
            f"{sorted(t.backend for t in TRANSCRIBERS)}"
        )
    return chosen


def default_transcriber() -> Transcriber | StreamingTranscriber:
    return selected_transcriber()()


class Utterance:
    """Audio since the last flush, with leading silence dropped. Whisper hallucinates on
    silence ("Thank you."), and the caller is silent for as long as the agent talks."""

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._samples = 0
        self.voiced = False

    def add(self, pcm: bytes) -> None:
        samples = to_samples(pcm)
        if not self.voiced:
            if not carries_speech(samples, VOICED_DBFS):
                return
            self.voiced = True
        self._chunks.append(samples)
        self._samples += samples.size
        while self._samples > MAX_UTTERANCE_S * SAMPLE_RATE and len(self._chunks) > 1:
            self._samples -= self._chunks.pop(0).size

    def audio(self) -> np.ndarray:
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)

    @property
    def seconds(self) -> float:
        return self._samples / SAMPLE_RATE


class Turn(Protocol):
    """One utterance on one socket. `add` runs on the event loop; the work it hands back
    runs on the service's pool."""

    def add(self, pcm: bytes) -> None: ...

    @property
    def ready(self) -> bool:
        """Whether enough speech has arrived for an interim to be worth asking for."""
        ...

    def interim(self) -> Work:
        """Only called once `ready`."""
        ...

    def final(self) -> Work | None:
        """None when the caller said nothing, so a flush is answered without the model."""
        ...


class BufferedTurn:
    """Whisper has no streaming mode: an interim and the final both transcribe the
    utterance buffer as it stands."""

    def __init__(self, transcribe: Callable[[np.ndarray, str], str], language: str) -> None:
        self._transcribe = transcribe
        self._language = language
        self._utterance = Utterance()

    def add(self, pcm: bytes) -> None:
        self._utterance.add(pcm)

    @property
    def ready(self) -> bool:
        return self._utterance.voiced and self._utterance.seconds >= INTERIM_AFTER_S

    def interim(self) -> Work:
        return self._work()

    def final(self) -> Work | None:
        return self._work() if self._utterance.voiced else None

    def _work(self) -> Work:
        audio = self._utterance.audio()
        return lambda: self._transcribe(audio, self._language)


class StreamingTurn:
    """Voxtral decodes while the caller is still speaking, so an interim reads off what
    its session has produced and the final closes the session and drains the rest. The
    session opens on the first frame that carries speech and not before, which is the
    engine's guard against spending the model on the silence while the agent talks."""

    def __init__(self, transcriber: StreamingTranscriber) -> None:
        self._transcriber = transcriber
        self._session: TranscriptionSession | None = None
        self._samples = 0

    def add(self, pcm: bytes) -> None:
        samples = to_samples(pcm)
        if self._session is None:
            if not carries_speech(samples, self._transcriber.speech_floor_dbfs):
                return
            self._session = self._transcriber.session()
        self._session.feed(samples)
        self._samples += samples.size

    @property
    def ready(self) -> bool:
        return self._samples >= INTERIM_AFTER_S * SAMPLE_RATE

    def interim(self) -> Work:
        return self._session.advance  # type: ignore[union-attr]  # `ready` opened it

    def final(self) -> Work | None:
        return self._session.finish if self._session is not None else None


def turns(transcriber: Transcriber | StreamingTranscriber) -> Callable[[str], Turn]:
    if isinstance(transcriber, StreamingTranscriber):
        return lambda language: StreamingTurn(transcriber)
    return lambda language: BufferedTurn(transcriber.transcribe, language)


def create_app(
    transcriber_factory: Callable[[], Transcriber | StreamingTranscriber] = default_transcriber,
    workers: int = int(os.environ.get("WHISPER_WORKERS", "3")),
    interim_interval_s: float = 0.5,
) -> FastAPI:
    registry = CollectorRegistry()
    in_flight = Gauge("whisper_in_flight", "Transcriptions running", registry=registry)
    waiting = Gauge("whisper_waiting", "Finals queued for a worker", registry=registry)
    wait_ms = Histogram(
        "whisper_queue_wait_ms",
        "Final wait for a worker",
        registry=registry,
        buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
    )
    run_ms = Histogram(
        "whisper_transcribe_ms",
        "Transcription time",
        ["kind"],
        registry=registry,
        buckets=(25, 50, 100, 150, 200, 300, 500, 750, 1000, 2000),
    )
    state: dict[str, object] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        transcriber = transcriber_factory()
        state["transcriber"] = transcriber
        new_turn = turns(transcriber)
        state["new_turn"] = new_turn
        # A streaming engine keeps decoder state between steps, so every session steps on
        # the one thread its model was built on. It also keeps a cancelled interim ordered
        # ahead of the flush behind it: cancelling releases the slot but not the thread.
        threads = 1 if isinstance(transcriber, StreamingTranscriber) else workers
        state["pool"] = ThreadPoolExecutor(max_workers=threads)
        state["slots"] = asyncio.Semaphore(threads)
        # Warm load: the first real call must not pay for kernel compilation. It runs on
        # the pool, like a call's work does, so an engine whose model answers over a socket
        # is not waiting on an event loop this is blocking.
        warm = new_turn("en")
        warm.add(WARMUP_PCM)
        if (work := warm.final()) is not None:
            await asyncio.get_running_loop().run_in_executor(state["pool"], work)  # type: ignore[arg-type]
        yield
        state["pool"].shutdown(wait=False)  # type: ignore[attr-defined]

    app = FastAPI(lifespan=lifespan)

    async def run(work: Work, kind: str) -> str:
        started = time.perf_counter()
        in_flight.inc()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(state["pool"], work)  # type: ignore[arg-type]
        finally:
            in_flight.dec()
            run_ms.labels(kind).observe((time.perf_counter() - started) * 1000)

    async def final(work: Work) -> str:
        slots: asyncio.Semaphore = state["slots"]  # type: ignore[assignment]
        queued = time.perf_counter()
        waiting.inc()
        async with slots:
            waiting.dec()
            wait_ms.observe((time.perf_counter() - queued) * 1000)
            return await run(work, "final")

    async def interim(work: Work) -> str | None:
        slots: asyncio.Semaphore = state["slots"]  # type: ignore[assignment]
        if slots.locked():
            return None  # a final needs the worker more than we need this interim
        async with slots:
            return await run(work, "interim")

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket, language: str = "en") -> None:
        await ws.accept()
        new_turn: Callable[[str], Turn] = state["new_turn"]  # type: ignore[assignment]
        turn = new_turn(language)
        interim_task: asyncio.Task[None] | None = None
        last_interim = time.monotonic()
        send_lock = asyncio.Lock()

        async def send(text: str, is_final: bool, flushed: bool) -> None:
            async with send_lock:
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "transcript",
                            "text": text,
                            "is_final": is_final,
                            "flushed": flushed,
                        }
                    )
                )

        async def emit_interim(work: Work) -> None:
            text = await interim(work)
            if text:
                await send(text, False, False)

        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    turn.add(message["bytes"])
                    idle = interim_task is None or interim_task.done()
                    due = time.monotonic() - last_interim >= interim_interval_s
                    if turn.ready and idle and due:
                        last_interim = time.monotonic()
                        interim_task = asyncio.create_task(emit_interim(turn.interim()))
                    continue
                command = json.loads(message.get("text") or "{}")
                if command.get("type") == "flush":
                    if interim_task is not None:
                        interim_task.cancel()
                    work = turn.final()
                    turn = new_turn(language)
                    await send(await final(work) if work else "", True, True)
                elif command.get("type") == "close":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            if interim_task is not None:
                interim_task.cancel()
        with contextlib.suppress(RuntimeError):  # already closed by the client
            await ws.close()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/info")
    async def info() -> dict[str, list[dict[str, str]]]:
        transcriber: Transcriber = state["transcriber"]  # type: ignore[assignment]
        return {"engines": [{"id": transcriber.engine, "model": transcriber.model}]}

    @app.get("/health/deep")
    async def deep() -> JSONResponse:
        """Runs the model. Liveness that doesn't would stay green with the model unloaded."""
        started = time.perf_counter()
        new_turn: Callable[[str], Turn] = state["new_turn"]  # type: ignore[assignment]
        turn = new_turn("en")
        turn.add(WARMUP_PCM)
        try:
            work = turn.final()
            if work is None:
                raise RuntimeError("the warm-up tone did not reach the model")
            await final(work)
        except Exception as exc:
            return JSONResponse({"status": "error", "error": str(exc)}, status_code=503)
        return JSONResponse(
            {"status": "ok", "ms": round((time.perf_counter() - started) * 1000, 1)}
        )

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(registry).decode())

    return app


def app() -> FastAPI:
    return create_app()
