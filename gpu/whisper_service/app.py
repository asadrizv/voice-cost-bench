"""Streaming STT over WebSocket, backed by faster-whisper.

Protocol (per call, one socket):
  -> binary frames: PCM16 LE, 16 kHz, mono
  -> {"type": "flush"}   finalise the current utterance
  -> {"type": "close"}
  <- {"type": "transcript", "text": str, "is_final": bool, "flushed": bool}

Whisper isn't a streaming model. Interims come from re-transcribing the growing utterance
buffer, and only when a worker is idle, so they never delay another call's final. Every
flush is answered exactly once, empty text included.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Protocol

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CollectorRegistry, Gauge, Histogram, generate_latest

log = logging.getLogger("whisper_service")

SAMPLE_RATE = 16_000
MAX_UTTERANCE_S = 30.0  # Whisper's window; older audio is dropped, not silently truncated
VOICED_DBFS = -45.0


MIN_AVG_LOGPROB = -1.0
"""Segments below this are dropped. Measured on large-v3-turbo: real speech scores about
-0.3, while the "what you" it invents from a breath scores about -2.8. Its no-speech
probability reads 0.0 for pure hiss, so it can't be used as the filter."""


def confident_text(segments: list[tuple[str, float]]) -> str:
    return " ".join(t.strip() for t, logprob in segments if logprob >= MIN_AVG_LOGPROB).strip()


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, language: str) -> str: ...


class FasterWhisperTranscriber:
    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            os.environ.get("WHISPER_MODEL", "large-v3-turbo"),
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

    def __init__(self) -> None:
        import mlx_whisper

        self._transcribe = mlx_whisper.transcribe
        self._repo = os.environ.get("WHISPER_MLX_REPO", "mlx-community/whisper-large-v3-turbo")

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        result = self._transcribe(
            audio,
            path_or_hf_repo=self._repo,
            language=language,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return confident_text([(s["text"], s["avg_logprob"]) for s in result["segments"]])


def default_transcriber() -> Transcriber:
    if os.environ.get("WHISPER_BACKEND", "faster-whisper") == "mlx":
        return MlxWhisperTranscriber()
    return FasterWhisperTranscriber()


class Utterance:
    """Audio since the last flush, with leading silence dropped. Whisper hallucinates on
    silence ("Thank you."), and the caller is silent for as long as the agent talks."""

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._samples = 0
        self.voiced = False

    def add(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if not self.voiced:
            rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
            if 20 * np.log10(max(rms, 1e-6)) < VOICED_DBFS:
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


def create_app(
    transcriber_factory: Callable[[], Transcriber] = default_transcriber,
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
        state["pool"] = ThreadPoolExecutor(max_workers=workers)
        state["slots"] = asyncio.Semaphore(workers)
        # Warm load: the first real call must not pay for CUDA kernel compilation.
        transcriber.transcribe(np.zeros(SAMPLE_RATE, np.float32), "en")
        yield
        state["pool"].shutdown(wait=False)  # type: ignore[attr-defined]

    app = FastAPI(lifespan=lifespan)

    async def run(audio: np.ndarray, language: str, kind: str) -> str:
        transcriber: Transcriber = state["transcriber"]  # type: ignore[assignment]
        started = time.perf_counter()
        in_flight.inc()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                state["pool"],
                transcriber.transcribe,
                audio,
                language,  # type: ignore[arg-type]
            )
        finally:
            in_flight.dec()
            run_ms.labels(kind).observe((time.perf_counter() - started) * 1000)

    async def final(audio: np.ndarray, language: str) -> str:
        slots: asyncio.Semaphore = state["slots"]  # type: ignore[assignment]
        queued = time.perf_counter()
        waiting.inc()
        async with slots:
            waiting.dec()
            wait_ms.observe((time.perf_counter() - queued) * 1000)
            return await run(audio, language, "final")

    async def interim(audio: np.ndarray, language: str) -> str | None:
        slots: asyncio.Semaphore = state["slots"]  # type: ignore[assignment]
        if slots.locked():
            return None  # a final needs the worker more than we need this interim
        async with slots:
            return await run(audio, language, "interim")

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket, language: str = "en") -> None:
        await ws.accept()
        utterance = Utterance()
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

        async def emit_interim(audio: np.ndarray) -> None:
            text = await interim(audio, language)
            if text:
                await send(text, False, False)

        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    utterance.add(message["bytes"])
                    idle = interim_task is None or interim_task.done()
                    due = time.monotonic() - last_interim >= interim_interval_s
                    if utterance.voiced and utterance.seconds >= 0.5 and idle and due:
                        last_interim = time.monotonic()
                        interim_task = asyncio.create_task(emit_interim(utterance.audio()))
                    continue
                command = json.loads(message.get("text") or "{}")
                if command.get("type") == "flush":
                    if interim_task is not None:
                        interim_task.cancel()
                    text = ""
                    if utterance.voiced:
                        text = await final(utterance.audio(), language)
                    utterance = Utterance()
                    await send(text, True, True)
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

    @app.get("/health/deep")
    async def deep() -> JSONResponse:
        """Runs the model. Liveness that doesn't would stay green with the model unloaded."""
        started = time.perf_counter()
        tone = 0.1 * np.sin(np.linspace(0, 440 * 2 * np.pi, SAMPLE_RATE // 2)).astype(np.float32)
        try:
            await final(tone, "en")
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
