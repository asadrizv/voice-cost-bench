"""Streaming TTS over HTTP, backed by Kokoro.

POST /v1/synthesize {"text": str, "voice": str, "speed": float}
  -> 200, body streams raw PCM16 LE, 24 kHz, mono, one segment at a time.

Kokoro doesn't batch across requests, so synthesis runs on a bounded pool; beyond it
requests queue here, visibly (kokoro_waiting), instead of inside the GPU.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Protocol

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CollectorRegistry, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

log = logging.getLogger("kokoro_service")
SAMPLE_RATE = 24_000


class Synthesizer(Protocol):
    def synthesize(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        """Yields float32 mono 24 kHz segments."""
        ...


class KokoroSynthesizer:
    def __init__(self) -> None:
        from kokoro import KPipeline

        self._pipelines: dict[str, object] = {}
        self._factory = KPipeline
        self._lock = threading.Lock()
        self._pipeline("a")

    def _pipeline(self, lang_code: str):  # type: ignore[no-untyped-def]
        with self._lock:
            if lang_code not in self._pipelines:
                self._pipelines[lang_code] = self._factory(
                    lang_code=lang_code, device=os.environ.get("KOKORO_DEVICE") or None
                )
            return self._pipelines[lang_code]

    def synthesize(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        # Kokoro voice ids start with their language code: af_heart -> "a" (US English).
        pipeline = self._pipeline(voice[0])
        for _, _, audio in pipeline(text, voice=voice, speed=speed):  # type: ignore[operator]
            yield np.asarray(audio, dtype=np.float32)


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    voice: str = "af_heart"
    speed: float = Field(1.0, ge=0.5, le=2.0)


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def create_app(
    synthesizer_factory: Callable[[], Synthesizer] = KokoroSynthesizer,
    workers: int = int(os.environ.get("KOKORO_WORKERS", "3")),
) -> FastAPI:
    registry = CollectorRegistry()
    waiting = Gauge("kokoro_waiting", "Requests waiting for a worker", registry=registry)
    in_flight = Gauge("kokoro_in_flight", "Syntheses running", registry=registry)
    first_byte_ms = Histogram(
        "kokoro_first_byte_ms",
        "Request to first audio",
        registry=registry,
        buckets=(10, 25, 50, 75, 100, 150, 200, 300, 500, 1000),
    )
    state: dict[str, object] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        synthesizer = synthesizer_factory()
        state["synth"] = synthesizer
        state["pool"] = ThreadPoolExecutor(max_workers=workers)
        state["slots"] = asyncio.Semaphore(workers)
        for _ in synthesizer.synthesize("Warming up.", "af_heart", 1.0):
            pass
        yield
        state["pool"].shutdown(wait=False)  # type: ignore[attr-defined]

    app = FastAPI(lifespan=lifespan)

    async def segments(req: SynthesizeRequest) -> AsyncIterator[bytes]:
        synth: Synthesizer = state["synth"]  # type: ignore[assignment]
        slots: asyncio.Semaphore = state["slots"]  # type: ignore[assignment]
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        waiting.inc()
        async with slots:
            waiting.dec()
            in_flight.inc()
            queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()

            def produce() -> None:
                try:
                    for audio in synth.synthesize(req.text, req.voice, req.speed):
                        loop.call_soon_threadsafe(queue.put_nowait, to_pcm16(audio))
                except BaseException as exc:  # surfaced to the async side below
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            loop.run_in_executor(state["pool"], produce)  # type: ignore[arg-type]
            first = True
            try:
                while (item := await queue.get()) is not None:
                    if isinstance(item, BaseException):
                        raise item
                    if first:
                        first_byte_ms.observe((time.perf_counter() - started) * 1000)
                        first = False
                    yield item
            finally:
                in_flight.dec()

    @app.post("/v1/synthesize", response_model=None)
    async def synthesize(req: SynthesizeRequest) -> StreamingResponse | JSONResponse:
        # Pull the first segment before committing to a 200: once headers are sent, a
        # model failure can only truncate the stream, which a client reads as silence.
        stream = segments(req)
        try:
            first = await anext(stream)
        except StopAsyncIteration:
            return JSONResponse({"error": "synthesised no audio"}, status_code=500)
        except Exception as exc:
            log.exception("synthesis failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

        async def body() -> AsyncIterator[bytes]:
            yield first
            async for chunk in stream:
                yield chunk

        return StreamingResponse(
            body(),
            media_type="audio/pcm",
            headers={"x-sample-rate": str(SAMPLE_RATE), "x-channels": "1"},
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/deep")
    async def deep() -> JSONResponse:
        started = time.perf_counter()
        try:
            total = 0
            async for chunk in segments(SynthesizeRequest(text="Hello.")):
                total += len(chunk)
            if total == 0:
                raise RuntimeError("synthesised no audio")
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
