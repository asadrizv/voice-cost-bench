"""Streaming TTS over HTTP, backed by Kokoro and Qwen3-TTS.

POST /v1/synthesize {"text": str, "voice": str, "speed": float}
  -> 200, body streams raw PCM16 LE, 24 kHz, mono, one segment at a time.

GET /v1/info -> {"engines": [{"id": str, "model": str}]}, every engine synthesis can route to.

The voice picks the engine: `<engine>:<voice>` goes to that engine, a bare id to Kokoro.
Kokoro doesn't batch across requests, so synthesis runs on a bounded pool; beyond it
requests queue here, visibly (kokoro_waiting), instead of inside the GPU.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CollectorRegistry, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

log = logging.getLogger("kokoro_service")
SAMPLE_RATE = 24_000
ENGINE_SEPARATOR = ":"
WARMUP_TEXT = "Warming up."


class UnknownVoice(ValueError):
    """No loaded engine speaks the requested voice."""


class Synthesizer(Protocol):
    engine: str
    """The engine's id in config/components.yaml, where /transparency names it from."""
    model: str
    warmup_voice: str
    """A voice this engine speaks, used to ready it before the first caller arrives."""

    def speaks(self, voice: str) -> bool:
        """Whether this engine can say `voice`, asked before any audio is promised."""
        ...

    def synthesize(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        """Yields float32 mono 24 kHz segments."""
        ...


class KokoroSynthesizer:
    engine = "kokoro"
    model = "hexgrad/Kokoro-82M"
    warmup_voice = "af_heart"

    def __init__(self) -> None:
        from kokoro import KPipeline

        self._pipelines: dict[str, object] = {}
        self._factory = KPipeline
        self._lock = threading.Lock()
        self._pipeline("a")

    def speaks(self, voice: str) -> bool:
        # Kokoro's voice packs are fetched by name on first use, so the set it can say is
        # not known up front; a name it has no pack for fails when synthesis starts.
        return bool(voice)

    def _pipeline(self, lang_code: str):  # type: ignore[no-untyped-def]
        with self._lock:
            if lang_code not in self._pipelines:
                self._pipelines[lang_code] = self._factory(
                    lang_code=lang_code,
                    repo_id=self.model,
                    device=os.environ.get("KOKORO_DEVICE") or None,
                )
            return self._pipelines[lang_code]

    def synthesize(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        # Kokoro voice ids start with their language code: af_heart -> "a" (US English).
        pipeline = self._pipeline(voice[0])
        for _, _, audio in pipeline(text, voice=voice, speed=speed):  # type: ignore[operator]
            yield np.asarray(audio, dtype=np.float32)


VOICES_PATH = Path(os.environ.get("QWEN3_TTS_VOICES") or Path(__file__).with_name("voices.yaml"))
STREAMING_INTERVAL_S = 0.5
"""Seconds of audio per streamed chunk. Below this mlx-audio's per-chunk vocoder overhead
starts to cost more than the earlier first chunk saves."""


class Qwen3TtsSynthesizer:
    """Apple Silicon: Qwen3-TTS VoiceDesign through mlx-audio, which is the only Qwen3-TTS
    runtime that streams (the official qwen-tts package does not, and pins transformers
    4.57). Voices are written descriptions in voices.yaml, so no recording is cloned. The
    CUDA backend is a separate engine; see gpu/README.md."""

    engine = "qwen3-tts"
    model = os.environ.get("QWEN3_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit")

    def __init__(self) -> None:
        from mlx_audio.tts.utils import load_model

        self._voices = load_voices(VOICES_PATH)
        self.warmup_voice = next(iter(self._voices))
        self._model = load_model(self.model)
        if self._model.sample_rate != SAMPLE_RATE:
            raise RuntimeError(
                f"{self.model} synthesises at {self._model.sample_rate} Hz; "
                f"this service's stream and its x-sample-rate header say {SAMPLE_RATE}"
            )

    def speaks(self, voice: str) -> bool:
        return voice in self._voices

    def synthesize(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        design = self._voices[voice]
        for result in self._model.generate(
            text=text,
            instruct=design["description"],
            lang_code=design["language"],
            speed=speed,
            stream=True,
            streaming_interval=STREAMING_INTERVAL_S,
        ):
            yield np.asarray(result.audio, dtype=np.float32)


def load_voices(path: Path) -> dict[str, dict[str, str]]:
    raw = yaml.safe_load(path.read_text())
    voices = {
        str(name): {"language": str(v["language"]), "description": str(v["description"])}
        for name, v in raw["voices"].items()
    }
    if not voices:
        raise RuntimeError(f"{path} defines no voices")
    return voices


SYNTHESIZERS: tuple[type[Synthesizer], ...] = (KokoroSynthesizer, Qwen3TtsSynthesizer)
"""Every engine this service can run; the API's catalogue must name each one."""


def enabled_synthesizers() -> tuple[type[Synthesizer], ...]:
    """TTS_ENGINES says which of SYNTHESIZERS this process loads, Kokoro alone by default:
    a deployment that needs no German never downloads Qwen3-TTS's several GB of weights."""
    by_id = {s.engine: s for s in SYNTHESIZERS}
    wanted = [e.strip() for e in os.environ.get("TTS_ENGINES", KokoroSynthesizer.engine).split(",")]
    unknown = [e for e in wanted if e not in by_id]
    if unknown:
        raise RuntimeError(f"TTS_ENGINES names {unknown}; this service runs {sorted(by_id)}")
    return tuple(by_id[e] for e in wanted)


def route(voice: str, engines: Sequence[str]) -> tuple[str, str]:
    """Splits a voice id into the engine that speaks it and the name that engine knows it
    by. A bare id belongs to the default engine, which is the first one loaded."""
    engine, separator, name = voice.partition(ENGINE_SEPARATOR)
    if not separator:
        return engines[0], voice
    if engine not in engines:
        raise UnknownVoice(f"voice {voice!r}: this service runs no engine {engine!r}")
    return engine, name


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    voice: str = "af_heart"
    speed: float = Field(1.0, ge=0.5, le=2.0)


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def create_app(
    engines: Sequence[type[Synthesizer]] | None = None,
    workers: int = int(os.environ.get("KOKORO_WORKERS", "3")),
) -> FastAPI:
    factories = enabled_synthesizers() if engines is None else tuple(engines)
    registry = CollectorRegistry()
    waiting = Gauge("kokoro_waiting", "Requests waiting for a worker", registry=registry)
    in_flight = Gauge("kokoro_in_flight", "Syntheses running", registry=registry)
    first_byte_ms = Histogram(
        "kokoro_first_byte_ms",
        "Request to first audio",
        ["engine"],
        registry=registry,
        buckets=(10, 25, 50, 75, 100, 150, 200, 300, 500, 1000),
    )
    synthesizers: dict[str, Synthesizer] = {}
    state: dict[str, object] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        state["pool"] = ThreadPoolExecutor(max_workers=workers)
        state["slots"] = asyncio.Semaphore(workers)
        for factory in factories:
            synth = synthesizers.setdefault(factory.engine, factory())
            for _ in synth.synthesize(WARMUP_TEXT, synth.warmup_voice, 1.0):
                pass
        yield
        state["pool"].shutdown(wait=False)  # type: ignore[attr-defined]

    app = FastAPI(lifespan=lifespan)

    async def segments(req: SynthesizeRequest) -> AsyncIterator[bytes]:
        engine, voice = route(req.voice, [f.engine for f in factories])
        synth = synthesizers[engine]
        if not synth.speaks(voice):
            raise UnknownVoice(f"voice {req.voice!r}: {engine} speaks no voice {voice!r}")
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
                    for audio in synth.synthesize(req.text, voice, req.speed):
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
                        first_byte_ms.labels(engine).observe((time.perf_counter() - started) * 1000)
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
        except UnknownVoice as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
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

    @app.get("/v1/info")
    async def info() -> dict[str, list[dict[str, str]]]:
        return {"engines": [{"id": f.engine, "model": f.model} for f in factories]}

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
