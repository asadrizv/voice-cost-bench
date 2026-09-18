"""TtsPort contract against ElevenLabs (mock transport) and Kokoro (real service code,
stub synthesizer)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import httpx
import numpy as np
import pytest

from backend.application.ports.tts_port import TtsPort
from backend.domain.value_objects.audio import PCM16_24K_MONO
from backend.infrastructure.tts.elevenlabs_tts import ElevenLabsTts
from backend.infrastructure.tts.kokoro_tts import KokoroTts
from gpu.kokoro_service.app import create_app, to_pcm16
from tests.integration.servers import run_asgi

SECONDS_PER_CHAR = 0.02


def fake_audio(text: str) -> np.ndarray:
    n = int(24000 * SECONDS_PER_CHAR * len(text))
    return (0.3 * np.sin(np.arange(n) * 2 * np.pi * 220 / 24000)).astype(np.float32)


class StubSynth:
    def synthesize(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        if text == "FAIL":
            raise RuntimeError("model crashed")
        audio = fake_audio(text)
        half = len(audio) // 2
        yield audio[:half]
        yield audio[half:]


def elevenlabs_handler(requests: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        text = json.loads(request.content)["text"]
        if text == "FAIL":
            return httpx.Response(401, json={"detail": {"status": "invalid_api_key"}})
        pcm = to_pcm16(fake_audio(text))

        async def body():  # type: ignore[no-untyped-def]
            # Odd-sized network chunks: the adapter must re-align to whole samples.
            for i in range(0, len(pcm), 777):
                yield pcm[i : i + 777]

        return httpx.Response(200, content=body())

    return handle


@asynccontextmanager
async def elevenlabs() -> AsyncIterator[TtsPort]:
    client = httpx.AsyncClient(
        base_url="https://api.elevenlabs.io", transport=httpx.MockTransport(elevenlabs_handler([]))
    )
    yield ElevenLabsTts("xi-test", client=client)


@asynccontextmanager
async def kokoro() -> AsyncIterator[TtsPort]:
    async with run_asgi(create_app(StubSynth, workers=2)) as host:
        yield KokoroTts(f"http://{host}")


IMPLEMENTATIONS: dict[str, Callable[[], AbstractAsyncContextManager[TtsPort]]] = {
    "elevenlabs": elevenlabs,
    "kokoro": kokoro,
}


@pytest.fixture(params=list(IMPLEMENTATIONS))
async def tts(request: pytest.FixtureRequest) -> AsyncIterator[TtsPort]:
    async with IMPLEMENTATIONS[request.param]() as impl:
        yield impl


async def test_streams_whole_sample_24k_pcm_of_the_right_length(tts: TtsPort) -> None:
    text = "Thank you, Ms. Weber."
    chunks = [c async for c in tts.synthesize(text, None)]
    assert len(chunks) > 1
    assert all(c.format == PCM16_24K_MONO and len(c.data) % 2 == 0 for c in chunks)
    total = b"".join(c.data for c in chunks)
    assert total == to_pcm16(fake_audio(text))


async def test_failure_raises(tts: TtsPort) -> None:
    with pytest.raises(Exception, match="401|model crashed|500"):
        _ = [c async for c in tts.synthesize("FAIL", None)]


async def test_elevenlabs_request_shape() -> None:
    requests: list[httpx.Request] = []
    client = httpx.AsyncClient(
        base_url="https://api.elevenlabs.io",
        transport=httpx.MockTransport(elevenlabs_handler(requests)),
    )
    tts = ElevenLabsTts("xi-test", model="eleven_flash_v2_5", client=client)
    _ = [c async for c in tts.synthesize("Hello.", "voice-123")]
    req = requests[0]
    assert req.url.path == "/v1/text-to-speech/voice-123/stream"
    assert req.url.params["output_format"] == "pcm_24000"
    assert req.headers["xi-api-key"] == "xi-test"
    assert json.loads(req.content) == {"text": "Hello.", "model_id": "eleven_flash_v2_5"}


async def test_kokoro_service_health_and_metrics() -> None:
    async with run_asgi(create_app(StubSynth, workers=1)) as host:
        async with httpx.AsyncClient(base_url=f"http://{host}") as client:
            assert (await client.get("/health/deep")).json()["status"] == "ok"
            metrics = (await client.get("/metrics")).text
    assert "kokoro_waiting" in metrics and "kokoro_first_byte_ms" in metrics
