from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from backend.domain.value_objects.audio import PCM16_24K_MONO, AudioChunk
from backend.infrastructure.tts.http_pcm_stream import stream_pcm


class KokoroTts:
    """Client for gpu/kokoro_service: POST /v1/synthesize streams raw 24 kHz PCM16."""

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url, timeout=httpx.Timeout(15.0, connect=2.0)
        )

    async def synthesize(self, text: str, voice: str | None) -> AsyncIterator[AudioChunk]:
        async with self._client.stream(
            "POST", "/v1/synthesize", json={"text": text, "voice": voice or "af_heart"}
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode(errors="replace")[:300]
                raise RuntimeError(f"Kokoro {response.status_code}: {body}")
            async for chunk in stream_pcm(response, PCM16_24K_MONO):
                yield chunk
