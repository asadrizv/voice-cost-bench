from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from backend.domain.value_objects.audio import PCM16_24K_MONO, AudioChunk
from backend.infrastructure.tts.http_pcm_stream import stream_pcm

ELEVENLABS_URL = "https://api.elevenlabs.io"
DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"


class ElevenLabsTts:
    """HTTP streaming TTS, one request per sentence, raw 24 kHz PCM so both pipelines hand
    the transport the same format."""

    def __init__(
        self,
        api_key: str,
        model: str = "eleven_flash_v2_5",
        base_url: str = ELEVENLABS_URL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client or httpx.AsyncClient(
            base_url=base_url, timeout=httpx.Timeout(15.0, connect=5.0)
        )

    async def synthesize(self, text: str, voice: str | None) -> AsyncIterator[AudioChunk]:
        async with self._client.stream(
            "POST",
            f"/v1/text-to-speech/{voice or DEFAULT_VOICE}/stream",
            params={"output_format": "pcm_24000"},
            headers={"xi-api-key": self._api_key},
            json={"text": text, "model_id": self._model},
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode(errors="replace")[:300]
                raise RuntimeError(f"ElevenLabs {response.status_code}: {body}")
            async for chunk in stream_pcm(response, PCM16_24K_MONO):
                yield chunk
