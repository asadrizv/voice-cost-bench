from __future__ import annotations

import json
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection

from backend.application.ports.stt_port import TranscriptEvent
from backend.infrastructure.stt.ws_stt_base import WebSocketStt

DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen"


class DeepgramStt(WebSocketStt):
    """Deepgram streaming. Their own endpointing is switched off: turn-taking is ours, so
    both pipelines are measured with the same endpointer. Billed per streamed audio second,
    silence included."""

    def __init__(self, api_key: str, model: str = "nova-3", url: str = DEEPGRAM_URL) -> None:
        self._api_key = api_key
        self._model = model
        self._base = url

    def _url(self, language: str) -> str:
        params = {
            "model": self._model,
            "language": language,
            "encoding": "linear16",
            "sample_rate": 16000,
            "channels": 1,
            "interim_results": "true",
            "punctuate": "true",
            "smart_format": "true",
            "endpointing": "false",
        }
        return f"{self._base}?{urlencode(params)}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self._api_key}"}

    async def _send_flush(self, ws: ClientConnection) -> None:
        await ws.send(json.dumps({"type": "Finalize"}))

    async def _send_close(self, ws: ClientConnection) -> None:
        await ws.send(json.dumps({"type": "CloseStream"}))

    def _parse(self, message: str | bytes) -> list[tuple[TranscriptEvent, bool]]:
        data = json.loads(message)
        if data.get("type") != "Results":
            return []
        alternatives = data.get("channel", {}).get("alternatives") or [{}]
        text = alternatives[0].get("transcript", "")
        is_final = bool(data.get("is_final"))
        return [(TranscriptEvent(text, is_final), bool(data.get("from_finalize")))]
