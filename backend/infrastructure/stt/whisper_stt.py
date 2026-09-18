from __future__ import annotations

import json
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection

from backend.application.ports.stt_port import TranscriptEvent
from backend.infrastructure.stt.ws_stt_base import WebSocketStt


class WhisperStt(WebSocketStt):
    """Client for gpu/whisper_service. Protocol: binary PCM16 16 kHz frames in;
    {"type":"flush"} finalises; {"type":"transcript","text","is_final","flushed"} out."""

    def __init__(self, url: str) -> None:
        self._base = url

    def _url(self, language: str) -> str:
        return f"{self._base}?{urlencode({'language': language})}"

    async def _send_flush(self, ws: ClientConnection) -> None:
        await ws.send(json.dumps({"type": "flush"}))

    async def _send_close(self, ws: ClientConnection) -> None:
        await ws.send(json.dumps({"type": "close"}))

    def _parse(self, message: str | bytes) -> list[tuple[TranscriptEvent, bool]]:
        data = json.loads(message)
        if data.get("type") != "transcript":
            return []
        event = TranscriptEvent(data.get("text", ""), bool(data.get("is_final")))
        return [(event, bool(data.get("flushed")))]
