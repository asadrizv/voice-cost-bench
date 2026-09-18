from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from websockets.asyncio.client import ClientConnection, connect

from backend.application.ports.stt_port import FlushSignal, TranscriptEvent
from backend.domain.value_objects.audio import PCM16_16K_MONO, AudioChunk

_END = object()


class WebSocketStt(ABC):
    """Shared plumbing for streaming STT over one WebSocket per call: a sender task pushes
    audio and flush requests while the generator yields transcripts as they arrive.

    Guarantees the port contract that every FlushSignal is answered: if the server stays
    silent past `flush_timeout_s` (Deepgram does when nothing was buffered), an empty
    flushed final is synthesised so the call never stalls waiting on it.
    """

    flush_timeout_s = 0.8

    @abstractmethod
    def _url(self, language: str) -> str: ...

    def _headers(self) -> dict[str, str]:
        return {}

    @abstractmethod
    async def _send_flush(self, ws: ClientConnection) -> None: ...

    @abstractmethod
    async def _send_close(self, ws: ClientConnection) -> None: ...

    @abstractmethod
    def _parse(self, message: str | bytes) -> list[tuple[TranscriptEvent, bool]]:
        """Events in `message`, each paired with whether it answers a flush."""

    async def _connect(self, language: str) -> ClientConnection:
        return await connect(
            self._url(language), additional_headers=self._headers(), max_size=2**22
        )

    async def stream(
        self, audio: AsyncIterable[AudioChunk | FlushSignal], language: str
    ) -> AsyncIterator[TranscriptEvent]:
        out: asyncio.Queue[Any] = asyncio.Queue()
        pending: list[asyncio.TimerHandle] = []
        loop = asyncio.get_running_loop()

        def on_flush_timeout(handle_ref: list[asyncio.TimerHandle]) -> None:
            if handle_ref and handle_ref[0] in pending:
                pending.remove(handle_ref[0])
                out.put_nowait(TranscriptEvent("", is_final=True, flushed=True))

        async with await self._connect(language) as ws:

            async def send() -> None:
                try:
                    async for item in audio:
                        if isinstance(item, FlushSignal):
                            ref: list[asyncio.TimerHandle] = []
                            ref.append(loop.call_later(self.flush_timeout_s, on_flush_timeout, ref))
                            pending.append(ref[0])
                            await self._send_flush(ws)
                        else:
                            if item.format != PCM16_16K_MONO:
                                raise ValueError(f"STT expects 16 kHz mono, got {item.format}")
                            await ws.send(item.data)
                    await self._send_close(ws)
                except Exception as exc:
                    out.put_nowait(exc)

            async def receive() -> None:
                try:
                    async for message in ws:
                        for event, answers_flush in self._parse(message):
                            if answers_flush and pending:
                                pending.pop(0).cancel()
                                event = TranscriptEvent(event.text, True, flushed=True)
                            out.put_nowait(event)
                except Exception as exc:
                    out.put_nowait(exc)
                finally:
                    out.put_nowait(_END)

            sender = asyncio.create_task(send())
            receiver = asyncio.create_task(receive())
            try:
                while (item := await out.get()) is not _END:
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                for handle in pending:
                    handle.cancel()
                for task in (sender, receiver):
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
