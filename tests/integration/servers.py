from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from websockets.asyncio.server import ServerConnection, serve

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@asynccontextmanager
async def run_asgi(app: Any) -> AsyncIterator[str]:
    """Serves an ASGI app on an ephemeral port for the duration of the block."""
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
    )
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            task.result()
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task


@dataclass
class FakeDeepgram:
    """Speaks Deepgram's live protocol from fixtures: one interim after the first audio,
    one final after 0.2 s of audio, and a from_finalize result per Finalize."""

    silent_on_finalize: bool = False
    audio_bytes: int = 0
    paths: list[str] = field(default_factory=list)
    auth: list[str] = field(default_factory=list)
    closed_cleanly: bool = False

    def __post_init__(self) -> None:
        self.messages = json.loads((FIXTURES / "deepgram" / "results.json").read_text())

    async def handler(self, ws: ServerConnection) -> None:
        self.paths.append(ws.request.path if ws.request else "")
        self.auth.append(ws.request.headers.get("Authorization", "") if ws.request else "")
        sent_interim = sent_final = False
        async for message in ws:
            if isinstance(message, bytes):
                self.audio_bytes += len(message)
                if not sent_interim:
                    sent_interim = True
                    await ws.send(json.dumps(self.messages["interim"]))
                if not sent_final and self.audio_bytes >= 6400:
                    sent_final = True
                    await ws.send(json.dumps(self.messages["final"]))
                continue
            command = json.loads(message)
            if command["type"] == "Finalize" and not self.silent_on_finalize:
                await ws.send(json.dumps(self.messages["finalize"]))
            elif command["type"] == "CloseStream":
                self.closed_cleanly = True
                await ws.send(json.dumps(self.messages["metadata"]))
                await ws.close()
                return

    @asynccontextmanager
    async def running(self) -> AsyncIterator[str]:
        async with serve(self.handler, "127.0.0.1", 0) as server:
            port = next(iter(server.sockets)).getsockname()[1]
            yield f"ws://127.0.0.1:{port}/v1/listen"
