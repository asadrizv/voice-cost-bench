from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
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


@dataclass
class FakeVllmRealtime:
    """Speaks vLLM's realtime transcription protocol, as its docs and server describe it:
    session.created on accept, session.update names the model, a commit starts a generation
    which streams transcription.delta as audio arrives, and a commit with final=true ends
    that generation with transcription.done.
    https://github.com/vllm-project/vllm/blob/main/docs/serving/online_serving/speech_to_text.md
    https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/speech_to_text/realtime/connection.py

    It answers one word of `text` per audio chunk, as a streaming model decodes what it has
    heard so far, and the rest when the generation ends."""

    text: str = ""
    corrected: str = ""
    """What transcription.done carries when the generation's final text differs from the
    deltas it streamed, as a model that revises its running transcript leaves it."""
    answer_delay_s: float = 0.0
    """How long the generation takes to finish after the audio ends."""
    silent: bool = False
    """Accepts audio and answers nothing, as a server with no speech to report does."""
    fail_after_chunks: int | None = None
    """Emits an error event instead of the next delta, as a dying engine does."""

    models: list[str] = field(default_factory=list)
    audio: bytearray = field(default_factory=bytearray)
    generations: int = 0

    def app(self) -> FastAPI:
        app = FastAPI()

        @app.websocket("/v1/realtime")
        async def realtime(ws: WebSocket) -> None:
            await ws.accept()
            await ws.send_json({"type": "session.created", "id": "sess-fake", "created": 0})
            words, said, chunks, generating = self.text.split(), 0, 0, False
            while True:
                try:
                    event = await ws.receive_json()
                except (WebSocketDisconnect, RuntimeError):
                    return
                kind = event.get("type")
                if kind == "session.update":
                    self.models.append(event.get("model"))
                elif kind == "input_audio_buffer.append":
                    self.audio += base64.b64decode(event["audio"])
                    chunks += 1
                    if chunks == self.fail_after_chunks:
                        await ws.send_json({"type": "error", "error": "engine died"})
                    elif generating and not self.silent and said < len(words):
                        await ws.send_json(
                            {"type": "transcription.delta", "delta": _spoken(words, said)}
                        )
                        said += 1
                elif kind == "input_audio_buffer.commit":
                    if not self.models:
                        await ws.send_json({"type": "error", "error": "model not validated"})
                    elif not event.get("final"):
                        generating, said = True, 0
                        self.generations += 1
                    elif not self.silent:
                        for index in range(said, len(words)):
                            await ws.send_json(
                                {"type": "transcription.delta", "delta": _spoken(words, index)}
                            )
                        await asyncio.sleep(self.answer_delay_s)
                        await ws.send_json(
                            {"type": "transcription.done", "text": self.corrected or self.text}
                        )
                        generating = False

        return app


def _spoken(words: list[str], index: int) -> str:
    return words[index] if index == 0 else f" {words[index]}"


@dataclass
class FakeVllmOmniSpeech:
    """Speaks vLLM-Omni's speech API as its docs and server describe it: POST
    /v1/audio/speech with stream_format="audio" answers 200 with raw PCM bytes and no
    framing, a request it will not serve is refused with a status before the body, and an
    engine that dies mid-stream can only truncate that body -- the raw form carries no
    error frame.
    https://github.com/vllm-project/vllm-omni/blob/main/docs/serving/speech_api.md

    `audio` stands in for the model, and the body is cut into chunks that do not fall on
    sample boundaries, as a network does."""

    audio: Callable[[str], bytes] = lambda text: b""
    chunk_bytes: int = 777
    refuses: str = "FAIL"
    """Input text the engine refuses, as a model that cannot serve the request does."""
    truncates: str = ""
    """Input text whose body stops after one chunk, as an engine that dies mid-stream
    leaves it: the raw form has no error frame to send instead."""
    requests: list[dict[str, Any]] = field(default_factory=list)

    def app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/v1/audio/speech")
        async def speech(request: Request) -> Response:
            payload = await request.json()
            self.requests.append(payload)
            if error := self._refusal(payload):
                return JSONResponse({"error": error}, status_code=error["code"])
            return StreamingResponse(self._body(payload["input"]), media_type="audio/pcm")

        return app

    def _refusal(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not payload.get("input"):
            return _omni_error("Input text cannot be empty", 400)
        if payload.get("stream_format") == "audio" or payload.get("stream"):
            if payload.get("response_format") not in ("pcm", "wav"):
                return _omni_error("Streaming requires response_format pcm or wav", 400)
            if payload.get("speed", 1.0) != 1.0:
                return _omni_error("Streaming requires speed 1.0", 400)
        if payload["input"] == self.refuses:
            return _omni_error("engine failed to start generation", 500)
        return None

    async def _body(self, text: str) -> AsyncIterator[bytes]:
        pcm = self.audio(text)
        for sent, start in enumerate(range(0, len(pcm), self.chunk_bytes)):
            if sent and text == self.truncates:
                raise RuntimeError("engine died mid-stream")
            yield pcm[start : start + self.chunk_bytes]


def _omni_error(message: str, code: int) -> dict[str, Any]:
    return {"message": message, "type": "BadRequestError", "param": None, "code": code}
