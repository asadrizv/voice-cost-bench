from __future__ import annotations

import asyncio
import time
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from backend.application.ports.llm_port import ChatMessage, TokenDelta
from backend.domain.entities.persona import SamplingParams
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.settings import Settings
from backend.interfaces.container import Container
from backend.interfaces.http.deps import container, settings

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/deep")
async def deep(
    c: Annotated[Container, Depends(container)],
    s: Annotated[Settings, Depends(settings)],
    pipeline: PipelineKind | None = None,
) -> JSONResponse:
    """Exercises the models, not just the processes: a one-token generation and a short
    synthesis. A liveness probe that skips them stays green with the model unloaded."""
    kind = pipeline or s.pipeline
    checks: dict[str, Any] = {}

    async def timed(name: str, coro: Any) -> None:
        started = time.perf_counter()
        try:
            await asyncio.wait_for(coro, timeout=15)
            checks[name] = {"ok": True, "ms": round((time.perf_counter() - started) * 1000)}
        except Exception as exc:
            checks[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}

    async def db() -> None:
        await c.repository.list(limit=1)

    async def llm() -> None:
        stream = c.pipelines.resolve(kind).llm.complete(
            [ChatMessage("user", "Say OK.")], SamplingParams(max_tokens=1)
        )
        async for event in stream:
            if isinstance(event, TokenDelta):
                break

    async def tts() -> None:
        async for chunk in c.pipelines.resolve(kind).tts.synthesize("OK.", None):
            if chunk.data:
                return
        raise RuntimeError("no audio")

    async def whisper() -> None:
        url = s.whisper_ws_url.replace("ws://", "http://").replace("wss://", "https://")
        base = url.rsplit("/v1/", 1)[0]
        async with httpx.AsyncClient(timeout=10) as client:
            (await client.get(f"{base}/health/deep")).raise_for_status()

    await timed("database", db())
    await timed("llm", llm())
    await timed("tts", tts())
    if kind is PipelineKind.SELFHOSTED:
        await timed("stt", whisper())
    ok = all(v["ok"] for v in checks.values())
    return JSONResponse(
        {"status": "ok" if ok else "degraded", "pipeline": kind.value, "checks": checks},
        status_code=200 if ok else 503,
    )
