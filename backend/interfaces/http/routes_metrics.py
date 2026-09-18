from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest
from sse_starlette.sse import EventSourceResponse

from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.settings import Settings
from backend.infrastructure.telemetry.event_codec import decode, encode
from backend.infrastructure.telemetry.inmemory_metrics_sink import InMemoryMetricsSink
from backend.infrastructure.telemetry.prometheus_metrics_sink import PrometheusMetricsSink
from backend.interfaces.container import Container
from backend.interfaces.http.deps import container, live, prometheus, settings

router = APIRouter()


@router.get("/metrics/live/{call_id}")
async def live_metrics(
    call_id: str, request: Request, sink: Annotated[InMemoryMetricsSink, Depends(live)]
) -> EventSourceResponse:
    """Server-sent events for one call: started, turn, tick (1 Hz), ended."""
    queue = sink.subscribe(call_id)

    async def events() -> AsyncIterator[dict[str, str]]:
        try:
            while not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue
                yield {"event": event.kind, "data": json.dumps(encode(event))}
                if event.kind == "ended":
                    return
        finally:
            sink.unsubscribe(call_id, queue)

    return EventSourceResponse(events())


@router.post("/internal/metrics/events", status_code=202)
async def ingest(
    body: dict[str, Any],
    c: Annotated[Container, Depends(container)],
    s: Annotated[Settings, Depends(settings)],
    x_internal_token: Annotated[str, Header()] = "",
) -> dict[str, str]:
    """Agent and load-harness processes forward their events here."""
    if x_internal_token != s.internal_token:
        raise HTTPException(401, "bad internal token")
    await c.metrics.emit(decode(body))
    return {"status": "accepted"}


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics(
    sink: Annotated[PrometheusMetricsSink, Depends(prometheus)],
    c: Annotated[Container, Depends(container)],
) -> str:
    for kind in PipelineKind:
        sink.set_spend(kind.value, await c.repository.total_spend_usd(kind))
    return generate_latest(sink.registry).decode()
