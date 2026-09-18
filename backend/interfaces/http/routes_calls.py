from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.application.ports.call_repository import CallNotFound
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.interfaces.container import Container
from backend.interfaces.http import serializers
from backend.interfaces.http.deps import container

router = APIRouter()


@router.get("/calls")
async def list_calls(
    c: Annotated[Container, Depends(container)],
    pipeline: PipelineKind | None = None,
    source: str | None = "browser",
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[dict[str, Any]]:
    """Synthetic load-test calls are hidden by default; pass source= (empty) for all."""
    calls = await c.repository.list(limit=limit, pipeline=pipeline, source=source or None)
    return [serializers.call_summary(call) for call in calls]


@router.get("/calls/compare")
async def compare(
    c: Annotated[Container, Depends(container)], source: str | None = "browser"
) -> list[dict[str, Any]]:
    summary = await c.compare.execute(source=source or None)
    return [serializers.pipeline_summary(s) for s in summary.values()]


@router.get("/calls/{call_id}")
async def get_call(call_id: str, c: Annotated[Container, Depends(container)]) -> dict[str, Any]:
    try:
        return serializers.call_detail(await c.repository.get(call_id))
    except CallNotFound:
        raise HTTPException(404, "call not found") from None


@router.get("/calls/{call_id}/cost")
async def get_call_cost(
    call_id: str, c: Annotated[Container, Depends(container)]
) -> dict[str, Any]:
    try:
        return serializers.cost_report(await c.compute_cost.execute(call_id))
    except CallNotFound:
        raise HTTPException(404, "call not found") from None
