from __future__ import annotations

from dataclasses import dataclass

from backend.application.ports.call_repository import CallRepository
from backend.domain.entities.call import CallStatus
from backend.domain.entities.cost import CostBreakdown
from backend.domain.entities.latency import LatencyStage
from backend.domain.services.latency_analyzer import LatencyAnalyzer, StageStats
from backend.domain.value_objects.pipeline_kind import PipelineKind


@dataclass(frozen=True)
class PipelineSummary:
    pipeline: PipelineKind
    calls: int
    turns: int
    total_minutes: float
    cost: CostBreakdown
    cost_per_minute_usd: float
    latency: dict[LatencyStage, StageStats]


class ComparePipelines:
    """Aggregates completed calls per pipeline. Cost per minute is total cost over total
    minutes, not a mean of per-call rates, so short calls don't dominate."""

    def __init__(self, repository: CallRepository) -> None:
        self._repository = repository
        self._latency = LatencyAnalyzer()

    async def execute(
        self, source: str | None = None, limit: int = 1000
    ) -> dict[PipelineKind, PipelineSummary]:
        result: dict[PipelineKind, PipelineSummary] = {}
        for kind in PipelineKind:
            calls = [
                c
                for c in await self._repository.list(limit=limit, pipeline=kind, source=source)
                if c.status is CallStatus.COMPLETED
            ]
            cost = CostBreakdown.zero()
            seconds = 0.0
            for call in calls:
                cost = cost + call.cost
                seconds += call.duration_seconds()
            samples = [lat for call in calls for lat in call.latencies()]
            result[kind] = PipelineSummary(
                pipeline=kind,
                calls=len(calls),
                turns=sum(len(c.turns) for c in calls),
                total_minutes=seconds / 60,
                cost=cost,
                cost_per_minute_usd=cost.per_minute(seconds).as_float(),
                latency=self._latency.stats(samples),
            )
        return result
