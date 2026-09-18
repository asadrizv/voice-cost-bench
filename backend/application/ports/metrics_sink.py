from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from backend.domain.entities.cost import CostBreakdown
from backend.domain.entities.latency import LatencyBreakdown
from backend.domain.value_objects.pipeline_kind import PipelineKind


@dataclass(frozen=True)
class CallMetricsEvent:
    kind: Literal["started", "turn", "tick", "ended"]
    call_id: str
    pipeline: PipelineKind
    elapsed_seconds: float
    running_cost: CostBreakdown
    turn_index: int | None = None
    latency: LatencyBreakdown | None = None
    turn_cost: CostBreakdown | None = None
    user_text: str = ""
    agent_text: str = ""
    interrupted: bool = False
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def cost_per_minute_usd(self) -> float:
        return self.running_cost.per_minute(self.elapsed_seconds).as_float()


class MetricsSink(Protocol):
    async def emit(self, event: CallMetricsEvent) -> None: ...
