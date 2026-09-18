from __future__ import annotations

from dataclasses import dataclass

from backend.application.ports.call_repository import CallRepository
from backend.domain.entities.call import Call
from backend.domain.entities.cost import CostBreakdown, UsageUnits
from backend.domain.services.cost_calculator import CostCalculator, PipelineRates
from backend.domain.services.latency_analyzer import LatencyAnalyzer


@dataclass(frozen=True)
class CallCostReport:
    call: Call
    cost: CostBreakdown
    usage: UsageUnits
    duration_seconds: float
    cost_per_minute_usd: float
    projected_per_1000_minutes_usd: float
    rates: PipelineRates
    rate_card_verified_on: str
    recomputed_total_usd: float
    """Stored usage re-priced with today's rate card; differs from the stored total only
    if rates.yaml changed since the call."""
    end_to_end_p95_ms: float
    perceived_delay_p95_ms: float


class ComputeCallCost:
    def __init__(self, repository: CallRepository, calculator: CostCalculator) -> None:
        self._repository = repository
        self._calculator = calculator
        self._latency = LatencyAnalyzer()

    async def execute(self, call_id: str) -> CallCostReport:
        call = await self._repository.get(call_id)
        return self.report(call)

    def report(self, call: Call) -> CallCostReport:
        cost = call.cost
        duration = call.duration_seconds()
        per_minute = cost.per_minute(duration).as_float()
        recomputed = self._calculator.calculate(call.pipeline, call.closing_usage)
        for turn in call.turns:
            recomputed = recomputed + self._calculator.calculate(call.pipeline, turn.usage)
        budget = self._latency.budget(call.latencies())
        card = self._calculator.rate_card
        return CallCostReport(
            call=call,
            cost=cost,
            usage=call.usage,
            duration_seconds=duration,
            cost_per_minute_usd=per_minute,
            projected_per_1000_minutes_usd=per_minute * 1000,
            rates=card.for_pipeline(call.pipeline),
            rate_card_verified_on=card.verified_on,
            recomputed_total_usd=recomputed.total.as_float(),
            end_to_end_p95_ms=budget.end_to_end_p95,
            perceived_delay_p95_ms=budget.perceived_delay_p95,
        )
