from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from backend.application.ports.component_catalogue import Component
from backend.application.ports.rate_card_provider import TelephonyQuote
from backend.application.use_cases.compare_pipelines import PipelineSummary
from backend.application.use_cases.compute_call_cost import CallCostReport
from backend.application.use_cases.describe_components import DescribedComponent
from backend.domain.entities.call import Call, Turn
from backend.domain.entities.cost import CostBreakdown, UsageUnits
from backend.domain.services.cost_calculator import PipelineRates
from backend.domain.services.latency_analyzer import LatencyAnalyzer


def cost(c: CostBreakdown) -> dict[str, float]:
    return c.as_dict()


def usage(u: UsageUnits) -> dict[str, float]:
    return asdict(u)


def rates(r: PipelineRates) -> dict[str, float]:
    return {k: float(v) for k, v in asdict(r).items()}


def telephony_quote(q: TelephonyQuote) -> dict[str, Any]:
    price = q.per_minute_usd
    return {**asdict(q), "per_minute_usd": None if price is None else float(price)}


def component(c: Component) -> dict[str, Any]:
    return {**asdict(c), "kind": c.kind.value}


def described_component(d: DescribedComponent) -> dict[str, Any]:
    """Shared by /transparency and the benchmark's provenance file: the two are meant to
    name the same components, so a new Component field reaches both or neither."""
    return {
        **component(d.component),
        "confirmed": d.confirmed,
        "unconfirmed_reason": d.unconfirmed_reason,
    }


def turn(t: Turn) -> dict[str, Any]:
    return {
        "index": t.index,
        "user_text": t.user_text,
        "agent_text": t.agent_text,
        "interrupted": t.interrupted,
        "started_at": t.started_at.isoformat() if t.started_at else None,
        "usage": usage(t.usage),
        "cost": cost(t.cost),
        "latency": t.latency.as_dict() if t.latency else None,
    }


def call_summary(c: Call) -> dict[str, Any]:
    now = datetime.now(UTC)
    duration = c.duration_seconds(now)
    budget = LatencyAnalyzer().budget(c.latencies())
    total = c.cost
    return {
        "id": c.id,
        "pipeline": c.pipeline.value,
        "persona": c.persona,
        "source": c.source,
        "status": c.status.value,
        "started_at": c.started_at.isoformat(),
        "ended_at": c.ended_at.isoformat() if c.ended_at else None,
        "duration_seconds": duration,
        "turns": len(c.turns),
        "cost": cost(total),
        "cost_per_minute_usd": total.per_minute(duration).as_float(),
        "end_to_end_p95_ms": budget.end_to_end_p95,
        "perceived_delay_p95_ms": budget.perceived_delay_p95,
    }


def call_detail(c: Call) -> dict[str, Any]:
    return {
        **call_summary(c),
        "usage": usage(c.usage),
        "closing": {"usage": usage(c.closing_usage), "cost": cost(c.closing_cost)},
        "turn_list": [turn(t) for t in c.turns],
    }


def cost_report(r: CallCostReport) -> dict[str, Any]:
    """The public per-call attribution surface: every dollar traced to units x rate."""
    c = r.call
    return {
        "call_id": c.id,
        "pipeline": c.pipeline.value,
        "status": c.status.value,
        "duration_seconds": r.duration_seconds,
        "currency": "USD",
        "total_usd": r.cost.total.as_float(),
        "cost_per_minute_usd": r.cost_per_minute_usd,
        "projected_per_1000_minutes_usd": r.projected_per_1000_minutes_usd,
        "stages": cost(r.cost),
        "units": usage(r.usage),
        "rates": rates(r.rates),
        "rate_card_verified_on": r.rate_card_verified_on,
        "recomputed_total_usd": r.recomputed_total_usd,
        "latency_p95_ms": {
            "end_to_end": r.end_to_end_p95_ms,
            "perceived_delay": r.perceived_delay_p95_ms,
        },
        "turns": [
            {"index": t.index, "units": usage(t.usage), "cost": cost(t.cost)} for t in c.turns
        ],
        "closing_segment": {"units": usage(c.closing_usage), "cost": cost(c.closing_cost)},
    }


def pipeline_summary(s: PipelineSummary) -> dict[str, Any]:
    return {
        "pipeline": s.pipeline.value,
        "calls": s.calls,
        "turns": s.turns,
        "total_minutes": s.total_minutes,
        "cost": cost(s.cost),
        "cost_per_minute_usd": s.cost_per_minute_usd,
        "latency": {stage.value: asdict(stats) for stage, stats in s.latency.items()},
    }
