from __future__ import annotations

from decimal import Decimal
from typing import Any

from backend.application.ports.metrics_sink import CallMetricsEvent
from backend.domain.entities.cost import CostBreakdown, Money
from backend.domain.entities.latency import LatencyBreakdown
from backend.domain.value_objects.pipeline_kind import PipelineKind


def encode(event: CallMetricsEvent) -> dict[str, Any]:
    """Wire format for SSE and for agent -> API forwarding. Money travels as strings so
    Decimal precision survives JSON."""
    return {
        "kind": event.kind,
        "call_id": event.call_id,
        "pipeline": event.pipeline.value,
        "elapsed_seconds": event.elapsed_seconds,
        "running_cost": _cost(event.running_cost),
        "cost_per_minute_usd": event.cost_per_minute_usd,
        "projected_per_1000_min_usd": event.cost_per_minute_usd * 1000,
        "turn_index": event.turn_index,
        "latency": event.latency.as_dict() if event.latency else None,
        "turn_cost": _cost(event.turn_cost) if event.turn_cost else None,
        "user_text": event.user_text,
        "agent_text": event.agent_text,
        "interrupted": event.interrupted,
        "extra": event.extra,
    }


def decode(data: dict[str, Any]) -> CallMetricsEvent:
    return CallMetricsEvent(
        kind=data["kind"],
        call_id=data["call_id"],
        pipeline=PipelineKind(data["pipeline"]),
        elapsed_seconds=float(data["elapsed_seconds"]),
        running_cost=_uncost(data["running_cost"]),
        turn_index=data.get("turn_index"),
        latency=LatencyBreakdown(**data["latency"]) if data.get("latency") else None,
        turn_cost=_uncost(data["turn_cost"]) if data.get("turn_cost") else None,
        user_text=data.get("user_text", ""),
        agent_text=data.get("agent_text", ""),
        interrupted=bool(data.get("interrupted")),
        extra={k: float(v) for k, v in (data.get("extra") or {}).items()},
    )


_STAGES = ("stt", "llm", "tts", "gpu", "telephony")


def _cost(cost: CostBreakdown) -> dict[str, str]:
    out = {k: str(getattr(cost, k).amount) for k in _STAGES}
    out["total"] = str(cost.total.amount)
    return out


def _uncost(data: dict[str, str]) -> CostBreakdown:
    return CostBreakdown(**{k: Money(Decimal(data.get(k, "0"))) for k in _STAGES})
