from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from decimal import Decimal
from typing import Any

from backend.application.ports.rate_card_provider import TelephonyQuote
from backend.domain.entities.call import Call
from backend.domain.entities.cost import CostBreakdown
from backend.domain.entities.latency import LatencyStage
from backend.domain.services.cost_calculator import RateCard
from backend.domain.services.latency_analyzer import (
    END_TO_END_P95_BUDGET_MS,
    PERCEIVED_DELAY_P95_BUDGET_MS,
    LatencyAnalyzer,
    percentile,
)
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.interfaces.cli.harness.caller import Conversation
from backend.interfaces.cli.harness.runner import LevelRun

UTILISATIONS = (0.10, 0.25, 0.50, 1.00)
MAX_HARNESS_LATENESS_MS = 20.0


def word_error_rate(reference: str, hypothesis: str) -> float:
    import jiwer

    ref = _normalise(reference)
    if not ref:
        return 0.0
    return float(jiwer.wer(ref, _normalise(hypothesis)))


def _normalise(text: str) -> str:
    keep = [ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text]
    return " ".join("".join(keep).split())


def _turn_rows(call: Call) -> list[dict[str, Any]]:
    """One row per timed turn: the numbers the percentiles above are computed from, so a
    reader who doubts a figure can recompute it and a tail nobody can see is not the only
    record of it. Untimed turns -- the greeting, and any the agent never answered -- carry
    no latency and are left out rather than entered as zero."""
    return [
        {
            "call": call.id,
            "index": turn.index,
            "started_at": turn.started_at.isoformat() if turn.started_at else None,
            "interrupted": turn.interrupted,
            "latency_ms": {
                stage.value: round(turn.latency.get(stage), 1) for stage in LatencyStage
            },
            "perceived_delay_ms": round(turn.latency.perceived_delay, 1),
            "cost_usd": turn.cost.as_dict(),
            "usage": asdict(turn.usage),
            "concurrent_calls": turn.usage.concurrent_calls,
        }
        for turn in call.turns
        if turn.latency is not None
    ]


def summarise_level(
    run: LevelRun,
    conversation: Conversation,
    simulated: bool = False,
    carriers: Sequence[TelephonyQuote] = (),
) -> dict[str, Any]:
    calls = run.calls
    cost = CostBreakdown.zero()
    seconds = 0.0
    gpu_seconds = share_seconds = telephony_seconds = 0.0
    wers: list[float] = []
    for call in calls:
        cost = cost + call.cost
        seconds += call.duration_seconds()
        segments = [t.usage for t in call.turns] + [call.closing_usage]
        for u in segments:
            gpu_seconds += u.gpu_seconds
            share_seconds += u.gpu_seconds / max(u.concurrent_calls, 1)
            telephony_seconds += u.telephony_seconds
        # Whole-call WER: a turn can split at a mid-utterance pause and be merged back by
        # carry-over, so pairing script lines with turns one-to-one would misalign.
        heard = " ".join(t.user_text for t in call.turns if t.user_text and not t.interrupted)
        wers.append(word_error_rate(" ".join(conversation.turns), heard))

    samples = [lat for call in calls for lat in call.latencies()]
    stats = LatencyAnalyzer().stats(samples)
    budget = LatencyAnalyzer().budget(samples)
    minutes = seconds / 60
    lateness_p95 = percentile(run.lateness_ms, 95) if run.lateness_ms else 0.0
    observed = run.caller_observed_ms
    per_minute = {k: (v * 60 / seconds if seconds else 0.0) for k, v in cost.as_dict().items()}
    # Telephony cost is linear in telephony seconds, so each carrier swaps only that line.
    without_telephony = per_minute["total"] - per_minute["telephony"]
    telephony_share = telephony_seconds / seconds if seconds else 0.0
    return {
        "concurrency": run.concurrency,
        "duration_s": run.duration_s,
        "wall_s": round(run.wall_s, 1),
        "calls_completed": len(calls),
        "calls_failed": run.failed,
        "calls_rejected": run.rejected,
        "reply_timeouts": run.reply_timeouts,
        "turns": len(samples),
        "turn_rows": [row for call in calls for row in _turn_rows(call)],
        "call_minutes": round(minutes, 3),
        "cost_usd": cost.as_dict(),
        "cost_per_minute_usd": per_minute["total"],
        "cost_per_minute_by_stage_usd": per_minute,
        "cost_per_minute_by_carrier_usd": {
            q.carrier: without_telephony + float(q.per_minute_usd) * telephony_share
            for q in carriers
            if q.per_minute_usd is not None
        },
        "effective_concurrency": gpu_seconds / share_seconds if share_seconds else 1.0,
        "latency_ms": {
            stage.value: {k: round(v, 1) for k, v in asdict(stats[stage]).items()}
            for stage in LatencyStage
        },
        "end_to_end_p95_ms": round(budget.end_to_end_p95, 1),
        "perceived_delay_p95_ms": round(budget.perceived_delay_p95, 1),
        **{
            f"caller_observed_p{pct}_ms": round(percentile(observed, pct), 1) if observed else None
            for pct in (50, 95, 99)
        },
        "endpoint_decisions": len(run.endpoint_inference_ms),
        "endpoint_failures": run.endpoint_failures,
        **{
            f"endpoint_inference_p{pct}_ms": round(percentile(run.endpoint_inference_ms, pct), 2)
            if run.endpoint_inference_ms
            else None
            for pct in (50, 95)
        },
        "caller_observed_turns": len(observed),
        "caller_observed_unanswered": run.unanswered_turns,
        "within_budget": budget.ok,
        "stt_wer": round(sum(wers) / len(wers), 4) if wers and not simulated else None,
        "harness_lateness_p95_ms": round(lateness_p95, 1),
        "harness_valid": lateness_p95 <= MAX_HARNESS_LATENESS_MS,
    }


def breaking_point(levels: list[dict[str, Any]]) -> dict[str, Any] | None:
    for level in levels:
        if level["turns"] == 0:
            continue
        if level["end_to_end_p95_ms"] >= END_TO_END_P95_BUDGET_MS:
            return {
                "concurrency": level["concurrency"],
                "end_to_end_p95_ms": level["end_to_end_p95_ms"],
                "perceived_delay_p95_ms": level["perceived_delay_p95_ms"],
                "last_within_budget": _last_ok(levels, level["concurrency"]),
            }
    return None


def _last_ok(levels: list[dict[str, Any]], before: int) -> int | None:
    ok = [lv["concurrency"] for lv in levels if lv["concurrency"] < before and lv["within_budget"]]
    return max(ok) if ok else None


def utilisation_curve(
    levels: list[dict[str, Any]], rate_card: RateCard, client_quotes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Measured runs keep the GPU ~100% busy. A box rented for an hour that serves twenty
    minutes still costs the hour, so the GPU component scales with 1/utilisation; nothing
    else does. Client EU prices rescale the GPU component by their hourly rate."""
    base_hourly = rate_card.for_pipeline(PipelineKind.SELFHOSTED).gpu_per_hour
    curve = []
    for level in levels:
        if not level["call_minutes"]:
            continue
        by_stage = level["cost_per_minute_by_stage_usd"]
        gpu = by_stage["gpu"]
        other = by_stage["total"] - gpu
        for u in UTILISATIONS:
            row = {
                "concurrency": level["concurrency"],
                "utilisation": u,
                "cost_per_minute_usd": other + gpu / u,
                "gpu_share_usd": gpu / u,
            }
            for quote in client_quotes:
                scale = float(Decimal(str(quote["hourly_usd"])) / base_hourly) if base_hourly else 0
                row[f"cost_per_minute_usd_{quote['provider']}"] = other + gpu * scale / u
            curve.append(row)
    return curve


def budgets() -> dict[str, float]:
    return {
        "end_to_end_p95_ms": END_TO_END_P95_BUDGET_MS,
        "perceived_delay_p95_ms": PERCEIVED_DELAY_P95_BUDGET_MS,
    }
