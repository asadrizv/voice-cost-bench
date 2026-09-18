from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from backend.domain.entities.latency import LatencyBreakdown, LatencyStage

END_TO_END_P95_BUDGET_MS = 900.0
PERCEIVED_DELAY_P95_BUDGET_MS = 1200.0


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear interpolation between closest ranks (numpy's default), without numpy."""
    if not values:
        return 0.0
    if not 0 <= pct <= 100:
        raise ValueError("pct must be within [0, 100]")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


@dataclass(frozen=True)
class StageStats:
    p50: float
    p95: float
    mean: float
    count: int


@dataclass(frozen=True)
class BudgetReport:
    end_to_end_p95: float
    perceived_delay_p95: float
    end_to_end_ok: bool
    perceived_delay_ok: bool

    @property
    def ok(self) -> bool:
        return self.end_to_end_ok and self.perceived_delay_ok


class LatencyAnalyzer:
    def __init__(
        self,
        end_to_end_budget_ms: float = END_TO_END_P95_BUDGET_MS,
        perceived_budget_ms: float = PERCEIVED_DELAY_P95_BUDGET_MS,
    ) -> None:
        self._e2e_budget = end_to_end_budget_ms
        self._perceived_budget = perceived_budget_ms

    def stats(self, samples: Sequence[LatencyBreakdown]) -> dict[LatencyStage, StageStats]:
        result: dict[LatencyStage, StageStats] = {}
        for stage in LatencyStage:
            values = [s.get(stage) for s in samples]
            result[stage] = StageStats(
                p50=percentile(values, 50),
                p95=percentile(values, 95),
                mean=sum(values) / len(values) if values else 0.0,
                count=len(values),
            )
        return result

    def budget(self, samples: Sequence[LatencyBreakdown]) -> BudgetReport:
        e2e = percentile([s.end_to_end for s in samples], 95)
        perceived = percentile([s.perceived_delay for s in samples], 95)
        return BudgetReport(
            end_to_end_p95=e2e,
            perceived_delay_p95=perceived,
            end_to_end_ok=e2e < self._e2e_budget,
            perceived_delay_ok=perceived < self._perceived_budget,
        )
