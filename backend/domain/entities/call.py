from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from backend.domain.entities.cost import CostBreakdown, UsageUnits
from backend.domain.entities.latency import LatencyBreakdown
from backend.domain.value_objects.pipeline_kind import PipelineKind


class CallStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class Turn:
    index: int
    user_text: str
    agent_text: str
    usage: UsageUnits
    cost: CostBreakdown
    latency: LatencyBreakdown | None = None
    """None for the greeting, which has no caller utterance to measure from."""
    interrupted: bool = False
    started_at: datetime | None = None


@dataclass
class Call:
    id: str
    pipeline: PipelineKind
    persona: str
    started_at: datetime
    status: CallStatus = CallStatus.ACTIVE
    ended_at: datetime | None = None
    turns: list[Turn] = field(default_factory=list)
    closing_usage: UsageUnits = field(default_factory=UsageUnits)
    """Wall time after the last turn (GPU and telephony only), accounted at call end."""
    closing_cost: CostBreakdown = field(default_factory=CostBreakdown)
    source: str = "browser"
    """browser | loadtest; the history view filters synthetic calls out by default."""

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)

    def next_turn_index(self) -> int:
        return len(self.turns)

    @property
    def cost(self) -> CostBreakdown:
        total = self.closing_cost
        for turn in self.turns:
            total = total + turn.cost
        return total

    @property
    def usage(self) -> UsageUnits:
        total = self.closing_usage
        for turn in self.turns:
            total = total + turn.usage
        return total

    def duration_seconds(self, now: datetime | None = None) -> float:
        end = self.ended_at or now
        if end is None:
            return 0.0
        return max(0.0, (end - self.started_at).total_seconds())

    def latencies(self) -> list[LatencyBreakdown]:
        return [t.latency for t in self.turns if t.latency is not None]

    def complete(self, at: datetime) -> None:
        self.status = CallStatus.COMPLETED
        self.ended_at = at

    def fail(self, at: datetime) -> None:
        self.status = CallStatus.FAILED
        self.ended_at = at
