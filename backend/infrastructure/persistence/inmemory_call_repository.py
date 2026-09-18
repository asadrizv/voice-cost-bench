from __future__ import annotations

import copy

from backend.application.ports.call_repository import CallNotFound
from backend.domain.entities.call import Call, Turn
from backend.domain.value_objects.pipeline_kind import PipelineKind


class InMemoryCallRepository:
    """For tests and for running without Postgres. Stores copies so callers can't mutate
    persisted state by accident, which is the behaviour the Postgres adapter has."""

    def __init__(self) -> None:
        self._calls: dict[str, Call] = {}

    async def add(self, call: Call) -> None:
        self._calls[call.id] = copy.deepcopy(call)

    async def add_turn(self, call_id: str, turn: Turn) -> None:
        stored = self._require(call_id)
        stored.turns = [t for t in stored.turns if t.index != turn.index] + [copy.deepcopy(turn)]
        stored.turns.sort(key=lambda t: t.index)

    async def update(self, call: Call) -> None:
        stored = self._require(call.id)
        stored.status = call.status
        stored.ended_at = call.ended_at
        stored.closing_usage = call.closing_usage
        stored.closing_cost = call.closing_cost

    async def get(self, call_id: str) -> Call:
        return copy.deepcopy(self._require(call_id))

    async def list(
        self, limit: int = 50, pipeline: PipelineKind | None = None, source: str | None = None
    ) -> list[Call]:
        calls = [
            c
            for c in self._calls.values()
            if (pipeline is None or c.pipeline is pipeline)
            and (source is None or c.source == source)
        ]
        calls.sort(key=lambda c: c.started_at, reverse=True)
        return [copy.deepcopy(c) for c in calls[:limit]]

    async def total_spend_usd(self, pipeline: PipelineKind) -> float:
        return sum(c.cost.total.as_float() for c in self._calls.values() if c.pipeline is pipeline)

    def _require(self, call_id: str) -> Call:
        try:
            return self._calls[call_id]
        except KeyError:
            raise CallNotFound(call_id) from None
