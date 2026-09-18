from __future__ import annotations

from typing import Protocol

from backend.domain.entities.call import Call, Turn
from backend.domain.value_objects.pipeline_kind import PipelineKind


class CallNotFound(Exception):
    pass


class CallRepository(Protocol):
    async def add(self, call: Call) -> None: ...

    async def add_turn(self, call_id: str, turn: Turn) -> None: ...

    async def update(self, call: Call) -> None:
        """Persists status, end time and closing usage; turns are written by add_turn."""
        ...

    async def get(self, call_id: str) -> Call:
        """Raises CallNotFound."""
        ...

    async def list(
        self, limit: int = 50, pipeline: PipelineKind | None = None, source: str | None = None
    ) -> list[Call]: ...

    async def total_spend_usd(self, pipeline: PipelineKind) -> float: ...
