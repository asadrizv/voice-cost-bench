from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from backend.application.ports.component_catalogue import ComponentKind


@dataclass(frozen=True)
class Engine:
    id: str
    model: str


class EngineReportUnavailable(RuntimeError):
    pass


class RunningEngines(Protocol):
    async def report(self, kind: ComponentKind) -> list[Engine]:
        """What the self-hosted service for that kind says it runs. Raises
        EngineReportUnavailable when the service can't be reached or answers malformed."""
        ...
