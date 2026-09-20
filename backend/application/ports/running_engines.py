from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from backend.application.ports.component_catalogue import ComponentKind


@dataclass(frozen=True)
class Engine:
    id: str
    model: str
    gpu_fraction: Decimal | None = None
    """The share of one GPU this engine's runtime reserves up front, None when it holds only
    its weights. A server reserves its share whether or not a call is in flight, so the GPU
    budget must read this rather than the catalogue's footprint for it."""


class EngineReportUnavailable(RuntimeError):
    pass


class RunningEngines(Protocol):
    async def report(self, kind: ComponentKind) -> list[Engine]:
        """What the self-hosted service for that kind says it runs. Raises
        EngineReportUnavailable when the service can't be reached or answers malformed."""
        ...
