from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GpuSnapshot:
    sku: str
    utilisation_pct: float
    memory_used_mb: float
    memory_total_mb: float
    busy_seconds: float
    """Cumulative seconds the device reported non-zero utilisation since telemetry start."""


class GpuTelemetry(Protocol):
    def snapshot(self) -> GpuSnapshot | None:
        """None when no GPU is visible to this process."""
        ...
