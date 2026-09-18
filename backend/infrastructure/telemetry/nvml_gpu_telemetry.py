from __future__ import annotations

import time
from typing import Any

from backend.application.ports.gpu_telemetry import GpuSnapshot


class NvmlGpuTelemetry:
    """Reads the first GPU via NVML. Only meaningful when this process runs on the GPU
    node, which colocation makes the normal case for the self-hosted pipeline."""

    def __init__(self, index: int = 0) -> None:
        import pynvml  # nvidia-ml-py; optional extra, absent on dev laptops

        pynvml.nvmlInit()
        self._nvml: Any = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(self._handle)
        self._sku = name.decode() if isinstance(name, bytes) else str(name)
        self._busy_seconds = 0.0
        self._last_sample = time.monotonic()

    def snapshot(self) -> GpuSnapshot | None:
        util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        now = time.monotonic()
        if util.gpu > 0:
            self._busy_seconds += now - self._last_sample
        self._last_sample = now
        return GpuSnapshot(
            sku=self._sku,
            utilisation_pct=float(util.gpu),
            memory_used_mb=mem.used / 2**20,
            memory_total_mb=mem.total / 2**20,
            busy_seconds=self._busy_seconds,
        )


class NullGpuTelemetry:
    def snapshot(self) -> GpuSnapshot | None:
        return None


def detect_gpu_telemetry() -> NvmlGpuTelemetry | NullGpuTelemetry:
    try:
        return NvmlGpuTelemetry()
    except Exception:
        return NullGpuTelemetry()
