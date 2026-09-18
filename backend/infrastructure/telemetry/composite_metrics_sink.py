from __future__ import annotations

import logging
from collections.abc import Sequence

from backend.application.ports.metrics_sink import CallMetricsEvent, MetricsSink

log = logging.getLogger(__name__)


class CompositeMetricsSink:
    """Fans out to every sink; one failing sink never breaks a call or the others."""

    def __init__(self, sinks: Sequence[MetricsSink]) -> None:
        self._sinks = list(sinks)

    async def emit(self, event: CallMetricsEvent) -> None:
        for sink in self._sinks:
            try:
                await sink.emit(event)
            except Exception:
                log.exception("metrics sink %s failed", type(sink).__name__)
