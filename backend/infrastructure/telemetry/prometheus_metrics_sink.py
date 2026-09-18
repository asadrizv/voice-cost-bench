from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from backend.application.ports.metrics_sink import CallMetricsEvent
from backend.domain.entities.latency import LatencyStage

# Buckets straddle the 900 ms / 1200 ms budgets so p95 against budget reads off directly.
_BUCKETS_MS = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000, 5000)


class PrometheusMetricsSink:
    """Ops view of live calls. Postgres stays the source of truth for cost; these counters
    reset on restart and exist for dashboards and alerts."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self._latency = Histogram(
            "voice_turn_latency_ms",
            "Per-turn latency by stage",
            ["pipeline", "stage"],
            buckets=_BUCKETS_MS,
            registry=self.registry,
        )
        self._cost = Counter(
            "voice_cost_usd",
            "Accrued cost by pipeline and component",
            ["pipeline", "component"],
            registry=self.registry,
        )
        self._turns = Counter(
            "voice_turns", "Completed turns", ["pipeline", "interrupted"], registry=self.registry
        )
        self._calls = Counter(
            "voice_calls", "Calls by lifecycle event", ["pipeline", "event"], registry=self.registry
        )
        self._active = Gauge(
            "voice_active_calls", "Calls in progress", ["pipeline"], registry=self.registry
        )
        self._cost_per_minute = Gauge(
            "voice_call_cost_per_minute_usd",
            "Running cost per minute of the most recently updated call",
            ["pipeline"],
            registry=self.registry,
        )
        self._closing_seen: set[str] = set()

    async def emit(self, event: CallMetricsEvent) -> None:
        pipeline = event.pipeline.value
        if event.kind == "started":
            self._calls.labels(pipeline, "started").inc()
            self._active.labels(pipeline).inc()
        elif event.kind == "turn":
            self._turns.labels(pipeline, str(event.interrupted).lower()).inc()
            if event.latency is not None:
                for stage in LatencyStage:
                    self._latency.labels(pipeline, stage.value).observe(event.latency.get(stage))
            if event.turn_cost is not None:
                self._add_cost(pipeline, event.turn_cost.as_dict())
        elif event.kind == "ended" and event.call_id not in self._closing_seen:
            self._closing_seen.add(event.call_id)
            self._calls.labels(pipeline, "ended").inc()
            self._active.labels(pipeline).dec()
        if event.elapsed_seconds > 0:
            self._cost_per_minute.labels(pipeline).set(event.cost_per_minute_usd)

    def _add_cost(self, pipeline: str, cost: dict[str, float]) -> None:
        for component, value in cost.items():
            if component != "total" and value > 0:
                self._cost.labels(pipeline, component).inc(value)
