from __future__ import annotations

from backend.application.dto.call_context import CallContext
from backend.application.ports.call_repository import CallRepository
from backend.application.ports.clock import Clock
from backend.application.ports.metrics_sink import CallMetricsEvent, MetricsSink
from backend.application.services.call_meter import segment_usage
from backend.domain.entities.call import Call
from backend.domain.services.cost_calculator import CostCalculator


class EndCall:
    def __init__(
        self,
        repository: CallRepository,
        metrics: MetricsSink,
        calculator: CostCalculator,
        clock: Clock,
    ) -> None:
        self._repository = repository
        self._metrics = metrics
        self._calculator = calculator
        self._clock = clock

    async def execute(self, ctx: CallContext, failed: bool = False) -> Call:
        """Bills the wall time after the last turn, then releases the concurrency slot.
        The segment must close before release, or the tail is divided by one call too few."""
        now = self._clock.monotonic()
        share, stt_seconds = ctx.meter.close_segment(now)
        if ctx.supervisor is not None:
            ctx.supervisor.release(ctx.call.id, now)

        ctx.call.closing_usage = segment_usage(share, stt_seconds, ctx.pipeline.uses_gpu)
        ctx.call.closing_cost = self._calculator.calculate(
            ctx.pipeline.kind, ctx.call.closing_usage
        )
        if failed:
            ctx.call.fail(self._clock.now())
        else:
            ctx.call.complete(self._clock.now())
        await self._repository.update(ctx.call)
        await self._metrics.emit(
            CallMetricsEvent(
                kind="ended",
                call_id=ctx.call.id,
                pipeline=ctx.pipeline.kind,
                elapsed_seconds=now - ctx.started_monotonic,
                running_cost=ctx.call.cost,
            )
        )
        return ctx.call
