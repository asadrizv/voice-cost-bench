from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping

from backend.application.dto.call_context import CallContext
from backend.application.ports.call_repository import CallRepository
from backend.application.ports.clock import Clock
from backend.application.ports.metrics_sink import CallMetricsEvent, MetricsSink
from backend.application.ports.persona_provider import PersonaProvider
from backend.application.ports.pipeline_provider import PipelineProvider
from backend.application.services.call_meter import CallMeter
from backend.application.services.concurrency_supervisor import ConcurrencySupervisor
from backend.domain.entities.call import Call
from backend.domain.entities.cost import CostBreakdown
from backend.domain.value_objects.pipeline_kind import PipelineKind


class BudgetExceeded(Exception):
    pass


class StartCall:
    def __init__(
        self,
        repository: CallRepository,
        pipelines: PipelineProvider,
        personas: PersonaProvider,
        supervisors: Mapping[PipelineKind, ConcurrencySupervisor],
        metrics: MetricsSink,
        clock: Clock,
        dev_spend_limit_usd: float | None = None,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._repository = repository
        self._pipelines = pipelines
        self._personas = personas
        self._supervisors = supervisors
        self._metrics = metrics
        self._clock = clock
        self._spend_limit = dev_spend_limit_usd
        self._id_factory = id_factory

    async def execute(
        self,
        pipeline_kind: PipelineKind,
        persona_id: str,
        source: str = "browser",
        call_id: str | None = None,
    ) -> CallContext:
        """Raises BudgetExceeded (API pipeline over DEV_SPEND_LIMIT_USD) or
        CapacityExceeded (pool at its concurrency ceiling). Neither persists a call."""
        if pipeline_kind is PipelineKind.API and self._spend_limit is not None:
            spent = await self._repository.total_spend_usd(PipelineKind.API)
            if spent >= self._spend_limit:
                raise BudgetExceeded(
                    f"API spend ${spent:.2f} has reached the ${self._spend_limit:.2f} limit"
                )

        persona = self._personas.get(persona_id)
        pipeline = self._pipelines.resolve(pipeline_kind)
        call_id = call_id or self._id_factory()
        now = self._clock.monotonic()
        supervisor = self._supervisors.get(pipeline_kind)
        if supervisor is not None:
            supervisor.acquire(call_id, now)

        try:
            call = Call(
                id=call_id,
                pipeline=pipeline_kind,
                persona=persona.id,
                started_at=self._clock.now(),
                source=source,
            )
            await self._repository.add(call)
        except BaseException:
            if supervisor is not None:
                supervisor.release(call_id, self._clock.monotonic())
            raise

        await self._metrics.emit(
            CallMetricsEvent(
                kind="started",
                call_id=call_id,
                pipeline=pipeline_kind,
                elapsed_seconds=0.0,
                running_cost=CostBreakdown.zero(),
            )
        )
        return CallContext(
            call=call,
            pipeline=pipeline,
            persona=persona,
            meter=CallMeter(now, supervisor if pipeline.uses_gpu else None),
            started_monotonic=now,
            supervisor=supervisor,
        )
