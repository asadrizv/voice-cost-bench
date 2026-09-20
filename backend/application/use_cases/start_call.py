from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping

from backend.application.dto.call_context import CallContext
from backend.application.ports.call_repository import CallRepository
from backend.application.ports.clock import Clock
from backend.application.ports.component_catalogue import ComponentKind
from backend.application.ports.metrics_sink import CallMetricsEvent, MetricsSink
from backend.application.ports.persona_provider import PersonaProvider
from backend.application.ports.pipeline_provider import PipelineProvider
from backend.application.ports.running_engines import EngineReportUnavailable, RunningEngines
from backend.application.services.call_meter import CallMeter
from backend.application.services.concurrency_supervisor import ConcurrencySupervisor
from backend.domain.entities.call import Call
from backend.domain.entities.cost import CostBreakdown
from backend.domain.entities.persona import Persona
from backend.domain.value_objects.pipeline_kind import PipelineKind


class BudgetExceeded(Exception):
    pass


class VoiceNotAvailable(Exception):
    """The persona's voice names a TTS engine this deployment is not running."""


ENGINE_SEPARATOR = ":"
"""How a voice id names its engine, as gpu/kokoro_service routes it: `<engine>:<voice>`.
A bare id belongs to whichever engine the service loaded first."""


class StartCall:
    def __init__(
        self,
        repository: CallRepository,
        pipelines: PipelineProvider,
        personas: PersonaProvider,
        supervisors: Mapping[PipelineKind, ConcurrencySupervisor],
        metrics: MetricsSink,
        clock: Clock,
        engines: RunningEngines | None = None,
        dev_spend_limit_usd: float | None = None,
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._repository = repository
        self._pipelines = pipelines
        self._personas = personas
        self._supervisors = supervisors
        self._metrics = metrics
        self._clock = clock
        self._engines = engines
        self._spend_limit = dev_spend_limit_usd
        self._id_factory = id_factory

    async def _require_a_voice(self, kind: PipelineKind, persona: Persona) -> None:
        """Refuses here rather than at synthesis: an engine the service never loaded fails
        on the greeting turn, where the caller hears silence and the operator sees one 400
        per call. An unreachable service is not evidence the voice is missing, so a failed
        probe lets the call through — the outage from refusing every call would be worse.
        """
        engine, separator, _ = persona.voices.get(kind.value, "").partition(ENGINE_SEPARATOR)
        if kind is not PipelineKind.SELFHOSTED or not separator or self._engines is None:
            return
        try:
            running = [e.id for e in await self._engines.report(ComponentKind.TTS)]
        except EngineReportUnavailable:
            return
        if engine not in running:
            raise VoiceNotAvailable(
                f"persona {persona.id!r} speaks with {engine!r}, which this deployment is "
                f"not running (TTS_ENGINES loaded {', '.join(running) or 'nothing'})"
            )

    async def execute(
        self,
        pipeline_kind: PipelineKind,
        persona_id: str,
        source: str = "browser",
        call_id: str | None = None,
    ) -> CallContext:
        """Raises BudgetExceeded (API pipeline over DEV_SPEND_LIMIT_USD), CapacityExceeded
        (pool at its concurrency ceiling) or VoiceNotAvailable (the persona's voice names a
        self-hosted engine that is not loaded). None persists a call."""
        if pipeline_kind is PipelineKind.API and self._spend_limit is not None:
            spent = await self._repository.total_spend_usd(PipelineKind.API)
            if spent >= self._spend_limit:
                raise BudgetExceeded(
                    f"API spend ${spent:.2f} has reached the ${self._spend_limit:.2f} limit"
                )

        persona = self._personas.get(persona_id)
        await self._require_a_voice(pipeline_kind, persona)
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
