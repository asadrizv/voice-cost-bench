"""Composition root: the one place that picks concrete adapters for every port."""

from __future__ import annotations

from dataclasses import dataclass

from backend.application.dto.call_context import CallContext
from backend.application.ports.audio_output import AudioOutput
from backend.application.ports.call_repository import CallRepository
from backend.application.ports.endpoint_detector import EndpointDetector
from backend.application.ports.metrics_sink import MetricsSink
from backend.application.ports.pipeline_provider import PipelineProvider
from backend.application.services.concurrency_supervisor import ConcurrencySupervisor
from backend.application.services.endpointing import (
    SemanticEndpointDetector,
    SilenceEndpointDetector,
)
from backend.application.use_cases.call_session import CallSession, SessionConfig
from backend.application.use_cases.compare_pipelines import ComparePipelines
from backend.application.use_cases.compute_call_cost import ComputeCallCost
from backend.application.use_cases.end_call import EndCall
from backend.application.use_cases.handle_call_turn import HandleCallTurn
from backend.application.use_cases.start_call import StartCall
from backend.domain.services.cost_calculator import CostCalculator
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.personas import YamlPersonaProvider
from backend.infrastructure.config.settings import Settings
from backend.infrastructure.persistence.inmemory_call_repository import InMemoryCallRepository
from backend.infrastructure.persistence.postgres_call_repository import SqlCallRepository
from backend.infrastructure.pipeline_factory import PipelineFactory
from backend.infrastructure.pricing.yaml_rate_card import YamlRateCardProvider
from backend.infrastructure.system_clock import SystemClock
from backend.infrastructure.vad.energy_vad import EnergyVad

_supervisors: dict[int, dict[PipelineKind, ConcurrencySupervisor]] = {}


def process_supervisors(ceiling: int) -> dict[PipelineKind, ConcurrencySupervisor]:
    """One supervisor per process, shared by every call in it. The agent runs jobs as
    threads, not processes, precisely so this stays the single authoritative count."""
    if ceiling not in _supervisors:
        _supervisors[ceiling] = {PipelineKind.SELFHOSTED: ConcurrencySupervisor(ceiling)}
    return _supervisors[ceiling]


@dataclass
class Container:
    settings: Settings
    clock: SystemClock
    repository: CallRepository
    rates: YamlRateCardProvider
    calculator: CostCalculator
    personas: YamlPersonaProvider
    pipelines: PipelineProvider
    supervisors: dict[PipelineKind, ConcurrencySupervisor]
    metrics: MetricsSink
    start_call: StartCall
    handle_turn: HandleCallTurn
    end_call: EndCall
    compute_cost: ComputeCallCost
    compare: ComparePipelines

    def endpointer(self, kind: str | None = None) -> EndpointDetector:
        if (kind or self.settings.endpointer) == "silence":
            return SilenceEndpointDetector()
        return SemanticEndpointDetector()

    def session(
        self,
        ctx: CallContext,
        output: AudioOutput,
        endpointer: str | None = None,
        config: SessionConfig | None = None,
    ) -> CallSession:
        return CallSession(
            ctx,
            self.handle_turn,
            self.end_call,
            EnergyVad(),
            self.endpointer(endpointer),
            output,
            self.clock,
            self.metrics,
            self.calculator,
            config,
        )


def build_repository(settings: Settings) -> CallRepository:
    if settings.database_url:
        return SqlCallRepository.from_url(settings.database_url)
    return InMemoryCallRepository()


def build_container(
    settings: Settings,
    metrics: MetricsSink,
    repository: CallRepository | None = None,
    pipelines: PipelineProvider | None = None,
    supervisors: dict[PipelineKind, ConcurrencySupervisor] | None = None,
) -> Container:
    clock = SystemClock()
    repository = repository or build_repository(settings)
    rates = YamlRateCardProvider(settings.rates_path)
    calculator = CostCalculator(rates.rate_card())
    personas = YamlPersonaProvider(settings.personas_dir)
    personas.validate_all()
    pipelines = pipelines or PipelineFactory(settings)
    supervisors = (
        supervisors
        if supervisors is not None
        else process_supervisors(settings.selfhosted_max_concurrency)
    )
    return Container(
        settings=settings,
        clock=clock,
        repository=repository,
        rates=rates,
        calculator=calculator,
        personas=personas,
        pipelines=pipelines,
        supervisors=supervisors,
        metrics=metrics,
        start_call=StartCall(
            repository,
            pipelines,
            personas,
            supervisors,
            metrics,
            clock,
            dev_spend_limit_usd=settings.dev_spend_limit_usd,
        ),
        handle_turn=HandleCallTurn(repository, metrics, calculator, clock),
        end_call=EndCall(repository, metrics, calculator, clock),
        compute_cost=ComputeCallCost(repository, calculator),
        compare=ComparePipelines(repository),
    )
