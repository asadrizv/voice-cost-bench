"""Composition root: the one place that picks concrete adapters for every port."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from backend.application.dto.call_context import CallContext
from backend.application.ports.audio_output import AudioOutput
from backend.application.ports.call_repository import CallRepository
from backend.application.ports.component_catalogue import ComponentCatalogue
from backend.application.ports.endpoint_detector import EndpointDetector
from backend.application.ports.metrics_sink import MetricsSink
from backend.application.ports.pipeline_provider import PipelineProvider
from backend.application.services.concurrency_supervisor import ConcurrencySupervisor
from backend.application.services.endpointing import (
    EndpointerKind,
    SemanticEndpointDetector,
    SilenceEndpointDetector,
    SmartTurnEndpointDetector,
)
from backend.application.use_cases.call_session import CallSession, SessionConfig
from backend.application.use_cases.check_gpu_budget import CheckGpuBudget
from backend.application.use_cases.compare_pipelines import ComparePipelines
from backend.application.use_cases.compute_call_cost import ComputeCallCost
from backend.application.use_cases.describe_components import DescribeComponents
from backend.application.use_cases.end_call import EndCall
from backend.application.use_cases.handle_call_turn import HandleCallTurn
from backend.application.use_cases.start_call import StartCall
from backend.domain.services.cost_calculator import CostCalculator
from backend.domain.services.gpu_memory_budget import BudgetVerdict, GpuMemoryBudget
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.component_catalogue import (
    YamlComponentCatalogue,
    require_eu_residency,
)
from backend.infrastructure.config.personas import YamlPersonaProvider
from backend.infrastructure.config.settings import Settings
from backend.infrastructure.persistence.inmemory_call_repository import InMemoryCallRepository
from backend.infrastructure.persistence.postgres_call_repository import SqlCallRepository
from backend.infrastructure.pipeline_factory import (
    SELFHOSTED_ENGINES,
    PipelineFactory,
    configured_components,
    llm_memory_share,
    service_info_urls,
)
from backend.infrastructure.pricing.yaml_rate_card import YamlRateCardProvider
from backend.infrastructure.running_engines import HttpRunningEngines
from backend.infrastructure.system_clock import SystemClock
from backend.infrastructure.vad.energy_vad import EnergyVad

log = logging.getLogger(__name__)

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
    catalogue: ComponentCatalogue
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
    describe_components: DescribeComponents
    gpu_budget: CheckGpuBudget
    turn_model: Callable[[bytes], float] | None = None
    """Smart Turn's model call; None loads the real one the first time it is selected."""

    def endpointer(self, kind: EndpointerKind | None = None) -> EndpointDetector:
        match kind or self.settings.endpointer:
            case EndpointerKind.SILENCE:
                return SilenceEndpointDetector()
            case EndpointerKind.SEMANTIC:
                return SemanticEndpointDetector()
            case EndpointerKind.SMART_TURN:
                # Imported here so processes that never select it pay neither onnxruntime's
                # import nor the model download.
                from backend.infrastructure.endpointing import smart_turn

                if self.turn_model is None:
                    smart_turn.warm()
                return SmartTurnEndpointDetector(
                    self.turn_model or smart_turn.probability, executor=smart_turn.pool()
                )

    def session(
        self,
        ctx: CallContext,
        output: AudioOutput,
        endpointer: EndpointDetector | None = None,
        config: SessionConfig | None = None,
    ) -> CallSession:
        return CallSession(
            ctx,
            self.handle_turn,
            self.end_call,
            EnergyVad(),
            endpointer or self.endpointer(),
            output,
            self.clock,
            self.metrics,
            self.calculator,
            config,
        )


@dataclass(frozen=True)
class StaticConfig:
    """The immutable config every call in a process shares."""

    rates: YamlRateCardProvider
    catalogue: YamlComponentCatalogue
    personas: YamlPersonaProvider
    gpu_budget: CheckGpuBudget


def load_static_config(settings: Settings) -> StaticConfig:
    """Parses and validates everything a container needs that a call cannot change, so a
    process fails at startup on config that would otherwise fail every call. Raises the
    loaders' errors.

    Hand the result to `build_container`: the agent builds a container per call on a thread
    executor, and re-reading these five YAML files there holds the GIL for ~5 ms — a quarter
    of an audio frame stolen from every other call in the process.
    """
    rates = YamlRateCardProvider(settings.rates_path)
    catalogue = build_catalogue(settings, rates)
    personas = YamlPersonaProvider(settings.personas_dir)
    personas.validate_all()
    gpu_budget = build_gpu_budget_check(settings, catalogue)
    warn_over_budget(
        gpu_budget.execute(catalogue.components(PipelineKind.SELFHOSTED)), settings.eu_only
    )
    return StaticConfig(rates, catalogue, personas, gpu_budget)


def build_gpu_budget_check(settings: Settings, catalogue: ComponentCatalogue) -> CheckGpuBudget:
    return CheckGpuBudget(catalogue.gpu_memory(), llm_memory_share(settings))


def warn_over_budget(budget: GpuMemoryBudget | None, eu_only: bool) -> None:
    """Warns rather than refuses: every figure behind the verdict is an estimate until #10
    measures one on an L40S, and a wrong estimate must not stop a run. Under the EU-only
    profile a definite overrun is an error: no other pipeline may take the calls, so a card
    that cannot hold the stack leaves nothing to answer them with. An unknown verdict stays
    a warning; an error nobody can act on teaches operators to ignore the ones they can."""
    if budget is None or budget.verdict is BudgetVerdict.FITS:
        return
    if eu_only and budget.verdict is BudgetVerdict.DOES_NOT_FIT:
        log.error("EU_ONLY has no pipeline to fall back on. %s", budget.summary())
    else:
        log.warning("GPU memory budget: %s", budget.summary())


def build_catalogue(settings: Settings, rates: YamlRateCardProvider) -> YamlComponentCatalogue:
    """Raises ComponentCatalogueError when a configured component, or an engine a self-hosted
    service can run, has no complete entry, and NotEuResident when EU_ONLY is set and one of
    them is not EU-resident."""
    catalogue = YamlComponentCatalogue(
        settings.components_path,
        configured_components(settings, rates.telephony_provider()),
        SELFHOSTED_ENGINES,
    )
    if settings.eu_only:
        require_eu_residency(catalogue, settings.selectable_pipelines, SELFHOSTED_ENGINES)
    return catalogue


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
    turn_model: Callable[[bytes], float] | None = None,
    static: StaticConfig | None = None,
) -> Container:
    clock = SystemClock()
    repository = repository or build_repository(settings)
    static = static or load_static_config(settings)
    rates, catalogue, personas = static.rates, static.catalogue, static.personas
    calculator = CostCalculator(rates.rate_card())
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
        catalogue=catalogue,
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
        turn_model=turn_model,
        gpu_budget=static.gpu_budget,
        describe_components=DescribeComponents(
            catalogue,
            None
            if settings.simulate_providers
            else HttpRunningEngines(service_info_urls(settings), clock),
        ),
    )
