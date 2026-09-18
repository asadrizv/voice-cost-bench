from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.application.ports.pipeline_provider import Pipeline
from backend.application.services.concurrency_supervisor import ConcurrencySupervisor
from backend.application.use_cases.end_call import EndCall
from backend.application.use_cases.handle_call_turn import HandleCallTurn
from backend.application.use_cases.start_call import StartCall
from backend.domain.services.cost_calculator import CostCalculator
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.persistence.inmemory_call_repository import InMemoryCallRepository
from tests.fakes import (
    PERSONA,
    RATE_CARD,
    FakeClock,
    FakeLlm,
    FakeTts,
    NullMetrics,
    ScriptedStt,
    StaticPersonas,
    StaticPipelines,
)


@dataclass
class World:
    clock: FakeClock
    repo: InMemoryCallRepository
    metrics: NullMetrics
    calculator: CostCalculator
    llm: FakeLlm
    tts: FakeTts
    stt: ScriptedStt
    supervisor: ConcurrencySupervisor
    start_call: StartCall
    handle_turn: HandleCallTurn
    end_call: EndCall


def make_world(
    replies: list[str] | None = None,
    utterances: list[str] | None = None,
    spend_limit: float | None = None,
    ceiling: int = 4,
    llm: FakeLlm | None = None,
    tts: FakeTts | None = None,
) -> World:
    clock = FakeClock()
    repo = InMemoryCallRepository()
    metrics = NullMetrics()
    calculator = CostCalculator(RATE_CARD)
    llm = llm or FakeLlm(replies or ["Certainly. May I have your name, please?"], clock)
    tts = tts or FakeTts(clock)
    stt = ScriptedStt(utterances or [], clock)
    pipelines = StaticPipelines(
        {
            PipelineKind.API: Pipeline(PipelineKind.API, stt, llm, tts, uses_gpu=False),
            PipelineKind.SELFHOSTED: Pipeline(
                PipelineKind.SELFHOSTED, stt, llm, tts, uses_gpu=True
            ),
        }
    )
    supervisor = ConcurrencySupervisor(ceiling)
    ids = iter(f"call-{i}" for i in range(1000))
    start = StartCall(
        repo,
        pipelines,
        StaticPersonas(PERSONA),
        {PipelineKind.SELFHOSTED: supervisor},
        metrics,
        clock,
        dev_spend_limit_usd=spend_limit,
        id_factory=lambda: next(ids),
    )
    return World(
        clock=clock,
        repo=repo,
        metrics=metrics,
        calculator=calculator,
        llm=llm,
        tts=tts,
        stt=stt,
        supervisor=supervisor,
        start_call=start,
        handle_turn=HandleCallTurn(repo, metrics, calculator, clock),
        end_call=EndCall(repo, metrics, calculator, clock),
    )


@pytest.fixture
def world() -> World:
    return make_world()
