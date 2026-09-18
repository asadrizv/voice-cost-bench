import pytest

from backend.application.use_cases.compare_pipelines import ComparePipelines
from backend.application.use_cases.compute_call_cost import ComputeCallCost
from backend.application.use_cases.handle_call_turn import TurnRequest
from backend.domain.entities.latency import LatencyStage, TurnTimeline
from backend.domain.value_objects.pipeline_kind import PipelineKind
from tests.fakes import RecordingOutput
from tests.unit.conftest import World


async def run_call(world: World, kind: PipelineKind, seconds: float, turns: int = 2) -> str:
    ctx = await world.start_call.execute(kind, "law_firm")
    for _ in range(turns):
        world.clock.advance(seconds / (turns + 1))
        t = world.clock.t
        await world.handle_turn.execute(
            ctx,
            TurnRequest(
                user_text="Hello.",
                output=RecordingOutput(),
                timeline=TurnTimeline(speech_end=t - 0.4, endpoint=t),
            ),
        )
    world.clock.advance(seconds / (turns + 1))
    await world.end_call.execute(ctx)
    return ctx.call.id


async def test_call_cost_report_is_traceable_to_units_and_rates(world: World) -> None:
    call_id = await run_call(world, PipelineKind.API, 60)
    report = await ComputeCallCost(world.repo, world.calculator).execute(call_id)
    assert report.cost.total.as_float() == pytest.approx(report.recomputed_total_usd)
    assert report.rate_card_verified_on == "2026-09-17"
    assert report.usage.llm_input_tokens == 1000
    assert report.cost_per_minute_usd == pytest.approx(
        report.cost.total.as_float() * 60 / report.duration_seconds
    )
    assert report.projected_per_1000_minutes_usd == pytest.approx(report.cost_per_minute_usd * 1000)
    assert report.perceived_delay_p95_ms > 0


async def test_compare_pipelines_aggregates_per_pipeline(world: World) -> None:
    await run_call(world, PipelineKind.API, 60)
    await run_call(world, PipelineKind.API, 120)
    await run_call(world, PipelineKind.SELFHOSTED, 60)
    active = await world.start_call.execute(PipelineKind.SELFHOSTED, "law_firm")  # excluded
    assert active

    summary = await ComparePipelines(world.repo).execute()
    api, selfhosted = summary[PipelineKind.API], summary[PipelineKind.SELFHOSTED]
    assert api.calls == 2 and api.turns == 4
    assert api.total_minutes == pytest.approx(3.0, abs=0.05)  # fakes add ~0.4s per turn
    assert api.cost_per_minute_usd == pytest.approx(api.cost.total.as_float() / api.total_minutes)
    assert selfhosted.calls == 1
    assert selfhosted.cost.gpu.amount > 0 and selfhosted.cost.llm.amount == 0
    assert api.latency[LatencyStage.PERCEIVED_DELAY].count == 4
