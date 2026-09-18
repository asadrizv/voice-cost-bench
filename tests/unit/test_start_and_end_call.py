import pytest

from backend.application.services.concurrency_supervisor import CapacityExceeded
from backend.application.use_cases.start_call import BudgetExceeded
from backend.domain.entities.call import CallStatus
from backend.domain.entities.cost import CostBreakdown, Money
from backend.domain.value_objects.pipeline_kind import PipelineKind
from tests.unit.conftest import World, make_world


async def test_start_call_persists_and_announces(world: World) -> None:
    ctx = await world.start_call.execute(PipelineKind.API, "law_firm")
    stored = await world.repo.get(ctx.call.id)
    assert stored.status is CallStatus.ACTIVE and stored.pipeline is PipelineKind.API
    assert world.metrics.events[0].kind == "started"  # type: ignore[attr-defined]
    assert world.supervisor.active == 0  # API calls don't occupy the GPU pool


async def test_selfhosted_call_takes_and_returns_a_slot(world: World) -> None:
    ctx = await world.start_call.execute(PipelineKind.SELFHOSTED, "law_firm")
    assert world.supervisor.active == 1
    world.clock.advance(30)
    call = await world.end_call.execute(ctx)
    assert world.supervisor.active == 0
    assert call.status is CallStatus.COMPLETED
    assert call.closing_usage.gpu_seconds == pytest.approx(30)
    assert call.closing_cost.gpu.amount > 0
    stored = await world.repo.get(ctx.call.id)
    assert stored.status is CallStatus.COMPLETED and stored.closing_cost == call.closing_cost


async def test_capacity_is_enforced_and_rejection_is_not_persisted() -> None:
    world = make_world(ceiling=1)
    await world.start_call.execute(PipelineKind.SELFHOSTED, "law_firm")
    with pytest.raises(CapacityExceeded):
        await world.start_call.execute(PipelineKind.SELFHOSTED, "law_firm")
    assert len(await world.repo.list()) == 1


async def test_budget_guard_blocks_api_but_not_selfhosted() -> None:
    world = make_world(spend_limit=1.0)
    ctx = await world.start_call.execute(PipelineKind.API, "law_firm")
    ctx.call.closing_cost = CostBreakdown(llm=Money.of("1.5"))
    await world.repo.update(ctx.call)
    with pytest.raises(BudgetExceeded):
        await world.start_call.execute(PipelineKind.API, "law_firm")
    await world.start_call.execute(PipelineKind.SELFHOSTED, "law_firm")


async def test_slot_is_released_if_persisting_fails(world: World) -> None:
    async def broken_add(call: object) -> None:
        raise RuntimeError("db down")

    world.repo.add = broken_add  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await world.start_call.execute(PipelineKind.SELFHOSTED, "law_firm")
    assert world.supervisor.active == 0


async def test_end_call_can_mark_failure(world: World) -> None:
    ctx = await world.start_call.execute(PipelineKind.API, "law_firm")
    call = await world.end_call.execute(ctx, failed=True)
    assert call.status is CallStatus.FAILED
    assert world.metrics.events[-1].kind == "ended"  # type: ignore[attr-defined]
