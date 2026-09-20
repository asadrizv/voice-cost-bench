from dataclasses import replace

import pytest

from backend.application.ports.component_catalogue import ComponentKind
from backend.application.ports.running_engines import Engine
from backend.application.services.concurrency_supervisor import CapacityExceeded
from backend.application.use_cases.start_call import BudgetExceeded, VoiceNotAvailable
from backend.domain.entities.call import CallStatus
from backend.domain.entities.cost import CostBreakdown, Money
from backend.domain.value_objects.pipeline_kind import PipelineKind
from tests.fakes import PERSONA, StaticEngines, UnreachableEngines
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


async def test_a_voice_no_loaded_engine_speaks_is_refused_before_the_call() -> None:
    """The German persona names a Qwen3-TTS voice, and TTS_ENGINES defaults to Kokoro
    alone. Left to synthesis, the mismatch raised on the greeting turn: the caller heard
    nothing at all and the operator saw one 400 per call."""
    world = make_world(
        persona=replace(PERSONA, voices={"api": "voice-a", "selfhosted": "qwen3-tts:clara_de"}),
        engines=StaticEngines({ComponentKind.TTS: [Engine("kokoro", "hexgrad/Kokoro-82M")]}),
    )

    with pytest.raises(VoiceNotAvailable, match="qwen3-tts"):
        await world.start_call.execute(PipelineKind.SELFHOSTED, "law_firm")

    assert await world.repo.list() == []
    assert world.supervisor.active == 0
    # The API pipeline speaks its own voice through a vendor, so it is unaffected.
    await world.start_call.execute(PipelineKind.API, "law_firm")


async def test_a_voice_the_service_does_speak_starts_the_call() -> None:
    world = make_world(
        persona=replace(PERSONA, voices={"api": "voice-a", "selfhosted": "qwen3-tts:clara_de"}),
        engines=StaticEngines(
            {
                ComponentKind.TTS: [
                    Engine("kokoro", "hexgrad/Kokoro-82M"),
                    Engine("qwen3-tts", "Qwen/Qwen3-TTS"),
                ]
            }
        ),
    )

    ctx = await world.start_call.execute(PipelineKind.SELFHOSTED, "law_firm")

    assert ctx.persona.voices["selfhosted"] == "qwen3-tts:clara_de"


async def test_a_service_that_cannot_be_asked_does_not_block_the_call(world: World) -> None:
    """An unreachable TTS service is not evidence that the voice is missing, and refusing
    every call on a failed probe would be a worse outage than the one being prevented."""
    world = make_world(
        persona=replace(PERSONA, voices={"api": "voice-a", "selfhosted": "qwen3-tts:clara_de"}),
        engines=UnreachableEngines(),
    )

    ctx = await world.start_call.execute(PipelineKind.SELFHOSTED, "law_firm")

    assert ctx.call.pipeline is PipelineKind.SELFHOSTED
