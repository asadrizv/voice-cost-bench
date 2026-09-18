import asyncio

import pytest

from backend.application.use_cases.handle_call_turn import TurnRequest
from backend.domain.entities.latency import TurnTimeline
from backend.domain.value_objects.pipeline_kind import PipelineKind
from tests.fakes import FakeLlm, FakeTts, RecordingOutput
from tests.unit.conftest import World, make_world


def timeline_at(clock_t: float) -> TurnTimeline:
    return TurnTimeline(speech_end=clock_t - 0.3, endpoint=clock_t, stt_final=clock_t + 0.05)


async def test_turn_streams_reply_and_measures_every_stage(world: World) -> None:
    ctx = await world.start_call.execute(PipelineKind.API, "law_firm")
    world.clock.advance(0.05)
    out = RecordingOutput()
    request = TurnRequest(
        user_text="I need help with a lease.", output=out, timeline=timeline_at(world.clock.t)
    )
    outcome = await world.handle_turn.execute(ctx, request)
    turn = outcome.turn

    assert outcome.audio_started and not turn.interrupted
    assert turn.agent_text.strip() == "Certainly. May I have your name, please?"
    assert world.tts.requests == ["Certainly.", "May I have your name, please?"]
    assert len(out.chunks) == 6

    lat = turn.latency
    assert lat is not None
    assert lat.endpoint_detected == pytest.approx(300)
    assert lat.llm_first_token == pytest.approx(200)
    assert lat.tts_first_byte == pytest.approx(100)
    # at least 300 endpoint + 200 first token + 10 (to "Certainly.") + 100 tts; the LLM keeps
    # streaming (and advancing the fake clock) while TTS starts, so allow a few tokens more
    assert 610 <= lat.perceived_delay <= 700
    assert lat.end_to_end == pytest.approx(lat.perceived_delay - 300)
    assert lat.llm_complete >= lat.llm_first_token


async def test_turn_records_usage_cost_and_history_inputs(world: World) -> None:
    ctx = await world.start_call.execute(PipelineKind.API, "law_firm")
    ctx.meter.add_stt_audio(4.0)
    outcome = await world.handle_turn.execute(
        ctx, TurnRequest(user_text="Hello.", output=RecordingOutput())
    )
    turn = outcome.turn
    assert turn.usage.llm_input_tokens == 500 and turn.usage.llm_output_tokens == 40
    assert turn.usage.tts_characters == len("Certainly.") + len("May I have your name, please?")
    assert turn.usage.stt_audio_seconds == 4.0
    assert turn.usage.gpu_seconds == 0
    assert turn.cost.stt.amount > 0 and turn.cost.llm.amount > 0 and turn.cost.tts.amount > 0
    stored = await world.repo.get(ctx.call.id)
    assert [t.index for t in stored.turns] == [0]
    assert world.metrics.events[-1].kind == "turn"  # type: ignore[attr-defined]
    sent = world.llm.calls[0]
    assert sent[0].role == "system" and sent[-1].content == "Hello."


async def test_greeting_skips_llm_and_has_no_latency(world: World) -> None:
    ctx = await world.start_call.execute(PipelineKind.SELFHOSTED, "law_firm")
    world.clock.advance(2)
    outcome = await world.handle_turn.execute(
        ctx, TurnRequest(user_text="", output=RecordingOutput(), fixed_reply="Hello. Welcome.")
    )
    assert world.llm.calls == []
    assert outcome.turn.latency is None
    assert outcome.turn.usage.llm_input_tokens == 0
    assert outcome.turn.usage.gpu_seconds == pytest.approx(2 + 0.1 * 2)
    assert outcome.turn.cost.gpu.amount > 0 and outcome.turn.cost.tts.amount == 0


async def test_interrupt_stops_playback_and_estimates_tokens() -> None:
    block = asyncio.Event()
    world = make_world(llm=None)
    world.tts = FakeTts(world.clock, block=block)
    world.llm = FakeLlm(["One. Two. Three."], world.clock, usage=None)
    ctx = await world.start_call.execute(PipelineKind.API, "law_firm")
    ctx.pipeline = type(ctx.pipeline)(
        ctx.pipeline.kind, ctx.pipeline.stt, world.llm, world.tts, ctx.pipeline.uses_gpu
    )
    out = RecordingOutput()
    request = TurnRequest(user_text="Hi there, I'm calling about my case.", output=out)
    task = asyncio.create_task(world.handle_turn.execute(ctx, request))
    while not out.chunks:
        await asyncio.sleep(0)
    request.interrupt.set()
    outcome = await task

    assert outcome.turn.interrupted and outcome.audio_started
    assert len(out.chunks) == 1
    assert outcome.turn.usage.llm_input_tokens > 0  # estimated, since no usage arrived


async def test_provider_error_is_recorded_then_raised() -> None:
    world = make_world()
    world.llm = FakeLlm(["x"], world.clock, error=RuntimeError("503 from provider"))
    ctx = await world.start_call.execute(PipelineKind.API, "law_firm")
    ctx.pipeline = type(ctx.pipeline)(
        ctx.pipeline.kind, ctx.pipeline.stt, world.llm, world.tts, ctx.pipeline.uses_gpu
    )
    with pytest.raises(RuntimeError, match="503"):
        await world.handle_turn.execute(ctx, TurnRequest(user_text="Hi.", output=RecordingOutput()))
    stored = await world.repo.get(ctx.call.id)
    assert len(stored.turns) == 1 and stored.turns[0].interrupted
