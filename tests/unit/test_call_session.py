from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from backend.application.dto.call_context import CallContext
from backend.application.ports.pipeline_provider import Pipeline
from backend.application.services.endpointing import SilenceEndpointDetector
from backend.application.use_cases.call_session import CallSession, SessionConfig
from backend.domain.entities.call import CallStatus
from backend.domain.value_objects.audio import AudioChunk
from backend.domain.value_objects.pipeline_kind import PipelineKind
from tests.fakes import ByteVad, FakeLlm, FakeTts, RecordingOutput, silence, speech
from tests.unit.conftest import World, make_world

FRAME_S = 0.02


class Caller:
    """Plays a scripted caller: waits for the agent to finish, speaks, falls silent."""

    def __init__(self, world: World, utterance_ms: list[int], settle_frames: int = 60) -> None:
        self.world = world
        self.utterance_ms = utterance_ms
        self.settle_frames = settle_frames
        self.session: CallSession | None = None

    async def frames(self) -> AsyncIterator[AudioChunk]:
        assert self.session is not None
        for ms in [*self.utterance_ms, None]:
            for _ in range(2000):
                if not self.session.agent_busy:
                    break
                yield await self._frame(silence())
            if ms is None:
                break
            for _ in range(ms // 20):
                yield await self._frame(speech())
            for _ in range(self.settle_frames):
                yield await self._frame(silence())

    async def _frame(self, chunk: AudioChunk) -> AudioChunk:
        self.world.clock.advance(FRAME_S)
        await asyncio.sleep(0)
        return chunk


async def make_session(
    world: World, kind: PipelineKind = PipelineKind.API, config: SessionConfig | None = None
) -> tuple[CallSession, RecordingOutput, CallContext]:
    ctx = await world.start_call.execute(kind, "law_firm")
    out = RecordingOutput()
    session = CallSession(
        ctx,
        world.handle_turn,
        world.end_call,
        ByteVad(),
        SilenceEndpointDetector(silence_ms=700),
        out,
        world.clock,
        world.metrics,
        world.calculator,
        config or SessionConfig(tick_interval_s=0.001),
    )
    return session, out, ctx


def swap_pipeline(ctx: CallContext, llm: FakeLlm | None = None, tts: FakeTts | None = None) -> None:
    p = ctx.pipeline
    ctx.pipeline = Pipeline(p.kind, p.stt, llm or p.llm, tts or p.tts, p.uses_gpu)


async def test_full_conversation_runs_and_is_accounted() -> None:
    world = make_world(
        replies=["May I have your name?", "Thank you, Ms. Weber. Tuesday at ten works."],
        utterances=["I need a lawyer for my lease.", "Anna Weber."],
    )
    session, out, ctx = await make_session(world, PipelineKind.SELFHOSTED)
    caller = Caller(world, [1000, 800])
    caller.session = session
    call = await session.run(caller.frames())

    assert call.status is CallStatus.COMPLETED
    assert [t.user_text for t in call.turns] == ["", "I need a lawyer for my lease.", "Anna Weber."]
    assert call.turns[0].latency is None
    for turn in call.turns[1:]:
        assert turn.latency is not None
        assert turn.latency.endpoint_detected == pytest.approx(700, abs=25)
        assert turn.latency.perceived_delay > turn.latency.end_to_end > 0
    assert [m.role for m in ctx.history] == ["assistant", "user", "assistant", "user", "assistant"]
    # every second of the call is billed exactly once across turns + closing segment
    billed = sum(t.usage.gpu_seconds for t in call.turns) + call.closing_usage.gpu_seconds
    assert billed == pytest.approx(call.duration_seconds(), abs=1e-6)
    assert call.usage.stt_audio_seconds == pytest.approx(world.stt.audio_seconds)
    assert world.supervisor.active == 0
    stored = await world.repo.get(call.id)
    assert len(stored.turns) == 3 and stored.status is CallStatus.COMPLETED
    kinds = {e.kind for e in world.metrics.events}  # type: ignore[attr-defined]
    assert {"started", "turn", "ended"} <= kinds
    assert out.chunks


async def test_empty_transcript_commits_no_turn() -> None:
    world = make_world(utterances=[""])
    session, _, _ = await make_session(world)
    caller = Caller(world, [600])
    caller.session = session
    call = await session.run(caller.frames())
    assert len(call.turns) == 1  # greeting only


async def test_barge_in_interrupts_agent_and_clears_playback() -> None:
    world = make_world(utterances=["Sorry, one more thing."])
    block = asyncio.Event()
    session, out, ctx = await make_session(world)
    swap_pipeline(ctx, tts=FakeTts(world.clock, block=block))

    async def frames() -> AsyncIterator[AudioChunk]:
        # greeting is stuck mid-playback; the caller talks over it
        while not out.chunks:
            world.clock.advance(FRAME_S)
            await asyncio.sleep(0)
            yield silence()
        for _ in range(20):
            world.clock.advance(FRAME_S)
            await asyncio.sleep(0)
            yield speech()
        block.set()
        for _ in range(60):
            world.clock.advance(FRAME_S)
            await asyncio.sleep(0)
            yield silence()
        while session.agent_busy:
            world.clock.advance(FRAME_S)
            await asyncio.sleep(0)
            yield silence()

    call = await session.run(frames())
    assert call.turns[0].interrupted
    assert out.clears == 1
    assert call.turns[1].user_text == "Sorry, one more thing."
    assert not call.turns[1].interrupted


async def test_words_cut_off_before_the_agent_spoke_carry_into_next_turn() -> None:
    world = make_world(utterances=["My landlord", "kept my deposit."])
    session, out, ctx = await make_session(
        world, config=SessionConfig(tick_interval_s=0.001, barge_in_min_speech_ms=100)
    )

    class SlowLlm(FakeLlm):
        """First reply never arrives before the caller resumes speaking."""

        gate = asyncio.Event()

        async def complete(self, messages, sampling):  # type: ignore[no-untyped-def,override]
            if len(self.calls) == 0:
                self.calls.append(list(messages))
                await self.gate.wait()
            async for e in super().complete(messages, sampling):
                yield e

    slow = SlowLlm(["x", "Understood, your deposit."], world.clock)
    swap_pipeline(ctx, llm=slow)
    caller = Caller(world, [600, 600], settle_frames=40)
    caller.session = session

    async def frames() -> AsyncIterator[AudioChunk]:
        async for f in caller.frames():
            yield f

    call = await session.run(frames())
    answered = [t for t in call.turns if not t.interrupted]
    assert answered[-1].user_text == "My landlord kept my deposit."
    assert any(t.interrupted and t.user_text == "My landlord" for t in call.turns)


async def test_provider_failure_fails_the_call_and_frees_the_slot() -> None:
    world = make_world(utterances=["Hello?"])
    session, _, ctx = await make_session(world, PipelineKind.SELFHOSTED)
    swap_pipeline(ctx, llm=FakeLlm(["x"], world.clock, error=RuntimeError("vLLM down")))
    caller = Caller(world, [600, 600])
    caller.session = session
    with pytest.raises(RuntimeError, match="vLLM down"):
        await session.run(caller.frames())
    stored = await world.repo.get(ctx.call.id)
    assert stored.status is CallStatus.FAILED
    assert world.supervisor.active == 0


async def test_flush_timeout_falls_back_to_collected_text() -> None:
    world = make_world()

    class DeafStt:
        async def stream(self, audio, language):  # type: ignore[no-untyped-def]
            from backend.application.ports.stt_port import TranscriptEvent

            yielded = False
            async for item in audio:
                if not yielded and not isinstance(item, AudioChunk):
                    continue
                if not yielded:
                    yielded = True
                    yield TranscriptEvent("partial words", is_final=False)

    session, _, ctx = await make_session(
        world, config=SessionConfig(tick_interval_s=0.001, flush_timeout_s=0.01)
    )
    ctx.pipeline = Pipeline(ctx.pipeline.kind, DeafStt(), ctx.pipeline.llm, ctx.pipeline.tts, False)
    caller = Caller(world, [600])
    caller.session = session
    call = await session.run(caller.frames())
    assert call.turns[-1].user_text == "partial words"


async def test_speech_during_the_agents_reply_is_not_billed_as_response_delay() -> None:
    """The caller talks (below barge-in) while the agent is still speaking; the agent then
    finishes and answers. The wait starts when the agent went quiet, not when they spoke."""
    world = make_world(utterances=["Thank you."])
    block = asyncio.Event()
    session, out, ctx = await make_session(
        world, config=SessionConfig(tick_interval_s=0.001, barge_in_min_speech_ms=10_000)
    )
    swap_pipeline(ctx, tts=FakeTts(world.clock, block=block))

    async def frames() -> AsyncIterator[AudioChunk]:
        async def tick(chunk: AudioChunk) -> AudioChunk:
            world.clock.advance(FRAME_S)
            await asyncio.sleep(0)
            return chunk

        while not out.chunks:
            yield await tick(silence())
        for _ in range(30):  # "thank you" over the greeting
            yield await tick(speech())
        for _ in range(200):  # the greeting keeps playing for 4 s
            yield await tick(silence())
        out.queued = 0.5  # half a second still buffered when the turn task ends
        block.set()
        for _ in range(100):
            yield await tick(silence())
        while session.agent_busy:
            yield await tick(silence())

    call = await session.run(frames())
    reply = call.turns[-1]
    assert reply.user_text == "Thank you."
    assert reply.latency is not None
    assert reply.latency.perceived_delay < 1500  # not the ~4.7 s since the caller spoke


async def test_endpointer_hears_what_the_agent_said() -> None:
    world = make_world(replies=["May I have your email?"], utterances=["Hello."])
    session, _, ctx = await make_session(world)
    heard: list[str] = []
    endpointer = session._endpointer  # noqa: SLF001
    original = endpointer.observe_agent_turn
    endpointer.observe_agent_turn = lambda text: (heard.append(text), original(text))[1]  # type: ignore[method-assign]
    caller = Caller(world, [600])
    caller.session = session
    await session.run(caller.frames())
    assert [h.strip() for h in heard] == [ctx.persona.greeting, "May I have your email?"]


async def test_llm_prompt_is_prewarmed_with_the_first_turns_prefix() -> None:
    world = make_world(utterances=["Hello."])
    session, _, ctx = await make_session(world)
    caller = Caller(world, [600])
    caller.session = session
    await session.run(caller.frames())
    (prefix,) = world.llm.prewarmed
    assert [m.content for m in prefix] == [ctx.persona.system_prompt, ctx.persona.greeting]
    first_turn = world.llm.calls[0]
    assert [m.content for m in first_turn[:2]] == [m.content for m in prefix]


async def test_prewarm_failure_does_not_affect_the_call() -> None:
    world = make_world(utterances=["Hello."])
    session, _, ctx = await make_session(world)

    async def broken(messages):  # type: ignore[no-untyped-def]
        raise RuntimeError("ollama restarting")

    world.llm.prewarm = broken  # type: ignore[method-assign]
    caller = Caller(world, [600])
    caller.session = session
    call = await session.run(caller.frames())
    assert call.status.value == "completed" and len(call.turns) == 2


async def test_agent_hangs_up_after_goodbyes_unless_the_caller_talks_over_it() -> None:
    world = make_world(
        replies=["You're welcome. Goodbye!", "Of course, what else?"],
        utterances=["That's all, thanks. Bye!", "Wait, one more thing."],
    )
    session, out, _ = await make_session(world)

    async def tick(chunk: AudioChunk) -> AudioChunk:
        world.clock.advance(FRAME_S)
        await asyncio.sleep(0)
        return chunk

    async def frames(interrupt: bool) -> AsyncIterator[AudioChunk]:
        while not out.chunks:  # greeting
            yield await tick(silence())
        while session.agent_busy:
            yield await tick(silence())
        for _ in range(30):  # "that's all, bye"
            yield await tick(speech())
        while len(session.call.turns) < 2 or session.agent_busy:
            yield await tick(silence())
        out.queued = 1.0  # the goodbye's last second is still in the speaker buffer
        if interrupt:
            for _ in range(20):  # "wait, one more thing" over the tail of the goodbye
                yield await tick(speech())
        for _ in range(200):  # 4 s: well past when the line would have been dropped
            out.queued = 0.0
            yield await tick(silence())
        while session.agent_busy:
            yield await tick(silence())

    call = await session.run(frames(interrupt=True))
    assert not session.agent_hung_up
    assert call.turns[-1].user_text == "Wait, one more thing."

    world2 = make_world(
        replies=["You're welcome. Goodbye!"], utterances=["That's all, thanks. Bye!"]
    )
    session, out, _ = await make_session(world2)
    world = world2
    call = await session.run(frames(interrupt=False))
    assert session.agent_hung_up and len(call.turns) == 2
