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
    assert [t.user_text for t in call.turns] == [
        "", "I need a lawyer for my lease.", "Anna Weber."
    ]
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
    ctx.pipeline = Pipeline(
        ctx.pipeline.kind, DeafStt(), ctx.pipeline.llm, ctx.pipeline.tts, False
    )
    caller = Caller(world, [600])
    caller.session = session
    call = await session.run(caller.frames())
    assert call.turns[-1].user_text == "partial words"
