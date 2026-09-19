from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field, replace
from datetime import datetime

from backend.application.dto.call_context import CallContext
from backend.application.ports.audio_output import AudioOutput
from backend.application.ports.call_repository import CallRepository
from backend.application.ports.clock import Clock
from backend.application.ports.llm_port import ChatMessage, TokenDelta, TokenUsage
from backend.application.ports.metrics_sink import CallMetricsEvent, MetricsSink
from backend.application.services.call_meter import segment_usage
from backend.application.services.sentence_chunker import SentenceChunker
from backend.domain.entities.call import Turn
from backend.domain.entities.latency import TurnTimeline
from backend.domain.services.cost_calculator import CostCalculator
from backend.domain.value_objects.audio import AudioChunk

_LOOKAHEAD_SENTENCES = 2


@dataclass
class TurnRequest:
    user_text: str
    """Empty for the greeting."""
    output: AudioOutput
    timeline: TurnTimeline = field(default_factory=TurnTimeline)
    fixed_reply: str | None = None
    """Speak this verbatim instead of asking the LLM (the greeting)."""
    interrupt: asyncio.Event = field(default_factory=asyncio.Event)
    """Set by the session on barge-in; the turn stops and is recorded as interrupted."""


@dataclass
class TurnOutcome:
    turn: Turn
    audio_started: bool


@dataclass
class _Progress:
    agent_text: str = ""
    spoken_chars: int = 0
    usage: TokenUsage | None = None
    audio_started: bool = False


class HandleCallTurn:
    """STT output -> LLM -> TTS -> audio out for one turn; records usage, cost and latency.

    TTS starts on the first sentence the LLM finishes, and the next sentences synthesise
    while earlier ones play, so the caller waits for one sentence, never the whole reply.
    """

    def __init__(
        self,
        repository: CallRepository,
        metrics: MetricsSink,
        calculator: CostCalculator,
        clock: Clock,
    ) -> None:
        self._repository = repository
        self._metrics = metrics
        self._calculator = calculator
        self._clock = clock

    async def execute(self, ctx: CallContext, request: TurnRequest) -> TurnOutcome:
        """Returns once the reply has played or the turn was interrupted. A provider error is
        re-raised, but only after the partial turn (and the tokens it spent) is recorded."""
        started_at = self._clock.now()
        progress = _Progress()
        work = asyncio.create_task(self._respond(ctx, request, progress))
        interrupt = asyncio.create_task(request.interrupt.wait())
        error: BaseException | None = None
        try:
            await asyncio.wait({work, interrupt}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            interrupted = not work.done()
            for task in (work, interrupt):
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await interrupt
            try:
                await work
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                error = _unwrap(exc)

        turn = await self._record(
            ctx, request, progress, interrupted or error is not None, started_at
        )
        if error is not None:
            raise error
        return TurnOutcome(turn=turn, audio_started=progress.audio_started)

    async def _respond(self, ctx: CallContext, request: TurnRequest, progress: _Progress) -> None:
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        playback: asyncio.Queue[asyncio.Queue[AudioChunk | None] | None] = asyncio.Queue(
            maxsize=_LOOKAHEAD_SENTENCES
        )
        async with asyncio.TaskGroup() as tg:
            if request.fixed_reply is not None:
                tg.create_task(self._fixed_text(request.fixed_reply, sentences, progress))
            else:
                tg.create_task(self._generate(ctx, request, sentences, progress))
            tg.create_task(self._synthesize(ctx, request, sentences, playback, progress, tg))
            tg.create_task(self._play(request, playback, progress))

    async def _fixed_text(
        self, text: str, sentences: asyncio.Queue[str | None], progress: _Progress
    ) -> None:
        progress.agent_text = text
        chunker = SentenceChunker()
        for sentence in [*chunker.push(text), *chunker.flush()]:
            await sentences.put(sentence)
        await sentences.put(None)

    async def _generate(
        self,
        ctx: CallContext,
        request: TurnRequest,
        sentences: asyncio.Queue[str | None],
        progress: _Progress,
    ) -> None:
        timeline = request.timeline
        messages = [
            ChatMessage("system", ctx.persona.system_prompt),
            *ctx.history,
            ChatMessage("user", request.user_text),
        ]
        chunker = SentenceChunker()
        timeline.llm_start = self._clock.monotonic()
        async for event in ctx.pipeline.llm.complete(messages, ctx.persona.sampling):
            if isinstance(event, TokenDelta):
                if not event.text:
                    continue
                if timeline.llm_first_token is None:
                    timeline.llm_first_token = self._clock.monotonic()
                progress.agent_text += event.text
                for sentence in chunker.push(event.text):
                    await sentences.put(sentence)
            elif isinstance(event, TokenUsage):
                progress.usage = event
        timeline.llm_complete = self._clock.monotonic()
        for sentence in chunker.flush():
            await sentences.put(sentence)
        await sentences.put(None)

    async def _synthesize(
        self,
        ctx: CallContext,
        request: TurnRequest,
        sentences: asyncio.Queue[str | None],
        playback: asyncio.Queue[asyncio.Queue[AudioChunk | None] | None],
        progress: _Progress,
        tg: asyncio.TaskGroup,
    ) -> None:
        voice = ctx.persona.voices.get(ctx.pipeline.kind.value)
        while (sentence := await sentences.get()) is not None:
            audio: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
            # put() blocks once the lookahead is full, so synthesis never runs further ahead
            # of playback than _LOOKAHEAD_SENTENCES.
            await playback.put(audio)
            progress.spoken_chars += len(sentence)
            tg.create_task(self._synthesize_one(ctx, request, sentence, voice, audio))
        await playback.put(None)

    async def _synthesize_one(
        self,
        ctx: CallContext,
        request: TurnRequest,
        sentence: str,
        voice: str | None,
        audio: asyncio.Queue[AudioChunk | None],
    ) -> None:
        timeline = request.timeline
        if timeline.tts_start is None:
            timeline.tts_start = self._clock.monotonic()
        try:
            async for chunk in ctx.pipeline.tts.synthesize(sentence, voice):
                if chunk.is_empty():
                    continue
                if timeline.tts_first_byte is None:
                    timeline.tts_first_byte = self._clock.monotonic()
                await audio.put(chunk)
        finally:
            audio.put_nowait(None)

    async def _play(
        self,
        request: TurnRequest,
        playback: asyncio.Queue[asyncio.Queue[AudioChunk | None] | None],
        progress: _Progress,
    ) -> None:
        while (audio := await playback.get()) is not None:
            while (chunk := await audio.get()) is not None:
                if not progress.audio_started:
                    progress.audio_started = True
                    request.timeline.audio_out = self._clock.monotonic()
                await request.output.write(chunk)

    async def _record(
        self,
        ctx: CallContext,
        request: TurnRequest,
        progress: _Progress,
        interrupted: bool,
        started_at: datetime,
    ) -> Turn:
        now = self._clock.monotonic()
        share, stt_seconds = ctx.meter.close_segment(now)
        usage_tokens = progress.usage or _estimate_usage(ctx, request, progress)
        usage = replace(
            segment_usage(share, stt_seconds, ctx.pipeline.uses_gpu),
            llm_input_tokens=usage_tokens.input_tokens,
            llm_output_tokens=usage_tokens.output_tokens,
            tts_characters=progress.spoken_chars,
        )
        cost = self._calculator.calculate(ctx.pipeline.kind, usage)
        is_greeting = request.fixed_reply is not None
        turn = Turn(
            index=ctx.call.next_turn_index(),
            user_text=request.user_text,
            agent_text=progress.agent_text,
            usage=usage,
            cost=cost,
            # No agent audio, no response to time: a zero here would drag percentiles down.
            latency=None
            if is_greeting or not progress.audio_started
            else request.timeline.breakdown(),
            interrupted=interrupted,
            started_at=started_at,
        )
        ctx.call.add_turn(turn)
        await self._repository.add_turn(ctx.call.id, turn)
        await self._metrics.emit(
            CallMetricsEvent(
                kind="turn",
                call_id=ctx.call.id,
                pipeline=ctx.pipeline.kind,
                elapsed_seconds=now - ctx.started_monotonic,
                running_cost=ctx.call.cost,
                turn_index=turn.index,
                latency=turn.latency,
                turn_cost=cost,
                user_text=turn.user_text,
                agent_text=turn.agent_text,
                interrupted=interrupted,
            )
        )
        return turn


def _estimate_usage(ctx: CallContext, request: TurnRequest, progress: _Progress) -> TokenUsage:
    """Used only when the stream ended before the provider reported usage (barge-in, or the
    greeting, which makes no LLM call). ~4 chars/token is the usual English BPE average."""
    if request.fixed_reply is not None:
        return TokenUsage(0, 0, estimated=True)
    prompt_chars = len(ctx.persona.system_prompt) + len(request.user_text)
    prompt_chars += sum(len(m.content) for m in ctx.history)
    return TokenUsage(prompt_chars // 4, len(progress.agent_text) // 4, estimated=True)


def _unwrap(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc
