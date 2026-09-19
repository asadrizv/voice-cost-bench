from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass

from backend.application.dto.call_context import CallContext
from backend.application.ports.audio_output import AudioOutput
from backend.application.ports.clock import Clock
from backend.application.ports.endpoint_detector import EndpointDetector
from backend.application.ports.llm_port import ChatMessage
from backend.application.ports.metrics_sink import CallMetricsEvent, MetricsSink
from backend.application.ports.stt_port import FlushSignal
from backend.application.ports.vad_port import VoiceActivityDetector
from backend.application.services.call_meter import segment_usage
from backend.application.use_cases.end_call import EndCall
from backend.application.use_cases.handle_call_turn import HandleCallTurn, TurnRequest
from backend.domain.entities.call import Call
from backend.domain.entities.latency import TurnTimeline
from backend.domain.services.cost_calculator import CostCalculator
from backend.domain.value_objects.audio import AudioChunk

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionConfig:
    barge_in_min_speech_ms: float = 300
    """Sustained caller speech needed to interrupt the agent; shorter trips on coughs."""
    barge_in_gap_reset_ms: float = 200
    flush_timeout_s: float = 1.0
    tick_interval_s: float = 1.0


class CallSession:
    """Runs one call from greeting to hang-up: caller audio in, agent audio out.

    Transport-agnostic on purpose: the LiveKit agent and the load harness drive the same
    object, so the benchmark measures exactly the code path a caller hears.
    """

    def __init__(
        self,
        ctx: CallContext,
        handle_turn: HandleCallTurn,
        end_call: EndCall,
        vad: VoiceActivityDetector,
        endpointer: EndpointDetector,
        output: AudioOutput,
        clock: Clock,
        metrics: MetricsSink,
        calculator: CostCalculator,
        config: SessionConfig | None = None,
    ) -> None:
        self._ctx = ctx
        self._handle_turn = handle_turn
        self._end_call = end_call
        self._vad = vad
        self._endpointer = endpointer
        self._output = output
        self._clock = clock
        self._metrics = metrics
        self._calculator = calculator
        self._config = config or SessionConfig()

        self._stt_in: asyncio.Queue[AudioChunk | FlushSignal | None] = asyncio.Queue()
        self._finals: list[str] = []
        self._interim = ""
        self._pending_flush: asyncio.Future[str] | None = None
        self._turn_task: asyncio.Task[None] | None = None
        self._turn_request: TurnRequest | None = None
        self._speech_run_ms = 0.0
        self._gap_ms = 0.0
        self._carry_text = ""
        self._agent_silent_at: float | None = None
        self._turn_error: BaseException | None = None

    @property
    def call(self) -> Call:
        return self._ctx.call

    @property
    def agent_busy(self) -> bool:
        return self._turn_task is not None and not self._turn_task.done()

    async def run(self, audio_in: AsyncIterable[AudioChunk]) -> Call:
        stt_task = asyncio.create_task(self._consume_stt())
        ticker = asyncio.create_task(self._tick())
        warmup = asyncio.create_task(self._prewarm_llm())
        failed = False
        try:
            self._start_turn(
                TurnRequest(
                    user_text="", output=self._output, fixed_reply=self._ctx.persona.greeting
                ),
                transcribe=False,
            )
            async for chunk in audio_in:
                await self._on_audio(chunk)
                if self._turn_error is not None:
                    raise self._turn_error
            await self._finish_turn()
            if self._turn_error is not None:
                raise self._turn_error
        except BaseException:
            failed = True
            raise
        finally:
            await self._finish_turn(interrupt=True)
            self._stt_in.put_nowait(None)
            for background in (ticker, warmup):
                background.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await background
            try:
                await asyncio.wait_for(stt_task, timeout=2.0)
            except (TimeoutError, asyncio.CancelledError):
                pass
            except Exception:
                failed = True
            await self._end_call.execute(self._ctx, failed=failed)
        return self._ctx.call

    async def _on_audio(self, chunk: AudioChunk) -> None:
        now = self._clock.monotonic()
        self._ctx.meter.add_stt_audio(chunk.duration_seconds)
        self._stt_in.put_nowait(chunk)
        speech = self._vad.is_speech(chunk)
        self._endpointer.observe_audio(speech, now)

        if self.agent_busy:
            if self._barge_in_detected(speech, chunk.duration_seconds * 1000):
                await self._barge_in()
            return
        self._speech_run_ms = self._gap_ms = 0.0
        if self._endpointer.should_commit(now):
            timeline = TurnTimeline(speech_end=self._listening_since(), endpoint=now)
            self._endpointer.reset()
            self._start_turn(TurnRequest(user_text="", output=self._output, timeline=timeline))

    def _listening_since(self) -> float | None:
        """When the caller's wait began: the end of their speech, or the moment the agent
        fell silent if they spoke over it. Without this, words said during the agent's
        reply would bill the agent's own talking time as response delay."""
        speech_end = self._endpointer.speech_end
        if speech_end is None or self._agent_silent_at is None:
            return speech_end
        return max(speech_end, self._agent_silent_at)

    def _barge_in_detected(self, speech: bool, frame_ms: float) -> bool:
        """Speech accumulates across short gaps between words and resets after a real pause,
        so a sentence trips barge-in and a cough doesn't."""
        if speech:
            self._speech_run_ms += frame_ms
            self._gap_ms = 0.0
        else:
            self._gap_ms += frame_ms
            if self._gap_ms >= self._config.barge_in_gap_reset_ms:
                self._speech_run_ms = 0.0
        return self._speech_run_ms >= self._config.barge_in_min_speech_ms

    async def _barge_in(self) -> None:
        if self._turn_request is not None:
            self._turn_request.interrupt.set()
        await self._output.clear()
        self._speech_run_ms = self._gap_ms = 0.0

    def _start_turn(self, request: TurnRequest, transcribe: bool = True) -> None:
        self._turn_request = request
        self._turn_task = asyncio.create_task(self._run_turn(request, transcribe))

    async def _finish_turn(self, interrupt: bool = False) -> None:
        if self._turn_task is None:
            return
        if interrupt and self._turn_request is not None:
            self._turn_request.interrupt.set()
        with contextlib.suppress(asyncio.CancelledError):
            await self._turn_task

    async def _run_turn(self, request: TurnRequest, transcribe: bool) -> None:
        try:
            await self._respond(request, transcribe)
        finally:
            self._agent_silent_at = self._clock.monotonic() + self._output.queued_seconds()

    async def _respond(self, request: TurnRequest, transcribe: bool) -> None:
        try:
            if transcribe:
                text = await self._flush_transcript()
                request.timeline.stt_final = self._clock.monotonic()
                request.user_text = f"{self._carry_text} {text}".strip()
                self._carry_text = ""
                if not request.user_text:
                    return
                if request.interrupt.is_set():
                    self._carry_text = request.user_text
                    return
            outcome = await self._handle_turn.execute(self._ctx, request)
        except Exception as exc:
            self._turn_error = exc
            return
        turn = outcome.turn
        if turn.interrupted and not outcome.audio_started and turn.user_text:
            # Cut off before saying anything: the caller hadn't finished, so their words
            # join the next utterance rather than becoming a turn the agent never answered.
            self._carry_text = turn.user_text
            return
        if turn.user_text:
            self._ctx.history.append(ChatMessage("user", turn.user_text))
        if turn.agent_text:
            self._ctx.history.append(ChatMessage("assistant", turn.agent_text))
            self._endpointer.observe_agent_turn(turn.agent_text)

    async def _flush_transcript(self) -> str:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._pending_flush = future
        self._stt_in.put_nowait(FlushSignal())
        try:
            return await asyncio.wait_for(asyncio.shield(future), self._config.flush_timeout_s)
        except TimeoutError:
            text = " ".join([*self._finals, self._interim]).strip()
            self._finals.clear()
            self._interim = ""
            return text
        finally:
            self._pending_flush = None

    async def _stt_input(self) -> AsyncIterator[AudioChunk | FlushSignal]:
        while (item := await self._stt_in.get()) is not None:
            yield item

    async def _consume_stt(self) -> None:
        stream = self._ctx.pipeline.stt.stream(self._stt_input(), self._ctx.persona.language)
        async for event in stream:
            self._endpointer.observe_transcript(event, self._clock.monotonic())
            if event.is_final:
                if event.text.strip():
                    self._finals.append(event.text.strip())
                self._interim = ""
            else:
                self._interim = event.text.strip()
            if event.flushed:
                text = " ".join(self._finals).strip()
                self._finals.clear()
                self._interim = ""
                if self._pending_flush is not None and not self._pending_flush.done():
                    self._pending_flush.set_result(text)

    async def _prewarm_llm(self) -> None:
        """Runs while the greeting plays; the first reply's prompt starts with exactly this."""
        persona = self._ctx.persona
        prefix = [
            ChatMessage("system", persona.system_prompt),
            ChatMessage("assistant", persona.greeting),
        ]
        try:
            await self._ctx.pipeline.llm.prewarm(prefix)
        except Exception:
            log.warning("LLM prewarm failed; first turn will pay a cold prompt", exc_info=True)

    async def _tick(self) -> None:
        ctx = self._ctx
        uses_gpu = ctx.pipeline.uses_gpu
        while True:
            await asyncio.sleep(self._config.tick_interval_s)
            now = self._clock.monotonic()
            share, stt_seconds = ctx.meter.peek(now)
            open_usage = segment_usage(share, stt_seconds, uses_gpu)
            running = ctx.call.cost + self._calculator.calculate(ctx.pipeline.kind, open_usage)
            extra = {"concurrency": float(ctx.supervisor.active)} if ctx.supervisor else {}
            await self._metrics.emit(
                CallMetricsEvent(
                    kind="tick",
                    call_id=ctx.call.id,
                    pipeline=ctx.pipeline.kind,
                    elapsed_seconds=now - ctx.started_monotonic,
                    running_cost=running,
                    extra=extra,
                )
            )
