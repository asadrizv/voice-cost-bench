"""Deterministic stand-ins for every port, so unit tests cost nothing and never sleep long."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from backend.application.ports.component_catalogue import ComponentKind
from backend.application.ports.llm_port import ChatMessage, LlmEvent, TokenDelta, TokenUsage
from backend.application.ports.pipeline_provider import Pipeline
from backend.application.ports.running_engines import Engine, EngineReportUnavailable
from backend.application.ports.stt_port import FlushSignal, TranscriptEvent
from backend.domain.entities.persona import Persona, SamplingParams
from backend.domain.services.cost_calculator import PipelineRates, RateCard
from backend.domain.value_objects.audio import AudioChunk
from backend.domain.value_objects.pipeline_kind import PipelineKind


class FakeClock:
    """Monotonic time only moves when a fake advances it, so latencies are exact."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start
        self._epoch = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
        self._start = start

    def monotonic(self) -> float:
        return self.t

    def now(self) -> datetime:
        return self._epoch + timedelta(seconds=self.t - self._start)

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeLlm:
    def __init__(
        self,
        replies: Sequence[str],
        clock: FakeClock,
        first_token_s: float = 0.2,
        per_token_s: float = 0.01,
        usage: tuple[int, int] | None = (500, 40),
        error: Exception | None = None,
    ) -> None:
        self._replies = list(replies)
        self._clock = clock
        self._first = first_token_s
        self._per = per_token_s
        self._usage = usage
        self._error = error
        self.calls: list[list[ChatMessage]] = []
        self.prewarmed: list[list[ChatMessage]] = []

    async def prewarm(self, messages: Sequence[ChatMessage]) -> None:
        self.prewarmed.append(list(messages))
        if self._error is not None:
            raise self._error

    async def complete(
        self, messages: Sequence[ChatMessage], sampling: SamplingParams
    ) -> AsyncIterator[LlmEvent]:
        self.calls.append(list(messages))
        reply = self._replies.pop(0) if self._replies else "Okay."
        self._clock.advance(self._first)
        await asyncio.sleep(0)
        if self._error is not None:
            raise self._error
        for word in reply.split(" "):
            yield TokenDelta(word + " ")
            self._clock.advance(self._per)
            await asyncio.sleep(0)
        if self._usage is not None:
            yield TokenUsage(*self._usage)


class FakeTts:
    def __init__(
        self,
        clock: FakeClock,
        first_byte_s: float = 0.1,
        chunks_per_sentence: int = 3,
        chunk_bytes: int = 640,
        block: asyncio.Event | None = None,
    ) -> None:
        self._clock = clock
        self._first = first_byte_s
        self._chunks = chunks_per_sentence
        self._bytes = chunk_bytes
        self._block = block
        self.requests: list[str] = []

    async def synthesize(self, text: str, voice: str | None) -> AsyncIterator[AudioChunk]:
        self.requests.append(text)
        self._clock.advance(self._first)
        for i in range(self._chunks):
            if self._block is not None and i > 0:
                await self._block.wait()
            yield AudioChunk(b"\x01\x00" * (self._bytes // 2))
            await asyncio.sleep(0)


class ScriptedStt:
    """Answers each FlushSignal with the next scripted utterance as a flushed final."""

    def __init__(self, utterances: Sequence[str], clock: FakeClock, flush_s: float = 0.05):
        self._utterances = list(utterances)
        self._clock = clock
        self._flush_s = flush_s
        self.audio_seconds = 0.0

    async def stream(
        self, audio: AsyncIterable[AudioChunk | FlushSignal], language: str
    ) -> AsyncIterator[TranscriptEvent]:
        async for item in audio:
            if isinstance(item, FlushSignal):
                self._clock.advance(self._flush_s)
                text = self._utterances.pop(0) if self._utterances else ""
                yield TranscriptEvent(text=text, is_final=True, flushed=True)
            else:
                self.audio_seconds += item.duration_seconds


@dataclass
class RecordingOutput:
    chunks: list[AudioChunk] = field(default_factory=list)
    clears: int = 0
    queued: float = 0.0

    async def write(self, chunk: AudioChunk) -> None:
        self.chunks.append(chunk)
        await asyncio.sleep(0)

    async def clear(self) -> None:
        self.clears += 1

    def queued_seconds(self) -> float:
        return self.queued


class ByteVad:
    """Any non-zero sample counts as speech; fixtures encode speech as 0x01 bytes."""

    def is_speech(self, chunk: AudioChunk) -> bool:
        return any(chunk.data)


class NullMetrics:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def emit(self, event: object) -> None:
        self.events.append(event)


class StaticPersonas:
    def __init__(self, persona: Persona) -> None:
        self._persona = persona

    def get(self, persona_id: str) -> Persona:
        return self._persona


class StaticPipelines:
    def __init__(self, pipelines: dict[PipelineKind, Pipeline]) -> None:
        self._pipelines = pipelines

    def resolve(self, kind: PipelineKind) -> Pipeline:
        return self._pipelines[kind]


PERSONA = Persona(
    id="test",
    language="en",
    greeting="Good morning, Hartley and Associates, this is the AI assistant. How can I help?",
    system_prompt="You are a receptionist.",
    voices={"api": "voice-a", "selfhosted": "af_heart"},
)

RATE_CARD = RateCard(
    verified_on="2026-09-17",
    rates={
        PipelineKind.API: PipelineRates(
            stt_per_minute=Decimal("0.0043"),
            llm_input_per_1k=Decimal("0.00015"),
            llm_output_per_1k=Decimal("0.0006"),
            tts_per_1k_chars=Decimal("0.05"),
            telephony_per_minute=Decimal("0.014"),
        ),
        PipelineKind.SELFHOSTED: PipelineRates(
            gpu_per_hour=Decimal("1.09"),
            telephony_per_minute=Decimal("0.014"),
        ),
    },
)


def speech(ms: int = 20) -> AudioChunk:
    return AudioChunk(b"\x01\x00" * (16 * ms))


def silence(ms: int = 20) -> AudioChunk:
    return AudioChunk(b"\x00\x00" * (16 * ms))


class StaticEngines:
    """What the self-hosted services report running, without asking one."""

    def __init__(self, engines: dict[ComponentKind, list[Engine]]) -> None:
        self._engines = engines

    async def report(self, kind: ComponentKind) -> list[Engine]:
        return self._engines[kind]


class UnreachableEngines:
    def __init__(self, reason: str = "ConnectError") -> None:
        self._reason = reason

    async def report(self, kind: ComponentKind) -> list[Engine]:
        raise EngineReportUnavailable(self._reason)
