"""A queueing model of one shared GPU, for exercising the harness without hardware.

Nothing here is a measurement. Numbers are chosen to resemble a 9B model with Whisper and
Kokoro on one L40S so the sweep, breaking-point search and utilisation curve have
something realistic-shaped to chew on; every result produced with it is stamped
"simulated" and the UI says so.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from dataclasses import dataclass

import numpy as np

from backend.application.ports.llm_port import ChatMessage, LlmEvent, TokenDelta, TokenUsage
from backend.application.ports.stt_port import FlushSignal, TranscriptEvent
from backend.domain.entities.persona import SamplingParams
from backend.domain.value_objects.audio import PCM16_24K_MONO, AudioChunk
from backend.infrastructure.audio.level import AUDIBLE_DBFS, dbfs, unit_samples


@dataclass(frozen=True)
class SimulatedGpuProfile:
    stt_final_ms: float = 90.0
    stt_workers: int = 3
    llm_ttft_ms: float = 140.0
    llm_ttft_per_active_ms: float = 9.0
    """Prefill contention: each concurrently decoding sequence adds this to TTFT."""
    llm_tokens_per_s: float = 90.0
    llm_decode_slowdown_per_active: float = 0.012
    tts_first_byte_ms: float = 80.0
    tts_workers: int = 3
    tts_realtime_factor: float = 0.08
    speech_s_per_char: float = 0.065
    interim_after_speech_s: float = 0.6
    jitter: float = 0.15


class SimulatedGpu:
    def __init__(self, profile: SimulatedGpuProfile | None = None, seed: int = 7) -> None:
        self.profile = profile or SimulatedGpuProfile()
        self._stt = asyncio.Semaphore(self.profile.stt_workers)
        self._tts = asyncio.Semaphore(self.profile.tts_workers)
        self.llm_active = 0
        self._rng = random.Random(seed)

    def jitter(self, ms: float) -> float:
        return max(0.0, ms * (1 + self._rng.uniform(-1, 1) * self.profile.jitter)) / 1000

    async def stt_final(self) -> None:
        async with self._stt:
            await asyncio.sleep(self.jitter(self.profile.stt_final_ms))


def _voiced(chunk: AudioChunk) -> bool:
    return dbfs(unit_samples(chunk.data)) > AUDIBLE_DBFS


# A quiet 40 ms, 220 Hz tone (whole cycles, so frames join without clicks): simulated
# speech you can hear in a browser demo, distinct from any real voice.
_TONE = (0.08 * 32767 * np.sin(2 * np.pi * 220 * np.arange(960) / 24000)).astype("<i2").tobytes()


class SimulatedStt:
    """Emits the caller's next scripted line as an interim once they've been speaking a
    moment (so semantic endpointing has words to judge, as with a real STT), and as the
    flushed final when the turn is committed."""

    def __init__(self, gpu: SimulatedGpu, script: Sequence[str]) -> None:
        self._gpu = gpu
        self._script = list(script)

    async def stream(
        self, audio: AsyncIterable[AudioChunk | FlushSignal], language: str
    ) -> AsyncIterator[TranscriptEvent]:
        script = list(self._script)  # per call: one adapter instance may serve many calls
        speech_s = 0.0
        announced = False
        async for item in audio:
            if isinstance(item, FlushSignal):
                await self._gpu.stt_final()
                text = script.pop(0) if script else ""
                speech_s, announced = 0.0, False
                yield TranscriptEvent(text, is_final=True, flushed=True)
            elif _voiced(item):
                speech_s += item.duration_seconds
                threshold = self._gpu.profile.interim_after_speech_s
                if not announced and script and speech_s >= threshold:
                    announced = True
                    yield TranscriptEvent(script[0], is_final=False)


_REPLIES = [
    "Thank you for calling. May I have your full name, please?",
    "Thank you. Is this about employment, tenancy, or family law?",
    "I understand, that sounds stressful. Is there a deadline or court date coming up?",
    "Noted. I can offer Tuesday at ten or Thursday at nine. Which suits you?",
    "Tuesday at ten is booked for you. What's the best email for the confirmation?",
    "Perfect, a lawyer will confirm by email today. Is there anything else?",
]


class SimulatedLlm:
    def __init__(self, gpu: SimulatedGpu) -> None:
        self._gpu = gpu

    async def prewarm(self, messages: Sequence[ChatMessage]) -> None:
        return None

    async def complete(
        self, messages: Sequence[ChatMessage], sampling: SamplingParams
    ) -> AsyncIterator[LlmEvent]:
        gpu, p = self._gpu, self._gpu.profile
        turn = sum(1 for m in messages if m.role == "user") - 1
        reply = _REPLIES[min(turn, len(_REPLIES) - 1)]
        gpu.llm_active += 1
        try:
            await asyncio.sleep(
                gpu.jitter(p.llm_ttft_ms + p.llm_ttft_per_active_ms * gpu.llm_active)
            )
            words = reply.split(" ")
            rate = p.llm_tokens_per_s / (1 + p.llm_decode_slowdown_per_active * gpu.llm_active)
            for i, word in enumerate(words):
                yield TokenDelta(word if i == 0 else " " + word)
                await asyncio.sleep(1.3 / rate)  # ~1.3 tokens per word
        finally:
            gpu.llm_active -= 1
        prompt_chars = sum(len(m.content) for m in messages)
        yield TokenUsage(prompt_chars // 4, int(len(words) * 1.3))


class SimulatedTts:
    def __init__(self, gpu: SimulatedGpu) -> None:
        self._gpu = gpu

    async def synthesize(self, text: str, voice: str | None) -> AsyncIterator[AudioChunk]:
        gpu, p = self._gpu, self._gpu.profile
        audio_s = p.speech_s_per_char * len(text)
        frame = PCM16_24K_MONO.byte_count(0.04)
        async with gpu._tts:  # noqa: SLF001
            await asyncio.sleep(gpu.jitter(p.tts_first_byte_ms))
            frames = max(1, int(audio_s / 0.04))
            for i in range(frames):
                yield AudioChunk(_TONE[:frame], PCM16_24K_MONO)
                if i % 10 == 9:
                    await asyncio.sleep(0.4 * p.tts_realtime_factor)
