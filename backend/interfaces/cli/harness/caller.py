from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from backend.application.use_cases.call_session import CallSession
from backend.domain.value_objects.audio import PCM16_16K_MONO, AudioChunk
from backend.infrastructure.audio.level import first_audible_s
from backend.infrastructure.audio.wav import frames, read_wav
from backend.infrastructure.transport.paced_output import PacedAudioOutput

FRAME_S = 0.02
SILENCE = AudioChunk(b"\x00\x00" * 320, PCM16_16K_MONO)


@dataclass(frozen=True)
class Conversation:
    name: str
    persona: str
    language: str
    turns: list[str]
    audio: list[list[AudioChunk]]

    @property
    def sha(self) -> str:
        import hashlib

        h = hashlib.sha256()
        for turn in self.audio:
            for chunk in turn:
                h.update(chunk.data)
        return h.hexdigest()[:16]


def load_conversation(fixtures: Path, name: str) -> Conversation:
    spec = yaml.safe_load((fixtures / "conversations.yaml").read_text())["conversations"][name]
    audio = []
    for i in range(len(spec["turns"])):
        pcm, fmt = read_wav(fixtures / "audio" / name / f"{i:02d}.wav")
        if fmt != PCM16_16K_MONO:
            raise ValueError(f"{name}/{i:02d}.wav must be 16 kHz mono; run `make fixtures`")
        audio.append(frames(pcm, fmt))
    return Conversation(name, spec["persona"], spec["language"], spec["turns"], audio)


@dataclass
class Pacer:
    """Absolute schedule, so a late frame doesn't push every later frame back. Lateness
    is recorded: if the harness itself falls behind real time, its latencies are fiction."""

    start: float = field(default_factory=time.monotonic)
    sent: int = 0
    lateness_ms: list[float] = field(default_factory=list)

    async def tick(self) -> None:
        due = self.start + self.sent * FRAME_S
        now = time.monotonic()
        if due > now:
            await asyncio.sleep(due - now)
        else:
            self.lateness_ms.append((now - due) * 1000)
        self.sent += 1


class SyntheticCaller:
    """Plays the caller side of a conversation into a CallSession in real time, waiting
    for each reply to finish playing before speaking again, then hangs up."""

    def __init__(
        self,
        conversation: Conversation,
        session: CallSession,
        output: PacedAudioOutput,
        reply_timeout_s: float = 20.0,
    ) -> None:
        self._conv = conversation
        self._session = session
        self._output = output
        self._reply_timeout = reply_timeout_s
        self.pacer = Pacer()
        self.timeouts = 0

    async def frames(self) -> AsyncIterator[AudioChunk]:
        async for chunk in self._await_reply():
            yield chunk
        for utterance in self._conv.audio:
            self._output.caller_speech_started()
            last_voiced = _last_audible_frame(utterance)
            for position, chunk in enumerate(utterance):
                await self.pacer.tick()
                if position == last_voiced:
                    # Stamped as the frame is handed over: CallSession reads its clock for
                    # this frame before anything else can run, so both sides agree.
                    self._output.caller_speech_ended()
                yield chunk
            async for chunk in self._await_reply():
                yield chunk

    async def _await_reply(self) -> AsyncIterator[AudioChunk]:
        """Silence until the agent has been heard answering and has finished playing (or
        gave up). Counting recorded turns instead mistook a turn cut off by a mid-sentence
        pause for an answer, so the caller spoke its next line over the agent."""
        deadline = time.monotonic() + self._reply_timeout
        while True:
            await self.pacer.tick()
            yield SILENCE
            heard = self._output.first_audio_at is not None and not self._output.awaiting_answer
            if heard and not self._session.agent_busy and self._output.drained():
                return
            if time.monotonic() > deadline:
                self.timeouts += 1
                return


def _last_audible_frame(utterance: list[AudioChunk]) -> int | None:
    audible = [i for i, chunk in enumerate(utterance) if first_audible_s(chunk) is not None]
    return audible[-1] if audible else None
