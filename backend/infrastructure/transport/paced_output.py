from __future__ import annotations

import asyncio

from backend.application.ports.clock import Clock
from backend.domain.value_objects.audio import AudioChunk
from backend.infrastructure.audio.level import first_audible_s


class PacedAudioOutput:
    """A virtual speaker for synthetic calls: audio "plays" in real time and writes block
    once `max_queue_s` is buffered, exactly as LiveKit's AudioSource behaves. Lets the load
    harness drive CallSession with the same back-pressure a browser call has."""

    def __init__(self, clock: Clock, max_queue_s: float = 1.0) -> None:
        self._clock = clock
        self._max_queue = max_queue_s
        self._until = clock.monotonic()
        self.first_audio_at: float | None = None
        self.seconds_played = 0.0
        self.caller_observed_ms: list[float] = []
        self._caller_stopped_at: float | None = None
        self._unanswered = 0

    def caller_speech_started(self) -> None:
        if self._caller_stopped_at is not None:
            self._unanswered += 1
            self._caller_stopped_at = None

    def caller_speech_ended(self) -> None:
        """Caller-observed delay runs from here to the first audible agent audio played
        after it, as Openbenchmarks' TTFAB does from a recording of both sides."""
        self._caller_stopped_at = self._clock.monotonic()

    @property
    def awaiting_answer(self) -> bool:
        """The caller has finished speaking and no audible agent audio has played since."""
        return self._caller_stopped_at is not None

    @property
    def unanswered_turns(self) -> int:
        return self._unanswered + int(self.awaiting_answer)

    async def write(self, chunk: AudioChunk) -> None:
        now = self._clock.monotonic()
        if self.first_audio_at is None:
            self.first_audio_at = now
        plays_at = max(self._until, now)
        if self._caller_stopped_at is not None and (offset := first_audible_s(chunk)) is not None:
            self.caller_observed_ms.append((plays_at + offset - self._caller_stopped_at) * 1000)
            self._caller_stopped_at = None
        self._until = plays_at + chunk.duration_seconds
        self.seconds_played += chunk.duration_seconds
        ahead = self._until - now - self._max_queue
        if ahead > 0:
            await asyncio.sleep(ahead)

    async def clear(self) -> None:
        self._until = self._clock.monotonic()

    def drained(self) -> bool:
        return self._clock.monotonic() >= self._until

    def queued_seconds(self) -> float:
        return self.remaining_seconds()

    def remaining_seconds(self) -> float:
        return max(0.0, self._until - self._clock.monotonic())
