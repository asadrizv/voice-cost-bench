from __future__ import annotations

import asyncio

from backend.application.ports.clock import Clock
from backend.domain.value_objects.audio import AudioChunk


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

    async def write(self, chunk: AudioChunk) -> None:
        now = self._clock.monotonic()
        if self.first_audio_at is None:
            self.first_audio_at = now
        self._until = max(self._until, now) + chunk.duration_seconds
        self.seconds_played += chunk.duration_seconds
        ahead = self._until - now - self._max_queue
        if ahead > 0:
            await asyncio.sleep(ahead)

    async def clear(self) -> None:
        self._until = self._clock.monotonic()

    def drained(self) -> bool:
        return self._clock.monotonic() >= self._until

    def remaining_seconds(self) -> float:
        return max(0.0, self._until - self._clock.monotonic())
