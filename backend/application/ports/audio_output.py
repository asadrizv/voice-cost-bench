from __future__ import annotations

from typing import Protocol

from backend.domain.value_objects.audio import AudioChunk


class AudioOutput(Protocol):
    async def write(self, chunk: AudioChunk) -> None:
        """May block to pace playback. The first write of a turn is taken as first audio out."""
        ...

    async def clear(self) -> None:
        """Drops queued audio immediately; used for barge-in."""
        ...

    def queued_seconds(self) -> float:
        """Audio written but not yet played."""
        ...
