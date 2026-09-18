from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from backend.domain.value_objects.audio import AudioChunk


@dataclass(frozen=True)
class FlushSignal:
    """Sent into an STT stream when the endpointer commits a turn: finalise what you have."""


@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    is_final: bool
    flushed: bool = False
    """True on the final that answers a FlushSignal; the turn's transcript is complete."""


class SttPort(Protocol):
    def stream(
        self, audio: AsyncIterable[AudioChunk | FlushSignal], language: str
    ) -> AsyncIterator[TranscriptEvent]:
        """Consumes audio until the input iterator ends. Each FlushSignal must eventually be
        answered by exactly one event with flushed=True, even when nothing was said."""
        ...
