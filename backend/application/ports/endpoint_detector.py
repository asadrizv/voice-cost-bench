from __future__ import annotations

from typing import Protocol

from backend.application.ports.stt_port import TranscriptEvent
from backend.domain.value_objects.audio import AudioChunk


class EndpointDetector(Protocol):
    """Decides when the caller has finished their turn. Fed by the call session every audio
    frame; must be cheap, since `should_commit` runs at frame rate."""

    def reset(self) -> None: ...

    def observe_audio(self, frame: AudioChunk, is_speech: bool, now: float) -> None:
        """One caller frame, in arrival order, with the VAD's verdict on it. Detectors that
        decide from timing alone may ignore `frame`; audio models read it."""
        ...

    def observe_transcript(self, event: TranscriptEvent, now: float) -> None: ...

    def should_commit(self, now: float) -> bool: ...

    def observe_agent_turn(self, text: str) -> None:
        """What the agent just said. Survives reset(): it frames the caller's next turn."""
        ...

    @property
    def speech_end(self) -> float | None:
        """Monotonic time the caller last stopped speaking, for perceived_delay."""
        ...
