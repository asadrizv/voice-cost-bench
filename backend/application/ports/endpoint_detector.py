from __future__ import annotations

from typing import Protocol

from backend.application.ports.stt_port import TranscriptEvent


class EndpointDetector(Protocol):
    """Decides when the caller has finished their turn. Fed by the call session every audio
    frame; must be cheap, since `should_commit` runs at frame rate."""

    def reset(self) -> None: ...

    def observe_audio(self, is_speech: bool, now: float) -> None: ...

    def observe_transcript(self, event: TranscriptEvent, now: float) -> None: ...

    def should_commit(self, now: float) -> bool: ...

    @property
    def speech_end(self) -> float | None:
        """Monotonic time the caller last stopped speaking, for perceived_delay."""
        ...
