from __future__ import annotations

from typing import Protocol, runtime_checkable

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


@runtime_checkable
class ReportsDecisions(Protocol):
    """A detector that spends measurable work deciding, and can say how much and how often
    that work failed. The benchmark publishes both, so a run whose model never answered
    cannot be read as one that did."""

    @property
    def inference_ms(self) -> tuple[float, ...]:
        """How long each decision took this call, in arrival order."""
        ...

    @property
    def failures(self) -> int:
        """Decisions the detector could not make, and fell back to its ceiling for."""
        ...
