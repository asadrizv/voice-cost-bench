from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class LatencyStage(StrEnum):
    ENDPOINT_DETECTED = "endpoint_detected"
    STT_FINAL = "stt_final"
    LLM_FIRST_TOKEN = "llm_first_token"
    LLM_COMPLETE = "llm_complete"
    TTS_FIRST_BYTE = "tts_first_byte"
    END_TO_END = "end_to_end"
    PERCEIVED_DELAY = "perceived_delay"


@dataclass(frozen=True)
class LatencyBreakdown:
    """Milliseconds per stage for one turn.

    endpoint_detected: caller stops speaking -> endpointer commits the turn
    stt_final:         endpoint -> final transcript available
    llm_first_token:   LLM request -> first token
    llm_complete:      LLM request -> last token
    tts_first_byte:    first TTS request -> first audio byte
    end_to_end:        endpoint -> first agent audio out (the pipeline's own latency)
    perceived_delay:   caller stops speaking -> first agent audio out (what the caller feels)
    """

    endpoint_detected: float
    stt_final: float
    llm_first_token: float
    llm_complete: float
    tts_first_byte: float
    end_to_end: float
    perceived_delay: float

    def get(self, stage: LatencyStage) -> float:
        value: float = getattr(self, stage.value)
        return value

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class LatencySample:
    call_id: str
    turn_index: int
    breakdown: LatencyBreakdown


@dataclass
class TurnTimeline:
    """Monotonic timestamps (seconds) captured while a turn runs.

    Any mark left unset collapses its stage to 0 rather than failing: a turn
    interrupted by barge-in still produces a usable, if partial, sample.
    """

    speech_end: float | None = None
    endpoint: float | None = None
    stt_final: float | None = None
    llm_start: float | None = None
    llm_first_token: float | None = None
    llm_complete: float | None = None
    tts_start: float | None = None
    tts_first_byte: float | None = None
    audio_out: float | None = None

    def breakdown(self) -> LatencyBreakdown:
        speech_end = self.speech_end if self.speech_end is not None else self.endpoint
        return LatencyBreakdown(
            endpoint_detected=_ms(speech_end, self.endpoint),
            stt_final=_ms(self.endpoint, self.stt_final),
            llm_first_token=_ms(self.llm_start, self.llm_first_token),
            llm_complete=_ms(self.llm_start, self.llm_complete),
            tts_first_byte=_ms(self.tts_start, self.tts_first_byte),
            end_to_end=_ms(self.endpoint, self.audio_out),
            perceived_delay=_ms(speech_end, self.audio_out),
        )


def _ms(start: float | None, end: float | None) -> float:
    if start is None or end is None:
        return 0.0
    return max(0.0, (end - start) * 1000.0)
