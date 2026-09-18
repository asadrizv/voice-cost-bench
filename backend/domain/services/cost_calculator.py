from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from backend.domain.entities.cost import CostBreakdown, Money, UsageUnits
from backend.domain.value_objects.pipeline_kind import PipelineKind


@dataclass(frozen=True)
class PipelineRates:
    """Unit prices for one pipeline. A component the pipeline does not pay for per unit
    (self-hosted STT/LLM/TTS, the API pipeline's GPU) is priced at zero."""

    stt_per_minute: Decimal = Decimal(0)
    llm_input_per_1k: Decimal = Decimal(0)
    llm_output_per_1k: Decimal = Decimal(0)
    tts_per_1k_chars: Decimal = Decimal(0)
    gpu_per_hour: Decimal = Decimal(0)
    telephony_per_minute: Decimal = Decimal(0)


@dataclass(frozen=True)
class RateCard:
    verified_on: str
    rates: dict[PipelineKind, PipelineRates]

    def for_pipeline(self, pipeline: PipelineKind) -> PipelineRates:
        return self.rates[pipeline]


class CostCalculator:
    def __init__(self, rate_card: RateCard) -> None:
        self._rate_card = rate_card

    @property
    def rate_card(self) -> RateCard:
        return self._rate_card

    def calculate(self, pipeline: PipelineKind, usage: UsageUnits) -> CostBreakdown:
        r = self._rate_card.for_pipeline(pipeline)
        return CostBreakdown(
            stt=Money(r.stt_per_minute * _d(usage.stt_audio_seconds) / 60),
            llm=Money(
                r.llm_input_per_1k * usage.llm_input_tokens / 1000
                + r.llm_output_per_1k * usage.llm_output_tokens / 1000
            ),
            tts=Money(r.tts_per_1k_chars * usage.tts_characters / 1000),
            gpu=gpu_cost(usage.gpu_seconds, r.gpu_per_hour, usage.concurrent_calls),
            telephony=Money(r.telephony_per_minute * _d(usage.telephony_seconds) / 60),
        )


def gpu_cost(gpu_seconds: float, hourly_rate: Decimal, concurrent_calls: float) -> Money:
    """The benchmark's honesty hinges on this line: one call's share of a rented GPU."""
    return Money(_d(gpu_seconds) * hourly_rate / 3600 / _d(max(concurrent_calls, 1)))


def gpu_cost_per_minute_at(
    hourly_rate: Decimal, concurrent_calls: float, utilisation: float
) -> Money:
    """Loaded GPU cost per call-minute when the box is busy only `utilisation` of the
    time it is rented. Idle time is paid for, so it is spread over the minutes served."""
    if not 0 < utilisation <= 1:
        raise ValueError("utilisation must be in (0, 1]")
    return gpu_cost(60, hourly_rate, concurrent_calls) / _d(utilisation)


def _d(value: float) -> Decimal:
    return Decimal(str(value))
