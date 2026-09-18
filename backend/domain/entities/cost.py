from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from typing import Self


@dataclass(frozen=True, order=True)
class Money:
    """USD amount. Decimal throughout so per-token prices don't drift when summed."""

    amount: Decimal = Decimal(0)

    @classmethod
    def of(cls, value: float | int | str | Decimal) -> Money:
        return cls(Decimal(str(value)))

    @classmethod
    def zero(cls) -> Money:
        return cls(Decimal(0))

    def __add__(self, other: Money) -> Money:
        return Money(self.amount + other.amount)

    def __sub__(self, other: Money) -> Money:
        return Money(self.amount - other.amount)

    def __mul__(self, factor: float | int | Decimal) -> Money:
        return Money(self.amount * Decimal(str(factor)))

    def __truediv__(self, divisor: float | int | Decimal) -> Money:
        return Money(self.amount / Decimal(str(divisor)))

    def as_float(self) -> float:
        return float(self.amount)

    def rounded(self, places: int = 6) -> Decimal:
        return self.amount.quantize(Decimal(1).scaleb(-places))


@dataclass(frozen=True)
class UsageUnits:
    stt_audio_seconds: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    tts_characters: int = 0
    gpu_seconds: float = 0.0
    """Wall-clock seconds this call held the GPU; 0.0 for the API pipeline."""
    concurrent_calls: float = 1
    """Calls sharing the GPU while these units were spent; the amortisation divisor.
    Fractional because concurrency changes mid-segment: it is the time-weighted
    harmonic mean from ConcurrencySupervisor, which keeps shares summing to wall time."""
    telephony_seconds: float = 0.0

    def __add__(self, other: UsageUnits) -> Self:
        """Sums counters. concurrent_calls is not additive, so the max is kept for display;
        cost must be computed per segment, never from a summed UsageUnits."""
        return type(self)(
            stt_audio_seconds=self.stt_audio_seconds + other.stt_audio_seconds,
            llm_input_tokens=self.llm_input_tokens + other.llm_input_tokens,
            llm_output_tokens=self.llm_output_tokens + other.llm_output_tokens,
            tts_characters=self.tts_characters + other.tts_characters,
            gpu_seconds=self.gpu_seconds + other.gpu_seconds,
            concurrent_calls=max(self.concurrent_calls, other.concurrent_calls),
            telephony_seconds=self.telephony_seconds + other.telephony_seconds,
        )


@dataclass(frozen=True)
class CostBreakdown:
    stt: Money = Money.zero()
    llm: Money = Money.zero()
    tts: Money = Money.zero()
    gpu: Money = Money.zero()
    telephony: Money = Money.zero()

    @classmethod
    def zero(cls) -> CostBreakdown:
        return cls()

    @property
    def total(self) -> Money:
        return self.stt + self.llm + self.tts + self.gpu + self.telephony

    def per_minute(self, call_seconds: float) -> Money:
        if call_seconds <= 0:
            return Money.zero()
        return self.total * 60 / call_seconds

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            **{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)}
        )

    def as_dict(self) -> dict[str, float]:
        stages = {f.name: getattr(self, f.name).as_float() for f in fields(self)}
        return {**stages, "total": self.total.as_float()}
