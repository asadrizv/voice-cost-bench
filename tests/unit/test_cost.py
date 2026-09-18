from decimal import Decimal

import pytest

from backend.domain.entities.cost import CostBreakdown, Money, UsageUnits
from backend.domain.services.cost_calculator import (
    CostCalculator,
    gpu_cost,
    gpu_cost_per_minute_at,
)
from backend.domain.value_objects.pipeline_kind import PipelineKind
from tests.fakes import RATE_CARD

HOURLY = Decimal("1.09")


class TestMoney:
    def test_arithmetic_stays_exact(self) -> None:
        total = Money.zero()
        for _ in range(10):
            total = total + Money.of("0.1")
        assert total == Money.of(1)

    def test_scaling(self) -> None:
        assert Money.of(3) * 2 == Money.of(6)
        assert Money.of(3) / 4 == Money.of("0.75")
        assert Money.of(3) - Money.of(1) == Money.of(2)
        assert Money.of("1.2345678").rounded(4) == Decimal("1.2346")


class TestCostBreakdown:
    def test_total_and_per_minute(self) -> None:
        cost = CostBreakdown(stt=Money.of("0.01"), llm=Money.of("0.02"), gpu=Money.of("0.03"))
        assert cost.total == Money.of("0.06")
        assert cost.per_minute(120) == Money.of("0.03")

    def test_per_minute_of_zero_length_call_is_zero(self) -> None:
        assert CostBreakdown(stt=Money.of(1)).per_minute(0) == Money.zero()

    def test_sum(self) -> None:
        a = CostBreakdown(stt=Money.of(1), tts=Money.of(2))
        b = CostBreakdown(stt=Money.of(3), telephony=Money.of(4))
        assert (a + b).as_dict() == {
            "stt": 4.0,
            "llm": 0.0,
            "tts": 2.0,
            "gpu": 0.0,
            "telephony": 4.0,
            "total": 10.0,
        }


def test_usage_units_sum_keeps_peak_concurrency() -> None:
    a = UsageUnits(stt_audio_seconds=1, llm_input_tokens=10, concurrent_calls=3)
    b = UsageUnits(stt_audio_seconds=2, llm_output_tokens=5, concurrent_calls=5)
    total = a + b
    assert total.stt_audio_seconds == 3
    assert total.llm_input_tokens == 10 and total.llm_output_tokens == 5
    assert total.concurrent_calls == 5


@pytest.mark.parametrize(
    ("gpu_seconds", "concurrent", "expected"),
    [
        (3600, 1, Decimal("1.09")),  # one call holds the whole box for an hour
        (3600, 10, Decimal("0.109")),  # ten calls split it
        (60, 1, Decimal("1.09") / 60),  # a minute alone
        (60, 20, Decimal("1.09") / 60 / 20),
        (60, 0, Decimal("1.09") / 60),  # zero concurrency is clamped, never divides by zero
        (60, 0.5, Decimal("1.09") / 60),  # sub-one averages are clamped too
        (0, 5, Decimal(0)),  # API pipeline: no GPU time, no GPU cost
        (90, 2.5, Decimal(90) * Decimal("1.09") / 3600 / Decimal("2.5")),  # fractional average
    ],
)
def test_gpu_amortisation(gpu_seconds: float, concurrent: float, expected: Decimal) -> None:
    assert gpu_cost(gpu_seconds, HOURLY, concurrent).amount == pytest.approx(expected)


@pytest.mark.parametrize(
    ("concurrent", "utilisation", "expected"),
    [
        (1, 1.0, Decimal("1.09") / 60),
        (10, 1.0, Decimal("1.09") / 600),
        (10, 0.25, Decimal("1.09") / 600 * 4),  # idle three quarters of the time: 4x
        (20, 0.1, Decimal("1.09") / 1200 * 10),
    ],
)
def test_loaded_gpu_cost_per_minute(concurrent: int, utilisation: float, expected: Decimal) -> None:
    got = gpu_cost_per_minute_at(HOURLY, concurrent, utilisation).amount
    assert got == pytest.approx(expected)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_loaded_gpu_cost_rejects_impossible_utilisation(bad: float) -> None:
    with pytest.raises(ValueError):
        gpu_cost_per_minute_at(HOURLY, 1, bad)


class TestCostCalculator:
    calc = CostCalculator(RATE_CARD)

    def test_api_pipeline_prices_every_component_and_no_gpu(self) -> None:
        usage = UsageUnits(
            stt_audio_seconds=60,
            llm_input_tokens=2000,
            llm_output_tokens=500,
            tts_characters=1000,
            gpu_seconds=0,
            telephony_seconds=60,
        )
        cost = self.calc.calculate(PipelineKind.API, usage)
        assert cost.stt.amount == Decimal("0.0043")
        assert cost.llm.amount == Decimal("0.0003") + Decimal("0.0003")
        assert cost.tts.amount == Decimal("0.05")
        assert cost.gpu.amount == 0
        assert cost.telephony.amount == Decimal("0.014")

    def test_api_pipeline_ignores_gpu_seconds_because_its_rate_is_zero(self) -> None:
        cost = self.calc.calculate(PipelineKind.API, UsageUnits(gpu_seconds=3600))
        assert cost.gpu.amount == 0

    def test_selfhosted_pays_only_gpu_share_and_telephony(self) -> None:
        usage = UsageUnits(
            stt_audio_seconds=60,
            llm_input_tokens=2000,
            llm_output_tokens=500,
            tts_characters=1000,
            gpu_seconds=60,
            concurrent_calls=10,
            telephony_seconds=60,
        )
        cost = self.calc.calculate(PipelineKind.SELFHOSTED, usage)
        assert cost.stt.amount == cost.llm.amount == cost.tts.amount == 0
        assert cost.gpu.amount == pytest.approx(Decimal("1.09") / 60 / 10)
        assert cost.telephony.amount == Decimal("0.014")

    def test_zero_usage_costs_nothing(self) -> None:
        for kind in PipelineKind:
            assert self.calc.calculate(kind, UsageUnits()).total == Money.zero()
