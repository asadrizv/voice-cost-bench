import pytest

from backend.domain.entities.latency import LatencyBreakdown, LatencyStage, TurnTimeline
from backend.domain.services.latency_analyzer import LatencyAnalyzer, percentile


@pytest.mark.parametrize(
    ("values", "pct", "expected"),
    [
        ([], 95, 0.0),
        ([5.0], 95, 5.0),
        ([1, 2, 3, 4, 5], 50, 3.0),
        ([1, 2, 3, 4], 50, 2.5),
        (list(range(1, 101)), 95, 95.05),  # numpy's linear method
        ([10, 1, 7], 0, 1.0),
        ([10, 1, 7], 100, 10.0),
    ],
)
def test_percentile(values: list[float], pct: float, expected: float) -> None:
    assert percentile(values, pct) == pytest.approx(expected)


def test_percentile_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_timeline_breakdown_measures_each_stage() -> None:
    t = TurnTimeline(
        speech_end=10.0,
        endpoint=10.3,
        stt_final=10.35,
        llm_start=10.35,
        llm_first_token=10.55,
        llm_complete=11.0,
        tts_start=10.6,
        tts_first_byte=10.7,
        audio_out=10.72,
    )
    b = t.breakdown()
    assert b.endpoint_detected == pytest.approx(300)
    assert b.stt_final == pytest.approx(50)
    assert b.llm_first_token == pytest.approx(200)
    assert b.llm_complete == pytest.approx(650)
    assert b.tts_first_byte == pytest.approx(100)
    assert b.end_to_end == pytest.approx(420)
    assert b.perceived_delay == pytest.approx(720)
    assert b.get(LatencyStage.PERCEIVED_DELAY) == b.perceived_delay


def test_partial_timeline_zeroes_missing_stages() -> None:
    b = TurnTimeline(endpoint=5.0, llm_start=5.1).breakdown()
    assert b.endpoint_detected == 0  # speech_end falls back to endpoint
    assert b.llm_first_token == 0
    assert b.perceived_delay == 0


def test_clock_skew_never_yields_negative_latency() -> None:
    assert TurnTimeline(endpoint=5.0, stt_final=4.9).breakdown().stt_final == 0


def _sample(e2e: float, perceived: float) -> LatencyBreakdown:
    return LatencyBreakdown(0, 0, 0, 0, 0, e2e, perceived)


def test_budget_passes_and_fails_on_p95() -> None:
    analyzer = LatencyAnalyzer()
    fast = [_sample(500, 800)] * 20
    assert analyzer.budget(fast).ok
    slow_tail = [_sample(500, 800)] * 18 + [_sample(1500, 2000)] * 2
    report = analyzer.budget(slow_tail)
    assert not report.end_to_end_ok and not report.perceived_delay_ok
    assert report.end_to_end_p95 > 900


def test_stats_cover_every_stage() -> None:
    stats = LatencyAnalyzer().stats([_sample(100, 200), _sample(300, 400)])
    assert set(stats) == set(LatencyStage)
    assert stats[LatencyStage.END_TO_END].mean == 200
    assert stats[LatencyStage.PERCEIVED_DELAY].p50 == 300
    assert stats[LatencyStage.END_TO_END].count == 2
