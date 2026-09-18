import pytest

from backend.application.services.call_meter import CallMeter, segment_usage
from backend.application.services.concurrency_supervisor import (
    CapacityExceeded,
    ConcurrencySupervisor,
    GpuShare,
)


def test_rejects_above_ceiling_instead_of_degrading() -> None:
    sup = ConcurrencySupervisor(ceiling=2)
    sup.acquire("a", 0)
    sup.acquire("b", 0)
    with pytest.raises(CapacityExceeded):
        sup.acquire("c", 0)
    sup.release("a", 1)
    sup.acquire("c", 1)
    assert sup.active == 2 and sup.peak == 2


def test_acquire_and_release_are_idempotent() -> None:
    sup = ConcurrencySupervisor(ceiling=1)
    sup.acquire("a", 0)
    sup.acquire("a", 0)
    sup.release("a", 1)
    sup.release("a", 1)
    assert sup.active == 0


def test_ceiling_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ConcurrencySupervisor(ceiling=0)


def test_shares_sum_to_busy_wall_time() -> None:
    """Three calls with staggered lifetimes: their GPU shares must add up to exactly the
    time the GPU was busy, or cost is being invented or lost."""
    sup = ConcurrencySupervisor(ceiling=10)
    meters: dict[str, CallMeter] = {}
    for call_id, start in (("a", 0.0), ("b", 10.0), ("c", 20.0)):
        sup.acquire(call_id, start)
        meters[call_id] = CallMeter(start, sup)
    shares: dict[str, GpuShare] = {}
    for call_id, end in (("a", 30.0), ("b", 40.0), ("c", 60.0)):
        shares[call_id] = meters[call_id].close_segment(end)[0]
        sup.release(call_id, end)

    busy = 60.0
    assert sum(s.share_seconds for s in shares.values()) == pytest.approx(busy)
    # a: 10s alone, 10s with b, 10s with b+c -> 10 + 5 + 10/3
    assert shares["a"].share_seconds == pytest.approx(10 + 5 + 10 / 3)
    assert shares["a"].effective_concurrency == pytest.approx(30 / (10 + 5 + 10 / 3))


def test_idle_time_is_attributed_to_nobody() -> None:
    sup = ConcurrencySupervisor(ceiling=2)
    sup.acquire("a", 0)
    sup.release("a", 10)
    cursor = sup.share_cursor(100)  # 90s idle
    assert cursor == pytest.approx(10)


def test_meter_segments_and_stt_seconds() -> None:
    sup = ConcurrencySupervisor(ceiling=4)
    sup.acquire("a", 0)
    sup.acquire("b", 0)
    meter = CallMeter(0, sup)
    meter.add_stt_audio(3.0)
    peek, stt_peek = meter.peek(10)
    share, stt = meter.close_segment(10)
    assert peek == share and stt_peek == stt == 3.0
    assert share.wall_seconds == 10 and share.effective_concurrency == pytest.approx(2)
    meter.add_stt_audio(1.0)
    share2, stt2 = meter.close_segment(15)
    assert share2.wall_seconds == 5 and stt2 == 1.0


def test_meter_without_supervisor_counts_whole_wall_time() -> None:
    meter = CallMeter(0)
    share, _ = meter.close_segment(8)
    assert share.share_seconds == 8 and share.effective_concurrency == 1


def test_segment_usage_zeroes_gpu_for_api() -> None:
    share = GpuShare(wall_seconds=10, share_seconds=5)
    api = segment_usage(share, 2.0, uses_gpu=False)
    gpu = segment_usage(share, 2.0, uses_gpu=True)
    assert api.gpu_seconds == 0 and api.concurrent_calls == 1 and api.telephony_seconds == 10
    assert gpu.gpu_seconds == 10 and gpu.concurrent_calls == 2


def test_zero_share_reports_single_concurrency() -> None:
    assert GpuShare(0, 0).effective_concurrency == 1
