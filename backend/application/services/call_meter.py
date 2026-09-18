from __future__ import annotations

from backend.application.services.concurrency_supervisor import ConcurrencySupervisor, GpuShare
from backend.domain.entities.cost import UsageUnits


class CallMeter:
    """Splits one call's wall time into billable segments (one per turn plus a closing one)
    so GPU and telephony time are attributed without gaps or overlap."""

    def __init__(self, started: float, supervisor: ConcurrencySupervisor | None = None) -> None:
        self._supervisor = supervisor
        self._segment_start = started
        self._share_cursor = supervisor.share_cursor(started) if supervisor else 0.0
        self._stt_seconds_total = 0.0
        self._stt_seconds_billed = 0.0

    def add_stt_audio(self, seconds: float) -> None:
        self._stt_seconds_total += seconds

    def open_segment_seconds(self, now: float) -> float:
        return max(0.0, now - self._segment_start)

    def peek(self, now: float) -> tuple[GpuShare, float]:
        """What close_segment would return now, without closing; for live running cost."""
        wall = self.open_segment_seconds(now)
        if self._supervisor is None:
            share = GpuShare(wall, wall)
        else:
            share = GpuShare(wall, self._supervisor.share_cursor(now) - self._share_cursor)
        return share, self._stt_seconds_total - self._stt_seconds_billed

    def close_segment(self, now: float) -> tuple[GpuShare, float]:
        """Returns the GPU share and STT audio seconds accrued since the previous close."""
        wall = self.open_segment_seconds(now)
        if self._supervisor is not None:
            cursor = self._supervisor.share_cursor(now)
            share = GpuShare(wall, cursor - self._share_cursor)
            self._share_cursor = cursor
        else:
            share = GpuShare(wall, wall)
        self._segment_start = now
        stt = self._stt_seconds_total - self._stt_seconds_billed
        self._stt_seconds_billed = self._stt_seconds_total
        return share, stt


def segment_usage(share: GpuShare, stt_seconds: float, uses_gpu: bool) -> UsageUnits:
    """Wall-time units for one segment; the caller adds tokens and characters."""
    return UsageUnits(
        stt_audio_seconds=stt_seconds,
        gpu_seconds=share.wall_seconds if uses_gpu else 0.0,
        concurrent_calls=share.effective_concurrency if uses_gpu else 1,
        telephony_seconds=share.wall_seconds,
    )
