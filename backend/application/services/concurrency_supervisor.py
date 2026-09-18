from __future__ import annotations

import threading
from dataclasses import dataclass


class CapacityExceeded(Exception):
    pass


@dataclass(frozen=True)
class GpuShare:
    wall_seconds: float
    share_seconds: float
    """This call's fair slice of GPU time: the integral of 1/active_calls over the segment."""

    @property
    def effective_concurrency(self) -> float:
        if self.share_seconds <= 0:
            return 1.0
        return self.wall_seconds / self.share_seconds


class ConcurrencySupervisor:
    """The single authoritative count of active calls on one GPU pool.

    Rejects admission above the ceiling instead of degrading everyone's latency, and is the
    only source of the GPU cost divisor. Because every active call accrues 1/n of each second,
    the shares of all calls sum exactly to the busy time: cost is conserved, never
    double-counted and never estimated per adapter.
    """

    def __init__(self, ceiling: int) -> None:
        if ceiling < 1:
            raise ValueError("ceiling must be >= 1")
        self._ceiling = ceiling
        self._active: set[str] = set()
        self._share_integral = 0.0
        self._last_update: float | None = None
        self._peak = 0
        # Admission is called from the agent's asyncio loop and the loadtest's; a lock keeps
        # the integral consistent if they ever run in separate threads.
        self._lock = threading.Lock()

    @property
    def ceiling(self) -> int:
        return self._ceiling

    @property
    def active(self) -> int:
        return len(self._active)

    @property
    def peak(self) -> int:
        return self._peak

    def acquire(self, call_id: str, now: float) -> None:
        with self._lock:
            if call_id in self._active:
                return
            if len(self._active) >= self._ceiling:
                raise CapacityExceeded(
                    f"{len(self._active)} active calls, ceiling is {self._ceiling}"
                )
            self._advance(now)
            self._active.add(call_id)
            self._peak = max(self._peak, len(self._active))

    def release(self, call_id: str, now: float) -> None:
        with self._lock:
            if call_id not in self._active:
                return
            self._advance(now)
            self._active.discard(call_id)

    def share_cursor(self, now: float) -> float:
        """Cumulative per-call share. The difference of two cursors taken while a call is
        active is that call's GPU share for the interval between them."""
        with self._lock:
            self._advance(now)
            return self._share_integral

    def _advance(self, now: float) -> None:
        if self._last_update is not None and self._active:
            self._share_integral += max(0.0, now - self._last_update) / len(self._active)
        self._last_update = now
