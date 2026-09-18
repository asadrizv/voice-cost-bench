from __future__ import annotations

import time
from datetime import UTC, datetime


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def now(self) -> datetime:
        return datetime.now(UTC)
