from __future__ import annotations

from datetime import datetime
from typing import Protocol


class Clock(Protocol):
    def monotonic(self) -> float: ...

    def now(self) -> datetime: ...
