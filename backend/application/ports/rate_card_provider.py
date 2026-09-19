from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from backend.domain.services.cost_calculator import RateCard


@dataclass(frozen=True)
class GpuQuote:
    provider: str
    sku: str
    region: str
    hourly_usd: Decimal


@dataclass(frozen=True)
class TelephonyQuote:
    """A carrier's per-minute price. None when the carrier publishes no per-minute figure
    for the priced call type; note then says why. An unverified price came from a fetch no
    human has checked, and must be marked wherever it is shown."""

    carrier: str
    per_minute_usd: Decimal | None
    source_url: str
    checked_on: str
    verified: bool
    selected: bool
    note: str = ""


class RateCardProvider(Protocol):
    def rate_card(self) -> RateCard: ...

    def client_gpu_quotes(self) -> list[GpuQuote]:
        """EU-entity prices for client quotes; never used for measured benchmark cost."""
        ...

    def telephony_quotes(self) -> list[TelephonyQuote]:
        """Every listed carrier; exactly one is selected, and its price is the rate card's
        telephony rate."""
        ...
