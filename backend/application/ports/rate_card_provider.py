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


class RateCardProvider(Protocol):
    def rate_card(self) -> RateCard: ...

    def client_gpu_quotes(self) -> list[GpuQuote]:
        """EU-entity prices for client quotes; never used for measured benchmark cost."""
        ...
