from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

MEASURED = "measured"
BASES = (MEASURED, "vendor-stated", "estimated")
"""How a memory figure was arrived at. Only MEASURED came off a GPU; the others are
advisory, which is why an over-committed card is reported and never refused."""

_GIB = Decimal("0.01")


class BudgetVerdict(StrEnum):
    FITS = "fits"
    DOES_NOT_FIT = "does not fit"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GpuCapacity:
    sku: str
    total_gib: Decimal
    basis: str
    source: str


@dataclass(frozen=True)
class MemoryClaim:
    """What one component is expected to hold on the GPU, and where that figure came from.
    gib is None when no figure is known: a missing figure must not be read as nothing."""

    component_id: str
    gib: Decimal | None
    basis: str
    source: str


@dataclass(frozen=True)
class GpuMemoryBudget:
    """Whether one GPU holds a set of components at once, with the arithmetic that says so."""

    gpu: GpuCapacity
    claims: tuple[MemoryClaim, ...]

    @property
    def claimed_gib(self) -> Decimal:
        """The claims that carry a figure. A lower bound while any claim is unknown."""
        return sum((c.gib for c in self.claims if c.gib is not None), Decimal(0))

    @property
    def unknown(self) -> tuple[str, ...]:
        return tuple(c.component_id for c in self.claims if c.gib is None)

    @property
    def headroom_gib(self) -> Decimal:
        return self.gpu.total_gib - self.claimed_gib

    @property
    def shortfall_gib(self) -> Decimal:
        return max(Decimal(0), -self.headroom_gib)

    @property
    def verdict(self) -> BudgetVerdict:
        if self.shortfall_gib > 0:
            return BudgetVerdict.DOES_NOT_FIT
        if self.unknown:
            return BudgetVerdict.UNKNOWN
        return BudgetVerdict.FITS

    @property
    def measured(self) -> bool:
        """False while any figure is an estimate, which every figure is until an L40S run
        measures one: the verdict is then a warning, not a fact."""
        bases = [self.gpu.basis] + [c.basis for c in self.claims if c.gib is not None]
        return all(basis == MEASURED for basis in bases)

    def summary(self) -> str:
        parts = " + ".join(
            f"{c.component_id} {_amount(c.gib)}" if c.gib is not None else f"{c.component_id} ?"
            for c in self.claims
        )
        room = (
            f"{_amount(self.shortfall_gib)} GiB short"
            if self.shortfall_gib > 0
            else f"{_amount(self.headroom_gib)} GiB spare"
        )
        reason = f"unknown: no figure for {', '.join(self.unknown)}"
        verdict = reason if self.verdict is BudgetVerdict.UNKNOWN else self.verdict.value
        return (
            f"{self.gpu.sku} {_amount(self.gpu.total_gib)} GiB: {parts} = "
            f"{_amount(self.claimed_gib)} GiB, {room} ({verdict})"
        )


def _amount(gib: Decimal) -> str:
    return str(gib.quantize(_GIB))
