from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from backend.application.ports.component_catalogue import (
    Component,
    ComponentKind,
    GpuMemoryProfile,
)
from backend.domain.services.gpu_memory_budget import GpuMemoryBudget, MemoryClaim


@dataclass(frozen=True)
class LlmShare:
    """The fraction of the card the LLM server reserves for itself, read from the serving
    configuration. None when nothing reserves a fixed share, as with Ollama."""

    utilisation: Decimal | None
    source: str


class CheckGpuBudget:
    """Whether the components that ran, or are configured to run, fit on one GPU together.
    The LLM's line is its configured share of the card, not a catalogue figure: that share
    is reserved whether or not the model needs it."""

    def __init__(self, profile: GpuMemoryProfile | None, llm: LlmShare) -> None:
        self._profile = profile
        self._llm = llm

    def execute(self, components: Iterable[Component]) -> GpuMemoryBudget | None:
        """None when no component runs on the GPU, or no GPU is described in the catalogue:
        there is then no card to divide up."""
        profile = self._profile
        if profile is None:
            return None
        claims = [
            self._reserved(profile, component)
            for component in components
            if component.id in profile.claims
        ]
        if not claims:
            return None
        return GpuMemoryBudget(profile.gpu, tuple(claims))

    def _reserved(self, profile: GpuMemoryProfile, component: Component) -> MemoryClaim:
        if component.kind is not ComponentKind.LLM:
            return profile.claims[component.id]
        share = self._llm.utilisation
        return MemoryClaim(
            component.id,
            None if share is None else profile.gpu.total_gib * share,
            profile.gpu.basis,
            self._llm.source,
        )
