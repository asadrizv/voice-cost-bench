from __future__ import annotations

from collections.abc import Iterable, Mapping
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

    What a component holds follows how it is served, not what kind it is. A server given a
    share of the card -- vLLM and its relatives -- reserves that share up front, whether or
    not a call is in flight, so its line is the share and never its weights. Everything else
    holds roughly what it loads, which is what the catalogue records.
    """

    def __init__(self, profile: GpuMemoryProfile | None, llm: LlmShare) -> None:
        self._profile = profile
        self._llm = llm

    def execute(
        self,
        components: Iterable[Component],
        reserved: Mapping[str, Decimal] | None = None,
    ) -> GpuMemoryBudget | None:
        """`reserved` maps a component id to the fraction of the card its runtime reserves,
        as the service running it reports at /v1/info. A component missing from it holds its
        catalogue footprint.

        None when no component runs on the GPU, or no GPU is described in the catalogue:
        there is then no card to divide up.
        """
        profile = self._profile
        if profile is None:
            return None
        claims = [
            self._claim(profile, component, (reserved or {}).get(component.id))
            for component in components
            if component.id in profile.claims
        ]
        if not claims:
            return None
        return GpuMemoryBudget(profile.gpu, tuple(claims))

    def _claim(
        self, profile: GpuMemoryProfile, component: Component, reserved: Decimal | None
    ) -> MemoryClaim:
        if component.kind is ComponentKind.LLM:
            return self._share(profile, component.id, self._llm.utilisation, self._llm.source)
        if reserved is not None:
            service = component.kind.value
            return self._share(
                profile,
                component.id,
                reserved,
                f"gpu-memory-utilization: {reserved} reported by the {service} service",
            )
        return profile.claims[component.id]

    @staticmethod
    def _share(
        profile: GpuMemoryProfile, component_id: str, share: Decimal | None, source: str
    ) -> MemoryClaim:
        return MemoryClaim(
            component_id,
            None if share is None else profile.gpu.total_gib * share,
            profile.gpu.basis,
            source,
        )
