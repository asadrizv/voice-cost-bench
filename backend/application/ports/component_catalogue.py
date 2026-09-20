from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from backend.domain.services.gpu_memory_budget import GpuCapacity, MemoryClaim
from backend.domain.value_objects.pipeline_kind import PipelineKind


class ComponentKind(StrEnum):
    TELEPHONY = "telephony"
    STT = "stt"
    LLM = "llm"
    TTS = "tts"
    ORCHESTRATION = "orchestration"
    STORAGE = "storage"


@dataclass(frozen=True)
class Component:
    kind: ComponentKind
    id: str
    vendor: str
    model: str
    version: str
    region: str
    licence: str
    leaves_eu: bool
    assumption: str
    """Empty when every value above is verified; otherwise names what is assumed."""


@dataclass(frozen=True)
class GpuMemoryProfile:
    """One GPU and every component the catalogue places on it. claims is keyed by component
    id and holds only components that run on that GPU, so an id missing from it takes no
    GPU memory at all, while a claim with no figure is one nobody has measured."""

    gpu: GpuCapacity
    claims: Mapping[str, MemoryClaim]


class ComponentCatalogue(Protocol):
    def version(self) -> int: ...

    def components(self, kind: PipelineKind) -> list[Component]:
        """Every component that touches a call on that pipeline, in ComponentKind order,
        as the running configuration selects them."""
        ...


class EngineCatalogue(ComponentCatalogue, Protocol):
    def engine(self, kind: ComponentKind, engine_id: str) -> Component | None:
        """The entry for an engine a self-hosted service can run, None for any other id."""
        ...

    def gpu_memory(self) -> GpuMemoryProfile | None:
        """The GPU the catalogue places components on, None when no host has one."""
        ...
