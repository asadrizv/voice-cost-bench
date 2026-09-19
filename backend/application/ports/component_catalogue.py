from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

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


class ComponentCatalogue(Protocol):
    def version(self) -> int: ...

    def components(self, kind: PipelineKind) -> list[Component]:
        """Every component that touches a call on that pipeline, in ComponentKind order,
        as the running configuration selects them."""
        ...
