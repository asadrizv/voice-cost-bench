from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from backend.application.ports.llm_port import LlmPort
from backend.application.ports.stt_port import SttPort
from backend.application.ports.tts_port import TtsPort
from backend.domain.value_objects.pipeline_kind import PipelineKind


@dataclass(frozen=True)
class Pipeline:
    kind: PipelineKind
    stt: SttPort
    llm: LlmPort
    tts: TtsPort
    uses_gpu: bool


class PipelineProvider(Protocol):
    def resolve(self, kind: PipelineKind) -> Pipeline: ...
