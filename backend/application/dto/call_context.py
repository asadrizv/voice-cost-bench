from __future__ import annotations

from dataclasses import dataclass, field

from backend.application.ports.llm_port import ChatMessage
from backend.application.ports.pipeline_provider import Pipeline
from backend.application.services.call_meter import CallMeter
from backend.application.services.concurrency_supervisor import ConcurrencySupervisor
from backend.domain.entities.call import Call
from backend.domain.entities.persona import Persona


@dataclass
class CallContext:
    call: Call
    pipeline: Pipeline
    persona: Persona
    meter: CallMeter
    started_monotonic: float
    supervisor: ConcurrencySupervisor | None = None
    history: list[ChatMessage] = field(default_factory=list)
