from __future__ import annotations

from typing import Protocol

from backend.domain.entities.persona import Persona


class PersonaProvider(Protocol):
    def get(self, persona_id: str) -> Persona: ...
