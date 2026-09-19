from __future__ import annotations

import re
from pathlib import Path

import yaml

from backend.domain.entities.persona import Persona, SamplingParams


class UnknownPersona(KeyError):
    pass


class MissingAiDisclosure(ValueError):
    pass


# The EU AI Act (Art. 50(1)) requires telling callers they are talking to an AI.
_AI_DISCLOSURE = {
    "en": re.compile(r"\b(AI|artificial intelligence)\b", re.IGNORECASE),
    "de": re.compile(r"\b(KI|künstliche[rn]? Intelligenz)\b", re.IGNORECASE),
}


class YamlPersonaProvider:
    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._cache: dict[str, Persona] = {}

    def get(self, persona_id: str) -> Persona:
        if persona_id not in self._cache:
            path = self._dir / f"{persona_id}.yaml"
            if not path.is_file():
                raise UnknownPersona(persona_id)
            self._cache[persona_id] = _load(path)
        return self._cache[persona_id]

    def available(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("*.yaml"))

    def validate_all(self) -> None:
        """Loads every persona so a non-compliant one fails at startup. Raises
        MissingAiDisclosure (or ValueError) naming the file."""
        for persona_id in self.available():
            self.get(persona_id)


def _load(path: Path) -> Persona:
    raw = yaml.safe_load(path.read_text())
    slots = ", ".join(raw.get("available_slots", []))
    sampling = raw.get("sampling", {})
    if sampling.get("enable_thinking"):
        raise ValueError(f"{path.name}: enable_thinking must be false for voice")
    language = raw.get("language", "en")
    greeting = raw["greeting"].strip()
    disclosure = _AI_DISCLOSURE.get(language)
    if disclosure is None or not disclosure.search(greeting):
        raise MissingAiDisclosure(
            f"{path.name}: greeting has no AI disclosure recognised for language {language!r}"
        )
    return Persona(
        id=raw["id"],
        language=language,
        greeting=greeting,
        system_prompt=raw["system_prompt"].replace("{available_slots}", slots).strip(),
        sampling=SamplingParams(**sampling),
        voices={str(k): str(v) for k, v in raw.get("voices", {}).items()},
    )
