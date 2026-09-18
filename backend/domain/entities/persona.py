from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 160
    enable_thinking: bool = False
    """Must stay False for voice: a reasoning trace before the first token blows the budget."""


@dataclass(frozen=True)
class Persona:
    id: str
    language: str
    greeting: str
    system_prompt: str
    sampling: SamplingParams = field(default_factory=SamplingParams)
    voices: dict[str, str] = field(default_factory=dict)
    """Voice id per pipeline kind; each TTS adapter interprets its own."""
