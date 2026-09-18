from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from backend.domain.entities.persona import SamplingParams

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class TokenDelta:
    text: str


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    estimated: bool = False
    """True when the provider reported no usage and counts were approximated from text."""


LlmEvent = TokenDelta | TokenUsage


class LlmPort(Protocol):
    def complete(
        self, messages: Sequence[ChatMessage], sampling: SamplingParams
    ) -> AsyncIterator[LlmEvent]:
        """Streams TokenDelta events and ends with exactly one TokenUsage."""
        ...
