from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from backend.application.ports.llm_port import ChatMessage
from backend.domain.entities.persona import SamplingParams
from backend.infrastructure.llm.openai_compatible import OpenAICompatibleLlm


class OllamaLlm(OpenAICompatibleLlm):
    """Ollama's OpenAI-compatible server, for running the self-hosted pipeline on a laptop.
    Ollama ignores vLLM's chat_template_kwargs; left thinking, Qwen spends the whole token
    budget reasoning and says nothing. It honours the standard reasoning_effort instead."""

    def _extra_body(self, sampling: SamplingParams) -> dict[str, Any] | None:
        return None if sampling.enable_thinking else {"reasoning_effort": "none"}

    async def prewarm(self, messages: Sequence[ChatMessage]) -> None:
        """Ollama keeps only the most recent prompt cached, so each new call would re-read
        the whole system prompt on its first turn (~0.6-1.3 s on an M5 Pro, vs ~45 ms
        cached). One token generated during the greeting moves that off the caller's clock."""
        async for _ in self.complete(messages, SamplingParams(max_tokens=1)):
            pass
