from __future__ import annotations

from typing import Any

from backend.domain.entities.persona import SamplingParams
from backend.infrastructure.llm.openai_compatible import OpenAICompatibleLlm


class OllamaLlm(OpenAICompatibleLlm):
    """Ollama's OpenAI-compatible server, for running the self-hosted pipeline on a laptop.
    Ollama ignores vLLM's chat_template_kwargs; left thinking, Qwen spends the whole token
    budget reasoning and says nothing. It honours the standard reasoning_effort instead."""

    def _extra_body(self, sampling: SamplingParams) -> dict[str, Any] | None:
        return None if sampling.enable_thinking else {"reasoning_effort": "none"}
