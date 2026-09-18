from __future__ import annotations

from typing import Any

from backend.domain.entities.persona import SamplingParams
from backend.infrastructure.llm.openai_compatible import OpenAICompatibleLlm


class VllmLlm(OpenAICompatibleLlm):
    """vLLM's OpenAI-compatible server. Thinking is switched off per request through the
    chat template, so the persona, not the server, owns it."""

    def _extra_body(self, sampling: SamplingParams) -> dict[str, Any] | None:
        return {"chat_template_kwargs": {"enable_thinking": sampling.enable_thinking}}
