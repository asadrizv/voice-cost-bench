from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
from openai import AsyncOpenAI

from backend.application.ports.llm_port import ChatMessage, LlmEvent, TokenDelta, TokenUsage
from backend.domain.entities.persona import SamplingParams


class OpenAICompatibleLlm:
    """Streaming chat completions against any OpenAI-compatible server."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key or "unused",
            base_url=base_url,
            http_client=http_client,
            timeout=timeout_s,
            max_retries=0,  # a retry mid-call costs seconds the caller hears; fail fast
        )

    def _extra_body(self, sampling: SamplingParams) -> dict[str, Any] | None:
        return None

    async def prewarm(self, messages: Sequence[ChatMessage]) -> None:
        return None

    async def complete(
        self, messages: Sequence[ChatMessage], sampling: SamplingParams
    ) -> AsyncIterator[LlmEvent]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],  # type: ignore[misc]
            stream=True,
            stream_options={"include_usage": True},
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            extra_body=self._extra_body(sampling),
        )
        output_chars = 0
        usage: TokenUsage | None = None
        async for chunk in stream:
            if chunk.usage is not None:
                usage = TokenUsage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
            for choice in chunk.choices:
                text = choice.delta.content if choice.delta else None
                if text:
                    output_chars += len(text)
                    yield TokenDelta(text)
        if usage is None:
            prompt_chars = sum(len(m.content) for m in messages)
            usage = TokenUsage(prompt_chars // 4, output_chars // 4, estimated=True)
        yield usage


class OpenAILlm(OpenAICompatibleLlm):
    """api.openai.com; no extra request body."""
