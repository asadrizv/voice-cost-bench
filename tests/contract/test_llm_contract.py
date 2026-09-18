"""LlmPort contract against both implementations, over recorded SSE fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from backend.application.ports.llm_port import ChatMessage, LlmPort, TokenDelta, TokenUsage
from backend.domain.entities.persona import SamplingParams
from backend.infrastructure.llm.openai_llm import OpenAILlm
from backend.infrastructure.llm.vllm_llm import VllmLlm

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "openai"
MESSAGES = [ChatMessage("system", "You are Clara."), ChatMessage("user", "Hello, I need help.")]


class Recorder:
    def __init__(self, fixture: str, status: int = 200) -> None:
        self.body = (FIXTURES / fixture).read_bytes()
        self.status = status
        self.requests: list[dict[str, object]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "overloaded"}})

        async def body():  # type: ignore[no-untyped-def]
            for line in self.body.split(b"\n\n"):
                if line.strip():
                    yield line + b"\n\n"

        return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})


def build(kind: str, recorder: Recorder) -> LlmPort:
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    if kind == "openai":
        return OpenAILlm("gpt-4o-mini", "sk-test", http_client=client)
    return VllmLlm("voice-llm", "unused", base_url="http://gpu:8000/v1", http_client=client)


@pytest.fixture(params=["openai", "vllm"])
def kind(request: pytest.FixtureRequest) -> str:
    return str(request.param)


async def run(llm: LlmPort) -> list[object]:
    return [e async for e in llm.complete(MESSAGES, SamplingParams(max_tokens=64))]


async def test_streams_deltas_then_exactly_one_usage(kind: str) -> None:
    events = await run(build(kind, Recorder("stream_with_usage.txt")))
    deltas = [e for e in events if isinstance(e, TokenDelta)]
    usages = [e for e in events if isinstance(e, TokenUsage)]
    assert "".join(d.text for d in deltas) == "Of course. May I have your name?"
    assert usages == [TokenUsage(412, 9)]
    assert events[-1] is usages[0]


async def test_missing_usage_is_estimated_and_flagged(kind: str) -> None:
    events = await run(build(kind, Recorder("stream_without_usage.txt")))
    usage = events[-1]
    assert isinstance(usage, TokenUsage) and usage.estimated
    assert usage.input_tokens > 0 and usage.output_tokens > 0


async def test_request_streams_with_usage_and_persona_sampling(kind: str) -> None:
    recorder = Recorder("stream_with_usage.txt")
    await run(build(kind, recorder))
    body = recorder.requests[0]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 64 and body["temperature"] == 0.4
    assert body["messages"][0] == {"role": "system", "content": "You are Clara."}


async def test_provider_error_raises(kind: str) -> None:
    import openai

    with pytest.raises(openai.APIStatusError):
        await run(build(kind, Recorder("stream_with_usage.txt", status=503)))


async def test_vllm_switches_thinking_off_per_request() -> None:
    recorder = Recorder("stream_with_usage.txt")
    await run(build("vllm", recorder))
    assert recorder.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    openai_recorder = Recorder("stream_with_usage.txt")
    await run(build("openai", openai_recorder))
    assert "chat_template_kwargs" not in openai_recorder.requests[0]
