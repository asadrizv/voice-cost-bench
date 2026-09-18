"""One real request per paid provider. Opt-in (`make test-paid`); costs a few cents.
Doubles as the way to re-record the fixtures the contract tests replay."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from backend.application.ports.llm_port import ChatMessage, TokenDelta, TokenUsage
from backend.application.ports.stt_port import FlushSignal
from backend.domain.entities.persona import SamplingParams
from backend.domain.value_objects.audio import AudioChunk
from backend.infrastructure.audio.wav import frames, read_wav
from backend.infrastructure.config.settings import get_settings
from backend.infrastructure.llm.openai_llm import OpenAILlm
from backend.infrastructure.stt.deepgram_stt import DeepgramStt
from backend.infrastructure.tts.elevenlabs_tts import ElevenLabsTts

pytestmark = pytest.mark.paid
AUDIO = Path(__file__).resolve().parents[2] / "fixtures" / "audio" / "intake_en" / "00.wav"
S = get_settings()


def need(key: str) -> None:
    if not getattr(S, key):
        pytest.skip(f"{key.upper()} not set")


async def test_deepgram_transcribes_fixture() -> None:
    need("deepgram_api_key")
    pcm, fmt = read_wav(AUDIO)

    async def audio() -> AsyncIterator[AudioChunk | FlushSignal]:
        for chunk in frames(pcm, fmt):
            yield chunk
            await asyncio.sleep(0.02)
        yield FlushSignal()
        await asyncio.sleep(2)

    text = []
    async for event in DeepgramStt(S.deepgram_api_key, S.deepgram_model).stream(audio(), "en"):
        if event.is_final:
            text.append(event.text)
        if event.flushed:
            break
    assert "landlord" in " ".join(text).lower()


async def test_openai_streams_with_usage() -> None:
    need("openai_api_key")
    events = [
        e
        async for e in OpenAILlm(S.openai_model, S.openai_api_key).complete(
            [ChatMessage("user", "Reply with the single word: ready.")],
            SamplingParams(max_tokens=5),
        )
    ]
    assert any(isinstance(e, TokenDelta) for e in events)
    usage = events[-1]
    assert isinstance(usage, TokenUsage) and not usage.estimated


async def test_elevenlabs_streams_pcm() -> None:
    need("elevenlabs_api_key")
    tts = ElevenLabsTts(S.elevenlabs_api_key, S.elevenlabs_model)
    chunks = [c async for c in tts.synthesize("Good morning.", None)]
    assert sum(len(c.data) for c in chunks) > 24000  # at least half a second of audio
