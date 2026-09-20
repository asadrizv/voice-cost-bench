"""The real engines behind gpu/whisper_service, driven through the service's own socket at
the pace a caller speaks. Skipped wherever an engine's runtime or weights are not already
here, so CI never downloads gigabytes to run the suite.

Word error rate and time to final are printed, not asserted at their measured values: the
fixture audio is synthetic (see fixtures/wer/README.md) and the machine is a laptop, so the
numbers compare the engines with each other and nothing else. What is asserted is that each
engine transcribes a scripted call at all."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from backend.application.ports.stt_port import FlushSignal, SttPort
from backend.domain.value_objects.audio import AudioChunk
from backend.infrastructure.config.settings import REPO_ROOT
from backend.infrastructure.stt.whisper_stt import WhisperStt
from backend.interfaces.cli.harness.caller import FRAME_S, Conversation, load_conversation
from backend.interfaces.cli.harness.report import word_error_rate
from gpu.whisper_service.app import MlxWhisperTranscriber, VoxtralTranscriber, create_app
from tests.integration.servers import run_asgi

ENGINES = (MlxWhisperTranscriber, VoxtralTranscriber)
RUNTIME = "mlx_audio.utils"
"""Where both engines' download patterns live. mlx-whisper fetches a whole snapshot, and
mlx-audio's DEFAULT_ALLOW_PATTERNS cover every file the Voxtral repo holds, so one
`local`-extra module answers for both."""

WORST_TOLERABLE_WER = 0.35
"""A working engine on clean synthetic speech scores far below this; the bound is here to
fail a load that returns silence or noise, not to record quality."""


def cached(engine: type[MlxWhisperTranscriber | VoxtralTranscriber]) -> None:
    runtime = pytest.importorskip(RUNTIME)
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            engine.model, allow_patterns=runtime.DEFAULT_ALLOW_PATTERNS, local_files_only=True
        )
    except Exception as exc:  # noqa: BLE001 - any lookup failure means "not downloaded here"
        pytest.skip(f"{engine.model} is not cached: {exc}")


async def say(stt: SttPort, language: str, audio: list[AudioChunk]) -> tuple[str, float]:
    """One turn, paced at real time because a streaming engine decodes while the caller is
    still speaking and a burst would hide that. Returns the final text and the seconds from
    the end of the speech to the answer the caller waits for."""
    spoken = 0.0

    async def turn() -> AsyncIterator[AudioChunk | FlushSignal]:
        nonlocal spoken
        started = time.perf_counter()
        for i, chunk in enumerate(audio):
            due = started + i * FRAME_S
            if (wait := due - time.perf_counter()) > 0:
                await asyncio.sleep(wait)
            yield chunk
        spoken = time.perf_counter()
        yield FlushSignal()
        await asyncio.sleep(0.2)

    async for event in stt.stream(turn(), language):
        if event.flushed:
            return event.text, time.perf_counter() - spoken
    raise AssertionError("the flush was never answered")


async def transcribe(
    engine: type[MlxWhisperTranscriber | VoxtralTranscriber], conversation: Conversation
) -> tuple[float, list[float]]:
    loading = time.perf_counter()
    async with run_asgi(create_app(engine, workers=1)) as host:
        print(f"{engine.engine}: {time.perf_counter() - loading:.1f} s to load and warm")
        stt = WhisperStt(f"ws://{host}/v1/stream")
        wers, to_final = [], []
        for reference, audio in zip(conversation.turns, conversation.audio, strict=True):
            text, seconds = await say(stt, conversation.language, audio)
            wers.append(word_error_rate(reference, text))
            to_final.append(seconds)
            print(f"  {engine.engine} {seconds * 1000:6.0f} ms  wer {wers[-1]:.2f}  {text!r}")
    return sum(wers) / len(wers), to_final


@pytest.mark.parametrize("conversation", ["intake_en", "intake_de"])
async def test_both_local_engines_transcribe_a_scripted_call(conversation: str) -> None:
    """Voxtral against the MLX Whisper engine on identical audio: the comparison #29 asks
    for, run through the unchanged WebSocket protocol rather than the model's own API."""
    for engine in ENGINES:
        cached(engine)
    scripted = load_conversation(REPO_ROOT / "fixtures", conversation)

    for engine in ENGINES:
        wer, to_final = await transcribe(engine, scripted)
        ordered = sorted(to_final)
        print(
            f"{conversation} {engine.engine}: mean wer {wer:.3f}, time to final "
            f"median {ordered[len(ordered) // 2] * 1000:.0f} ms, worst {ordered[-1] * 1000:.0f} ms"
        )
        assert wer <= WORST_TOLERABLE_WER
