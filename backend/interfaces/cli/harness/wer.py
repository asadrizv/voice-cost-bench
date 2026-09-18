from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import yaml

from backend.application.ports.stt_port import FlushSignal
from backend.domain.value_objects.audio import AudioChunk
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.audio.wav import frames, read_wav
from backend.infrastructure.config.settings import Settings
from backend.infrastructure.pipeline_factory import PipelineFactory
from backend.interfaces.cli.harness.report import word_error_rate


async def transcribe(stt: Any, pcm_frames: list[AudioChunk], language: str) -> str:
    async def audio() -> AsyncIterator[AudioChunk | FlushSignal]:
        for chunk in pcm_frames:
            yield chunk
        yield FlushSignal()
        await asyncio.sleep(1.5)  # keep the socket open for the flushed final

    text: list[str] = []
    async for event in stt.stream(audio(), language):
        if event.is_final and event.text:
            text.append(event.text)
        if event.flushed:
            break
    return " ".join(text)


async def run_wer(
    settings: Settings, fixtures: Path, pipeline: str, language: str, limit: int | None
) -> dict[str, Any]:
    spec = yaml.safe_load((fixtures / "wer" / "sentences.yaml").read_text())[language]
    stt = PipelineFactory(settings).resolve(PipelineKind(pipeline)).stt
    rows = []
    wav_dir = fixtures / "audio" / "wer" / language
    files = sorted(wav_dir.glob("*.wav"))[:limit] if limit else sorted(wav_dir.glob("*.wav"))
    if not files:
        raise SystemExit(f"no audio in {wav_dir}; run `make wer-audio`")
    for path in files:
        sentence = spec["sentences"][int(path.stem.split("_")[0])]
        pcm, fmt = read_wav(path)
        hypothesis = await transcribe(stt, frames(pcm, fmt), language)
        rows.append(
            {
                "file": path.name,
                "reference": sentence,
                "hypothesis": hypothesis,
                "wer": word_error_rate(sentence, hypothesis),
            }
        )
    mean = sum(r["wer"] for r in rows) / len(rows)
    return {
        "pipeline": pipeline,
        "language": language,
        "utterances": len(rows),
        "wer_mean": round(mean, 4),
        "synthetic_audio": True,
        "note": "macOS TTS audio: optimistic, not publishable. See fixtures/wer/README.md.",
        "rows": rows,
    }
