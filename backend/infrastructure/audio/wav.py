from __future__ import annotations

import wave
from pathlib import Path

from backend.domain.value_objects.audio import AudioChunk, AudioFormat


def read_wav(path: Path) -> tuple[bytes, AudioFormat]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        return w.readframes(w.getnframes()), AudioFormat(w.getframerate(), w.getnchannels())


def write_wav(path: Path, pcm: bytes, fmt: AudioFormat) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(fmt.channels)
        w.setsampwidth(2)
        w.setframerate(fmt.sample_rate)
        w.writeframes(pcm)


def frames(pcm: bytes, fmt: AudioFormat, frame_ms: int = 20) -> list[AudioChunk]:
    size = fmt.byte_count(frame_ms / 1000)
    return [AudioChunk(pcm[i : i + size], fmt) for i in range(0, len(pcm), size)]
