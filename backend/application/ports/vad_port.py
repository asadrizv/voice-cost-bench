from __future__ import annotations

from typing import Protocol

from backend.domain.value_objects.audio import AudioChunk


class VoiceActivityDetector(Protocol):
    def is_speech(self, chunk: AudioChunk) -> bool: ...
