from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from backend.domain.value_objects.audio import AudioChunk


class TtsPort(Protocol):
    def synthesize(self, text: str, voice: str | None) -> AsyncIterator[AudioChunk]:
        """Streams audio for `text`. Billed characters are len(text)."""
        ...
