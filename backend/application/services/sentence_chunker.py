from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"([.!?…]+[\"')\]]*)(\s+|$)")
_CLAUSE_END = re.compile(r"([,;:–—])(\s+)")


class SentenceChunker:
    """Cuts a token stream into speakable chunks so TTS starts on the first sentence, not
    the whole reply. The first chunk may break early at a clause boundary: its length is
    the part of the LLM's time the caller waits through in silence."""

    def __init__(self, first_chunk_min_chars: int = 24, min_chars: int = 40) -> None:
        self._buffer = ""
        self._emitted_any = False
        self._first_min = first_chunk_min_chars
        self._min = min_chars

    def push(self, text: str) -> list[str]:
        self._buffer += text
        chunks: list[str] = []
        while True:
            chunk = self._take()
            if chunk is None:
                return chunks
            chunks.append(chunk)

    def flush(self) -> list[str]:
        rest = self._buffer.strip()
        self._buffer = ""
        return [rest] if rest else []

    def _take(self) -> str | None:
        min_len = self._min if self._emitted_any else self._first_min
        for match in _SENTENCE_END.finditer(self._buffer):
            # A sentence end needs trailing whitespace to be sure it isn't "3.5" or "Dr."
            # mid-stream; at the buffer's end we wait for more text.
            if match.end(2) == len(self._buffer) and not match.group(2):
                break
            if match.end(1) >= min_len or not self._emitted_any:
                return self._cut(match.end(1), match.end(2))
        if not self._emitted_any:
            for match in _CLAUSE_END.finditer(self._buffer):
                if match.end(1) >= self._first_min:
                    return self._cut(match.end(1), match.end(2))
        return None

    def _cut(self, text_end: int, consumed_end: int) -> str | None:
        chunk = self._buffer[:text_end].strip()
        self._buffer = self._buffer[consumed_end:]
        if not chunk:
            return None
        self._emitted_any = True
        return chunk
