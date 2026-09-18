from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from backend.domain.value_objects.audio import AudioChunk, AudioFormat


async def stream_pcm(
    response: httpx.Response, fmt: AudioFormat, frame_ms: int = 40
) -> AsyncIterator[AudioChunk]:
    """Re-frames an HTTP byte stream into whole-sample PCM chunks. Network chunk boundaries
    fall anywhere, including mid-sample, which would shift every later sample into noise."""
    frame_bytes = fmt.byte_count(frame_ms / 1000)
    buffer = bytearray()
    async for data in response.aiter_bytes():
        buffer.extend(data)
        while len(buffer) >= frame_bytes:
            yield AudioChunk(bytes(buffer[:frame_bytes]), fmt)
            del buffer[:frame_bytes]
    usable = len(buffer) - len(buffer) % (2 * fmt.channels)
    if usable:
        yield AudioChunk(bytes(buffer[:usable]), fmt)
