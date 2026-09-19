from __future__ import annotations

from collections.abc import AsyncIterator

from livekit import rtc

from backend.domain.value_objects.audio import PCM16_16K_MONO, AudioChunk

OUTPUT_SAMPLE_RATE = 24_000


class LiveKitAudioOutput:
    """AudioOutput over a LiveKit AudioSource. capture_frame blocks once ~1 s is queued,
    which paces playback; clear() drops the queue for barge-in. "First audio out" is
    therefore when the first frame is queued, which an empty queue plays immediately."""

    def __init__(self, source: rtc.AudioSource) -> None:
        self._source = source
        self._resamplers: dict[int, rtc.AudioResampler] = {}

    async def write(self, chunk: AudioChunk) -> None:
        fmt = chunk.format
        frame = rtc.AudioFrame(
            data=chunk.data,
            sample_rate=fmt.sample_rate,
            num_channels=fmt.channels,
            samples_per_channel=len(chunk.data) // (2 * fmt.channels),
        )
        if fmt.sample_rate == self._source.sample_rate:
            await self._source.capture_frame(frame)
            return
        resampler = self._resamplers.get(fmt.sample_rate)
        if resampler is None:
            resampler = rtc.AudioResampler(fmt.sample_rate, self._source.sample_rate)
            self._resamplers[fmt.sample_rate] = resampler
        for out in resampler.push(frame):
            await self._source.capture_frame(out)

    async def clear(self) -> None:
        self._source.clear_queue()

    def queued_seconds(self) -> float:
        return float(self._source.queued_duration)


async def caller_audio(track: rtc.Track) -> AsyncIterator[AudioChunk]:
    """The caller's microphone as 20 ms, 16 kHz mono chunks; ends when the track does."""
    stream = rtc.AudioStream.from_track(
        track=track, sample_rate=16_000, num_channels=1, frame_size_ms=20
    )
    try:
        async for event in stream:
            yield AudioChunk(bytes(event.frame.data), PCM16_16K_MONO)
    finally:
        await stream.aclose()
