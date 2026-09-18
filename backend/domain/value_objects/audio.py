from dataclasses import dataclass


@dataclass(frozen=True)
class AudioFormat:
    """Interleaved signed 16-bit little-endian PCM."""

    sample_rate: int = 16_000
    channels: int = 1

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * 2

    def duration_seconds(self, byte_count: int) -> float:
        return byte_count / self.bytes_per_second

    def byte_count(self, seconds: float) -> int:
        frames = int(round(seconds * self.sample_rate))
        return frames * self.channels * 2


PCM16_16K_MONO = AudioFormat(16_000, 1)
PCM16_24K_MONO = AudioFormat(24_000, 1)
PCM16_48K_MONO = AudioFormat(48_000, 1)


@dataclass(frozen=True)
class AudioChunk:
    data: bytes
    format: AudioFormat = PCM16_16K_MONO

    @property
    def duration_seconds(self) -> float:
        return self.format.duration_seconds(len(self.data))

    def is_empty(self) -> bool:
        return not self.data
