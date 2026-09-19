"""The real engines behind gpu/kokoro_service, driven through the service itself. Skipped
wherever a model's weights are not already cached, so CI never downloads several GB to run
the suite.

Time to first audio is printed, not asserted: it is a figure about this machine, and the
numbers that matter come from the L40S run (#10)."""

from __future__ import annotations

import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import numpy as np
import pytest

from backend.infrastructure.config.personas import YamlPersonaProvider
from backend.infrastructure.config.settings import REPO_ROOT
from gpu.kokoro_service.app import (
    SAMPLE_RATE,
    KokoroSynthesizer,
    Qwen3TtsSynthesizer,
    Synthesizer,
    create_app,
)
from tests.integration.servers import run_asgi

GERMAN = "Guten Tag, Kanzlei Hartley und Weber, hier ist Clara."
ENGLISH = "Good morning, Hartley and Weber, this is Clara."
AUDIBLE = REPO_ROOT / "results" / "tts"


RUNTIMES: dict[str, str] = {
    KokoroSynthesizer.engine: "kokoro",
    Qwen3TtsSynthesizer.engine: "mlx_audio.utils",
}
"""The module each engine's runtime lives in, both from the project's `local` extra.
mlx-audio's is named one level down because that is where DEFAULT_ALLOW_PATTERNS is."""


@pytest.fixture
def engine(request: pytest.FixtureRequest) -> type[Synthesizer]:
    """Sync, because pytest-asyncio turns a skip raised inside an async fixture into an
    error. Skips only when the runtime or the weights are missing here; an engine that
    then fails to load or speak is a failure."""
    chosen: type[Synthesizer] = request.param
    runtime = pytest.importorskip(RUNTIMES[chosen.engine])
    from huggingface_hub import snapshot_download

    # Each loader fetches its own subset of its repo, so asking for more than it would
    # download reads a usable cache as incomplete.
    patterns = getattr(runtime, "DEFAULT_ALLOW_PATTERNS", None)
    try:
        snapshot_download(chosen.model, allow_patterns=patterns, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - any lookup failure means "not downloaded here"
        pytest.skip(f"{chosen.model} is not cached: {exc}")
    return chosen


@pytest.fixture
async def service(engine: type[Synthesizer]) -> AsyncIterator[httpx.AsyncClient]:
    started = time.perf_counter()
    async with run_asgi(create_app((engine,), workers=1)) as host:
        print(f"{engine.engine}: {time.perf_counter() - started:.1f} s to load and warm")
        async with httpx.AsyncClient(base_url=f"http://{host}", timeout=120.0) as client:
            yield client


def save(name: str, pcm: bytes) -> Path:
    """Writes the synthesis out so a human can hear whether the German is native."""
    AUDIBLE.mkdir(parents=True, exist_ok=True)
    path = AUDIBLE / f"{name}.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm)
    return path


@pytest.mark.parametrize(
    ("engine", "text", "voice"),
    [
        pytest.param(KokoroSynthesizer, ENGLISH, "af_heart", id="kokoro"),
        pytest.param(Qwen3TtsSynthesizer, GERMAN, "qwen3-tts:clara_de", id="qwen3-tts"),
    ],
    indirect=["engine"],
)
async def test_the_engine_streams_a_sentence_as_24k_pcm(
    service: httpx.AsyncClient, text: str, voice: str
) -> None:
    started = time.perf_counter()
    first_ms, pcm = 0.0, b""
    async with service.stream(
        "POST", "/v1/synthesize", json={"text": text, "voice": voice}
    ) as response:
        assert response.status_code == 200
        assert response.headers["x-sample-rate"] == str(SAMPLE_RATE)
        async for chunk in response.aiter_bytes():
            if chunk and not pcm:
                # The service holds the headers back until the first segment exists, so
                # this is the whole wait a caller hears, not just the body's start.
                first_ms = (time.perf_counter() - started) * 1000
            pcm += chunk

    samples = np.frombuffer(pcm, dtype="<i2")
    seconds = len(samples) / SAMPLE_RATE
    engine = voice.partition(":")[0] if ":" in voice else KokoroSynthesizer.engine
    print(f"{engine}: time to first audio {first_ms:.0f} ms, {seconds:.2f} s spoken")
    print(f"{engine}: listen to {save(engine, pcm)}")
    assert len(pcm) % 2 == 0
    # A sentence this long takes at least this long to say, and silence is not speech.
    assert seconds > len(text) * 0.03
    assert np.abs(samples).mean() > 300


@pytest.mark.parametrize("engine", [Qwen3TtsSynthesizer], indirect=True)
async def test_the_german_persona_greeting_is_synthesised_by_the_voice_it_configures(
    service: httpx.AsyncClient,
) -> None:
    persona = YamlPersonaProvider(REPO_ROOT / "config" / "personas").get("law_firm_de")
    response = await service.post(
        "/v1/synthesize", json={"text": persona.greeting, "voice": persona.voices["selfhosted"]}
    )
    assert response.status_code == 200
    print(f"german persona greeting: listen to {save('law_firm_de_greeting', response.content)}")
    assert len(response.content) / 2 / SAMPLE_RATE > 3.0
