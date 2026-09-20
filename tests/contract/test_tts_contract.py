"""TtsPort contract against ElevenLabs (mock transport), both Apple Silicon engines of the
TTS service (real service code, stub synthesizers) and its CUDA engine (real backend code
against a server speaking vLLM-Omni's speech API)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any, NamedTuple

import httpx
import numpy as np
import pytest

from backend.application.ports.tts_port import TtsPort
from backend.domain.value_objects.audio import PCM16_24K_MONO
from backend.infrastructure.tts.elevenlabs_tts import ElevenLabsTts
from backend.infrastructure.tts.kokoro_tts import KokoroTts
from gpu.kokoro_service.app import (
    VOICES_PATH,
    Qwen3TtsSynthesizer,
    VllmOmniQwen3TtsSynthesizer,
    create_app,
    enabled_synthesizers,
    load_voices,
    to_pcm16,
)
from tests.integration.servers import FakeVllmOmniSpeech, run_asgi

SECONDS_PER_CHAR = 0.02
GERMAN_VOICE = "qwen3-tts:clara_de"


def fake_audio(text: str, hz: float = 220) -> np.ndarray:
    n = int(24000 * SECONDS_PER_CHAR * len(text))
    return (0.3 * np.sin(np.arange(n) * 2 * np.pi * hz / 24000)).astype(np.float32)


class StubSynth:
    """Kokoro's stub. Its tone says which engine served a request."""

    engine, model, warmup_voice, hz = "kokoro", "hexgrad/Kokoro-82M", "af_heart", 220.0
    voices = ("af_heart",)

    def speaks(self, voice: str) -> bool:
        return voice in self.voices

    def synthesize(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        if text == "FAIL":
            raise RuntimeError("model crashed")
        audio = fake_audio(text, self.hz)
        half = len(audio) // 2
        yield audio[:half]
        yield audio[half:]


class StubQwen(StubSynth):
    engine, model, warmup_voice, hz = (
        "qwen3-tts",
        Qwen3TtsSynthesizer.model,
        GERMAN_VOICE.split(":")[1],
        440.0,
    )
    voices = (GERMAN_VOICE.split(":")[1],)


BOTH_ENGINES = (StubSynth, StubQwen)


def elevenlabs_handler(requests: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        text = json.loads(request.content)["text"]
        if text == "FAIL":
            return httpx.Response(401, json={"detail": {"status": "invalid_api_key"}})
        pcm = to_pcm16(fake_audio(text))

        async def body():  # type: ignore[no-untyped-def]
            # Odd-sized network chunks: the adapter must re-align to whole samples.
            for i in range(0, len(pcm), 777):
                yield pcm[i : i + 777]

        return httpx.Response(200, content=body())

    return handle


class Impl(NamedTuple):
    tts: TtsPort
    voice: str | None
    hz: float
    """The tone the engine behind `voice` produces, so its audio is told from the others'."""


@asynccontextmanager
async def elevenlabs() -> AsyncIterator[Impl]:
    client = httpx.AsyncClient(
        base_url="https://api.elevenlabs.io", transport=httpx.MockTransport(elevenlabs_handler([]))
    )
    yield Impl(ElevenLabsTts("xi-test", client=client), None, 220.0)


@asynccontextmanager
async def kokoro() -> AsyncIterator[Impl]:
    async with run_asgi(create_app(BOTH_ENGINES, workers=2)) as host:
        yield Impl(KokoroTts(f"http://{host}"), None, StubSynth.hz)


@asynccontextmanager
async def qwen3_tts() -> AsyncIterator[Impl]:
    async with run_asgi(create_app(BOTH_ENGINES, workers=2)) as host:
        yield Impl(KokoroTts(f"http://{host}"), GERMAN_VOICE, StubQwen.hz)


def omni_server(**kwargs: Any) -> FakeVllmOmniSpeech:
    """A vLLM-Omni server whose model is the tone this suite tells Qwen3-TTS apart by."""
    return FakeVllmOmniSpeech(audio=lambda text: to_pcm16(fake_audio(text, StubQwen.hz)), **kwargs)


@asynccontextmanager
async def omni_service(server: FakeVllmOmniSpeech) -> AsyncIterator[str]:
    """The TTS service running Kokoro's stub beside its CUDA engine, whose model is a
    vLLM-Omni server away."""
    async with run_asgi(server.app()) as omni:
        engines = (StubSynth, lambda: VllmOmniQwen3TtsSynthesizer(base_url=f"http://{omni}"))
        async with run_asgi(create_app(engines, workers=2)) as host:
            yield host


@asynccontextmanager
async def qwen3_tts_vllm() -> AsyncIterator[Impl]:
    async with omni_service(omni_server()) as host:
        yield Impl(KokoroTts(f"http://{host}"), GERMAN_VOICE, StubQwen.hz)


IMPLEMENTATIONS: dict[str, Callable[[], AbstractAsyncContextManager[Impl]]] = {
    "elevenlabs": elevenlabs,
    "kokoro": kokoro,
    "qwen3-tts": qwen3_tts,
    "qwen3-tts-vllm": qwen3_tts_vllm,
}


@pytest.fixture(params=list(IMPLEMENTATIONS))
async def tts(request: pytest.FixtureRequest) -> AsyncIterator[Impl]:
    async with IMPLEMENTATIONS[request.param]() as impl:
        yield impl


async def test_streams_whole_sample_24k_pcm_of_the_right_length(tts: Impl) -> None:
    text = "Thank you, Ms. Weber."
    chunks = [c async for c in tts.tts.synthesize(text, tts.voice)]
    assert len(chunks) > 1
    assert all(c.format == PCM16_24K_MONO and len(c.data) % 2 == 0 for c in chunks)
    total = b"".join(c.data for c in chunks)
    assert total == to_pcm16(fake_audio(text, tts.hz))


async def test_failure_raises(tts: Impl) -> None:
    with pytest.raises(Exception, match="401|model crashed|500"):
        _ = [c async for c in tts.tts.synthesize("FAIL", tts.voice)]


async def test_elevenlabs_request_shape() -> None:
    requests: list[httpx.Request] = []
    client = httpx.AsyncClient(
        base_url="https://api.elevenlabs.io",
        transport=httpx.MockTransport(elevenlabs_handler(requests)),
    )
    tts = ElevenLabsTts("xi-test", model="eleven_flash_v2_5", client=client)
    _ = [c async for c in tts.synthesize("Hello.", "voice-123")]
    req = requests[0]
    assert req.url.path == "/v1/text-to-speech/voice-123/stream"
    assert req.url.params["output_format"] == "pcm_24000"
    assert req.headers["xi-api-key"] == "xi-test"
    assert json.loads(req.content) == {"text": "Hello.", "model_id": "eleven_flash_v2_5"}


async def test_kokoro_service_health_and_metrics() -> None:
    async with (
        run_asgi(create_app((StubSynth,), workers=1)) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        assert (await client.get("/health/deep")).json()["status"] == "ok"
        metrics = (await client.get("/metrics")).text
    assert "kokoro_waiting" in metrics and "kokoro_first_byte_ms" in metrics


async def test_health_is_probed_with_a_voice_the_loaded_engine_speaks() -> None:
    """gpu/README.md recommends running one CUDA speech engine, and Qwen3-TTS alone speaks
    no `af_heart`. Probing the request default made /health/deep 503 forever there, and
    runpod/start.sh waits on it without a timeout."""
    async with (
        run_asgi(create_app((StubQwen,), workers=1)) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        assert (await client.get("/health/deep")).json()["status"] == "ok"


@pytest.mark.parametrize(("voice", "hz"), [("af_heart", StubSynth.hz), (GERMAN_VOICE, StubQwen.hz)])
async def test_the_voice_chooses_the_engine_that_speaks_it(voice: str, hz: float) -> None:
    async with (
        run_asgi(create_app(BOTH_ENGINES, workers=1)) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        response = await client.post("/v1/synthesize", json={"text": "Guten Tag.", "voice": voice})

    assert response.status_code == 200
    assert response.headers["x-sample-rate"] == "24000"
    assert response.content == to_pcm16(fake_audio("Guten Tag.", hz))


@pytest.mark.parametrize("voice", ["orpheus:clara", "qwen3-tts:nobody"])
async def test_a_voice_no_engine_speaks_is_refused_before_any_audio_is_promised(
    voice: str,
) -> None:
    """A 200 that then stops reads to a caller as silence, so the refusal has to come
    with the status code."""
    async with (
        run_asgi(create_app(BOTH_ENGINES, workers=1)) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        response = await client.post("/v1/synthesize", json={"text": "Guten Tag.", "voice": voice})

    assert response.status_code == 400
    assert voice in response.json()["error"]


async def test_each_engine_loads_its_model_once_at_startup_rather_than_per_request() -> None:
    """Qwen3-TTS's weights are gigabytes: a caller must never wait for them, and two
    copies would not fit beside Kokoro and the LLM."""
    loads: list[str] = []

    class CountedKokoro(StubSynth):
        def __init__(self) -> None:
            loads.append(self.engine)

    class CountedQwen(StubQwen):
        def __init__(self) -> None:
            loads.append(self.engine)

    async with (
        run_asgi(create_app((CountedKokoro, CountedQwen), workers=1)) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        assert sorted(loads) == ["kokoro", "qwen3-tts"]
        for voice in (GERMAN_VOICE, "af_heart", GERMAN_VOICE):
            await client.post("/v1/synthesize", json={"text": "Guten Tag.", "voice": voice})

    assert sorted(loads) == ["kokoro", "qwen3-tts"]


async def test_the_service_reports_every_engine_it_can_route_to() -> None:
    async with (
        run_asgi(create_app(BOTH_ENGINES, workers=1)) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        engines = (await client.get("/v1/info")).json()["engines"]

    assert engines == [
        {"id": "kokoro", "model": "hexgrad/Kokoro-82M"},
        {"id": "qwen3-tts", "model": Qwen3TtsSynthesizer.model},
    ]


@pytest.mark.parametrize(
    ("setting", "engines"),
    [
        (None, ["kokoro"]),
        ("qwen3-tts", ["qwen3-tts"]),
        ("kokoro, qwen3-tts", ["kokoro", "qwen3-tts"]),
        ("kokoro,qwen3-tts-vllm", ["kokoro", "qwen3-tts"]),
    ],
)
def test_the_environment_says_which_engines_a_process_loads(
    monkeypatch: pytest.MonkeyPatch, setting: str | None, engines: list[str]
) -> None:
    monkeypatch.delenv("TTS_ENGINES", raising=False)
    if setting is not None:
        monkeypatch.setenv("TTS_ENGINES", setting)
    assert [s.engine for s in enabled_synthesizers()] == engines


def test_an_engine_this_service_cannot_run_is_refused_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TTS_ENGINES", "kokoro,orpheus")
    with pytest.raises(RuntimeError, match="orpheus"):
        enabled_synthesizers()


async def test_the_cuda_engine_asks_vllm_omni_to_stream_the_designed_voice_as_pcm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VoiceDesign carries the voice in `instructions`, and the stream has to be raw PCM at
    this service's rate: a stream_format the server frames differently, or a rate it
    resamples to, would reach the caller as noise under the x-sample-rate this service sets."""
    monkeypatch.delenv("QWEN3_TTS_VLLM_MODEL", raising=False)
    server = omni_server()
    async with omni_service(server) as host, httpx.AsyncClient(base_url=f"http://{host}") as client:
        await client.post("/v1/synthesize", json={"text": "Guten Tag.", "voice": GERMAN_VOICE})

    clara = load_voices(VOICES_PATH)["clara_de"]
    assert server.requests[-1] == {
        # Qwen's VoiceDesign checkpoint, the one this backend asks a server to be serving.
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "input": "Guten Tag.",
        "response_format": "pcm",
        "stream": True,
        "stream_format": "audio",
        "sample_rate": 24000,
        "task_type": "VoiceDesign",
        "instructions": clara["description"],
        "language": "German",
    }


async def test_a_vllm_omni_failure_before_any_audio_is_refused_with_a_status() -> None:
    """A 200 whose body then stops reads to a caller as silence, so an engine that fails to
    start generating must be answered with the status code, not with an empty stream."""
    async with (
        omni_service(omni_server()) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        response = await client.post("/v1/synthesize", json={"text": "FAIL", "voice": GERMAN_VOICE})

    assert response.status_code == 500
    assert "engine failed to start generation" in response.json()["error"]


async def test_a_request_vllm_omni_refuses_is_not_played_to_the_caller_as_audio() -> None:
    """A refusal comes back as JSON under a 4xx -- a checkpoint that cannot do VoiceDesign
    is the documented case -- and a client reading it as PCM would play it as noise."""
    async with (
        omni_service(omni_server(refusal_status=400)) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        response = await client.post("/v1/synthesize", json={"text": "FAIL", "voice": GERMAN_VOICE})

    assert response.status_code == 500
    assert "does not support task_type" in response.json()["error"]


async def test_a_vllm_omni_stream_that_dies_mid_body_reaches_the_caller_as_a_failure() -> None:
    """Raw PCM carries no error frame, so a dying engine can only cut the body. The caller
    has to see a broken stream rather than a short, clean one it would play as the whole
    utterance."""
    async with omni_service(omni_server(truncates="Guten Tag, Frau Weber.")) as host:
        tts = KokoroTts(f"http://{host}")
        with pytest.raises(Exception, match="incomplete|peer closed|ChunkedEncoding"):
            _ = [chunk async for chunk in tts.synthesize("Guten Tag, Frau Weber.", GERMAN_VOICE)]


async def test_the_cuda_engine_refuses_a_speed_vllm_omni_cannot_stream() -> None:
    """vLLM-Omni streams at speed 1.0 only; a request for another speed is refused here,
    with the constraint named, rather than served at a speed the caller did not ask for."""
    async with (
        omni_service(omni_server()) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        response = await client.post(
            "/v1/synthesize", json={"text": "Guten Tag.", "voice": GERMAN_VOICE, "speed": 1.5}
        )

    assert response.status_code == 500
    assert "1.0" in response.json()["error"]


async def test_the_cuda_engine_is_reported_as_qwen3_tts_by_the_runtime_that_served_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine is the one the Apple Silicon backend runs, so /transparency names the same
    catalogue entry; the model says which runtime produced the numbers, which the
    catalogue's version -- fixed configuration -- cannot."""
    monkeypatch.delenv("QWEN3_TTS_VLLM_MODEL", raising=False)
    async with (
        omni_service(omni_server()) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        engines = (await client.get("/v1/info")).json()["engines"]

    assert engines == [
        {"id": "kokoro", "model": "hexgrad/Kokoro-82M"},
        {"id": "qwen3-tts", "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign (vLLM-Omni)"},
    ]


def test_two_runtimes_of_one_engine_cannot_be_loaded_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both would answer to the same voice prefix and the same catalogue entry, so which of
    them served a call -- and which one /v1/info names -- would be arbitrary."""
    monkeypatch.setenv("TTS_ENGINES", "qwen3-tts,qwen3-tts-vllm")
    with pytest.raises(RuntimeError, match="qwen3-tts"):
        enabled_synthesizers()


def test_the_shipped_voice_file_designs_a_german_voice() -> None:
    """The description is the whole voice: VoiceDesign has no German preset to fall back
    on, and an empty or mislabelled one gives a Chinese-accented Clara."""
    clara = load_voices(VOICES_PATH)["clara_de"]
    assert clara["language"] == "german"
    assert "German" in clara["description"]


def test_a_voice_file_with_no_voices_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "voices.yaml"
    empty.write_text("version: 1\nvoices: {}\n")
    with pytest.raises(RuntimeError, match=str(empty)):
        load_voices(empty)
