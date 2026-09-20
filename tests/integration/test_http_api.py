from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import jwt
import numpy as np
import pytest
from fastapi import FastAPI

from backend.application.ports.component_catalogue import ComponentKind
from backend.application.ports.pipeline_provider import Pipeline
from backend.application.use_cases.handle_call_turn import TurnRequest
from backend.application.use_cases.start_call import VoiceNotAvailable
from backend.domain.entities.latency import TurnTimeline
from backend.domain.services.gpu_memory_budget import BudgetVerdict
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.component_catalogue import (
    ComponentCatalogueError,
    NotEuResident,
)
from backend.infrastructure.config.settings import Settings
from backend.infrastructure.persistence.inmemory_call_repository import InMemoryCallRepository
from backend.infrastructure.pipeline_factory import SELFHOSTED_ENGINES
from backend.interfaces.container import build_container, load_static_config
from backend.interfaces.http.app import create_app
from gpu.kokoro_service.app import SYNTHESIZERS, KokoroSynthesizer
from gpu.kokoro_service.app import create_app as kokoro_app
from gpu.whisper_service.app import TRANSCRIBERS, FasterWhisperTranscriber
from gpu.whisper_service.app import create_app as whisper_app
from tests.config_fixtures import config_copy
from tests.fakes import (
    FakeClock,
    FakeLlm,
    FakeTts,
    NullMetrics,
    RecordingOutput,
    ScriptedStt,
    StaticPipelines,
)
from tests.integration.servers import run_asgi

SETTINGS = Settings(
    database_url="",
    livekit_api_key="lk-key",
    livekit_api_secret="lk-secret-that-is-long-enough-for-hs256",
    livekit_url="wss://example.livekit.cloud",
    internal_token="internal-test",
    # Nothing listens on port 9, so no test asks whichever GPU services this machine runs.
    whisper_ws_url="ws://127.0.0.1:9/v1/stream",
    kokoro_url="http://127.0.0.1:9",
)


@pytest.fixture
async def api() -> AsyncIterator[tuple[httpx.AsyncClient, object]]:
    clock = FakeClock()
    pipelines = StaticPipelines(
        {
            kind: Pipeline(
                kind,
                ScriptedStt([], clock),
                FakeLlm(["Of course. May I have your name?"] * 10, clock),
                FakeTts(clock),
                uses_gpu=kind is PipelineKind.SELFHOSTED,
            )
            for kind in PipelineKind
        }
    )
    app = create_app(SETTINGS, pipelines=pipelines)
    async with client_for(app) as client:
        yield client, app.state.container


@asynccontextmanager
async def client_for(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """An HTTP client onto an app already built — for tests that keep one app across two
    clients, where rebuilding it would throw away the container's caches."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@asynccontextmanager
async def api_client(settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    """The API under these settings, served in-process with no pipelines behind it."""
    async with client_for(create_app(settings, pipelines=StaticPipelines({}))) as client:
        yield client


async def make_call(container, kind: PipelineKind) -> str:  # type: ignore[no-untyped-def]
    ctx = await container.start_call.execute(kind, "law_firm")
    ctx.meter.add_stt_audio(5.0)
    now = container.clock.monotonic()
    await container.handle_turn.execute(
        ctx,
        TurnRequest(
            user_text="I need a lawyer.",
            output=RecordingOutput(),
            timeline=TurnTimeline(speech_end=now - 0.4, endpoint=now - 0.1, stt_final=now),
        ),
    )
    await container.end_call.execute(ctx)
    return ctx.call.id


async def test_call_history_detail_and_cost_attribution(api) -> None:  # type: ignore[no-untyped-def]
    client, container = api
    api_id = await make_call(container, PipelineKind.API)
    await make_call(container, PipelineKind.SELFHOSTED)

    calls = (await client.get("/calls")).json()
    assert {c["pipeline"] for c in calls} == {"api", "selfhosted"}
    assert all(c["status"] == "completed" and c["turns"] == 1 for c in calls)

    detail = (await client.get(f"/calls/{api_id}")).json()
    assert detail["turn_list"][0]["user_text"] == "I need a lawyer."
    assert detail["turn_list"][0]["latency"]["perceived_delay"] > 0

    cost = (await client.get(f"/calls/{api_id}/cost")).json()
    assert set(cost["stages"]) == {"stt", "llm", "tts", "gpu", "telephony", "total"}
    assert cost["stages"]["stt"] > 0 and cost["stages"]["llm"] > 0 and cost["stages"]["tts"] > 0
    assert cost["stages"]["gpu"] == 0
    assert cost["units"]["llm_input_tokens"] == 500
    assert cost["rates"]["stt_per_minute"] == pytest.approx(0.0043)
    assert cost["total_usd"] == pytest.approx(cost["recomputed_total_usd"])
    assert cost["rate_card_verified_on"] == "2026-09-17"

    assert (await client.get("/calls/nope")).status_code == 404
    assert (await client.get("/calls/nope/cost")).status_code == 404

    compare = {row["pipeline"]: row for row in (await client.get("/calls/compare")).json()}
    assert compare["api"]["calls"] == 1 and compare["selfhosted"]["cost"]["gpu"] > 0


async def test_loadtest_calls_are_hidden_from_history_by_default(api) -> None:  # type: ignore[no-untyped-def]
    client, container = api
    ctx = await container.start_call.execute(PipelineKind.API, "law_firm", source="loadtest")
    await container.end_call.execute(ctx)
    assert (await client.get("/calls")).json() == []
    assert len((await client.get("/calls", params={"source": ""})).json()) == 1


async def test_live_metrics_stream_replays_the_call(api) -> None:  # type: ignore[no-untyped-def]
    client, container = api
    call_id = await make_call(container, PipelineKind.API)
    kinds: list[str] = []
    async with client.stream("GET", f"/metrics/live/{call_id}") as response:
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                kinds.append(line.split(":", 1)[1].strip())
            if line.startswith("data:") and kinds and kinds[-1] == "turn":
                data = json.loads(line[5:])
                assert data["latency"]["perceived_delay"] > 0
                assert float(data["running_cost"]["total"]) > 0
    assert kinds == ["started", "turn", "ended"]


async def test_internal_event_ingest_requires_token(api) -> None:  # type: ignore[no-untyped-def]
    client, _ = api
    event = {
        "kind": "started",
        "call_id": "remote-1",
        "pipeline": "selfhosted",
        "elapsed_seconds": 0,
        "running_cost": {"total": "0"},
    }
    assert (await client.post("/internal/metrics/events", json=event)).status_code == 401
    ok = await client.post(
        "/internal/metrics/events", json=event, headers={"x-internal-token": "internal-test"}
    )
    assert ok.status_code == 202
    metrics = (await client.get("/metrics")).text
    assert 'voice_calls_total{event="started",pipeline="selfhosted"} 1.0' in metrics


async def test_prometheus_exposes_latency_and_cost(api) -> None:  # type: ignore[no-untyped-def]
    client, container = api
    await make_call(container, PipelineKind.API)
    metrics = (await client.get("/metrics")).text
    assert 'voice_turn_latency_ms_bucket{le="900.0",pipeline="api",stage="end_to_end"}' in metrics
    assert 'voice_cost_usd_total{component="llm",pipeline="api"}' in metrics


async def test_token_dispatches_our_agent_with_the_pipeline_choice(api) -> None:  # type: ignore[no-untyped-def]
    client, _ = api
    body = (await client.post("/token", json={"pipeline": "selfhosted"})).json()
    assert body["room"] == body["call_id"] and body["url"] == "wss://example.livekit.cloud"
    claims = jwt.decode(body["token"], SETTINGS.livekit_api_secret, algorithms=["HS256"])
    assert claims["video"]["room"] == body["room"] and claims["video"]["roomJoin"]
    room_config = claims["roomConfig"]
    dispatch = room_config["agents"][0]
    assert dispatch["agentName"] == "voice-cost-bench"
    assert json.loads(dispatch["metadata"])["pipeline"] == "selfhosted"
    assert json.loads(dispatch["metadata"])["endpointer"] == "semantic"

    for choice in ("silence", "smart_turn"):
        body = (await client.post("/token", json={"endpointer": choice})).json()
        claims = jwt.decode(body["token"], SETTINGS.livekit_api_secret, algorithms=["HS256"])
        metadata = json.loads(claims["roomConfig"]["agents"][0]["metadata"])
        assert metadata["endpointer"] == choice

    assert (await client.post("/token", json={"persona": "nope"})).status_code == 404
    assert (await client.post("/token", json={"endpointer": "vibes"})).status_code == 422


async def test_config_and_benchmark(api, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client, _ = api
    config = (await client.get("/config")).json()
    assert config["rates"]["selfhosted"]["gpu_per_hour"] == 1.09
    assert "law_firm_de" in config["personas"]
    assert {q["provider"] for q in config["client_gpu_quotes"]} >= {"scaleway", "ovhcloud"}

    from backend.interfaces.http import routes_benchmark

    monkeypatch.setattr(routes_benchmark, "RESULTS", tmp_path)
    assert (await client.get("/benchmark")).status_code == 404
    (tmp_path / "benchmark.json").write_text(json.dumps({"levels": []}))
    assert (await client.get("/benchmark")).json() == {"levels": []}
    assert (await client.get("/benchmark", params={"name": "../pyproject.toml"})).status_code == 404
    assert (await client.get("/benchmark/runs")).json() == ["benchmark.json"]


async def test_deep_health_exercises_models(api) -> None:  # type: ignore[no-untyped-def]
    client, _ = api
    body = (await client.get("/health/deep", params={"pipeline": "api"})).json()
    assert body["status"] == "ok"
    assert set(body["checks"]) == {"database", "llm", "tts"}


COMPONENT_FIELDS = {
    "kind",
    "id",
    "vendor",
    "model",
    "version",
    "region",
    "licence",
    "leaves_eu",
    "assumption",
    "confirmed",
    "unconfirmed_reason",
}
KINDS = ["telephony", "stt", "llm", "tts", "orchestration", "storage"]


async def test_transparency_lists_every_component_of_both_pipelines(api) -> None:  # type: ignore[no-untyped-def]
    client, _ = api
    body = (await client.get("/transparency")).json()

    assert body["catalogue_version"] == 1
    assert set(body["pipelines"]) == {"api", "selfhosted"}
    for components in body["pipelines"].values():
        assert [c["kind"] for c in components] == KINDS
        for c in components:
            assert set(c) == COMPONENT_FIELDS
            assert all(
                c[f]
                for f in COMPONENT_FIELDS
                - {"leaves_eu", "assumption", "confirmed", "unconfirmed_reason"}
            )
            assert isinstance(c["leaves_eu"], bool)

    api_stack = {c["kind"]: c for c in body["pipelines"]["api"]}
    assert (api_stack["stt"]["vendor"], api_stack["stt"]["model"]) == ("Deepgram", "nova-3")
    assert (api_stack["llm"]["vendor"], api_stack["llm"]["model"]) == ("OpenAI", "gpt-4o-mini")
    assert api_stack["tts"]["model"] == "eleven_flash_v2_5"
    assert api_stack["tts"]["licence"] == "proprietary" and api_stack["tts"]["leaves_eu"]
    assert api_stack["stt"]["region"] == "us"
    assert api_stack["stt"]["assumption"] == (
        "Deepgram's default US endpoint; no EU endpoint is configured."
    )

    selfhosted = {c["kind"]: c for c in body["pipelines"]["selfhosted"]}
    assert selfhosted["llm"]["model"] == "Qwen/Qwen3.5-9B"
    assert selfhosted["llm"]["version"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert selfhosted["stt"]["id"] == "faster-whisper" and selfhosted["stt"]["licence"] == "MIT"
    assert selfhosted["tts"]["licence"] == "Apache-2.0" and not selfhosted["tts"]["leaves_eu"]
    assert selfhosted["tts"]["region"] == "gpu-host (see deployment)"
    assert selfhosted["tts"]["assumption"].startswith("Runs wherever the GPU runs.")
    assert selfhosted["stt"]["assumption"].startswith(
        "model is whisper_service's default; WHISPER_MODEL on the GPU host overrides it. Runs"
    )
    assert selfhosted["orchestration"]["id"] == "livekit-cloud"
    assert selfhosted["storage"]["id"] == "in-memory"


async def transparency_of(settings: Settings) -> dict[str, dict[str, dict[str, object]]]:
    body = await transparency_body(settings)
    return {p: {c["kind"]: c for c in cs} for p, cs in body["pipelines"].items()}


async def test_transparency_follows_the_catalogue_and_the_running_configuration(
    tmp_path: Path,
) -> None:
    config = config_copy(tmp_path)
    catalogue = config / "components.yaml"
    catalogue.write_text(
        catalogue.read_text().replace(
            "kokoro>=0.9.4\n    host: gpu_host\n    licence: Apache-2.0",
            "kokoro>=0.9.4\n    host: gpu_host\n    licence: X",
        )
    )
    settings = SETTINGS.model_copy(
        update={
            "config_dir": config,
            "deepgram_model": "nova-3-medical",
            "llm_server": "ollama",
            "vllm_model": "qwen3.5:9b",
            "livekit_url": "ws://localhost:7880",
        }
    )

    stacks = await transparency_of(settings)

    assert stacks["selfhosted"]["tts"]["licence"] == "X"
    assert stacks["api"]["stt"]["model"] == "nova-3-medical"
    assert stacks["selfhosted"]["llm"]["id"] == "ollama:qwen3.5:9b"
    assert stacks["selfhosted"]["llm"]["model"] == "qwen3.5:9b"
    assert stacks["api"]["orchestration"]["id"] == "livekit-server"
    assert stacks["api"]["orchestration"]["leaves_eu"] is False


class IdentifiedTranscriber:
    gpu_fraction = None

    def __init__(self, engine: str, model: str) -> None:
        self.engine, self.model = engine, model

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        return ""


async def test_transparency_names_the_stt_engine_the_whisper_service_runs() -> None:
    transcriber = IdentifiedTranscriber("mlx", "mlx-community/whisper-large-v3-turbo")
    async with run_asgi(whisper_app(lambda: transcriber, workers=1)) as host:
        stacks = await transparency_of(
            SETTINGS.model_copy(update={"whisper_ws_url": f"ws://{host}/v1/stream"})
        )

    stt = stacks["selfhosted"]["stt"]
    assert (stt["id"], stt["vendor"]) == ("mlx", "MLX Whisper (OpenAI Whisper weights)")
    assert stt["model"] == "mlx-community/whisper-large-v3-turbo"
    assert (stt["confirmed"], stt["unconfirmed_reason"]) == (True, "")


async def test_a_service_running_an_engine_without_a_catalogue_entry_is_unconfirmed() -> None:
    transcriber = IdentifiedTranscriber("parakeet", "nvidia/parakeet-tdt-0.6b-v3")
    async with run_asgi(whisper_app(lambda: transcriber, workers=1)) as host:
        stacks = await transparency_of(
            SETTINGS.model_copy(update={"whisper_ws_url": f"ws://{host}/v1/stream"})
        )

    stt = stacks["selfhosted"]["stt"]
    assert (stt["id"], stt["model"]) == ("faster-whisper", "large-v3-turbo")
    assert stt["confirmed"] is False
    assert stt["unconfirmed_reason"] == (
        "the stt service runs 'parakeet', which has no catalogue entry; "
        "listed from the catalogue's default"
    )


class KokoroStub:
    engine, model = KokoroSynthesizer.engine, KokoroSynthesizer.model
    gpu_fraction = KokoroSynthesizer.gpu_fraction
    warmup_voice = KokoroSynthesizer.warmup_voice

    def speaks(self, voice: str) -> bool:
        return True

    def synthesize(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        yield np.zeros(240, np.float32)


async def test_transparency_confirms_the_tts_engine_the_kokoro_service_runs() -> None:
    async with run_asgi(kokoro_app((KokoroStub,), workers=1)) as host:
        stacks = await transparency_of(SETTINGS.model_copy(update={"kokoro_url": f"http://{host}"}))

    tts = stacks["selfhosted"]["tts"]
    assert (tts["id"], tts["vendor"]) == ("kokoro", "hexgrad Kokoro")
    assert tts["model"] == "hexgrad/Kokoro-82M"
    assert (tts["confirmed"], tts["unconfirmed_reason"]) == (True, "")


async def serving(engines: list[dict[str, Any]]) -> FastAPI:
    info = FastAPI()

    @info.get("/v1/info")
    async def report() -> dict[str, list[dict[str, Any]]]:
        return {"engines": engines}

    return info


async def transparency_components(engines: list[dict[str, str]]) -> list[dict[str, object]]:
    async with run_asgi(await serving(engines)) as host:
        body = await transparency_body(SETTINGS.model_copy(update={"kokoro_url": f"http://{host}"}))
    return [c for c in body["pipelines"]["selfhosted"] if c["kind"] == "tts"]


async def test_every_engine_a_service_runs_is_listed_and_confirmed() -> None:
    """A German deployment runs Kokoro and Qwen3-TTS together; listing one per kind left
    the public page saying the whole row was unconfirmed."""
    tts = await transparency_components(
        [{"id": "kokoro", "model": "hexgrad/Kokoro-82M"}, {"id": "qwen3-tts", "model": "q-8bit"}]
    )

    assert [(c["id"], c["model"], c["confirmed"]) for c in tts] == [
        ("kokoro", "hexgrad/Kokoro-82M", True),
        ("qwen3-tts", "q-8bit", True),
    ]


@pytest.mark.parametrize(
    ("engines", "reason"),
    [
        ([], "the tts service reports no engine; listed from the catalogue's default"),
        (
            [{"id": "kokoro", "model": "k"}, {"id": "orpheus", "model": "o"}],
            "the tts service runs 'orpheus', which has no catalogue entry; "
            "listed from the catalogue's default",
        ),
    ],
)
async def test_a_service_running_something_uncatalogued_leaves_its_default_unconfirmed(
    engines: list[dict[str, str]], reason: str
) -> None:
    [tts] = await transparency_components(engines)

    assert (tts["id"], tts["model"], tts["confirmed"]) == ("kokoro", "Kokoro-82M", False)
    assert tts["unconfirmed_reason"] == reason


async def test_a_hung_service_leaves_its_default_unconfirmed_within_seconds() -> None:
    info, release = FastAPI(), asyncio.Event()

    @info.get("/v1/info")
    async def hang() -> None:
        await release.wait()

    async with run_asgi(info) as host:
        try:
            stacks = await asyncio.wait_for(
                transparency_of(SETTINGS.model_copy(update={"kokoro_url": f"http://{host}"})),
                2.5,
            )
        finally:
            release.set()

    tts = stacks["selfhosted"]["tts"]
    assert (tts["id"], tts["confirmed"]) == ("kokoro", False)
    assert "(ReadTimeout)" in tts["unconfirmed_reason"]


def test_startup_checks_the_catalogue_for_every_engine_the_services_can_run() -> None:
    """A backend is a runtime, an engine is what a call is served by and what the catalogue
    names, so the two CUDA backends add runtimes to engines already listed, not engines."""
    assert list(dict.fromkeys(t.engine for t in TRANSCRIBERS)) == list(
        SELFHOSTED_ENGINES[ComponentKind.STT]
    )
    assert list(dict.fromkeys(s.engine for s in SYNTHESIZERS)) == list(
        SELFHOSTED_ENGINES[ComponentKind.TTS]
    )
    assert SELFHOSTED_ENGINES[ComponentKind.STT][0] == FasterWhisperTranscriber.engine


async def test_simulated_calls_leave_the_services_unasked() -> None:
    transcriber = IdentifiedTranscriber("mlx", "mlx-community/whisper-large-v3-turbo")
    async with run_asgi(whisper_app(lambda: transcriber, workers=1)) as host:
        stacks = await transparency_of(
            SETTINGS.model_copy(
                update={"whisper_ws_url": f"ws://{host}/v1/stream", "simulate_providers": True}
            )
        )

    stt = stacks["selfhosted"]["stt"]
    assert (stt["id"], stt["confirmed"]) == ("faster-whisper", False)
    assert stt["unconfirmed_reason"] == "calls are simulated; the stt service was not asked"


async def test_a_service_report_is_reused_for_a_minute() -> None:
    transcriber = IdentifiedTranscriber("mlx", "mlx-community/whisper-large-v3-turbo")
    async with run_asgi(whisper_app(lambda: transcriber, workers=1)) as host:
        settings = SETTINGS.model_copy(update={"whisper_ws_url": f"ws://{host}/v1/stream"})
        app = create_app(settings, pipelines=StaticPipelines({}))
        async with client_for(app) as client:
            first = (await client.get("/transparency")).json()
    # The same app, so the same container: a second client must not re-probe the service.
    async with client_for(app) as client:
        after_the_service_stopped = (await client.get("/transparency")).json()

    for body in (first, after_the_service_stopped):
        stt = next(c for c in body["pipelines"]["selfhosted"] if c["kind"] == "stt")
        assert (stt["id"], stt["confirmed"]) == ("mlx", True)


async def test_requests_arriving_together_share_one_probe_of_the_service() -> None:
    """The cache is what keeps a public endpoint from becoming load on the GPU services,
    and a cache that writes only after the answer arrives protects nothing: every request
    in the first window misses and probes. Ten callers, one probe."""
    probes = 0
    released = asyncio.Event()
    info = FastAPI()

    @info.get("/v1/info")
    async def report() -> dict[str, list[dict[str, str]]]:
        nonlocal probes
        probes += 1
        await released.wait()
        return {"engines": [{"id": "kokoro", "model": "hexgrad/Kokoro-82M"}]}

    async with run_asgi(info) as host:
        settings = SETTINGS.model_copy(update={"kokoro_url": f"http://{host}"})
        async with api_client(settings) as client:
            waiting = [asyncio.create_task(client.get("/transparency")) for _ in range(10)]
            while probes == 0:
                await asyncio.sleep(0.01)
            released.set()
            answers = await asyncio.gather(*waiting)

    assert probes == 1
    for answer in answers:
        tts = [c for c in answer.json()["pipelines"]["selfhosted"] if c["kind"] == "tts"]
        assert [(c["id"], c["confirmed"]) for c in tts] == [("kokoro", True)]


async def test_an_unreachable_service_leaves_its_default_listed_but_unconfirmed() -> None:
    async with api_client(SETTINGS) as client:
        response = await client.get("/transparency")

    assert response.status_code == 200
    pipelines = response.json()["pipelines"]
    selfhosted = {c["kind"]: c for c in pipelines["selfhosted"]}
    assert (selfhosted["stt"]["id"], selfhosted["tts"]["id"]) == ("faster-whisper", "kokoro")
    for kind in ("stt", "tts"):
        assert selfhosted[kind]["confirmed"] is False
        assert selfhosted[kind]["unconfirmed_reason"] == (
            f"the {kind} service did not report what it runs (ConnectError); "
            "listed from the catalogue's default"
        )
    assert selfhosted["llm"]["confirmed"] and selfhosted["llm"]["unconfirmed_reason"] == ""
    assert all(c["confirmed"] for c in pipelines["api"])


@pytest.mark.parametrize(
    ("update", "edit", "named"),
    [
        ({"llm_server": "ollama", "vllm_model": "llama3.2"}, None, "'ollama:llama3.2'"),
        ({}, ("  mlx:\n    kind: stt", "  mlx-old:\n    kind: stt"), "'mlx'"),
        (
            {"database_url": "postgresql://x/y"},
            ("  postgres:\n", "  postgres-old:\n"),
            "'postgres'",
        ),
        (
            {},
            (
                "kokoro>=0.9.4\n    host: gpu_host\n    licence: Apache-2.0",
                "kokoro>=0.9.4\n    host: gpu_host",
            ),
            "'kokoro': needs licence",
        ),
        ({}, ("  qwen3-tts:\n    kind: tts", "  qwen3-tts-old:\n    kind: tts"), "'qwen3-tts'"),
        ({}, ("vendor: Deepgram\n", "vendor: Deepgram\n    model: nova-2\n"), "model comes"),
        ({}, ("host: gpu_host\n    licence: Apache", "host: mars\n    licence: Apache"), "'mars'"),
        (
            {},
            (
                "leaves_eu: true\n    assumption: Deepgram",
                "assumption: Deepgram",
            ),
            "needs leaves_eu",
        ),
        ({}, ("  kokoro:\n    kind: tts", "  kokoro:\n    kind: stt"), "kind 'stt'"),
    ],
)
def test_a_configured_component_without_a_complete_catalogue_entry_fails_startup(
    tmp_path: Path, update: dict[str, str], edit: tuple[str, str] | None, named: str
) -> None:
    config = config_copy(tmp_path)
    if edit is not None:
        catalogue = config / "components.yaml"
        text = catalogue.read_text()
        assert edit[0] in text
        catalogue.write_text(text.replace(edit[0], edit[1], 1))
    settings = SETTINGS.model_copy(update={"config_dir": config, **update})

    with pytest.raises(ComponentCatalogueError, match=named):
        load_static_config(settings)
    with pytest.raises(ComponentCatalogueError, match=named):
        create_app(settings, pipelines=StaticPipelines({}))


def select_carrier(config: Path, carrier: str) -> None:
    rates = config / "rates.yaml"
    rates.write_text(rates.read_text().replace("selected: twilio", f"selected: {carrier}", 1))


async def test_config_lists_every_carrier_and_the_selected_one_prices_telephony(
    tmp_path: Path,
) -> None:
    config = config_copy(tmp_path)
    select_carrier(config, "telnyx")
    async with api_client(SETTINGS.model_copy(update={"config_dir": config})) as client:
        body = (await client.get("/config")).json()
        transparency = (await client.get("/transparency")).json()

    carriers = {q["carrier"]: q for q in body["telephony_quotes"]}
    assert set(carriers) == {"twilio", "telnyx", "sipgate"}
    assert [c for c, q in carriers.items() if q["selected"]] == ["telnyx"]
    assert carriers["telnyx"]["per_minute_usd"] == 0.0032
    assert carriers["telnyx"]["verified"] is False
    assert carriers["telnyx"]["source_url"].startswith("https://telnyx.com/")
    assert (carriers["twilio"]["verified"], carriers["twilio"]["checked_on"]) == (
        True,
        "2026-09-17",
    )
    assert carriers["sipgate"]["per_minute_usd"] is None and carriers["sipgate"]["note"]
    assert body["rates"]["api"]["telephony_per_minute"] == 0.0032
    assert body["rates"]["selfhosted"]["telephony_per_minute"] == 0.0032
    assert body["rates"]["selfhosted"]["gpu_per_hour"] == 1.09
    for components in transparency["pipelines"].values():
        telephony = next(c for c in components if c["kind"] == "telephony")
        assert telephony["id"] == "telnyx"


def test_a_selected_carrier_without_a_catalogue_entry_fails_startup(tmp_path: Path) -> None:
    config = config_copy(tmp_path)
    select_carrier(config, "telnyx")
    catalogue = config / "components.yaml"
    catalogue.write_text(catalogue.read_text().replace("  telnyx:\n", "  telnyx-old:\n", 1))
    settings = SETTINGS.model_copy(update={"config_dir": config})

    with pytest.raises(ComponentCatalogueError, match="'telnyx'"):
        create_app(settings, pipelines=StaticPipelines({}))


GPU_HOST_IN_THE_EU = """  gpu_host:
    region: "gpu-host (see deployment)"
    leaves_eu: false"""
VOXTRAL_ON_ITS_VENDORS_CLOUD = """    host: gpu_host
    licence: Apache-2.0
    gpu_memory:
      gib: 9.98"""


def eu_config(
    tmp_path: Path, carrier: str = "sipgate", edit: tuple[str, str] | None = None
) -> Path:
    """The shipped configuration with the German carrier selected: everything the selfhosted
    pipeline touches is then EU-resident, so each edit isolates one component that isn't.
    sipgate sells inbound by the month, so an operator enters their contracted per-minute
    rate before they can select it; this stands in for that."""
    config = config_copy(tmp_path)
    rates = config / "rates.yaml"
    rates.write_text(
        rates.read_text().replace(
            "    sipgate:\n      unit: minute\n",
            "    sipgate:\n      unit: minute\n      price_usd: 0.006\n",
            1,
        )
    )
    select_carrier(config, carrier)
    if edit is not None:
        catalogue = config / "components.yaml"
        text = catalogue.read_text()
        assert edit[0] in text
        catalogue.write_text(text.replace(edit[0], edit[1], 1))
    return config


def eu_settings(config: Path, **update: object) -> Settings:
    return SETTINGS.model_copy(
        update={
            "config_dir": config,
            "eu_only": True,
            "pipeline": PipelineKind.SELFHOSTED,
            "livekit_url": "ws://localhost:7880",
            **update,
        }
    )


@pytest.mark.parametrize(
    ("update", "carrier", "edit", "named"),
    [
        (
            {"pipeline": PipelineKind.API},
            "twilio",
            None,
            [
                "api telephony 'twilio' (Twilio, us)",
                "api stt 'deepgram' (Deepgram, us)",
                "api llm 'openai' (OpenAI, us)",
                "api tts 'elevenlabs' (ElevenLabs, us)",
            ],
        ),
        ({}, "twilio", None, ["selfhosted telephony 'twilio' (Twilio, us)"]),
        (
            {},
            "telnyx",
            None,
            ["selfhosted telephony 'telnyx' (Telnyx, global (nearest point of presence))"],
        ),
        (
            {"livekit_url": "wss://example.livekit.cloud"},
            "sipgate",
            None,
            ["selfhosted orchestration 'livekit-cloud' (LiveKit Cloud, global (nearest edge))"],
        ),
        (
            {},
            "sipgate",
            (GPU_HOST_IN_THE_EU, GPU_HOST_IN_THE_EU.replace("false", "true")),
            [
                "selfhosted stt 'faster-whisper'",
                "selfhosted llm 'vllm:Qwen/Qwen3.5-9B'",
                "selfhosted tts 'kokoro'",
                "selfhosted stt engine 'mlx'",
                "selfhosted stt engine 'voxtral'",
                "selfhosted tts engine 'qwen3-tts'",
            ],
        ),
        (
            {},
            "sipgate",
            (
                VOXTRAL_ON_ITS_VENDORS_CLOUD,
                VOXTRAL_ON_ITS_VENDORS_CLOUD.replace(
                    "host: gpu_host", "region: us\n    leaves_eu: true"
                ),
            ),
            ["selfhosted stt engine 'voxtral' (Mistral AI Voxtral, us)"],
        ),
    ],
)
def test_the_eu_only_profile_refuses_to_start_naming_every_component_that_leaves_the_eu(
    tmp_path: Path,
    update: dict[str, object],
    carrier: str,
    edit: tuple[str, str] | None,
    named: list[str],
) -> None:
    settings = eu_settings(eu_config(tmp_path, carrier, edit), **update)

    for start in (load_static_config, lambda s: create_app(s, pipelines=StaticPipelines({}))):
        with pytest.raises(NotEuResident) as raised:
            start(settings)
        message = str(raised.value)
        for component in named:
            assert component in message
        assert message.endswith("Select EU-resident components or unset EU_ONLY.")


def test_a_deployment_without_the_profile_may_leave_the_eu(tmp_path: Path) -> None:
    """The profile is opt-in: the shipped configuration keeps working untouched."""
    settings = SETTINGS.model_copy(update={"config_dir": config_copy(tmp_path)})

    load_static_config(settings)
    create_app(settings, pipelines=StaticPipelines({}))


async def transparency_body(settings: Settings) -> dict[str, Any]:
    async with api_client(settings) as client:
        body: dict[str, Any] = (await client.get("/transparency")).json()
    return body


async def test_an_eu_only_deployment_starts_and_transparency_shows_nothing_leaving_the_eu(
    tmp_path: Path,
) -> None:
    settings = eu_settings(eu_config(tmp_path))

    load_static_config(settings)
    body = await transparency_body(settings)

    assert body["eu_only"] is True
    assert set(body["pipelines"]) == {"selfhosted"}
    components = body["pipelines"]["selfhosted"]
    assert [c["kind"] for c in components] == KINDS
    assert [c["leaves_eu"] for c in components] == [False] * len(KINDS)
    assert {c["id"] for c in components} == {
        "sipgate",
        "faster-whisper",
        "vllm:Qwen/Qwen3.5-9B",
        "kokoro",
        "livekit-server",
        "in-memory",
    }


async def test_a_deployment_without_the_profile_still_publishes_both_pipelines(
    tmp_path: Path,
) -> None:
    body = await transparency_body(SETTINGS.model_copy(update={"config_dir": eu_config(tmp_path)}))

    assert body["eu_only"] is False
    assert set(body["pipelines"]) == {"api", "selfhosted"}


async def test_a_caller_cannot_ask_for_a_pipeline_the_eu_only_profile_refused(
    tmp_path: Path,
) -> None:
    """The browser picks a pipeline per call, so the profile has to close that choice:
    otherwise a toggle would route the caller straight to the vendors it refused."""
    async with api_client(eu_settings(eu_config(tmp_path))) as client:
        refused = await client.post("/token", json={"pipeline": "api"})
        allowed = await client.post("/token", json={"pipeline": "selfhosted"})
        default = await client.post("/token", json={})

    assert refused.status_code == 409
    assert refused.json()["detail"] == (
        "EU_ONLY serves the selfhosted pipeline only; the api pipeline is not available"
    )
    assert allowed.json()["pipeline"] == "selfhosted"
    assert default.json()["pipeline"] == "selfhosted"


def over_committed(config: Path) -> Path:
    """The LLM's share raised until the engines beside it no longer fit the card."""
    serving = config / "serving" / "qwen-9b-l40s.yaml"
    serving.write_text(
        serving.read_text().replace("gpu-memory-utilization: 0.72", "gpu-memory-utilization: 0.95")
    )
    return config


def test_the_eu_only_profile_has_no_pipeline_to_fall_back_on_when_the_card_overruns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Without the profile an overrun degrades to the API pipeline; under it there is
    nothing else to take the call, so the same estimate is worth more than a warning."""
    settings = eu_settings(over_committed(eu_config(tmp_path)))

    with caplog.at_level(logging.WARNING):
        load_static_config(settings)

    [record] = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert record.levelno == logging.ERROR
    assert record.getMessage().startswith("EU_ONLY has no pipeline to fall back on.")
    assert record.getMessage().endswith("1.60 GiB short (does not fit)")


def test_a_deployment_without_the_profile_only_warns_about_the_same_card(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = SETTINGS.model_copy(update={"config_dir": over_committed(eu_config(tmp_path))})

    with caplog.at_level(logging.WARNING):
        load_static_config(settings)

    [record] = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert record.levelno == logging.WARNING
    assert record.getMessage().startswith("GPU memory budget: L40S 48.00 GiB:")


def test_the_eu_only_error_names_every_offender_in_one_sentence(tmp_path: Path) -> None:
    """A data-protection officer has to act on this line: every subprocessor that would
    see the call, with the vendor and region that disqualify it (§43e BRAO)."""
    settings = eu_settings(eu_config(tmp_path, "twilio"), pipeline=PipelineKind.API)

    with pytest.raises(NotEuResident) as raised:
        load_static_config(settings)

    assert str(raised.value) == (
        "EU_ONLY is set, but these components leave the EU: "
        "api telephony 'twilio' (Twilio, us), api stt 'deepgram' (Deepgram, us), "
        "api llm 'openai' (OpenAI, us), api tts 'elevenlabs' (ElevenLabs, us). "
        "Select EU-resident components or unset EU_ONLY."
    )


def test_an_unproven_card_stays_a_warning_even_under_the_eu_only_profile(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unknown verdict is not an overrun: an Ollama deployment reserves no fixed share,
    and logging that as an error teaches operators to ignore the one that means it."""
    settings = eu_settings(eu_config(tmp_path)).model_copy(
        update={"llm_server": "ollama", "vllm_model": "qwen3.5:9b"}
    )

    with caplog.at_level(logging.WARNING):
        load_static_config(settings)

    [record] = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert record.levelno == logging.WARNING
    assert "unknown" in record.getMessage()


async def test_a_deployment_without_the_german_voice_refuses_the_call_not_the_greeting(
    tmp_path: Path,
) -> None:
    """The German persona asks for qwen3-tts:clara_de and TTS_ENGINES defaults to Kokoro
    alone. Left to synthesis this failed on the greeting turn, so the caller heard nothing;
    now the container refuses the call with a reason the browser can show."""
    async with run_asgi(await serving([{"id": "kokoro", "model": "hexgrad/Kokoro-82M"}])) as host:
        settings = SETTINGS.model_copy(
            update={"config_dir": config_copy(tmp_path), "kokoro_url": f"http://{host}"}
        )
        container = build_container(settings, NullMetrics(), repository=InMemoryCallRepository())  # type: ignore[arg-type]

        with pytest.raises(VoiceNotAvailable, match="qwen3-tts"):
            await container.start_call.execute(PipelineKind.SELFHOSTED, "law_firm_de")

        # English is spoken by the engine that is running, so it is unaffected.
        english = await container.start_call.execute(PipelineKind.SELFHOSTED, "law_firm")

    assert english.persona.id == "law_firm"
    assert await container.repository.list() == [english.call]


async def test_the_budget_reads_the_share_a_cuda_runtime_reserves_not_its_weights() -> None:
    """End to end for #34: the services report what their vLLM processes reserve, and the
    check that guards the single-GPU claim reads that. Budgeting the catalogue's footprints
    instead answered 'fits' for exactly the deployment the check exists to guard."""
    reported = [
        {"id": "voxtral", "model": "mistralai/Voxtral-Mini-4B-Realtime-2602 (vLLM realtime)"},
        {"id": "qwen3-tts", "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign (vLLM-Omni)"},
    ]
    async with (
        run_asgi(await serving([{**reported[0], "gpu_fraction": 0.34}])) as stt,
        run_asgi(await serving([{**reported[1], "gpu_fraction": 0.6}])) as tts,
    ):
        settings = SETTINGS.model_copy(
            update={"whisper_ws_url": f"ws://{stt}/v1/stream", "kokoro_url": f"http://{tts}"}
        )
        container = build_container(settings, NullMetrics())  # type: ignore[arg-type]
        components = await container.describe_components.execute(PipelineKind.SELFHOSTED)

    budget = container.gpu_budget.execute(
        [d.component for d in components],
        reserved={d.component.id: d.gpu_fraction for d in components if d.gpu_fraction},
    )

    assert budget is not None
    assert budget.verdict is BudgetVerdict.DOES_NOT_FIT
    # 0.72 + 0.34 + 0.60 of a 48 GiB card: 34.56 + 16.32 + 28.80.
    assert budget.claimed_gib == Decimal("79.68")
    assert "voxtral 16.32" in budget.summary() and "qwen3-tts 28.80" in budget.summary()
