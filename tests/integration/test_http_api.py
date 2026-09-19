from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import jwt
import numpy as np
import pytest
from fastapi import FastAPI

from backend.application.ports.component_catalogue import ComponentKind
from backend.application.ports.pipeline_provider import Pipeline
from backend.application.use_cases.handle_call_turn import TurnRequest
from backend.domain.entities.latency import TurnTimeline
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.component_catalogue import ComponentCatalogueError
from backend.infrastructure.config.settings import Settings
from backend.infrastructure.pipeline_factory import SELFHOSTED_ENGINES
from backend.interfaces.container import validate_static_config
from backend.interfaces.http.app import create_app
from gpu.kokoro_service.app import SYNTHESIZERS, KokoroSynthesizer
from gpu.kokoro_service.app import create_app as kokoro_app
from gpu.whisper_service.app import TRANSCRIBERS, FasterWhisperTranscriber
from gpu.whisper_service.app import create_app as whisper_app
from tests.fakes import FakeClock, FakeLlm, FakeTts, RecordingOutput, ScriptedStt, StaticPipelines
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
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, app.state.container


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


def config_copy(tmp_path: Path) -> Path:
    config = tmp_path / "config"
    shutil.copytree(SETTINGS.config_dir, config)
    return config


async def transparency_of(settings: Settings) -> dict[str, dict[str, dict[str, object]]]:
    app = create_app(settings, pipelines=StaticPipelines({}))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = (await client.get("/transparency")).json()
    return {p: {c["kind"]: c for c in cs} for p, cs in body["pipelines"].items()}


async def test_transparency_follows_the_catalogue_and_the_running_configuration(
    tmp_path: Path,
) -> None:
    config = config_copy(tmp_path)
    catalogue = config / "components.yaml"
    catalogue.write_text(
        catalogue.read_text().replace(
            "licence: Apache-2.0\n\n  livekit-server", "licence: X\n\n  livekit-server"
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
    transcriber = IdentifiedTranscriber("voxtral", "mistralai/Voxtral-Mini-3B")
    async with run_asgi(whisper_app(lambda: transcriber, workers=1)) as host:
        stacks = await transparency_of(
            SETTINGS.model_copy(update={"whisper_ws_url": f"ws://{host}/v1/stream"})
        )

    stt = stacks["selfhosted"]["stt"]
    assert (stt["id"], stt["model"]) == ("faster-whisper", "large-v3-turbo")
    assert stt["confirmed"] is False
    assert stt["unconfirmed_reason"] == (
        "the stt service runs 'voxtral', which has no catalogue entry; "
        "listed from the catalogue's default"
    )


class KokoroStub:
    engine, model = KokoroSynthesizer.engine, KokoroSynthesizer.model

    def synthesize(self, text: str, voice: str, speed: float) -> Iterator[np.ndarray]:
        yield np.zeros(240, np.float32)


async def test_transparency_confirms_the_tts_engine_the_kokoro_service_runs() -> None:
    async with run_asgi(kokoro_app(KokoroStub, workers=1)) as host:
        stacks = await transparency_of(SETTINGS.model_copy(update={"kokoro_url": f"http://{host}"}))

    tts = stacks["selfhosted"]["tts"]
    assert (tts["id"], tts["vendor"]) == ("kokoro", "hexgrad Kokoro")
    assert tts["model"] == "hexgrad/Kokoro-82M"
    assert (tts["confirmed"], tts["unconfirmed_reason"]) == (True, "")


@pytest.mark.parametrize(
    ("engines", "named"),
    [
        ([{"id": "kokoro", "model": "k"}, {"id": "orpheus", "model": "o"}], "'kokoro', 'orpheus'"),
        ([], "none"),
    ],
)
async def test_a_service_not_reporting_exactly_one_engine_leaves_its_default_unconfirmed(
    engines: list[dict[str, str]], named: str
) -> None:
    info = FastAPI()

    @info.get("/v1/info")
    async def report() -> dict[str, list[dict[str, str]]]:
        return {"engines": engines}

    async with run_asgi(info) as host:
        stacks = await transparency_of(SETTINGS.model_copy(update={"kokoro_url": f"http://{host}"}))

    tts = stacks["selfhosted"]["tts"]
    assert (tts["id"], tts["model"], tts["confirmed"]) == ("kokoro", "Kokoro-82M", False)
    assert tts["unconfirmed_reason"] == (
        f"the tts service reports engines {named}, and this lists one per kind; "
        "listed from the catalogue's default"
    )


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
    assert [t.engine for t in TRANSCRIBERS] == list(SELFHOSTED_ENGINES[ComponentKind.STT])
    assert [s.engine for s in SYNTHESIZERS] == list(SELFHOSTED_ENGINES[ComponentKind.TTS])
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
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = (await client.get("/transparency")).json()
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        after_the_service_stopped = (await client.get("/transparency")).json()

    for body in (first, after_the_service_stopped):
        stt = next(c for c in body["pipelines"]["selfhosted"] if c["kind"] == "stt")
        assert (stt["id"], stt["confirmed"]) == ("mlx", True)


async def test_an_unreachable_service_leaves_its_default_listed_but_unconfirmed() -> None:
    app = create_app(SETTINGS, pipelines=StaticPipelines({}))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
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
        ({}, ("licence: Apache-2.0\n\n  livekit", "\n\n  livekit"), "'kokoro': needs licence"),
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
        validate_static_config(settings)
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
    app = create_app(
        SETTINGS.model_copy(update={"config_dir": config}), pipelines=StaticPipelines({})
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
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
