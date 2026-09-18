from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import jwt
import pytest

from backend.application.ports.pipeline_provider import Pipeline
from backend.application.use_cases.handle_call_turn import TurnRequest
from backend.domain.entities.latency import TurnTimeline
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.settings import Settings
from backend.interfaces.http.app import create_app
from tests.fakes import FakeClock, FakeLlm, FakeTts, RecordingOutput, ScriptedStt, StaticPipelines

SETTINGS = Settings(
    database_url="",
    livekit_api_key="lk-key",
    livekit_api_secret="lk-secret-that-is-long-enough-for-hs256",
    livekit_url="wss://example.livekit.cloud",
    internal_token="internal-test",
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
