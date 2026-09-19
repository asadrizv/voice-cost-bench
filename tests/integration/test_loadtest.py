"""The harness end to end on the simulated GPU: real CallSession, real VAD and endpointer
on real fixture audio, real-time pacing. Short conversation to keep it under ~15 s."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from backend.application.ports.component_catalogue import Component, ComponentKind
from backend.application.ports.rate_card_provider import TelephonyQuote
from backend.application.services.concurrency_supervisor import ConcurrencySupervisor
from backend.application.services.endpointing import EndpointerKind
from backend.application.use_cases.describe_components import DescribedComponent
from backend.domain.value_objects.audio import PCM16_24K_MONO, AudioChunk
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.settings import MissingServingConfig, Settings
from backend.infrastructure.persistence.inmemory_call_repository import InMemoryCallRepository
from backend.infrastructure.simulated.gpu_model import SimulatedGpu, SimulatedGpuProfile
from backend.infrastructure.transport.paced_output import PacedAudioOutput
from backend.interfaces.cli import loadtest
from backend.interfaces.cli.harness import provenance, report
from backend.interfaces.cli.harness.caller import Conversation, load_conversation
from backend.interfaces.cli.harness.runner import LevelRun, LevelRunner
from backend.interfaces.container import build_container
from tests.fakes import FakeClock, NullMetrics

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
needs_audio = pytest.mark.skipif(
    not (FIXTURES / "audio" / "intake_en").is_dir(), reason="no fixture audio: run `make fixtures`"
)


@needs_audio
async def test_level_run_produces_costed_timed_calls() -> None:
    conversation = load_conversation(FIXTURES, "intake_en")
    conversation = replace(
        conversation, turns=conversation.turns[1:2], audio=conversation.audio[1:2]
    )
    supervisor = ConcurrencySupervisor(ceiling=3)
    container = build_container(
        Settings(database_url=""),
        NullMetrics(),  # type: ignore[arg-type]
        repository=InMemoryCallRepository(),
        supervisors={PipelineKind.SELFHOSTED: supervisor},
    )
    gpu = SimulatedGpu(SimulatedGpuProfile(speech_s_per_char=0.01))
    runner = LevelRunner(
        container, PipelineKind.SELFHOSTED, conversation, EndpointerKind.SEMANTIC, gpu
    )

    run = await runner.run(concurrency=3, duration_s=1)

    assert run.failed == 0 and run.rejected == 0
    assert len(run.calls) == 3
    assert supervisor.peak == 3 and supervisor.active == 0
    for call in run.calls:
        assert [t.user_text for t in call.turns] == ["", "My name is Anna Weber."]
        billed = sum(t.usage.gpu_seconds for t in call.turns) + call.closing_usage.gpu_seconds
        assert abs(billed - call.duration_seconds()) < 0.05

    summary = report.summarise_level(run, conversation, simulated=True)
    assert summary["calls_completed"] == 3 and summary["turns"] == 3
    assert summary["effective_concurrency"] > 1.5  # three calls overlapped on one GPU
    assert summary["cost_per_minute_by_stage_usd"]["gpu"] > 0
    assert summary["perceived_delay_p95_ms"] > summary["end_to_end_p95_ms"] > 0
    assert summary["harness_valid"]

    # One utterance per call, so each call's single caller-observed delay pairs with its
    # one answered turn. Caller-observed can't exceed internal perceived delay here: the
    # simulated TTS opens with audible tone, so both end at the same agent audio, and the
    # fixture's speech end is never earlier than the endpointer's (see AUDIBLE_DBFS).
    assert run.unanswered_turns == 0
    for call, observed in zip(run.calls, run.caller_observed_ms, strict=True):
        assert 0 < observed <= call.latencies()[0].perceived_delay + 1
    assert summary["caller_observed_turns"] == 3
    assert summary["caller_observed_unanswered"] == 0
    assert 0 < summary["caller_observed_p95_ms"] <= summary["perceived_delay_p95_ms"] + 1

    curve = report.utilisation_curve(
        [summary], container.calculator.rate_card, [{"provider": "scaleway", "hourly_usd": 1.40}]
    )
    by_u = {row["utilisation"]: row for row in curve}
    assert by_u[0.25]["gpu_share_usd"] == 4 * by_u[1.0]["gpu_share_usd"]
    assert by_u[1.0]["cost_per_minute_usd_scaleway"] > by_u[1.0]["cost_per_minute_usd"]


@needs_audio
async def test_each_priced_carrier_gets_a_cost_row_differing_only_in_telephony() -> None:
    conversation = load_conversation(FIXTURES, "intake_en")
    conversation = replace(
        conversation, turns=conversation.turns[1:2], audio=conversation.audio[1:2]
    )
    container = build_container(
        Settings(database_url=""),
        NullMetrics(),  # type: ignore[arg-type]
        repository=InMemoryCallRepository(),
        supervisors={PipelineKind.SELFHOSTED: ConcurrencySupervisor(ceiling=2)},
    )
    gpu = SimulatedGpu(SimulatedGpuProfile(speech_s_per_char=0.01))
    runner = LevelRunner(
        container, PipelineKind.SELFHOSTED, conversation, EndpointerKind.SEMANTIC, gpu
    )
    double = TelephonyQuote("double", Decimal("0.028"), "https://x", "2026-09-19", False, False)
    carriers = [*container.rates.telephony_quotes(), double]

    run = await runner.run(concurrency=2, duration_s=1)
    summary = report.summarise_level(run, conversation, True, carriers)

    rows = summary["cost_per_minute_by_carrier_usd"]
    assert set(rows) == {"twilio", "telnyx", "double"}  # sipgate publishes no per-minute price
    telephony = summary["cost_per_minute_by_stage_usd"]["telephony"]
    assert telephony > 0
    assert rows["twilio"] == pytest.approx(summary["cost_per_minute_usd"])
    assert rows["double"] == pytest.approx(summary["cost_per_minute_usd"] + telephony)
    assert rows["telnyx"] == pytest.approx(
        summary["cost_per_minute_usd"] - telephony * (1 - 0.0032 / 0.014)
    )


def _tone_utterance(speech_s: float, pause_s: float) -> list[AudioChunk]:
    tone = (3000 * np.sin(2 * np.pi * 440 * np.arange(320) / 16_000)).astype("<i2").tobytes()
    frames = int(speech_s / 0.02) * [AudioChunk(tone)]
    return frames + int(pause_s / 0.02) * [AudioChunk(bytes(640))]


async def test_caller_observed_delay_matches_a_pipeline_of_known_latency() -> None:
    conversation = Conversation(
        "tones",
        "law_firm",
        "en",
        ["My name is Anna Weber.", "It's a tenancy issue."],
        [_tone_utterance(1.0, 0.2), _tone_utterance(0.6, 0.2)],
    )
    container = build_container(
        Settings(database_url=""),
        NullMetrics(),  # type: ignore[arg-type]
        repository=InMemoryCallRepository(),
    )
    after_commit_ms = 300 + 100  # LLM first token + TTS first byte; STT and decode free
    gpu = SimulatedGpu(
        SimulatedGpuProfile(
            stt_final_ms=0,
            llm_ttft_ms=300,
            llm_ttft_per_active_ms=0,
            llm_tokens_per_s=1e6,
            tts_first_byte_ms=100,
            speech_s_per_char=0.002,
            jitter=0,
        )
    )
    runner = LevelRunner(
        container, PipelineKind.SELFHOSTED, conversation, EndpointerKind.SILENCE, gpu
    )

    run = await runner.run(concurrency=1, duration_s=0.1)

    assert run.failed == 0 and run.unanswered_turns == 0
    [call] = run.calls
    answered = call.latencies()
    assert len(run.caller_observed_ms) == len(answered) == 2
    for observed, latency in zip(run.caller_observed_ms, answered, strict=True):
        # The 700 ms silence endpointer notices on a frame boundary, so its wait is read
        # from the turn; everything after the commit is fixed by the profile.
        assert 700 <= latency.endpoint_detected <= 700 + 20 + 5
        assert observed == pytest.approx(latency.endpoint_detected + after_commit_ms, abs=20)


def _agent_audio(*parts: tuple[float, float]) -> AudioChunk:
    """Concatenated (seconds, dBFS) sections of a 440 Hz tone at 24 kHz, as TTS emits."""
    pcm = []
    for seconds, dbfs in parts:
        t = np.arange(int(24_000 * seconds)) / 24_000
        amplitude = 32767 * np.sqrt(2) * 10 ** (dbfs / 20)  # sine RMS = peak / sqrt(2)
        pcm.append((amplitude * np.sin(2 * np.pi * 440 * t)).astype("<i2"))
    return AudioChunk(np.concatenate(pcm).tobytes(), PCM16_24K_MONO)


@pytest.mark.parametrize(
    "chunks",
    [
        [_agent_audio((0.2, -60)), _agent_audio((0.04, -20))],
        [_agent_audio((0.2, -60), (0.04, -20))],
        [_agent_audio((0.2, -46), (0.04, -44))],
    ],
    ids=["separate chunks", "one chunk", "either side of -45 dBFS"],
)
async def test_near_silent_tts_lead_in_is_not_the_agent_speaking(chunks: list[AudioChunk]) -> None:
    clock = FakeClock()
    output = PacedAudioOutput(clock)
    output.caller_speech_started()
    output.caller_speech_ended()
    clock.advance(0.5)

    for chunk in chunks:
        await output.write(chunk)

    assert output.caller_observed_ms == [pytest.approx(700)]
    assert output.unanswered_turns == 0


async def test_a_turn_the_agent_never_answers_is_counted_unanswered_not_timed() -> None:
    clock = FakeClock()
    output = PacedAudioOutput(clock)
    await output.write(_agent_audio((0.5, -20)))  # the greeting answers no caller turn
    clock.advance(1.0)
    output.caller_speech_started()
    output.caller_speech_ended()
    clock.advance(2.0)
    output.caller_speech_started()  # spoke again with no answer: the first turn is lost
    output.caller_speech_ended()
    clock.advance(2.0)
    output.caller_speech_started()  # and again; the agent talks over this utterance
    await output.write(_agent_audio((0.2, -20)))
    output.caller_speech_ended()
    clock.advance(0.3)
    await output.write(_agent_audio((0.5, -20)))
    await output.write(_agent_audio((0.5, -20)))
    clock.advance(1.0)
    output.caller_speech_started()
    output.caller_speech_ended()  # then hung up on

    assert output.caller_observed_ms == [pytest.approx(300)]
    assert output.unanswered_turns == 3


def test_a_level_where_every_call_failed_costs_nothing_under_any_carrier() -> None:
    conversation = Conversation("intake_en", "law_firm", "en", ["Hello."], [[]])
    twilio = TelephonyQuote("twilio", Decimal("0.014"), "https://x", "2026-09-17", True, True)

    summary = report.summarise_level(LevelRun(2, 1.0, failed=2), conversation, True, [twilio])

    assert summary["cost_per_minute_by_carrier_usd"] == {"twilio": 0.0}
    assert summary["caller_observed_p50_ms"] is None
    assert summary["caller_observed_p99_ms"] is None
    assert summary["caller_observed_turns"] == 0


def test_caller_observed_percentiles_summarise_every_timed_turn() -> None:
    conversation = Conversation("intake_en", "law_firm", "en", ["Hello."], [[]])
    delays = [v + 0.04 for v in range(100, 1100, 10)]
    run = LevelRun(1, 1.0, caller_observed_ms=delays, unanswered_turns=4)

    summary = report.summarise_level(run, conversation, simulated=True)

    assert summary["caller_observed_p50_ms"] == 595.0
    assert summary["caller_observed_p95_ms"] == 1040.5
    assert summary["caller_observed_p99_ms"] == 1080.1  # 1080.14, to one decimal
    assert summary["caller_observed_turns"] == 100
    assert summary["caller_observed_unanswered"] == 4


def test_breaking_point_is_first_level_over_budget() -> None:
    levels = [
        {
            "concurrency": c,
            "turns": 10,
            "end_to_end_p95_ms": e2e,
            "perceived_delay_p95_ms": 0,
            "within_budget": e2e < 900,
        }
        for c, e2e in ((1, 400), (10, 600), (20, 880), (40, 950), (60, 1400))
    ]
    point = report.breaking_point(levels)
    assert point == {
        "concurrency": 40,
        "end_to_end_p95_ms": 950,
        "perceived_delay_p95_ms": 0,
        "last_within_budget": 20,
    }
    assert report.breaking_point(levels[:3]) is None


def test_wer_normalises_case_and_punctuation() -> None:
    assert report.word_error_rate("Tuesday at ten, please.", "tuesday at ten please") == 0
    assert report.word_error_rate("a b c d", "a x c d") == 0.25
    assert report.word_error_rate("", "anything") == 0


def test_harness_rejects_an_unknown_endpointer(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        loadtest.main(["sweep", "--pipeline", "simulated", "--endpointer", "vibes"])
    assert exit_info.value.code == 2
    assert "--endpointer" in capsys.readouterr().err


class OneStackCatalogue:
    def __init__(self, kind: PipelineKind, components: list[Component]) -> None:
        self._kind, self._components = kind, components

    def version(self) -> int:
        return 7

    def components(self, kind: PipelineKind) -> list[Component]:
        return self._components if kind is self._kind else []


@pytest.mark.parametrize(
    ("pipeline", "kind", "model_facts"),
    [
        ("api", PipelineKind.API, {"listed_in_rate_card"}),
        (
            "selfhosted",
            PipelineKind.SELFHOSTED,
            {"vllm_version", "serving_config", "serving_config_sha256"},
        ),
        (
            "simulated",
            PipelineKind.SELFHOSTED,
            {"vllm_version", "serving_config", "serving_config_sha256"},
        ),
    ],
)
def test_provenance_records_what_ran_and_whether_it_was_confirmed(
    pipeline: str, kind: PipelineKind, model_facts: set[str]
) -> None:
    stt = Component(ComponentKind.STT, "acme", "Acme", "ears-2", "r9", "eu-west", "MIT", False, "")
    tts = Component(ComponentKind.TTS, "voxy", "Voxy", "v1", "1", "eu-west", "MIT", False, "")
    conversation = Conversation("intake_en", "law_firm", "en", ["Hello."], [[]])
    settings = Settings(database_url="", vllm_base_url="http://127.0.0.1:9/v1")

    info = provenance.collect(
        settings,
        pipeline,
        conversation,
        EndpointerKind.SEMANTIC,
        simulated=True,
        rates_raw={"api": {"stt": {"model": "ears-2"}}},
        components=[
            DescribedComponent(stt, ""),
            DescribedComponent(tts, "the tts service did not report what it runs"),
        ],
        catalogue_version=7,
    )

    assert info["component_catalogue_version"] == 7
    assert info["components"] == [
        {
            "kind": "stt",
            "id": "acme",
            "vendor": "Acme",
            "model": "ears-2",
            "version": "r9",
            "region": "eu-west",
            "licence": "MIT",
            "leaves_eu": False,
            "assumption": "",
            "confirmed": True,
            "unconfirmed_reason": "",
        },
        {
            "kind": "tts",
            "id": "voxy",
            "vendor": "Voxy",
            "model": "v1",
            "version": "1",
            "region": "eu-west",
            "licence": "MIT",
            "leaves_eu": False,
            "assumption": "",
            "confirmed": False,
            "unconfirmed_reason": "the tts service did not report what it runs",
        },
    ]
    assert set(info["models"]) == model_facts
    assert "whisper" not in str(info["models"]) and "deepgram" not in str(info["models"])


def test_provenance_hashes_the_serving_config_vllm_starts_from() -> None:
    conversation = Conversation("intake_en", "law_firm", "en", ["Hello."], [[]])
    settings = Settings(database_url="", serving_config="llama-8b-l40s.yaml")

    info = provenance.collect(
        settings,
        "selfhosted",
        conversation,
        EndpointerKind.SEMANTIC,
        simulated=True,
        rates_raw={},
        components=[],
        catalogue_version=7,
    )

    served = (settings.config_dir / "serving" / "llama-8b-l40s.yaml").read_bytes()
    assert info["models"]["serving_config"] == "llama-8b-l40s.yaml"
    assert info["models"]["serving_config_sha256"] == hashlib.sha256(served).hexdigest()


def test_provenance_refuses_to_record_a_serving_config_that_does_not_exist() -> None:
    """A benchmark whose provenance says "missing" can't be reproduced or challenged."""
    conversation = Conversation("intake_en", "law_firm", "en", ["Hello."], [[]])
    settings = Settings(database_url="", serving_config="serving/qwen-9b-l40s.yaml")

    with pytest.raises(MissingServingConfig, match="serving/qwen-9b-l40s.yaml"):
        provenance.collect(
            settings,
            "selfhosted",
            conversation,
            EndpointerKind.SEMANTIC,
            simulated=True,
            rates_raw={},
            components=[],
            catalogue_version=7,
        )


async def test_the_caller_waits_for_an_answer_when_a_pause_splits_its_turn() -> None:
    """A mid-utterance pause lets semantic endpointing commit early; the caller talks on,
    barges in, and the cut-off turn is recorded. A caller that counted recorded turns as
    answers then spoke its next line over the agent, leaving turns unanswered."""
    paused = _tone_utterance(1.0, 0.35) + _tone_utterance(1.0, 0.2)
    conversation = Conversation(
        "paused",
        "law_firm",
        "en",
        ["It's a tenancy issue.", "He's refusing to return my deposit.", "No deadline."],
        [paused, _tone_utterance(0.8, 0.2)],
    )
    container = build_container(
        Settings(database_url=""),
        NullMetrics(),  # type: ignore[arg-type]
        repository=InMemoryCallRepository(),
    )
    gpu = SimulatedGpu(
        SimulatedGpuProfile(
            stt_final_ms=0,
            llm_ttft_ms=400,
            llm_ttft_per_active_ms=0,
            llm_tokens_per_s=1e6,
            tts_first_byte_ms=100,
            speech_s_per_char=0.002,
            interim_after_speech_s=0.5,
            jitter=0,
        )
    )
    runner = LevelRunner(
        container, PipelineKind.SELFHOSTED, conversation, EndpointerKind.SEMANTIC, gpu
    )

    run = await runner.run(concurrency=1, duration_s=0.1)

    [call] = run.calls
    assert any(t.interrupted for t in call.turns), "the pause should split the first turn"
    assert run.unanswered_turns == 0
    assert all(t.latency is None for t in call.turns if t.interrupted and not t.agent_text)


def test_provenance_of_an_ollama_run_records_no_vllm_serving_config() -> None:
    """The laptop demo serves the LLM from Ollama; a vLLM model, revision and config hash in
    its provenance would describe a server that took no part in the numbers."""
    conversation = Conversation("intake_en", "law_firm", "en", ["Hello."], [[]])
    settings = Settings(
        database_url="",
        llm_server="ollama",
        vllm_model="qwen3.5:9b",
        vllm_base_url="http://127.0.0.1:9/v1",
    )

    info = provenance.collect(
        settings,
        "selfhosted",
        conversation,
        EndpointerKind.SEMANTIC,
        simulated=False,
        rates_raw={},
        components=[],
        catalogue_version=1,
    )

    assert info["models"] == {"llm_server": "ollama", "ollama_model": "qwen3.5:9b"}
