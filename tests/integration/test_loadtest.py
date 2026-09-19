"""The harness end to end on the simulated GPU: real CallSession, real VAD and endpointer
on real fixture audio, real-time pacing. Short conversation to keep it under ~15 s."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from backend.application.ports.component_catalogue import Component, ComponentKind
from backend.application.services.concurrency_supervisor import ConcurrencySupervisor
from backend.application.services.endpointing import EndpointerKind
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.settings import Settings
from backend.infrastructure.persistence.inmemory_call_repository import InMemoryCallRepository
from backend.infrastructure.simulated.gpu_model import SimulatedGpu, SimulatedGpuProfile
from backend.interfaces.cli import loadtest
from backend.interfaces.cli.harness import provenance, report
from backend.interfaces.cli.harness.caller import Conversation, load_conversation
from backend.interfaces.cli.harness.runner import LevelRunner
from backend.interfaces.container import build_container
from tests.fakes import NullMetrics

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

    curve = report.utilisation_curve(
        [summary], container.calculator.rate_card, [{"provider": "scaleway", "hourly_usd": 1.40}]
    )
    by_u = {row["utilisation"]: row for row in curve}
    assert by_u[0.25]["gpu_share_usd"] == 4 * by_u[1.0]["gpu_share_usd"]
    assert by_u[1.0]["cost_per_minute_usd_scaleway"] > by_u[1.0]["cost_per_minute_usd"]


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
def test_provenance_names_the_components_from_the_catalogue(
    pipeline: str, kind: PipelineKind, model_facts: set[str]
) -> None:
    stt = Component(ComponentKind.STT, "acme", "Acme", "ears-2", "r9", "eu-west", "MIT", False, "")
    conversation = Conversation("intake_en", "law_firm", "en", ["Hello."], [[]])
    settings = Settings(database_url="", vllm_base_url="http://127.0.0.1:9/v1")

    info = provenance.collect(
        settings,
        pipeline,
        conversation,
        EndpointerKind.SEMANTIC,
        simulated=True,
        rates_raw={"api": {"stt": {"model": "ears-2"}}},
        catalogue=OneStackCatalogue(kind, [stt]),
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
        }
    ]
    assert set(info["models"]) == model_facts
    assert "whisper" not in str(info["models"]) and "deepgram" not in str(info["models"])
