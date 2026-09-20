from __future__ import annotations

import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from backend.application.ports.component_catalogue import (
    Component,
    ComponentKind,
    GpuMemoryProfile,
)
from backend.application.use_cases.check_gpu_budget import CheckGpuBudget, LlmShare
from backend.domain.services.gpu_memory_budget import (
    BASES,
    BudgetVerdict,
    GpuCapacity,
    MemoryClaim,
)
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.component_catalogue import (
    ComponentCatalogueError,
    YamlComponentCatalogue,
)
from backend.infrastructure.config.settings import REPO_ROOT, Settings
from backend.infrastructure.pricing.yaml_rate_card import YamlRateCardProvider
from backend.interfaces.container import build_catalogue, build_gpu_budget_check

GPU = "L40S"


def config_with(tmp_path: Path, utilisation: str, **figures: dict[str, Any]) -> Path:
    """The shipped configuration with the LLM's memory share, and any named component's
    memory figure, replaced."""
    config = tmp_path / "config"
    shutil.copytree(REPO_ROOT / "config", config)
    serving = config / "serving" / "qwen-9b-l40s.yaml"
    served = yaml.safe_load(serving.read_text())
    served["gpu-memory-utilization"] = float(utilisation)
    serving.write_text(yaml.safe_dump(served))
    components = config / "components.yaml"
    raw = yaml.safe_load(components.read_text())
    for name, figure in figures.items():
        raw["components"][name]["gpu_memory"] = figure
    components.write_text(yaml.safe_dump(raw))
    return config


def component(kind: ComponentKind, id: str) -> Component:
    return Component(kind, id, "Acme", "m", "1", "eu-west", "MIT", False, "")


def profile(**figures: str | None) -> GpuMemoryProfile:
    return GpuMemoryProfile(
        gpu=GpuCapacity(GPU, Decimal("48.0"), "vendor-stated", "the data sheet"),
        claims={
            engine: MemoryClaim(engine, None if gib is None else Decimal(gib), "estimated", "recon")
            for engine, gib in figures.items()
        },
    )


def check(share: str | None = "0.72", **figures: str | None) -> CheckGpuBudget:
    claims = {"llm": None, **figures}
    return CheckGpuBudget(
        profile(**claims),
        LlmShare(None if share is None else Decimal(share), "gpu-memory-utilization"),
    )


def selection(*engines: str) -> list[Component]:
    kinds = {"whisper": ComponentKind.STT, "voxtral": ComponentKind.STT}
    return [component(ComponentKind.LLM, "llm")] + [
        component(kinds.get(engine, ComponentKind.TTS), engine) for engine in engines
    ]


def test_the_default_engines_fit_beside_the_llms_reserved_share() -> None:
    budget = check(whisper="3.0", kokoro="1.0").execute(selection("whisper", "kokoro"))

    assert budget is not None
    assert budget.gpu.sku == GPU
    assert [(c.component_id, c.gib) for c in budget.claims] == [
        ("llm", Decimal("34.56")),
        ("whisper", Decimal("3.0")),
        ("kokoro", Decimal("1.0")),
    ]
    assert budget.claimed_gib == Decimal("38.56")
    assert budget.headroom_gib == Decimal("9.44")
    assert budget.shortfall_gib == Decimal(0)
    assert budget.verdict is BudgetVerdict.FITS


def test_engines_that_overrun_the_card_report_the_shortfall() -> None:
    budget = check(voxtral="9.98", qwen3_tts="5.48").execute(selection("voxtral", "qwen3_tts"))

    assert budget is not None
    assert budget.claimed_gib == Decimal("50.02")
    assert budget.headroom_gib == Decimal("-2.02")
    assert budget.shortfall_gib == Decimal("2.02")
    assert budget.verdict is BudgetVerdict.DOES_NOT_FIT


def test_a_combination_that_exactly_fills_the_card_fits() -> None:
    budget = check(voxtral="9.44", kokoro="4.0").execute(selection("voxtral", "kokoro"))

    assert budget is not None
    assert budget.claimed_gib == Decimal("48.00")
    assert budget.headroom_gib == Decimal(0)
    assert budget.verdict is BudgetVerdict.FITS


def test_an_engine_with_no_figure_is_unknown_rather_than_free() -> None:
    """Counting a missing figure as zero would let an over-committed card look fine."""
    budget = check(whisper="3.0", mlx=None).execute(selection("whisper", "mlx"))

    assert budget is not None
    assert budget.unknown == ("mlx",)
    assert budget.claimed_gib == Decimal("37.56")
    assert budget.verdict is BudgetVerdict.UNKNOWN


def test_a_card_already_overrun_does_not_fit_whatever_the_unknown_engine_holds() -> None:
    budget = check(voxtral="9.98", qwen3_tts="5.48", mlx=None).execute(
        selection("voxtral", "qwen3_tts", "mlx")
    )

    assert budget is not None
    assert budget.unknown == ("mlx",)
    assert budget.shortfall_gib == Decimal("2.02")
    assert budget.verdict is BudgetVerdict.DOES_NOT_FIT


def test_an_llm_that_reserves_no_share_leaves_the_budget_unsettled() -> None:
    """Ollama takes what it needs when it needs it, so nothing here can be added up."""
    budget = check(share=None, whisper="3.0", kokoro="1.0").execute(selection("whisper", "kokoro"))

    assert budget is not None
    assert budget.unknown == ("llm",)
    assert budget.verdict is BudgetVerdict.UNKNOWN


def test_components_that_never_touch_the_gpu_are_left_out_of_the_budget() -> None:
    budget = check(whisper="3.0").execute(
        [
            *selection("whisper"),
            component(ComponentKind.TELEPHONY, "twilio"),
            component(ComponentKind.STORAGE, "postgres"),
        ]
    )

    assert budget is not None
    assert [c.component_id for c in budget.claims] == ["llm", "whisper"]


def test_a_pipeline_with_nothing_on_the_gpu_has_no_budget() -> None:
    budget = check().execute([component(ComponentKind.LLM, "openai")])

    assert budget is None


def test_the_summary_shows_the_arithmetic_behind_the_verdict() -> None:
    fits = check(whisper="3.0", kokoro="1.0").execute(selection("whisper", "kokoro"))
    over = check(voxtral="9.98", qwen3_tts="5.48").execute(selection("voxtral", "qwen3_tts"))
    unsure = check(whisper="3.0", mlx=None).execute(selection("whisper", "mlx"))

    assert fits is not None and over is not None and unsure is not None
    assert fits.summary() == (
        "L40S 48.00 GiB: llm 34.56 + whisper 3.00 + kokoro 1.00 = 38.56 GiB, 9.44 GiB spare (fits)"
    )
    assert over.summary() == (
        "L40S 48.00 GiB: llm 34.56 + voxtral 9.98 + qwen3_tts 5.48 = 50.02 GiB, "
        "2.02 GiB short (does not fit)"
    )
    assert unsure.summary() == (
        "L40S 48.00 GiB: llm 34.56 + whisper 3.00 + mlx ? = 37.56 GiB, "
        "10.44 GiB spare (unknown: no figure for mlx)"
    )


@pytest.mark.parametrize(
    ("utilisation", "verdict"),
    [("0.72", BudgetVerdict.FITS), ("0.80", BudgetVerdict.DOES_NOT_FIT)],
)
def test_the_llms_configured_share_decides_what_is_left_for_the_engines(
    utilisation: str, verdict: BudgetVerdict
) -> None:
    budget = check(share=utilisation, whisper="3.0", kokoro="7.0").execute(
        selection("whisper", "kokoro")
    )

    assert budget is not None and budget.verdict is verdict


def shipped() -> tuple[YamlComponentCatalogue, CheckGpuBudget]:
    settings = Settings(database_url="")
    catalogue = build_catalogue(settings, YamlRateCardProvider(settings.rates_path))
    return catalogue, build_gpu_budget_check(settings, catalogue)


def engines(catalogue: YamlComponentCatalogue, stt: str, tts: str) -> list[Component]:
    """The LLM the selfhosted pipeline configures, beside a chosen STT and TTS engine."""
    llm = [c for c in catalogue.components(PipelineKind.SELFHOSTED) if c.kind is ComponentKind.LLM]
    chosen = [catalogue.engine(ComponentKind.STT, stt), catalogue.engine(ComponentKind.TTS, tts)]
    return llm + [c for c in chosen if c is not None]


def test_every_engine_on_the_gpu_carries_a_figure_with_the_basis_it_came_from() -> None:
    catalogue, _ = shipped()

    profile = catalogue.gpu_memory()

    assert profile is not None
    assert profile.gpu.sku == "L40S"
    assert profile.gpu.total_gib == Decimal("48.0")
    assert profile.gpu.basis in BASES and profile.gpu.source
    claims = profile.claims
    assert set(claims) >= {"faster-whisper", "kokoro", "voxtral", "qwen3-tts", "mlx"}
    assert "deepgram" not in claims and "postgres" not in claims
    for engine in ("faster-whisper", "kokoro", "voxtral", "qwen3-tts"):
        claim = claims[engine]
        assert claim.gib is not None and claim.gib > 0
        assert claim.basis in BASES and claim.source


def test_an_engine_that_never_runs_on_the_gpu_has_no_figure_to_read() -> None:
    """MLX is the Apple Silicon path: a CUDA footprint for it would be invented."""
    catalogue, _ = shipped()

    profile = catalogue.gpu_memory()

    assert profile is not None
    assert profile.claims["mlx"].gib is None


def test_the_default_engines_fit_one_l40s_beside_the_configured_llm() -> None:
    catalogue, check = shipped()

    budget = check.execute(engines(catalogue, "faster-whisper", "kokoro"))

    assert budget is not None
    assert budget.verdict is BudgetVerdict.FITS
    assert budget.headroom_gib > 0
    assert not budget.measured


def test_voxtral_and_qwen3_tts_do_not_fit_one_l40s_beside_the_configured_llm() -> None:
    catalogue, check = shipped()

    budget = check.execute(engines(catalogue, "voxtral", "qwen3-tts"))

    assert budget is not None
    assert budget.verdict is BudgetVerdict.DOES_NOT_FIT
    assert budget.shortfall_gib == Decimal("2.02")
    assert "2.02 GiB short" in budget.summary()


def test_the_serving_configs_memory_share_decides_the_shipped_verdict(tmp_path: Path) -> None:
    settings = Settings(database_url="", config_dir=config_with(tmp_path, "0.50"))
    catalogue = build_catalogue(settings, YamlRateCardProvider(settings.rates_path))

    budget = build_gpu_budget_check(settings, catalogue).execute(
        engines(catalogue, "voxtral", "qwen3-tts")
    )

    assert budget is not None
    assert budget.verdict is BudgetVerdict.FITS


def test_a_memory_figure_without_a_stated_basis_fails_at_startup(tmp_path: Path) -> None:
    settings = Settings(
        database_url="", config_dir=config_with(tmp_path, "0.72", kokoro={"gib": 1.0})
    )

    with pytest.raises(ComponentCatalogueError, match="kokoro.*basis"):
        build_catalogue(settings, YamlRateCardProvider(settings.rates_path))


def without_the_gpu(config: Path, **hosts: Any) -> Path:
    raw = yaml.safe_load((config / "components.yaml").read_text())
    gpu = raw["hosts"]["gpu_host"].pop("gpu")
    for name in hosts:
        raw["hosts"][name]["gpu"] = gpu
    (config / "components.yaml").write_text(yaml.safe_dump(raw))
    return config


def test_a_deployment_with_no_gpu_has_nothing_to_budget(tmp_path: Path) -> None:
    settings = Settings(database_url="", config_dir=without_the_gpu(config_with(tmp_path, "0.72")))
    catalogue = build_catalogue(settings, YamlRateCardProvider(settings.rates_path))

    assert catalogue.gpu_memory() is None
    assert (
        build_gpu_budget_check(settings, catalogue).execute(
            engines(catalogue, "faster-whisper", "kokoro")
        )
        is None
    )


def test_a_second_host_with_a_gpu_fails_at_startup(tmp_path: Path) -> None:
    """One card is the product's claim; two would each need their own budget."""
    config = config_with(tmp_path, "0.72")
    settings = Settings(
        database_url="", config_dir=without_the_gpu(config, gpu_host=True, app_host=True)
    )

    with pytest.raises(ComponentCatalogueError, match="app_host, gpu_host"):
        build_catalogue(settings, YamlRateCardProvider(settings.rates_path))
