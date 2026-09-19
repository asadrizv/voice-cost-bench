"""Compose, GPU compose, the RunPod script and native runs read the same settings with the
same meaning (#26)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from backend.infrastructure.config.settings import REPO_ROOT, MissingServingConfig, Settings


def test_the_shipped_serving_config_resolves_to_a_file_in_config_serving() -> None:
    settings = Settings(database_url="")
    assert settings.serving_config_path == REPO_ROOT / "config" / "serving" / "qwen-9b-l40s.yaml"
    assert settings.serving_config_path.is_file()


@pytest.mark.parametrize("value", ["llama-8b-l40s.yaml", "qwen-9b-l40s.yaml"])
def test_serving_config_names_a_file_in_config_serving_as_the_gpu_services_do(value: str) -> None:
    settings = Settings(database_url="", serving_config=value)
    assert settings.serving_config_path == REPO_ROOT / "config" / "serving" / value


@pytest.mark.parametrize("value", ["serving/qwen-9b-l40s.yaml", "qwen-9b-h100.yaml"])
def test_a_serving_config_that_names_no_file_fails_naming_the_value(value: str) -> None:
    settings = Settings(database_url="", serving_config=value)
    with pytest.raises(MissingServingConfig, match=re.escape(f"SERVING_CONFIG={value}")):
        _ = settings.serving_config_path


def _documented_names(env_example: Path) -> set[str]:
    assignment = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=", re.MULTILINE)
    return set(assignment.findall(env_example.read_text()))


def test_the_env_example_lists_every_setting_the_backend_reads() -> None:
    settings_names = {name.upper() for name in Settings.model_fields}
    assert settings_names - _documented_names(REPO_ROOT / ".env.example") == set()


def test_the_env_example_lists_every_setting_the_gpu_services_read() -> None:
    read = re.compile(r"os\.environ\.get\(\"([A-Z][A-Z0-9_]*)\"")
    service_names = {
        name
        for app in (REPO_ROOT / "gpu").glob("*_service/app.py")
        for name in read.findall(app.read_text())
    }
    assert service_names
    assert service_names - _documented_names(REPO_ROOT / ".env.example") == set()
