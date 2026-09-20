"""Compose, GPU compose, the RunPod script and native runs read the same settings with the
same meaning (#26)."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml
from dotenv import dotenv_values

from backend.infrastructure.config.settings import REPO_ROOT, MissingServingConfig, Settings
from gpu.kokoro_service.app import VllmOmniQwen3TtsSynthesizer
from gpu.whisper_service.app import VllmVoxtralTranscriber


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


def test_no_env_example_value_is_a_stray_comment() -> None:
    """After an empty value, python-dotenv and compose read an inline comment as the value."""
    values = dotenv_values(REPO_ROOT / ".env.example")
    assert {name: v for name, v in values.items() if v and v.startswith("#")} == {}


def _compose_defaults(environment: dict[str, str]) -> dict[str, str]:
    """`${NAME:-default}` as compose resolves it with NAME unset, which is how a GPU node
    runs unless its operator overrides one."""
    assignment = re.compile(r"^\$\{[A-Z0-9_]+:-(.*)\}$")
    return {
        name: matched.group(1)
        for name, value in environment.items()
        if (matched := assignment.match(str(value)))
    }


def test_the_gpu_compose_points_each_cuda_backend_at_the_server_that_serves_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name the service does not read, or a port the server does not publish, would leave
    the backend on its localhost default -- which inside compose is its own container."""
    services = yaml.safe_load((REPO_ROOT / "gpu" / "docker-compose.gpu.yml").read_text())[
        "services"
    ]
    for service in ("whisper", "kokoro"):
        for name, value in _compose_defaults(services[service]["environment"]).items():
            monkeypatch.setenv(name, value)

    realtime_port = services["voxtral-vllm"]["ports"][0].split(":")[1]
    speech_port = services["qwen3-tts-vllm"]["ports"][0].split(":")[1]
    assert VllmVoxtralTranscriber().url == f"ws://voxtral-vllm:{realtime_port}/v1/realtime"
    assert VllmOmniQwen3TtsSynthesizer().base_url == f"http://qwen3-tts-vllm:{speech_port}"


def test_the_runpod_script_serves_each_cuda_backend_on_the_port_its_default_url_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pod runs every server on localhost, so there the backends' own defaults are the
    configuration."""
    for name in ("VOXTRAL_VLLM_URL", "QWEN3_TTS_VLLM_URL"):
        monkeypatch.delenv(name, raising=False)
    script = (REPO_ROOT / "gpu" / "runpod" / "start.sh").read_text()

    for url in (VllmVoxtralTranscriber().url, VllmOmniQwen3TtsSynthesizer().base_url):
        parsed = urlparse(url)
        assert parsed.hostname == "127.0.0.1"
        assert f"--port {parsed.port}" in script
