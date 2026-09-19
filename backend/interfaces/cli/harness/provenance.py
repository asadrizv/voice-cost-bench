from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from backend.application.services.endpointing import EndpointerKind
from backend.application.use_cases.describe_components import DescribedComponent
from backend.infrastructure.config.settings import REPO_ROOT, Settings
from backend.infrastructure.telemetry.nvml_gpu_telemetry import detect_gpu_telemetry
from backend.interfaces.cli.harness.caller import Conversation


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"


def _probe(url: str) -> Any:
    try:
        response = httpx.get(url, timeout=3)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": type(exc).__name__}


def collect(
    settings: Settings,
    pipeline: str,
    conversation: Conversation,
    endpointer: EndpointerKind,
    simulated: bool,
    rates_raw: dict[str, Any],
    components: list[DescribedComponent],
    catalogue_version: int,
) -> dict[str, Any]:
    """Everything needed to reproduce or challenge a number, written with the number."""
    gpu = detect_gpu_telemetry().snapshot()
    info: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "simulated": simulated,
        "pipeline": pipeline,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "rate_card_verified_on": rates_raw.get("verified_on"),
        "rate_card_sha256": _sha256(settings.rates_path),
        "gpu": {
            **rates_raw.get("selfhosted", {}).get("gpu", {}),
            "detected": gpu.sku if gpu else None,
        },
        "fixture_set": {
            "conversation": conversation.name,
            "language": conversation.language,
            "turns": len(conversation.turns),
            "audio_sha256": conversation.sha,
        },
        "persona": conversation.persona,
        "persona_sha256": _sha256(settings.personas_dir / f"{conversation.persona}.yaml"),
        "endpointer": endpointer.value,
        "component_catalogue_version": catalogue_version,
        "components": [
            {
                **asdict(d.component),
                "kind": d.component.kind.value,
                "confirmed": d.confirmed,
                "unconfirmed_reason": d.unconfirmed_reason,
            }
            for d in components
        ],
        "harness_host": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
    }
    if pipeline == "api":
        api = rates_raw.get("api", {})
        info["models"] = {
            "listed_in_rate_card": {k: v.get("model") for k, v in api.items()},
        }
    elif settings.llm_server == "ollama":
        info["models"] = {"llm_server": "ollama", "ollama_model": settings.vllm_model}
    else:
        vllm_root = settings.vllm_base_url.rsplit("/v1", 1)[0]
        info["models"] = {
            "vllm_version": None if simulated else _probe(f"{vllm_root}/version"),
            "serving_config": settings.serving_config,
            "serving_config_sha256": _sha256(settings.serving_config_path),
        }
    return info
