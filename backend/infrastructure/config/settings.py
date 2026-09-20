from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.application.services.endpointing import EndpointerKind
from backend.domain.value_objects.pipeline_kind import PipelineKind

REPO_ROOT = Path(__file__).resolve().parents[3]


class MissingServingConfig(RuntimeError):
    pass


class Settings(BaseSettings):
    """Secrets and endpoints only. Tuning lives in versioned YAML under config/."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    pipeline: PipelineKind = PipelineKind.API
    persona: str = "law_firm"
    endpointer: EndpointerKind = EndpointerKind.SEMANTIC
    eu_only: bool = False
    """EU-only deployment profile: startup fails unless every catalogue component a call
    could touch is EU-resident, and no call may run on a pipeline other than the one
    above."""

    livekit_url: str = "ws://localhost:7880"
    livekit_public_url: str = ""
    """What the browser dials, when it differs from the agent's address (Docker)."""
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "secret"

    deepgram_api_key: str = ""
    deepgram_model: str = "nova-3"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    elevenlabs_api_key: str = ""
    elevenlabs_model: str = "eleven_flash_v2_5"

    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "voice-llm"
    llm_server: str = Field("vllm", description="vllm | ollama (laptop demo)")
    whisper_ws_url: str = "ws://localhost:8001/v1/stream"
    kokoro_url: str = "http://localhost:8002"

    database_url: str = ""
    """Empty runs on the in-memory repository: fine for a demo, gone on restart."""
    api_base_url: str = "http://localhost:8080"
    """Where the agent worker forwards live metrics for SSE and Prometheus."""
    internal_token: str = "dev-internal-token"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    selfhosted_on_local_machine: bool = False
    """Self-hosted models are running on a laptop, not the benchmark GPU: the UI says so,
    because latency and the L40S-priced cost then describe hardware nobody will deploy."""
    simulate_providers: bool = False
    """Both pipelines resolve to the simulated GPU model: runs the whole stack (LiveKit,
    agent, browser) with no keys and no GPU. Every cost it produces is fiction."""
    dev_spend_limit_usd: float = 25.0
    selfhosted_max_concurrency: int = 40

    config_dir: Path = REPO_ROOT / "config"
    serving_config: str = "qwen-9b-l40s.yaml"
    """A file name in config/serving/: the meaning gpu/docker-compose.gpu.yml and
    gpu/runpod/start.sh give SERVING_CONFIG when they start vLLM from it."""

    @property
    def selectable_pipelines(self) -> tuple[PipelineKind, ...]:
        """The pipelines a call may run on, which the browser and the harness choose from
        per call. Under EU_ONLY only the configured one, so a per-call choice cannot reach
        components the profile refused to start with."""
        return (self.pipeline,) if self.eu_only else tuple(PipelineKind)

    @property
    def rates_path(self) -> Path:
        return self.config_dir / "rates.yaml"

    @property
    def components_path(self) -> Path:
        return self.config_dir / "components.yaml"

    @property
    def personas_dir(self) -> Path:
        return self.config_dir / "personas"

    @property
    def serving_config_path(self) -> Path:
        """Raises MissingServingConfig when SERVING_CONFIG names no file, so a benchmark
        can't record provenance for a vLLM configuration nobody can find."""
        path = self.config_dir / "serving" / self.serving_config
        if not path.is_file():
            raise MissingServingConfig(
                f"SERVING_CONFIG={self.serving_config}: no file at {path}; "
                "name a file in config/serving/"
            )
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
