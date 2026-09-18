from __future__ import annotations

from collections.abc import Callable

from backend.application.ports.pipeline_provider import Pipeline
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.settings import Settings
from backend.infrastructure.llm.openai_llm import OpenAILlm
from backend.infrastructure.llm.vllm_llm import VllmLlm
from backend.infrastructure.stt.deepgram_stt import DeepgramStt
from backend.infrastructure.stt.whisper_stt import WhisperStt
from backend.infrastructure.tts.elevenlabs_tts import ElevenLabsTts
from backend.infrastructure.tts.kokoro_tts import KokoroTts


class MissingCredentials(RuntimeError):
    pass


def _api(s: Settings) -> Pipeline:
    missing = [
        name
        for name, value in (
            ("DEEPGRAM_API_KEY", s.deepgram_api_key),
            ("OPENAI_API_KEY", s.openai_api_key),
            ("ELEVENLABS_API_KEY", s.elevenlabs_api_key),
        )
        if not value
    ]
    if missing:
        raise MissingCredentials(f"api pipeline needs {', '.join(missing)}")
    return Pipeline(
        kind=PipelineKind.API,
        stt=DeepgramStt(s.deepgram_api_key, s.deepgram_model),
        llm=OpenAILlm(s.openai_model, s.openai_api_key),
        tts=ElevenLabsTts(s.elevenlabs_api_key, s.elevenlabs_model),
        uses_gpu=False,
    )


def _selfhosted(s: Settings) -> Pipeline:
    return Pipeline(
        kind=PipelineKind.SELFHOSTED,
        stt=WhisperStt(s.whisper_ws_url),
        llm=VllmLlm(s.vllm_model, "unused", base_url=s.vllm_base_url),
        tts=KokoroTts(s.kokoro_url),
        uses_gpu=True,
    )


def _simulated(kind: PipelineKind) -> Callable[[Settings], Pipeline]:
    def build(s: Settings) -> Pipeline:
        import yaml

        from backend.infrastructure.simulated.gpu_model import (
            SimulatedGpu,
            SimulatedLlm,
            SimulatedStt,
            SimulatedTts,
        )

        spec = yaml.safe_load((s.config_dir.parent / "fixtures" / "conversations.yaml").read_text())
        gpu = SimulatedGpu()
        script = spec["conversations"]["intake_en"]["turns"]
        return Pipeline(kind, SimulatedStt(gpu, script), SimulatedLlm(gpu), SimulatedTts(gpu), True)

    return build


BUILDERS: dict[PipelineKind, Callable[[Settings], Pipeline]] = {
    PipelineKind.API: _api,
    PipelineKind.SELFHOSTED: _selfhosted,
}


class PipelineFactory:
    """Resolves a PipelineKind to its adapter triple. The only place that knows which
    vendor sits behind which kind; use cases see ports only."""

    def __init__(
        self,
        settings: Settings,
        builders: dict[PipelineKind, Callable[[Settings], Pipeline]] | None = None,
    ) -> None:
        self._settings = settings
        if builders is None and settings.simulate_providers:
            builders = {kind: _simulated(kind) for kind in PipelineKind}
        self._builders = builders or BUILDERS
        self._cache: dict[PipelineKind, Pipeline] = {}

    def resolve(self, kind: PipelineKind) -> Pipeline:
        if kind not in self._cache:
            self._cache[kind] = self._builders[kind](self._settings)
        return self._cache[kind]
