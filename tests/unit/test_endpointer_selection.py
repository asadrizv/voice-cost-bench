import re

import pytest
from pydantic import ValidationError

from backend.application.services.endpointing import (
    EndpointerKind,
    SemanticEndpointDetector,
    SilenceEndpointDetector,
    SmartTurnEndpointDetector,
)
from backend.domain.value_objects.audio import AudioChunk
from backend.infrastructure.config.settings import REPO_ROOT, Settings
from backend.interfaces.container import build_container
from tests.fakes import NullMetrics
from tests.unit.test_endpointing import speak_then_silence


def test_settings_read_an_endpointer_name_and_reject_unknown_ones() -> None:
    assert Settings(endpointer="silence").endpointer is EndpointerKind.SILENCE  # type: ignore[arg-type]
    assert Settings(endpointer="smart_turn").endpointer is EndpointerKind.SMART_TURN  # type: ignore[arg-type]
    assert Settings(_env_file=None).endpointer is EndpointerKind.SEMANTIC
    with pytest.raises(ValidationError, match="endpointer"):
        Settings(endpointer="vibes")  # type: ignore[arg-type]


def test_container_builds_the_named_endpointer_defaulting_to_settings() -> None:
    container = build_container(
        Settings(database_url="", endpointer=EndpointerKind.SILENCE),
        NullMetrics(),  # type: ignore[arg-type]
    )
    assert isinstance(container.endpointer(), SilenceEndpointDetector)
    assert isinstance(container.endpointer(EndpointerKind.SEMANTIC), SemanticEndpointDetector)
    assert isinstance(container.endpointer(EndpointerKind.SILENCE), SilenceEndpointDetector)


def test_container_builds_smart_turn_from_the_injected_model() -> None:
    container = build_container(
        Settings(database_url="", endpointer=EndpointerKind.SMART_TURN),
        NullMetrics(),  # type: ignore[arg-type]
        turn_model=lambda audio: 0.9,
    )
    detector = container.endpointer()
    assert isinstance(detector, SmartTurnEndpointDetector)
    last = speak_then_silence(detector)
    detector.observe_audio(AudioChunk(b"\x00\x00" * 320), False, last + 0.25)
    assert detector.should_commit(last + 0.25)


def test_the_browser_offers_every_endpointer_the_backend_knows() -> None:
    api_ts = (REPO_ROOT / "frontend" / "lib" / "api.ts").read_text()
    listed = re.search(r"export const ENDPOINTERS = \{(.*?)\} as const;", api_ts, re.S)
    assert listed is not None
    assert set(re.findall(r"(\w+):", listed.group(1))) == {kind.value for kind in EndpointerKind}
