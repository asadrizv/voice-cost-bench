import pytest
from pydantic import ValidationError

from backend.application.services.endpointing import (
    EndpointerKind,
    SemanticEndpointDetector,
    SilenceEndpointDetector,
)
from backend.infrastructure.config.settings import Settings
from backend.interfaces.container import build_container
from tests.fakes import NullMetrics


def test_settings_read_an_endpointer_name_and_reject_unknown_ones() -> None:
    assert Settings(endpointer="silence").endpointer is EndpointerKind.SILENCE  # type: ignore[arg-type]
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
