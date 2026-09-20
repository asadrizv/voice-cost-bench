"""Which pipeline a call may run on, and who decides (#11)."""

import re

import pytest

from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.settings import Settings
from backend.infrastructure.pipeline_factory import PipelineFactory, PipelineNotSelectable


def test_a_call_may_run_on_either_pipeline_until_the_eu_only_profile_narrows_it() -> None:
    assert Settings(database_url="").selectable_pipelines == tuple(PipelineKind)
    assert Settings(database_url="", eu_only=True).selectable_pipelines == (PipelineKind.API,)
    assert Settings(
        database_url="", eu_only=True, pipeline=PipelineKind.SELFHOSTED
    ).selectable_pipelines == (PipelineKind.SELFHOSTED,)


def test_the_factory_refuses_a_pipeline_the_eu_only_profile_does_not_serve() -> None:
    """The agent takes its pipeline from the call's job metadata, not from the token route,
    so the refusal has to sit where every call resolves its adapters."""
    factory = PipelineFactory(
        Settings(database_url="", eu_only=True, pipeline=PipelineKind.SELFHOSTED)
    )

    with pytest.raises(
        PipelineNotSelectable,
        match=re.escape("EU_ONLY serves the selfhosted pipeline only; the api pipeline"),
    ):
        factory.resolve(PipelineKind.API)
