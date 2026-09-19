from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.settings import Settings
from backend.interfaces.container import Container
from backend.interfaces.http.deps import container, settings
from backend.interfaces.http.serializers import component

router = APIRouter()


@router.get("/transparency")
async def transparency(
    c: Annotated[Container, Depends(container)], s: Annotated[Settings, Depends(settings)]
) -> dict[str, Any]:
    """Public and read-only: every component that touches a call, per pipeline. Self-hosted
    STT and TTS are named from what their services report running (asked at most once a
    minute); everything else from the configuration this process runs on. Stable shape
    (catalogue_version bumps on a breaking change):

        {"catalogue_version": 1, "simulated": bool, "selfhosted_on_local_machine": bool,
         "pipelines": {"api" | "selfhosted": [
            {"kind": "telephony" | "stt" | "llm" | "tts" | "orchestration" | "storage",
             "id": str, "vendor": str, "model": str, "version": str, "region": str,
             "licence": str, "leaves_eu": bool,
             "assumption": str,  # "" when nothing is assumed
             "confirmed": bool,  # false: the running engine is unknown, so this is the
                                 # catalogue's default for the slot
             "unconfirmed_reason": str  # "" when confirmed
            }, ...  # one per kind, in that order
         ]}}

    simulated true means calls run on a simulated GPU model, not these components.
    """
    return {
        "catalogue_version": c.catalogue.version(),
        "simulated": s.simulate_providers,
        "selfhosted_on_local_machine": s.selfhosted_on_local_machine,
        "pipelines": {
            kind.value: [
                {
                    **component(d.component),
                    "confirmed": d.confirmed,
                    "unconfirmed_reason": d.unconfirmed_reason,
                }
                for d in await c.describe_components.execute(kind)
            ]
            for kind in PipelineKind
        },
    }
