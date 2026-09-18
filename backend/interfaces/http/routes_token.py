from __future__ import annotations

import json
import secrets
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from livekit import api
from pydantic import BaseModel

from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.settings import Settings
from backend.interfaces.container import Container
from backend.interfaces.http.deps import container, settings

router = APIRouter()
AGENT_NAME = "voice-cost-bench"


class TokenRequest(BaseModel):
    pipeline: PipelineKind | None = None
    persona: str | None = None
    endpointer: str | None = None


class TokenResponse(BaseModel):
    token: str
    url: str
    room: str
    call_id: str
    pipeline: PipelineKind
    persona: str


@router.post("/token")
def create_token(
    body: TokenRequest,
    s: Annotated[Settings, Depends(settings)],
    c: Annotated[Container, Depends(container)],
) -> TokenResponse:
    """One room per call; its name is the call id, so the browser can subscribe to live
    metrics before the agent has even joined. The pipeline choice rides in room metadata
    carried by the token's room configuration: an explicit dispatch to our agent with the
    choice as job metadata, so no other agent on the LiveKit project picks the call up."""
    pipeline = body.pipeline or s.pipeline
    persona = body.persona or s.persona
    if persona not in c.personas.available():
        raise HTTPException(404, f"unknown persona {persona}")
    endpointer = body.endpointer or s.endpointer
    if endpointer not in ("semantic", "silence"):
        raise HTTPException(422, "endpointer must be semantic or silence")

    call_id = f"call-{secrets.token_hex(6)}"
    metadata = json.dumps(
        {"pipeline": pipeline.value, "persona": persona, "endpointer": endpointer}
    )
    token = (
        api.AccessToken(s.livekit_api_key, s.livekit_api_secret)
        .with_identity(f"caller-{secrets.token_hex(3)}")
        .with_name("Caller")
        .with_ttl(timedelta(hours=1))
        .with_grants(api.VideoGrants(room_join=True, room=call_id))
        .with_room_config(
            api.RoomConfiguration(
                metadata=metadata,
                agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME, metadata=metadata)],
            )
        )
        .to_jwt()
    )
    return TokenResponse(
        token=token,
        url=s.livekit_public_url or s.livekit_url,
        room=call_id,
        call_id=call_id,
        pipeline=pipeline,
        persona=persona,
    )
