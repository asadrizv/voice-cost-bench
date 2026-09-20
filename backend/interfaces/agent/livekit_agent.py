"""LiveKit worker. Thin by design: maps room events onto use cases and nothing else.

Run: uv run python -m backend.interfaces.agent.livekit_agent dev
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from collections.abc import AsyncIterator
from typing import Any

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobExecutorType, WorkerOptions, cli

from backend.application.services.concurrency_supervisor import CapacityExceeded
from backend.application.services.endpointing import EndpointerKind
from backend.application.use_cases.start_call import BudgetExceeded
from backend.domain.value_objects.audio import AudioChunk
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.settings import get_settings
from backend.infrastructure.persistence.postgres_call_repository import SqlCallRepository
from backend.infrastructure.pipeline_factory import MissingCredentials, PipelineNotSelectable
from backend.infrastructure.telemetry.http_metrics_sink import HttpMetricsSink
from backend.infrastructure.transport.livekit_audio import (
    OUTPUT_SAMPLE_RATE,
    LiveKitAudioOutput,
    caller_audio,
)
from backend.interfaces.container import build_container, validate_static_config
from backend.interfaces.http.routes_token import AGENT_NAME

log = logging.getLogger("voice-cost-bench.agent")

# livekit-rtc 1.1.18 panics ("timed out waiting for ReadyForRoomEventRequest after
# ConnectCallback") when job threads of a fresh worker join rooms at the same moment:
# reproduced 4/4 with three simultaneous calls, 0/3 with the process executor. We need
# threads so calls share one ConcurrencySupervisor, so room joins take turns instead.
_ROOM_JOIN = threading.Lock()


@contextlib.asynccontextmanager
async def _connect_lock() -> AsyncIterator[None]:
    # Acquired off the event loop: each job thread has its own loop, and blocking one
    # while another joins would stall that call's audio.
    await asyncio.to_thread(_ROOM_JOIN.acquire)
    try:
        yield
    finally:
        _ROOM_JOIN.release()


def _call_options(ctx: JobContext) -> dict[str, Any]:
    raw = ctx.job.metadata or (ctx.room.metadata if ctx.room else "") or "{}"
    try:
        options: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        options = {}
    return options


async def _status(ctx: JobContext, **payload: Any) -> None:
    with contextlib.suppress(Exception):
        await ctx.room.local_participant.publish_data(
            json.dumps(payload), reliable=True, topic="call-status"
        )


async def _first_audio_track(ctx: JobContext, participant: rtc.RemoteParticipant) -> rtc.Track:
    for publication in participant.track_publications.values():
        if publication.track is not None and publication.kind == rtc.TrackKind.KIND_AUDIO:
            return publication.track
    subscribed: asyncio.Future[rtc.Track] = asyncio.get_running_loop().create_future()

    def on_subscribed(track: rtc.Track, pub: Any, who: rtc.RemoteParticipant) -> None:
        is_caller_audio = (
            who.identity == participant.identity and track.kind == rtc.TrackKind.KIND_AUDIO
        )
        if is_caller_audio and not subscribed.done():
            subscribed.set_result(track)

    ctx.room.on("track_subscribed", on_subscribed)
    try:
        return await asyncio.wait_for(subscribed, timeout=30)
    finally:
        ctx.room.off("track_subscribed", on_subscribed)


async def entrypoint(ctx: JobContext) -> None:
    settings = get_settings()
    options = _call_options(ctx)
    pipeline = PipelineKind(options.get("pipeline", settings.pipeline.value))
    persona = options.get("persona", settings.persona)
    endpointer = EndpointerKind(options.get("endpointer", settings.endpointer))

    async with _connect_lock():
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    track = await _first_audio_track(ctx, participant)

    metrics = HttpMetricsSink(settings.api_base_url, settings.internal_token)
    container = build_container(settings, metrics)
    try:
        try:
            call_ctx = await container.start_call.execute(
                pipeline, persona, source="browser", call_id=ctx.room.name
            )
        except (
            BudgetExceeded,
            CapacityExceeded,
            MissingCredentials,
            PipelineNotSelectable,
        ) as exc:
            log.warning("call rejected: %s", exc)
            await _status(ctx, state="rejected", reason=str(exc))
            await asyncio.sleep(1)  # let the data message reach the browser
            ctx.shutdown(reason=str(exc))
            return

        await _status(ctx, state="started", call_id=call_ctx.call.id, pipeline=pipeline.value)
        source = rtc.AudioSource(OUTPUT_SAMPLE_RATE, 1)
        agent_track = rtc.LocalAudioTrack.create_audio_track("agent-voice", source)
        await ctx.room.local_participant.publish_track(
            agent_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        session = container.session(
            call_ctx, LiveKitAudioOutput(source), container.endpointer(endpointer)
        )

        # The mic stream doesn't reliably end on disconnect, so a pump feeds a queue that
        # hang-up can terminate even while no frames are arriving.
        frames: asyncio.Queue[AudioChunk | None] = asyncio.Queue()

        async def pump() -> None:
            async for chunk in caller_audio(track):
                frames.put_nowait(chunk)
            frames.put_nowait(None)

        def on_disconnect(p: rtc.RemoteParticipant) -> None:
            if p.identity == participant.identity:
                frames.put_nowait(None)

        ctx.room.on("participant_disconnected", on_disconnect)
        pump_task = asyncio.create_task(pump())

        async def until_hang_up() -> AsyncIterator[AudioChunk]:
            while (chunk := await frames.get()) is not None:
                yield chunk

        try:
            call = await session.run(until_hang_up())
            ended_by = "agent" if session.agent_hung_up else "caller"
            await _status(ctx, state="ended", call_id=call.id, ended_by=ended_by)
        except Exception:
            log.exception("call %s failed", call_ctx.call.id)
            await _status(ctx, state="failed", call_id=call_ctx.call.id)
        finally:
            pump_task.cancel()
            await source.aclose()
    finally:
        await metrics.aclose()
        if isinstance(container.repository, SqlCallRepository):
            await container.repository.dispose()
        ctx.shutdown()


def main() -> None:
    settings = get_settings()
    validate_static_config(settings)
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=AGENT_NAME,
            # Threads, not processes: every call must share one ConcurrencySupervisor,
            # the single source of the GPU cost divisor.
            job_executor_type=JobExecutorType.THREAD,
            ws_url=settings.livekit_url,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
        )
    )


if __name__ == "__main__":
    main()
