from __future__ import annotations

import asyncio
import contextlib
import logging

import httpx

from backend.application.ports.metrics_sink import CallMetricsEvent
from backend.infrastructure.telemetry.event_codec import encode

log = logging.getLogger(__name__)


class HttpMetricsSink:
    """Forwards events from the agent (or load harness) process to the API, which owns SSE
    and the Prometheus registry. Fire-and-forget through a bounded queue: a slow or down
    API drops metrics, never audio."""

    def __init__(self, api_base_url: str, token: str, queue_size: int = 2000) -> None:
        self._client = httpx.AsyncClient(base_url=api_base_url, timeout=2.0)
        self._token = token
        self._queue: asyncio.Queue[CallMetricsEvent] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task[None] | None = None
        self.dropped = 0

    async def emit(self, event: CallMetricsEvent) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drain())
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1

    async def _drain(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self._client.post(
                    "/internal/metrics/events",
                    json=encode(event),
                    headers={"x-internal-token": self._token},
                )
            except httpx.HTTPError as exc:
                log.debug("metrics forward failed: %s", exc)
            finally:
                self._queue.task_done()

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._queue.join(), timeout=2.0)
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self._client.aclose()
