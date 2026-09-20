from __future__ import annotations

import asyncio
from collections.abc import Mapping
from decimal import Decimal

import httpx

from backend.application.ports.clock import Clock
from backend.application.ports.component_catalogue import ComponentKind
from backend.application.ports.running_engines import Engine, EngineReportUnavailable


def _engine(reported: dict[str, object]) -> Engine:
    """A service that predates gpu_fraction reports none, which reads as "holds its weights"
    — the same answer the budget gave before any service reported a share."""
    share = reported.get("gpu_fraction")
    return Engine(
        str(reported["id"]),
        str(reported["model"]),
        None if share is None else Decimal(str(share)),
    )


class HttpRunningEngines:
    """Asks each self-hosted service's GET /v1/info. Answers, failures included, are kept
    for ttl_s, and callers that arrive while a probe is in flight wait on that one probe,
    so a public endpoint can't turn its traffic into load on the GPU services."""

    def __init__(
        self,
        urls: Mapping[ComponentKind, str],
        clock: Clock,
        ttl_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._urls = urls
        self._clock = clock
        self._ttl_s = ttl_s
        self._given_client = client
        self._client: httpx.AsyncClient | None = client
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._answers: dict[ComponentKind, tuple[float, asyncio.Task[list[Engine]]]] = {}

    async def report(self, kind: ComponentKind) -> list[Engine]:
        now = self._clock.monotonic()
        loop = asyncio.get_running_loop()
        answered = self._answers.get(kind)
        # A task belongs to the loop that made it, so a container reused across loops asks again.
        fresh = answered is not None and answered[1].get_loop() is loop
        if answered is None or not fresh or now - answered[0] >= self._ttl_s:
            answered = (now, loop.create_task(self._ask(self._urls[kind])))
            self._answers[kind] = answered
        # Shielded: one caller giving up must not cancel the probe the others are waiting on.
        return await asyncio.shield(answered[1])

    def _http(self, loop: asyncio.AbstractEventLoop) -> httpx.AsyncClient:
        """Built on the first probe, not in __init__: the agent builds a container per call
        and never asks, so an eager client would leak a connection pool per call. Rebuilt
        with the loop for the same reason the answers are — a client bound to a loop that
        has closed raises RuntimeError, which is not an answer about the service."""
        if self._given_client is not None:
            return self._given_client
        if self._client is None or self._client_loop is not loop:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(2.0))
            self._client_loop = loop
        return self._client

    async def _ask(self, url: str) -> list[Engine]:
        try:
            response = await self._http(asyncio.get_running_loop()).get(url)
            response.raise_for_status()
            return [_engine(e) for e in response.json()["engines"]]
        except httpx.HTTPStatusError as exc:
            raise EngineReportUnavailable(f"HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise EngineReportUnavailable(type(exc).__name__) from exc
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise EngineReportUnavailable("malformed report") from exc
