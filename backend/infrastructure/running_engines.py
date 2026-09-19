from __future__ import annotations

from collections.abc import Mapping

import httpx

from backend.application.ports.clock import Clock
from backend.application.ports.component_catalogue import ComponentKind
from backend.application.ports.running_engines import Engine, EngineReportUnavailable


class HttpRunningEngines:
    """Asks each self-hosted service's GET /v1/info. Answers, failures included, are kept
    for ttl_s so a public endpoint can't turn its traffic into load on the GPU services."""

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
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(2.0))
        self._answers: dict[ComponentKind, tuple[float, list[Engine] | str]] = {}

    async def report(self, kind: ComponentKind) -> list[Engine]:
        now = self._clock.monotonic()
        answered = self._answers.get(kind)
        if answered is None or now - answered[0] >= self._ttl_s:
            answered = (now, await self._ask(self._urls[kind]))
            self._answers[kind] = answered
        answer = answered[1]
        if isinstance(answer, str):
            raise EngineReportUnavailable(answer)
        return answer

    async def _ask(self, url: str) -> list[Engine] | str:
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            return [Engine(str(e["id"]), str(e["model"])) for e in response.json()["engines"]]
        except httpx.HTTPStatusError as exc:
            return f"HTTP {exc.response.status_code}"
        except httpx.HTTPError as exc:
            return type(exc).__name__
        except (ValueError, KeyError, TypeError):
            return "malformed report"
