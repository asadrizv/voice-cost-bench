from __future__ import annotations

import httpx
import pytest

from backend.application.ports.component_catalogue import ComponentKind
from backend.application.ports.running_engines import Engine, EngineReportUnavailable
from backend.infrastructure.running_engines import HttpRunningEngines
from tests.fakes import FakeClock

INFO = "http://stt.test/v1/info"
MLX = {"engines": [{"id": "mlx", "model": "mlx-community/whisper-large-v3-turbo"}]}


def engines_answering(
    responses: list[httpx.Response], clock: FakeClock
) -> tuple[HttpRunningEngines, list[httpx.Request]]:
    asked: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        asked.append(request)
        return responses[min(len(asked), len(responses)) - 1]

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return HttpRunningEngines({ComponentKind.STT: INFO}, clock, client=client), asked


async def test_a_report_is_asked_again_once_a_minute_has_passed() -> None:
    clock = FakeClock()
    faster = {"engines": [{"id": "faster-whisper", "model": "large-v3-turbo"}]}
    engines, asked = engines_answering(
        [httpx.Response(200, json=MLX), httpx.Response(200, json=faster)], clock
    )

    first = await engines.report(ComponentKind.STT)
    clock.advance(59.9)
    within_the_minute = await engines.report(ComponentKind.STT)
    clock.advance(0.1)
    after_it = await engines.report(ComponentKind.STT)

    assert first == within_the_minute == [Engine("mlx", "mlx-community/whisper-large-v3-turbo")]
    assert after_it == [Engine("faster-whisper", "large-v3-turbo")]
    assert [str(r.url) for r in asked] == [INFO, INFO]


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (httpx.Response(503, text="loading"), "HTTP 503"),
        (httpx.Response(200, text="not json"), "malformed report"),
        (httpx.Response(200, json={"engine": "mlx"}), "malformed report"),
        (httpx.Response(200, json={"engines": [{"id": "mlx"}]}), "malformed report"),
        (httpx.Response(200, json={"engines": [None]}), "malformed report"),
    ],
)
async def test_a_failed_report_is_unavailable_and_remembered_for_a_minute(
    response: httpx.Response, reason: str
) -> None:
    clock = FakeClock()
    engines, asked = engines_answering([response, httpx.Response(200, json=MLX)], clock)

    for _ in range(2):
        with pytest.raises(EngineReportUnavailable, match=f"^{reason}$"):
            await engines.report(ComponentKind.STT)
    clock.advance(60)

    assert await engines.report(ComponentKind.STT) == [
        Engine("mlx", "mlx-community/whisper-large-v3-turbo")
    ]
    assert len(asked) == 2
