from __future__ import annotations

import asyncio
from collections import defaultdict, deque

from backend.application.ports.metrics_sink import CallMetricsEvent


class InMemoryMetricsSink:
    """Fan-out of live call events to SSE subscribers, with a short replay buffer so a
    browser that connects a moment after the call starts still sees the greeting turn."""

    def __init__(self, replay: int = 200, subscriber_queue: int = 500) -> None:
        self._history: dict[str, deque[CallMetricsEvent]] = defaultdict(
            lambda: deque(maxlen=replay)
        )
        self._subscribers: dict[str, set[asyncio.Queue[CallMetricsEvent]]] = defaultdict(set)
        self._queue_size = subscriber_queue

    async def emit(self, event: CallMetricsEvent) -> None:
        self._history[event.call_id].append(event)
        for queue in list(self._subscribers[event.call_id]):
            if queue.full():
                # A stalled browser must not back-pressure the call; drop its oldest event.
                queue.get_nowait()
            queue.put_nowait(event)

    def subscribe(self, call_id: str) -> asyncio.Queue[CallMetricsEvent]:
        queue: asyncio.Queue[CallMetricsEvent] = asyncio.Queue(maxsize=self._queue_size)
        for event in self._history.get(call_id, ()):
            queue.put_nowait(event)
        self._subscribers[call_id].add(queue)
        return queue

    def unsubscribe(self, call_id: str, queue: asyncio.Queue[CallMetricsEvent]) -> None:
        self._subscribers[call_id].discard(queue)

    def history(self, call_id: str) -> list[CallMetricsEvent]:
        return list(self._history.get(call_id, ()))

    def latest_call_id(self) -> str | None:
        return next(reversed(self._history), None) if self._history else None
