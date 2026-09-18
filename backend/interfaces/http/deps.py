from __future__ import annotations

from fastapi import Request

from backend.infrastructure.config.settings import Settings
from backend.infrastructure.telemetry.inmemory_metrics_sink import InMemoryMetricsSink
from backend.infrastructure.telemetry.prometheus_metrics_sink import PrometheusMetricsSink
from backend.interfaces.container import Container


def container(request: Request) -> Container:
    c: Container = request.app.state.container
    return c


def settings(request: Request) -> Settings:
    s: Settings = request.app.state.settings
    return s


def live(request: Request) -> InMemoryMetricsSink:
    sink: InMemoryMetricsSink = request.app.state.live
    return sink


def prometheus(request: Request) -> PrometheusMetricsSink:
    sink: PrometheusMetricsSink = request.app.state.prometheus
    return sink
