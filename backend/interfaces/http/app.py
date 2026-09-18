from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.application.ports.pipeline_provider import PipelineProvider
from backend.infrastructure.config.settings import Settings, get_settings
from backend.infrastructure.persistence.postgres_call_repository import SqlCallRepository
from backend.infrastructure.telemetry.composite_metrics_sink import CompositeMetricsSink
from backend.infrastructure.telemetry.inmemory_metrics_sink import InMemoryMetricsSink
from backend.infrastructure.telemetry.prometheus_metrics_sink import PrometheusMetricsSink
from backend.interfaces.container import build_container
from backend.interfaces.http import (
    routes_benchmark,
    routes_calls,
    routes_health,
    routes_metrics,
    routes_token,
)


def create_app(
    settings: Settings | None = None, pipelines: PipelineProvider | None = None
) -> FastAPI:
    settings = settings or get_settings()
    live = InMemoryMetricsSink()
    prometheus = PrometheusMetricsSink()
    container = build_container(
        settings, CompositeMetricsSink([live, prometheus]), pipelines=pipelines
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        if isinstance(container.repository, SqlCallRepository):
            await container.repository.dispose()

    app = FastAPI(title="Voice Cost Bench", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.container = container
    app.state.live = live
    app.state.prometheus = prometheus
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for module in (routes_token, routes_calls, routes_metrics, routes_benchmark, routes_health):
        app.include_router(module.router)
    return app


def app() -> FastAPI:
    """uvicorn --factory entry point."""
    return create_app()
