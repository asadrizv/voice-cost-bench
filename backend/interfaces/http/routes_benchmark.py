from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from backend.infrastructure.config.settings import REPO_ROOT, Settings
from backend.interfaces.container import Container
from backend.interfaces.http.deps import container, settings

router = APIRouter()
RESULTS = REPO_ROOT / "results"


@router.get("/benchmark")
def benchmark(name: str = "benchmark.json") -> dict[str, Any]:
    path = (RESULTS / name).resolve()
    if path.parent != RESULTS.resolve() or not path.is_file():
        raise HTTPException(404, "no benchmark yet: run `make benchmark`")
    data: dict[str, Any] = json.loads(path.read_text())
    return data


@router.get("/benchmark/runs")
def runs() -> list[str]:
    return sorted(p.name for p in RESULTS.glob("*.json")) if RESULTS.is_dir() else []


@router.get("/config")
async def config(
    c: Annotated[Container, Depends(container)], s: Annotated[Settings, Depends(settings)]
) -> dict[str, Any]:
    from backend.domain.value_objects.pipeline_kind import PipelineKind
    from backend.interfaces.http.serializers import rates, telephony_quote

    card = c.calculator.rate_card
    return {
        "default_pipeline": s.pipeline.value,
        "selectable_pipelines": [k.value for k in s.selectable_pipelines],
        "eu_only": s.eu_only,
        "personas": c.personas.available(),
        "endpointer": s.endpointer.value,
        "rate_card_verified_on": card.verified_on,
        "rates": {k.value: rates(card.for_pipeline(k)) for k in PipelineKind},
        "client_gpu_quotes": [
            {
                "provider": q.provider,
                "sku": q.sku,
                "region": q.region,
                "hourly_usd": float(q.hourly_usd),
            }
            for q in c.rates.client_gpu_quotes()
        ],
        "telephony_quotes": [telephony_quote(q) for q in c.rates.telephony_quotes()],
        "api_spend_usd": await c.repository.total_spend_usd(PipelineKind.API),
        "api_spend_limit_usd": s.dev_spend_limit_usd,
        "selfhosted_ceiling": s.selfhosted_max_concurrency,
        "selfhosted_on_local_machine": s.selfhosted_on_local_machine,
        "simulated": s.simulate_providers,
    }
