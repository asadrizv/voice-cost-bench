from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from backend.application.ports.rate_card_provider import GpuQuote
from backend.domain.services.cost_calculator import PipelineRates, RateCard
from backend.domain.value_objects.pipeline_kind import PipelineKind


class RateCardError(ValueError):
    pass


class YamlRateCardProvider:
    """Reads config/rates.yaml once. A malformed or incomplete card fails at startup rather
    than silently pricing a component at zero."""

    def __init__(self, path: Path) -> None:
        self._path = path
        raw = yaml.safe_load(path.read_text())
        self._card = _parse(raw)
        self._quotes = [
            GpuQuote(q["provider"], q["sku"], q.get("region", ""), _price(q, "hourly_usd"))
            for q in raw.get("selfhosted", {}).get("gpu_client_eu", [])
        ]
        self._raw = raw

    def rate_card(self) -> RateCard:
        return self._card

    def client_gpu_quotes(self) -> list[GpuQuote]:
        return list(self._quotes)

    def gpu_sku(self) -> str:
        gpu = self._raw["selfhosted"]["gpu"]
        return f"{gpu.get('provider', '')} {gpu.get('sku', '')} {gpu.get('region', '')}".strip()

    def telephony_provider(self) -> str:
        return str(self._raw["telephony"]["provider"])

    def raw(self) -> dict[str, Any]:
        return dict(self._raw)


def _parse(raw: dict[str, Any]) -> RateCard:
    try:
        api = raw["api"]
        telephony = _price(raw["telephony"], "price_usd")
        stt_unit = api["stt"]["unit"]
        tts_unit = api["tts"]["unit"]
        if stt_unit != "audio_minute" or tts_unit != "1k_characters":
            raise RateCardError(f"unsupported units: stt={stt_unit} tts={tts_unit}")
        api_rates = PipelineRates(
            stt_per_minute=_price(api["stt"], "price_usd"),
            llm_input_per_1k=_price(api["llm"], "input_per_1k_usd"),
            llm_output_per_1k=_price(api["llm"], "output_per_1k_usd"),
            tts_per_1k_chars=_price(api["tts"], "price_usd"),
            telephony_per_minute=telephony,
        )
        selfhosted_rates = PipelineRates(
            gpu_per_hour=_price(raw["selfhosted"]["gpu"], "hourly_usd"),
            telephony_per_minute=telephony,
        )
        return RateCard(
            verified_on=str(raw["verified_on"]),
            rates={PipelineKind.API: api_rates, PipelineKind.SELFHOSTED: selfhosted_rates},
        )
    except KeyError as exc:
        raise RateCardError(f"rates.yaml is missing {exc}") from exc


def _price(node: dict[str, Any], key: str) -> Decimal:
    value = node[key]
    price = Decimal(str(value))
    if price < 0:
        raise RateCardError(f"negative price for {key}: {value}")
    return price
