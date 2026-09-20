from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from backend.application.ports.rate_card_provider import GpuQuote, TelephonyQuote
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
        self._carriers, telephony = _carriers(raw)
        self._card = _parse(raw, telephony)
        self._quotes = [
            GpuQuote(q["provider"], q["sku"], q.get("region", ""), _price(q, "hourly_usd"))
            for q in raw.get("selfhosted", {}).get("gpu_client_eu", [])
        ]
        self._raw = raw

    def rate_card(self) -> RateCard:
        return self._card

    def client_gpu_quotes(self) -> list[GpuQuote]:
        return list(self._quotes)

    def telephony_quotes(self) -> list[TelephonyQuote]:
        return list(self._carriers)

    def gpu_sku(self) -> str:
        gpu = self._raw["selfhosted"]["gpu"]
        return f"{gpu.get('provider', '')} {gpu.get('sku', '')} {gpu.get('region', '')}".strip()

    def telephony_provider(self) -> str:
        return next(q.carrier for q in self._carriers if q.selected)

    def raw(self) -> dict[str, Any]:
        return dict(self._raw)


def _parse(raw: dict[str, Any], telephony: Decimal) -> RateCard:
    try:
        api = raw["api"]
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


def _carriers(raw: dict[str, Any]) -> tuple[list[TelephonyQuote], Decimal]:
    """Every listed carrier, and the selected carrier's price per minute."""
    node = raw.get("telephony")
    if not isinstance(node, dict) or not isinstance(node.get("carriers"), dict):
        raise RateCardError("rates.yaml telephony needs a carriers mapping and a selected carrier")
    selected = node.get("selected")
    quotes = [
        _carrier(str(name), entry, name == selected) for name, entry in node["carriers"].items()
    ]
    chosen = [q for q in quotes if q.selected]
    if not chosen:
        raise RateCardError(f"selected telephony carrier {selected!r} is not listed")
    price = chosen[0].per_minute_usd
    if price is None:
        raise RateCardError(f"selected telephony carrier {selected!r} has no price_usd")
    return quotes, price


def _carrier(name: str, entry: Any, selected: bool) -> TelephonyQuote:
    where = f"telephony carrier {name!r}"
    if not isinstance(entry, dict):
        raise RateCardError(f"{where} must be a mapping")
    if entry.get("unit") != "minute":
        raise RateCardError(f"{where}: unsupported unit {entry.get('unit')!r}")
    missing = [f for f in ("source", "checked_on") if not entry.get(f)]
    if missing or not isinstance(entry.get("verified"), bool):
        raise RateCardError(f"{where}: needs {', '.join(missing or ['verified (true/false)'])}")
    priced = entry.get("price_usd") is not None
    if not priced and not entry.get("note"):
        raise RateCardError(f"{where}: without price_usd, note must say why")
    return TelephonyQuote(
        carrier=name,
        per_minute_usd=_price(entry, "price_usd") if priced else None,
        source_url=str(entry["source"]),
        checked_on=str(entry["checked_on"]),
        verified=entry["verified"],
        selected=selected,
        note=str(entry.get("note", "")),
    )


def _price(node: dict[str, Any], key: str) -> Decimal:
    value = node[key]
    price = Decimal(str(value))
    if price < 0:
        raise RateCardError(f"negative price for {key}: {value}")
    return price
