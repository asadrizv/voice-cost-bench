from __future__ import annotations

import re
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.settings import REPO_ROOT
from backend.infrastructure.pricing.yaml_rate_card import RateCardError, YamlRateCardProvider

TWILIO = {
    "unit": "minute",
    "price_usd": 0.014,
    "source": "https://example.com/twilio",
    "checked_on": "2026-09-17",
    "verified": True,
}
TELNYX = {
    "unit": "minute",
    "price_usd": 0.0032,
    "source": "https://example.com/telnyx",
    "checked_on": "2026-09-19",
    "verified": False,
    "note": "inbound local",
}


def card_with(tmp_path: Path, telephony: dict[str, Any]) -> Path:
    raw = yaml.safe_load((REPO_ROOT / "config" / "rates.yaml").read_text())
    raw["telephony"] = telephony
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "rates.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_the_selected_carrier_prices_telephony_and_nothing_else(tmp_path: Path) -> None:
    carriers = {"twilio": TWILIO, "telnyx": TELNYX}
    on_twilio = YamlRateCardProvider(
        card_with(tmp_path / "a", {"selected": "twilio", "carriers": carriers})
    ).rate_card()
    on_telnyx = YamlRateCardProvider(
        card_with(tmp_path / "b", {"selected": "telnyx", "carriers": carriers})
    ).rate_card()

    for kind in PipelineKind:
        twilio_rates, telnyx_rates = on_twilio.for_pipeline(kind), on_telnyx.for_pipeline(kind)
        assert twilio_rates.telephony_per_minute == Decimal("0.014")
        assert telnyx_rates.telephony_per_minute == Decimal("0.0032")
        assert replace(telnyx_rates, telephony_per_minute=Decimal("0.014")) == twilio_rates
    assert on_telnyx.verified_on == on_twilio.verified_on


def test_every_carrier_is_listed_with_its_source_and_whether_it_was_verified(
    tmp_path: Path,
) -> None:
    unpriced = {**TELNYX, "note": "flat monthly channels"}
    del unpriced["price_usd"]
    carriers = {"twilio": TWILIO, "telnyx": TELNYX, "sipgate": unpriced}
    rates = YamlRateCardProvider(card_with(tmp_path, {"selected": "twilio", "carriers": carriers}))

    quotes = {q.carrier: q for q in rates.telephony_quotes()}

    assert set(quotes) == {"twilio", "telnyx", "sipgate"}
    assert [q.carrier for q in quotes.values() if q.selected] == ["twilio"]
    assert rates.telephony_provider() == "twilio"
    telnyx = quotes["telnyx"]
    assert (telnyx.per_minute_usd, telnyx.verified) == (Decimal("0.0032"), False)
    assert (telnyx.source_url, telnyx.checked_on) == ("https://example.com/telnyx", "2026-09-19")
    assert telnyx.note == "inbound local"
    assert quotes["twilio"].verified
    assert quotes["sipgate"].per_minute_usd is None
    assert quotes["sipgate"].note == "flat monthly channels"


def without(entry: dict[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in entry.items() if k != key}


@pytest.mark.parametrize(
    ("telephony", "named"),
    [
        ({"selected": "vonage", "carriers": {"twilio": TWILIO}}, "'vonage' is not listed"),
        (
            {
                "selected": "telnyx",
                "carriers": {"twilio": TWILIO, "telnyx": without(TELNYX, "price_usd")},
            },
            "'telnyx' has no price_usd",
        ),
        ({"provider": "twilio", "unit": "minute", "price_usd": 0.014}, "needs a carriers mapping"),
        ({"selected": "twilio", "carriers": {"twilio": "0.014"}}, "'twilio' must be a mapping"),
        (
            {"selected": "twilio", "carriers": {"twilio": {**TWILIO, "unit": "second"}}},
            "'twilio': unsupported unit 'second'",
        ),
        (
            {"selected": "twilio", "carriers": {"twilio": without(TWILIO, "source")}},
            "'twilio': needs source",
        ),
        (
            {"selected": "twilio", "carriers": {"twilio": without(TWILIO, "checked_on")}},
            "'twilio': needs checked_on",
        ),
        (
            {"selected": "twilio", "carriers": {"twilio": {**TWILIO, "verified": "yes"}}},
            "'twilio': needs verified (true/false)",
        ),
        (
            {"selected": "twilio", "carriers": {"twilio": {**TWILIO, "price_usd": -0.01}}},
            "negative price",
        ),
        (
            {
                "selected": "twilio",
                "carriers": {
                    "twilio": TWILIO,
                    "telnyx": without(without(TELNYX, "price_usd"), "note"),
                },
            },
            "'telnyx': without price_usd, note must say why",
        ),
    ],
)
def test_an_unknown_selected_carrier_or_malformed_entry_fails_at_load(
    tmp_path: Path, telephony: dict[str, Any], named: str
) -> None:
    with pytest.raises(RateCardError, match=re.escape(named)):
        YamlRateCardProvider(card_with(tmp_path, telephony))
