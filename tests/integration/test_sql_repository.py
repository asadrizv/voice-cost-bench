"""Runs against SQLite always, and against Postgres when TEST_DATABASE_URL is set (CI and
`make test` start one), so the adapter is proven on the dialect production uses."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import BINARY, VARBINARY, LargeBinary, MetaData, TypeDecorator, text

from backend.application.ports.call_repository import CallNotFound
from backend.domain.entities.call import Call, CallStatus, Turn
from backend.domain.entities.cost import CostBreakdown, Money, UsageUnits
from backend.domain.entities.latency import LatencyBreakdown
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.persistence.inmemory_call_repository import InMemoryCallRepository
from backend.infrastructure.persistence.models import Base
from backend.infrastructure.persistence.postgres_call_repository import SqlCallRepository

T0 = datetime(2026, 9, 18, 10, 0, tzinfo=UTC)
BACKENDS = ["sqlite", "memory"] + (["postgres"] if os.environ.get("TEST_DATABASE_URL") else [])


@pytest.fixture(params=BACKENDS)
async def repo(request: pytest.FixtureRequest, tmp_path) -> AsyncIterator[object]:  # type: ignore[no-untyped-def]
    if request.param == "memory":
        yield InMemoryCallRepository()
        return
    if request.param == "sqlite":
        r = SqlCallRepository.from_url(f"sqlite+aiosqlite:///{tmp_path / 'calls.db'}")
        await r.create_schema()
    else:
        r = SqlCallRepository.from_url(os.environ["TEST_DATABASE_URL"])
        async with r._engine.begin() as conn:  # noqa: SLF001
            await conn.execute(text("TRUNCATE turns, calls"))
    yield r
    await r.dispose()


def make_call(call_id: str, pipeline: PipelineKind, minutes_ago: int = 0) -> Call:
    return Call(
        id=call_id,
        pipeline=pipeline,
        persona="law_firm",
        started_at=T0 - timedelta(minutes=minutes_ago),
    )


def make_turn(index: int, llm_cost: str = "0.0001234567") -> Turn:
    return Turn(
        index=index,
        user_text="Hello" if index else "",
        agent_text="Good morning.",
        usage=UsageUnits(
            stt_audio_seconds=3.5,
            llm_input_tokens=420,
            llm_output_tokens=31,
            tts_characters=13,
            gpu_seconds=4.0,
            concurrent_calls=2.5,
            telephony_seconds=4.0,
        ),
        cost=CostBreakdown(llm=Money.of(llm_cost), gpu=Money.of("0.0000484444")),
        latency=None if index == 0 else LatencyBreakdown(250, 40, 180, 600, 90, 380, 630),
        interrupted=index == 2,
        started_at=T0,
    )


async def test_round_trip_preserves_every_field(repo) -> None:  # type: ignore[no-untyped-def]
    call = make_call("c1", PipelineKind.SELFHOSTED)
    await repo.add(call)
    for i in range(3):
        await repo.add_turn("c1", make_turn(i))
    call.closing_usage = UsageUnits(gpu_seconds=1.5, concurrent_calls=3, telephony_seconds=1.5)
    call.closing_cost = CostBreakdown(gpu=Money.of("0.0000151389"))
    call.complete(T0 + timedelta(seconds=90))
    await repo.update(call)

    got = await repo.get("c1")
    assert got.status is CallStatus.COMPLETED
    assert got.ended_at == T0 + timedelta(seconds=90)
    assert got.pipeline is PipelineKind.SELFHOSTED
    assert [t.index for t in got.turns] == [0, 1, 2]
    assert got.turns[0].latency is None
    assert got.turns[1].latency == LatencyBreakdown(250, 40, 180, 600, 90, 380, 630)
    assert got.turns[2].interrupted
    assert got.turns[1].usage == make_turn(1).usage
    assert got.turns[1].cost.llm.amount == Decimal("0.0001234567")
    assert got.closing_usage.concurrent_calls == 3
    assert got.cost.total.amount == (
        Decimal("0.0001234567") * 3 + Decimal("0.0000484444") * 3 + Decimal("0.0000151389")
    )


async def test_add_turn_is_idempotent_per_index(repo) -> None:  # type: ignore[no-untyped-def]
    await repo.add(make_call("c1", PipelineKind.API))
    await repo.add_turn("c1", make_turn(1, "0.1"))
    await repo.add_turn("c1", make_turn(1, "0.2"))
    got = await repo.get("c1")
    assert len(got.turns) == 1 and got.turns[0].cost.llm == Money.of("0.2")


async def test_list_filters_and_orders_newest_first(repo) -> None:  # type: ignore[no-untyped-def]
    await repo.add(make_call("old", PipelineKind.API, minutes_ago=10))
    await repo.add(make_call("new", PipelineKind.API, minutes_ago=1))
    synthetic = make_call("load", PipelineKind.SELFHOSTED)
    synthetic.source = "loadtest"
    await repo.add(synthetic)

    assert [c.id for c in await repo.list()] == ["load", "new", "old"]
    assert [c.id for c in await repo.list(pipeline=PipelineKind.API)] == ["new", "old"]
    assert [c.id for c in await repo.list(source="loadtest")] == ["load"]
    assert [c.id for c in await repo.list(limit=1)] == ["load"]


async def test_total_spend_sums_turns_and_closing_per_pipeline(repo) -> None:  # type: ignore[no-untyped-def]
    api = make_call("a", PipelineKind.API)
    await repo.add(api)
    await repo.add_turn("a", make_turn(0, "0.5"))
    api.closing_cost = CostBreakdown(telephony=Money.of("0.25"))
    await repo.update(api)
    await repo.add(make_call("s", PipelineKind.SELFHOSTED))
    await repo.add_turn("s", make_turn(0, "9"))

    spent = await repo.total_spend_usd(PipelineKind.API)
    assert spent == pytest.approx(0.5 + 0.0000484444 + 0.25)


async def test_missing_call_raises(repo) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(CallNotFound):
        await repo.get("nope")
    with pytest.raises(CallNotFound):
        await repo.update(make_call("nope", PipelineKind.API))


def binary_columns(metadata: MetaData) -> list[str]:
    found = []
    for table in metadata.sorted_tables:
        for column in table.columns:
            kind = column.type
            if isinstance(kind, TypeDecorator):  # PickleType and friends store bytes
                kind = kind.impl_instance
            if isinstance(kind, LargeBinary | BINARY | VARBINARY):
                found.append(f"{table.name}.{column.name}")
    return found


def test_no_persisted_table_can_hold_audio() -> None:
    assert Base.metadata.sorted_tables
    assert binary_columns(Base.metadata) == []


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="needs the migrated Postgres")
async def test_the_migrated_schema_has_no_column_that_can_hold_audio() -> None:
    """Production runs the Alembic migrations, not the models, so a hand-written migration
    adding a bytea column would slip past the model check above."""
    r = SqlCallRepository.from_url(os.environ["TEST_DATABASE_URL"])
    async with r._engine.connect() as conn:  # noqa: SLF001
        tables = (
            await conn.execute(
                text("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'")
            )
        ).scalar_one()
        binary = (
            (
                await conn.execute(
                    text(
                        "SELECT table_name || '.' || column_name FROM information_schema.columns "
                        "WHERE table_schema = 'public' AND data_type = 'bytea'"
                    )
                )
            )
            .scalars()
            .all()
        )
    await r.dispose()
    assert tables >= 2
    assert binary == []
