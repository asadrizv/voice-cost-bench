from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from backend.application.ports.call_repository import CallNotFound
from backend.domain.entities.call import Call, CallStatus, Turn
from backend.domain.entities.cost import CostBreakdown, Money, UsageUnits
from backend.domain.entities.latency import LatencyBreakdown
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.persistence.models import Base, CallRow, TurnRow, _UsageAndCost

_LATENCY_FIELDS = (
    "endpoint_detected",
    "stt_final",
    "llm_first_token",
    "llm_complete",
    "tts_first_byte",
    "end_to_end",
    "perceived_delay",
)


class SqlCallRepository:
    """Postgres in deployment; any SQLAlchemy async dialect works (tests use SQLite)."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def from_url(cls, url: str) -> SqlCallRepository:
        return cls(create_async_engine(_async_url(url), pool_pre_ping=True))

    async def create_schema(self) -> None:
        """For tests and SQLite only; Postgres is migrated by Alembic."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self._engine.dispose()

    async def add(self, call: Call) -> None:
        async with self._sessions.begin() as session:
            row = CallRow(
                id=call.id,
                pipeline=call.pipeline.value,
                persona=call.persona,
                source=call.source,
                status=call.status.value,
                started_at=call.started_at,
                ended_at=call.ended_at,
            )
            _write_usage(row, call.closing_usage, call.closing_cost)
            session.add(row)
            for turn in call.turns:
                session.add(_turn_row(call.id, turn))

    async def add_turn(self, call_id: str, turn: Turn) -> None:
        async with self._sessions.begin() as session:
            await session.merge(_turn_row(call_id, turn))

    async def update(self, call: Call) -> None:
        async with self._sessions.begin() as session:
            row = await session.get(CallRow, call.id)
            if row is None:
                raise CallNotFound(call.id)
            row.status = call.status.value
            row.ended_at = call.ended_at
            _write_usage(row, call.closing_usage, call.closing_cost)

    async def get(self, call_id: str) -> Call:
        async with self._sessions() as session:
            row = await session.get(CallRow, call_id, options=[selectinload(CallRow.turns)])
            if row is None:
                raise CallNotFound(call_id)
            return _to_call(row)

    async def list(
        self, limit: int = 50, pipeline: PipelineKind | None = None, source: str | None = None
    ) -> list[Call]:
        query = select(CallRow).options(selectinload(CallRow.turns))
        if pipeline is not None:
            query = query.where(CallRow.pipeline == pipeline.value)
        if source is not None:
            query = query.where(CallRow.source == source)
        query = query.order_by(CallRow.started_at.desc()).limit(limit)
        async with self._sessions() as session:
            rows = (await session.scalars(query)).all()
            return [_to_call(r) for r in rows]

    async def total_spend_usd(self, pipeline: PipelineKind) -> float:
        cost_cols = ("cost_stt", "cost_llm", "cost_tts", "cost_gpu", "cost_telephony")

        def total(model: type[_UsageAndCost]) -> object:
            return func.coalesce(func.sum(sum(getattr(model, c) for c in cost_cols)), 0)

        async with self._sessions() as session:
            closing = await session.scalar(
                select(total(CallRow)).where(CallRow.pipeline == pipeline.value)  # type: ignore[call-overload]
            )
            turns = await session.scalar(
                select(total(TurnRow))  # type: ignore[call-overload]
                .join(CallRow, TurnRow.call_id == CallRow.id)
                .where(CallRow.pipeline == pipeline.value)
            )
        return float(Decimal(str(closing or 0)) + Decimal(str(turns or 0)))


def _async_url(url: str) -> str:
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url.split("://", 1)[1]
    return url


def _write_usage(row: _UsageAndCost, usage: UsageUnits, cost: CostBreakdown) -> None:
    row.stt_audio_seconds = usage.stt_audio_seconds
    row.llm_input_tokens = usage.llm_input_tokens
    row.llm_output_tokens = usage.llm_output_tokens
    row.tts_characters = usage.tts_characters
    row.gpu_seconds = usage.gpu_seconds
    row.concurrent_calls = usage.concurrent_calls
    row.telephony_seconds = usage.telephony_seconds
    row.cost_stt = cost.stt.amount
    row.cost_llm = cost.llm.amount
    row.cost_tts = cost.tts.amount
    row.cost_gpu = cost.gpu.amount
    row.cost_telephony = cost.telephony.amount


def _read_usage(row: _UsageAndCost) -> tuple[UsageUnits, CostBreakdown]:
    usage = UsageUnits(
        stt_audio_seconds=row.stt_audio_seconds,
        llm_input_tokens=row.llm_input_tokens,
        llm_output_tokens=row.llm_output_tokens,
        tts_characters=row.tts_characters,
        gpu_seconds=row.gpu_seconds,
        concurrent_calls=row.concurrent_calls,
        telephony_seconds=row.telephony_seconds,
    )
    cost = CostBreakdown(
        stt=_money(row.cost_stt),
        llm=_money(row.cost_llm),
        tts=_money(row.cost_tts),
        gpu=_money(row.cost_gpu),
        telephony=_money(row.cost_telephony),
    )
    return usage, cost


def _money(value: Decimal | float | None) -> Money:
    return Money(Decimal(str(value or 0)))


def _turn_row(call_id: str, turn: Turn) -> TurnRow:
    row = TurnRow(
        call_id=call_id,
        index=turn.index,
        user_text=turn.user_text,
        agent_text=turn.agent_text,
        interrupted=turn.interrupted,
        started_at=turn.started_at,
    )
    _write_usage(row, turn.usage, turn.cost)
    if turn.latency is not None:
        for name in _LATENCY_FIELDS:
            setattr(row, f"{name}_ms", getattr(turn.latency, name))
    return row


def _to_call(row: CallRow) -> Call:
    closing_usage, closing_cost = _read_usage(row)
    call = Call(
        id=row.id,
        pipeline=PipelineKind(row.pipeline),
        persona=row.persona,
        started_at=_aware(row.started_at),
        status=CallStatus(row.status),
        ended_at=_aware(row.ended_at) if row.ended_at else None,
        closing_usage=closing_usage,
        closing_cost=closing_cost,
        source=row.source,
    )
    for t in row.turns:
        usage, cost = _read_usage(t)
        latency = None
        if t.perceived_delay_ms is not None:
            latency = LatencyBreakdown(**{n: getattr(t, f"{n}_ms") or 0.0 for n in _LATENCY_FIELDS})
        call.turns.append(
            Turn(
                index=t.index,
                user_text=t.user_text,
                agent_text=t.agent_text,
                usage=usage,
                cost=cost,
                latency=latency,
                interrupted=t.interrupted,
                started_at=_aware(t.started_at) if t.started_at else None,
            )
        )
    return call


def _aware(value: datetime) -> datetime:
    """SQLite drops tzinfo; everything in the domain is UTC-aware."""
    return value if value.tzinfo else value.replace(tzinfo=UTC)
