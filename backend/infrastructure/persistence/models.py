from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Per-token prices reach 1e-7 USD; 10 decimal places keep sums exact.
Usd = Numeric(18, 10)


class Base(DeclarativeBase):
    pass


class _UsageAndCost:
    stt_audio_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    llm_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    tts_characters: Mapped[int] = mapped_column(Integer, default=0)
    gpu_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    concurrent_calls: Mapped[float] = mapped_column(Float, default=1.0)
    telephony_seconds: Mapped[float] = mapped_column(Float, default=0.0)

    cost_stt: Mapped[Decimal] = mapped_column(Usd, default=Decimal(0))
    cost_llm: Mapped[Decimal] = mapped_column(Usd, default=Decimal(0))
    cost_tts: Mapped[Decimal] = mapped_column(Usd, default=Decimal(0))
    cost_gpu: Mapped[Decimal] = mapped_column(Usd, default=Decimal(0))
    cost_telephony: Mapped[Decimal] = mapped_column(Usd, default=Decimal(0))


class CallRow(_UsageAndCost, Base):
    """Usage and cost columns here hold the closing segment only; turns hold the rest."""

    __tablename__ = "calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    pipeline: Mapped[str] = mapped_column(String(16), index=True)
    persona: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(16), default="browser", index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    turns: Mapped[list[TurnRow]] = relationship(
        back_populates="call", order_by="TurnRow.index", cascade="all, delete-orphan"
    )


class TurnRow(_UsageAndCost, Base):
    __tablename__ = "turns"

    call_id: Mapped[str] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), primary_key=True
    )
    index: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_text: Mapped[str] = mapped_column(Text, default="")
    agent_text: Mapped[str] = mapped_column(Text, default="")
    interrupted: Mapped[bool] = mapped_column(Boolean, default=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    endpoint_detected_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    stt_final_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_first_token_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_complete_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tts_first_byte_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_to_end_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    perceived_delay_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    call: Mapped[CallRow] = relationship(back_populates="turns")
