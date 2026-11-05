from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, JSON, UniqueConstraint, ForeignKey, Index, func
from sqlalchemy.orm import Mapped, mapped_column
from app.db.models import Base, Vector

class MemoryTurn(Base):
    __tablename__ = "memory_turns"
    turn_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(100), unique=True)
    conversation_id: Mapped[str] = mapped_column(String(100), index=True)
    messages: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="in_progress")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    window_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieval_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_embedding: Mapped[list | None] = mapped_column(Vector(1024) if Vector else JSON, nullable=True)
    embedding_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    index_status: Mapped[str] = mapped_column(String(20), default="pending_window")
    memory_epoch: Mapped[int] = mapped_column(Integer, default=0)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # User-facing soft delete: hidden turns keep messages/response/facts on disk
    # but disappear from every user-facing read path. ``deleted_at`` remains the
    # physical purge marker used by ``delete_memory``.
    hidden_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    response: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Stable execution plan/operation state for retried requests (P1-2).
    operations: Mapped[dict | None] = mapped_column(JSON, nullable=True)

class UserFact(Base):
    __tablename__ = "user_facts"
    fact_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    field: Mapped[str] = mapped_column(String(160), index=True)
    value: Mapped[str] = mapped_column(Text)
    source_turn_id: Mapped[str] = mapped_column(String(36), index=True)
    source_message_id: Mapped[str] = mapped_column(String(100))
    source_quote: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="active")
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    memory_epoch: Mapped[int] = mapped_column(Integer, default=0)

class MemoryOutbox(Base):
    __tablename__ = "memory_outbox"
    __table_args__ = (UniqueConstraint("source_event_id", "op", name="uq_memory_event_op"),)
    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_event_id: Mapped[str] = mapped_column(String(120))
    turn_id: Mapped[str] = mapped_column(String(36), index=True)
    op: Mapped[str] = mapped_column(String(20), default="index")
    content_hash: Mapped[str] = mapped_column(String(64))
    memory_epoch: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)


class MemoryState(Base):
    """Derived context versions; permanent facts remain authoritative in UserFact."""
    __tablename__ = 'memory_states'
    __table_args__ = (
        UniqueConstraint('owner_id', 'conversation_id', 'scope', 'logical_key', 'memory_epoch', 'version', name='uq_memory_state_version'),
        Index('ix_memory_state_source', 'source_turn_id'),
        Index('ix_memory_state_scope', 'owner_id', 'conversation_id', 'scope', 'logical_key'),
        Index('ix_memory_state_order', 'owner_id', 'memory_epoch', 'version'),
    )
    state_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(100))
    conversation_id: Mapped[str] = mapped_column(String(100), default='')
    scope: Mapped[str] = mapped_column(String(20))
    logical_key: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(20))
    domain: Mapped[str] = mapped_column(String(20), default='general')
    version: Mapped[int] = mapped_column(Integer)
    value: Mapped[str] = mapped_column(Text, default='')
    status: Mapped[str] = mapped_column(String(20), default='active')
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    memory_epoch: Mapped[int] = mapped_column(Integer)
    source_turn_id: Mapped[str] = mapped_column(ForeignKey('memory_turns.turn_id', ondelete='CASCADE'))
    source_message_id: Mapped[str] = mapped_column(String(100))
    source_fact_id: Mapped[str | None] = mapped_column(ForeignKey('user_facts.fact_id', ondelete='CASCADE'), nullable=True)
