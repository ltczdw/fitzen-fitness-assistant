"""C1 v2 execution ledger tables (new tables only).

These four tables are additive: they do not modify, reference or cascade into
any legacy business or memory table. They are registered on the existing
``app.db.models.Base`` so ``Base.metadata.create_all`` creates them during the
normal initialization transaction; shape verification lives in
``app.db.execution_ledger_schema``.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKeyConstraint, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models import Base


class ExecutionRequestModel(Base):
    __tablename__ = "agent_execution_requests"

    user_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(Text)
    source_digest: Mapped[str] = mapped_column(String(64))
    requested_mode: Mapped[str] = mapped_column(String(32))
    effective_mode: Mapped[str] = mapped_column(String(32))
    schema_version: Mapped[int] = mapped_column(Integer)
    tool_version: Mapped[str] = mapped_column(String(64))
    time_base: Mapped[str] = mapped_column(String(64))
    budget_scope_id: Mapped[str] = mapped_column(String(128))
    request_kind: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    revision: Mapped[int] = mapped_column(Integer, default=0)
    lease_worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lease_deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    checkpoint: Mapped[dict] = mapped_column(JSON)
    response: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ExecutionTaskModel(Base):
    __tablename__ = "agent_execution_tasks"

    user_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_start: Mapped[int] = mapped_column(Integer)
    source_end: Mapped[int] = mapped_column(Integer)
    source_digest: Mapped[str] = mapped_column(String(64))
    task_type: Mapped[str] = mapped_column(String(64))
    dependencies: Mapped[list] = mapped_column(JSON)
    parameters: Mapped[dict] = mapped_column(JSON)
    required: Mapped[bool] = mapped_column(Boolean)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    current_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["user_id", "request_id"],
            ["agent_execution_requests.user_id", "agent_execution_requests.request_id"],
            name="fk_agent_exec_tasks_request",
        ),
    )


class ExecutionOperationModel(Base):
    __tablename__ = "agent_execution_operations"

    user_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    op_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(150), unique=True)
    task_id: Mapped[str] = mapped_column(String(128))
    position: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(64))
    parameters: Mapped[dict] = mapped_column(JSON)
    parameters_digest: Mapped[str] = mapped_column(String(64))
    source_digest: Mapped[str] = mapped_column(String(64))
    time_base: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="reserved")
    confirmation_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ["user_id", "request_id", "task_id"],
            ["agent_execution_tasks.user_id", "agent_execution_tasks.request_id", "agent_execution_tasks.task_id"],
            name="fk_agent_exec_ops_task",
        ),
        UniqueConstraint("user_id", "request_id", "task_id", "position", name="uq_agent_exec_ops_position"),
    )


class ExecutionEventModel(Base):
    __tablename__ = "agent_execution_events"

    user_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    sequence: Mapped[int] = mapped_column(Integer)
    event_key: Mapped[str] = mapped_column(String(128))
    tool_call_id: Mapped[str] = mapped_column(String(128))
    batch_id: Mapped[str] = mapped_column(String(128))
    task_id: Mapped[str] = mapped_column(String(128))
    op_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    phase: Mapped[str] = mapped_column(String(64))
    observation: Mapped[dict] = mapped_column(JSON)
    checkpoint: Mapped[dict] = mapped_column(JSON)
    task_status: Mapped[str] = mapped_column(String(32))
    operation_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confirmed: Mapped[bool] = mapped_column(Boolean)

    __table_args__ = (
        ForeignKeyConstraint(
            ["user_id", "request_id", "task_id"],
            ["agent_execution_tasks.user_id", "agent_execution_tasks.request_id", "agent_execution_tasks.task_id"],
            name="fk_agent_exec_events_task",
        ),
        ForeignKeyConstraint(
            ["user_id", "request_id", "op_id"],
            ["agent_execution_operations.user_id", "agent_execution_operations.request_id", "agent_execution_operations.op_id"],
            name="fk_agent_exec_events_operation",
        ),
        UniqueConstraint("user_id", "request_id", "sequence", name="uq_agent_exec_events_sequence"),
        UniqueConstraint("user_id", "request_id", "event_key", name="uq_agent_exec_events_key"),
    )


LEDGER_MODELS = (
    ExecutionRequestModel,
    ExecutionTaskModel,
    ExecutionOperationModel,
    ExecutionEventModel,
)
