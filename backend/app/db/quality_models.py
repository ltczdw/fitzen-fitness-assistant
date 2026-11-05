"""Local request diagnostics, feedback and review audit records."""
from datetime import datetime
from sqlalchemy import String, Text, Integer, DateTime, Float, Boolean, JSON, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column
from app.db.models import Base


class ChatRun(Base):
    __tablename__ = "chat_runs"
    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(100), index=True)
    message_id: Mapped[str] = mapped_column(String(160), index=True)
    conversation_id: Mapped[str] = mapped_column(String(100), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    intent: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(30))
    metrics: Mapped[dict] = mapped_column(JSON)
    citations: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class FailureCase(Base):
    __tablename__ = "failure_cases"
    case_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("chat_runs.run_id"), unique=True)
    category: Mapped[str] = mapped_column(String(80))
    reasons: Mapped[list] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    approved_document_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    review_digest: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AnswerFeedback(Base):
    __tablename__ = "answer_feedback"
    feedback_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("chat_runs.run_id"), unique=True)
    rating: Mapped[str] = mapped_column(String(10))
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReviewAudit(Base):
    __tablename__ = "review_audits"
    audit_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("failure_cases.case_id"), index=True)
    decision: Mapped[str] = mapped_column(String(20))
    note: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer: Mapped[str] = mapped_column(String(50), default="local-user")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
