"""Durable provider quotas, attempt audit and scoped successful-result cache."""
from datetime import datetime
from sqlalchemy import String, Integer, BigInteger, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.db.models import Base


class ModelQuota(Base):
    __tablename__='model_quotas'
    quota_id: Mapped[str]=mapped_column(String(80),primary_key=True)
    kind: Mapped[str]=mapped_column(String(24))
    calls: Mapped[int]=mapped_column(Integer,default=0)
    tokens: Mapped[int]=mapped_column(BigInteger,default=0)
    max_calls: Mapped[int]=mapped_column(Integer)
    max_tokens: Mapped[int]=mapped_column(BigInteger)
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime]=mapped_column(DateTime(timezone=True))


class ModelCallLedger(Base):
    __tablename__='model_call_ledger'
    call_id: Mapped[str]=mapped_column(String(36),primary_key=True)
    scope_id: Mapped[str]=mapped_column(String(80),index=True)
    kind: Mapped[str]=mapped_column(String(24))
    capability: Mapped[str]=mapped_column(String(20))
    model: Mapped[str]=mapped_column(String(100))
    stage_id: Mapped[str]=mapped_column(String(80))
    quota_ids: Mapped[list]=mapped_column(JSON)
    reserved_tokens: Mapped[int]=mapped_column(BigInteger)
    status: Mapped[str]=mapped_column(String(24),default='started')
    usage: Mapped[dict | None]=mapped_column(JSON,nullable=True)
    error_code: Mapped[str | None]=mapped_column(String(80),nullable=True)
    started_at: Mapped[datetime]=mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None]=mapped_column(DateTime(timezone=True),nullable=True)


class ModelResultCache(Base):
    __tablename__='model_result_cache'
    cache_id: Mapped[str]=mapped_column(String(64),primary_key=True)
    scope_id: Mapped[str]=mapped_column(String(80),index=True)
    capability: Mapped[str]=mapped_column(String(20))
    owner_id: Mapped[str | None]=mapped_column(String(64),nullable=True,index=True)
    result: Mapped[dict]=mapped_column(JSON)
    created_at: Mapped[datetime]=mapped_column(DateTime(timezone=True))
