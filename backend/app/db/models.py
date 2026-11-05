from datetime import date, datetime

from sqlalchemy import BigInteger, Boolean, Date, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = None


class Base(DeclarativeBase):
    pass


class WorkoutRecordModel(Base):
    __tablename__ = "workout_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(100), index=True, default="demo-user")
    workout_date: Mapped[date] = mapped_column(Date, index=True)
    exercise: Mapped[str] = mapped_column(String(200))
    sets: Mapped[int] = mapped_column(Integer)
    reps: Mapped[int] = mapped_column(Integer)
    weight_kg: Mapped[float] = mapped_column(Float)
    difficulty: Mapped[str] = mapped_column(String(20), default="未记录")
    completed: Mapped[bool] = mapped_column(Boolean, default=True)
    note: Mapped[str] = mapped_column(Text, default="")
    set_details: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(150), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserProfileModel(Base):
    __tablename__ = "user_profiles"
    user_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    memory_epoch: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    display_name: Mapped[str] = mapped_column(String(100), default="我的训练")
    goal: Mapped[str] = mapped_column(String(100), default="保持健康")
    experience: Mapped[str] = mapped_column(String(100), default="初学者")
    available_days: Mapped[int] = mapped_column(Integer, default=3)
    body_weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    height_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    body_fat_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class NutritionLogModel(Base):
    __tablename__ = "nutrition_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(100), index=True, default="demo-user")
    log_date: Mapped[date] = mapped_column(Date, index=True)
    meal_type: Mapped[str] = mapped_column(String(50), default="其他")
    description: Mapped[str] = mapped_column(Text)
    calories: Mapped[float | None] = mapped_column(Float, nullable=True)
    protein_g: Mapped[float | None] = mapped_column(Float, nullable=True)
    carbs_g: Mapped[float | None] = mapped_column(Float, nullable=True)
    fat_g: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TrainingPlanModel(Base):
    __tablename__ = "training_plans"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    plan_name: Mapped[str] = mapped_column(String(200), default="基础全身训练")
    goal: Mapped[str] = mapped_column(String(100), default="保持健康")
    days_per_week: Mapped[int] = mapped_column(Integer, default=3)
    plan_json: Mapped[str] = mapped_column(Text, default="{}")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class KnowledgeIndexVersionModel(Base):
    __tablename__ = "knowledge_index_versions"
    corpus: Mapped[str] = mapped_column(String(50), primary_key=True)
    revision: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")


class KnowledgeDocumentModel(Base):
    __tablename__ = "knowledge_documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(300))
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(500), default="seed")
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KnowledgeChunkModel(Base):
    __tablename__ = "knowledge_chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(300), index=True)
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(500), default="seed")
    document_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    parent_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=1)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(256) if Vector else Text, nullable=True)
    semantic_embedding: Mapped[list[float] | None] = mapped_column(Vector(1024) if Vector else Text, nullable=True)
    embedding_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
