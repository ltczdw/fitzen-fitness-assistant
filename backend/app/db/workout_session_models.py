from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models import Base


class WorkoutSessionModel(Base):
    __tablename__ = "workout_sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(100), index=True)
    source_record_id: Mapped[int | None] = mapped_column(ForeignKey("workout_records.id", ondelete="SET NULL"), unique=True, nullable=True)
    session_date: Mapped[date] = mapped_column(Date, index=True)
    title: Mapped[str] = mapped_column(String(200), default="训练会话")
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class WorkoutExerciseModel(Base):
    __tablename__ = "workout_session_exercises"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("workout_sessions.id", ondelete="CASCADE"), index=True)
    exercise_name: Mapped[str] = mapped_column(String(200))
    difficulty: Mapped[str] = mapped_column(String(20), default="未记录")
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    note: Mapped[str] = mapped_column(Text, default="")
    # R012: the persistent binding between a source record and the one action
    # that represents it inside the session. Name/order never identify it.
    source_record_id: Mapped[int | None] = mapped_column(ForeignKey("workout_records.id", ondelete="SET NULL"), unique=True, nullable=True)


class WorkoutSetModel(Base):
    __tablename__ = "workout_session_sets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exercise_id: Mapped[int] = mapped_column(ForeignKey("workout_session_exercises.id", ondelete="CASCADE"), index=True)
    set_index: Mapped[int] = mapped_column(Integer, default=0)
    weight_kg: Mapped[float] = mapped_column(Float, default=0.0)
    reps: Mapped[int] = mapped_column(Integer)
    distance_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    difficulty: Mapped[str] = mapped_column(String(20), default="未记录")
    completed: Mapped[bool] = mapped_column(Boolean, default=True)
    note: Mapped[str] = mapped_column(Text, default="")


class UserCustomExerciseModel(Base):
    __tablename__ = "user_custom_exercises"
    __table_args__ = (UniqueConstraint("user_id", "exercise_name", name="uq_user_custom_exercise_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(100), index=True)
    exercise_name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(80), default="自定义")
    target_muscles: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
