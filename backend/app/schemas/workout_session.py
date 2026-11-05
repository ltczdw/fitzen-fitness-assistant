from app.core import clock
from datetime import date

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class WorkoutSetPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int | None = Field(default=None, ge=1)
    weight_kg: float = Field(default=0, ge=0, le=1000)
    reps: int = Field(default=0, ge=0, le=1000)
    distance_km: float | None = Field(default=None, ge=0, le=10000, allow_inf_nan=False)
    duration_minutes: float | None = Field(default=None, gt=0, le=10080, allow_inf_nan=False)
    difficulty: Literal["未记录", "轻松", "正常", "困难"] = "未记录"
    completed: bool = True
    note: str = Field(default="", max_length=1000)

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str) -> str:
        return value.strip()


    @model_validator(mode="after")
    def check_measurement_units(self):
        cardio = self.distance_km is not None or self.duration_minutes is not None
        if cardio:
            if self.distance_km is None or self.duration_minutes is None:
                raise ValueError("有氧训练请填写距离（公里）和时间（分钟）")
            if self.weight_kg != 0 or self.reps != 0:
                raise ValueError("有氧训练不能同时填写重量或次数")
        elif self.reps < 1:
            raise ValueError("力量训练次数至少为1")
        return self


class WorkoutExercisePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int | None = Field(default=None, ge=1)
    exercise_name: str = Field(min_length=1, max_length=200)
    difficulty: Literal["未记录", "轻松", "正常", "困难"] = "未记录"
    note: str = Field(default="", max_length=1000)
    sets: list[WorkoutSetPayload] = Field(min_length=1, max_length=100)

    @field_validator("exercise_name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("exercise_name cannot be empty")
        return value

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str) -> str:
        return value.strip()


    @model_validator(mode="after")
    def consistent_measurements(self):
        if len({item.duration_minutes is not None for item in self.sets}) != 1:
            raise ValueError("同一个动作不能混用有氧和力量计量方式")
        return self


class WorkoutSessionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = "demo-user"
    session_date: date = Field(default_factory=clock.today)
    title: str = Field(default="训练会话", min_length=1, max_length=200)
    note: str = Field(default="", max_length=2000)
    exercises: list[WorkoutExercisePayload] = Field(min_length=1, max_length=30)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title cannot be empty")
        return value

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str) -> str:
        return value.strip()


class WorkoutSetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    difficulty: Literal["未记录", "轻松", "正常", "困难"] | None = None
    completed: bool | None = None
    weight_kg: float | None = Field(default=None, ge=0, le=1000)
    reps: int | None = Field(default=None, ge=0, le=1000)
    distance_km: float | None = Field(default=None, ge=0, le=10000, allow_inf_nan=False)
    duration_minutes: float | None = Field(default=None, gt=0, le=10080, allow_inf_nan=False)
    note: str | None = Field(default=None, max_length=1000)

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class CustomExercisePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = "demo-user"
    exercise_name: str = Field(min_length=1, max_length=200)
    category: str = Field(default="自定义", min_length=1, max_length=80)
    target_muscles: str = Field(default="", max_length=200)

    @field_validator("exercise_name", "category", "target_muscles")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(value.strip().split())

    @field_validator("exercise_name")
    @classmethod
    def require_exercise_name(cls, value: str) -> str:
        if not value:
            raise ValueError("动作名称不能为空")
        return value
