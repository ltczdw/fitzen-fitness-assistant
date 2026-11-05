from datetime import date as date_type
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class WorkoutSetDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight_kg: float = Field(ge=0, le=1000)
    reps: int = Field(ge=1, le=1000)
    difficulty: Literal["未记录", "轻松", "正常", "困难"] = "未记录"
    completed: bool = True
    note: str = Field(default="", max_length=2000)

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str) -> str:
        return value.strip()


class WorkoutRecordPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: date_type | None = None
    exercise: str = Field(min_length=1, max_length=80)
    sets: int = Field(ge=0, le=100)
    reps: int = Field(ge=1, le=1000)
    weight_kg: float = Field(ge=0, le=1000)
    difficulty: Literal["未记录", "轻松", "正常", "困难"] = "未记录"
    completed: bool = True
    note: str = Field(default="", max_length=2000)
    set_details: list[WorkoutSetDetail] | None = None
    # Internal retry key used by the MCP write boundary.  It is optional so
    # direct UI/API writes keep their existing behavior.
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=150)

    @model_validator(mode="after")
    def validate_set_details(self):
        if self.set_details is not None and len(self.set_details) != self.sets:
            raise ValueError("set_details length must match sets")
        return self

    @field_validator("exercise")
    @classmethod
    def normalize_exercise(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("exercise cannot be empty")
        return value

    @field_validator("note")
    @classmethod
    def normalize_note(cls, value: str) -> str:
        return value.strip()


class WorkoutRecord(WorkoutRecordPayload):
    user_id: str = "demo-user"
