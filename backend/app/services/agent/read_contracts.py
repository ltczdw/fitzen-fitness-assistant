"""One canonical contract for application-owned business reads.

The model selects a resource and, for the profile, a finite field enum. It never
supplies account identity, SQL or stored values. A discriminated union keeps the
resource-specific shape closed: profile has no dates, a saved plan has no field
projection, and a workout read has no profile fields. Unknown names therefore
fail at the schema boundary instead of reaching a business read.
"""
from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class ProfileField(StrEnum):
    DISPLAY_NAME = "display_name"
    GOAL = "goal"
    EXPERIENCE = "experience"
    AVAILABLE_DAYS = "available_days"
    BODY_WEIGHT_KG = "body_weight_kg"
    TARGET_WEIGHT_KG = "target_weight_kg"
    HEIGHT_CM = "height_cm"
    BODY_FAT_PCT = "body_fat_pct"


# Single field registry: labels/units for rendering, enum values for validation.
PROFILE_FIELD_LABELS: dict[str, tuple[str, str]] = {
    "display_name": ("显示名", ""),
    "goal": ("训练目标", ""),
    "experience": ("训练经验", ""),
    "available_days": ("每周可训练天数", "天"),
    "body_weight_kg": ("当前体重", "kg"),
    "target_weight_kg": ("目标体重", "kg"),
    "height_cm": ("身高", "cm"),
    "body_fat_pct": ("体脂率", "%"),
}

PROFILE_FIELD_NAMES: tuple[str, ...] = tuple(item.value for item in ProfileField)


class ProfileReadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resource: Literal["profile"]
    fields: list[ProfileField] = Field(default_factory=list, max_length=8)


class PlanReadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resource: Literal["training_plan"]
    # A stored plan is always rendered completely; no field projection exists.


class WorkoutRecordsReadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resource: Literal["workout_records"]
    date_start: date
    date_end: date
    exercises: list[str] = Field(default_factory=list, max_length=20)


ReadSpec = Annotated[
    Union[ProfileReadSpec, PlanReadSpec, WorkoutRecordsReadSpec],
    Field(discriminator="resource"),
]

_read_spec_adapter = TypeAdapter(ReadSpec)


def validate_read_spec(value) -> BaseModel:
    """Validate a model-owned or legacy spec without guessing aliases.

    Raises ``pydantic.ValidationError``; callers convert it to their own bounded
    error code. A malformed legacy dict must not reach a repository read.
    """
    if isinstance(value, (ProfileReadSpec, PlanReadSpec, WorkoutRecordsReadSpec)):
        return value
    return _read_spec_adapter.validate_python(value)


__all__ = [
    "PROFILE_FIELD_LABELS",
    "PROFILE_FIELD_NAMES",
    "PlanReadSpec",
    "ProfileField",
    "ProfileReadSpec",
    "ReadSpec",
    "WorkoutRecordsReadSpec",
    "validate_read_spec",
]
