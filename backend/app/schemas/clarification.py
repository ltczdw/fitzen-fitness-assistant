"""Shared, additive API contract for missing input and ambiguous references."""
from typing import Literal
from pydantic import BaseModel, Field


class MissingField(BaseModel):
    name: str
    label: str
    entity: str | None = None


class TaskClarification(BaseModel):
    task_id: str = ""
    task_type: str
    category: Literal["missing_information", "ambiguous_reference", "invalid_parameter", "service_unavailable"] = "missing_information"
    fields: list[MissingField] = Field(default_factory=list)
    question: str

