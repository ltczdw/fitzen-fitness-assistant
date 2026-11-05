from fastapi import APIRouter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.db.plan_repository import get_plan, save_plan


router = APIRouter(tags=["plans"])


class PlanSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    day: str = Field(min_length=1, max_length=30)
    name: str = Field(min_length=1, max_length=200)
    items: list[str] = Field(default_factory=list, max_length=30)
    status: Literal["待开始", "今天", "已完成"] = "待开始"

    @field_validator("day", "name")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("训练日和训练名称不能为空")
        return value

    @field_validator("items")
    @classmethod
    def normalize_items(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        if len(normalized) > 30:
            raise ValueError("单个训练日最多 30 个动作")
        return normalized


class PlanUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = "demo-user"
    plan_name: str = Field(default="基础全身训练", max_length=200)
    goal: str = Field(default="保持健康", max_length=100)
    days_per_week: int = Field(default=3, ge=1, le=7)
    sessions: list[PlanSession] = Field(default_factory=list, max_length=14)

    @field_validator("plan_name", "goal")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("计划名称和目标不能为空")
        return value


@router.get("/plans")
def read_plan(user_id: str = "demo-user") -> dict:
    return get_plan(user_id)


@router.put("/plans")
def update_plan(plan: PlanUpdate) -> dict:
    return save_plan(plan.user_id, plan.model_dump(exclude={"user_id"}))
