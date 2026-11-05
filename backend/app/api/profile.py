from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from app.db.profile_repository import get_profile, save_profile
from app.services.memory.redis_memory import memory_service


router = APIRouter(tags=["profile"])


class ProfileUpdate(BaseModel):
    user_id: str = "demo-user"
    display_name: str = Field(default="我的训练", max_length=100)
    goal: str = Field(default="保持健康", max_length=100)
    experience: str = Field(default="初学者", max_length=100)
    available_days: int = Field(default=3, ge=1, le=7)
    body_weight_kg: float | None = Field(default=None, ge=20, le=400, allow_inf_nan=False)
    target_weight_kg: float | None = Field(default=None, ge=20, le=400, allow_inf_nan=False)
    height_cm: float | None = Field(default=None, ge=100, le=250, allow_inf_nan=False)
    body_fat_pct: float | None = Field(default=None, gt=0, lt=100, allow_inf_nan=False)


    @field_validator("display_name", "goal", "experience")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("请填写称呼、训练目标和训练经验")
        return value


@router.get("/profile")
def read_profile(user_id: str = "demo-user") -> dict:
    return get_profile(user_id)


@router.put("/profile")
async def update_profile(profile: ProfileUpdate) -> dict:
    saved = save_profile(profile.user_id, profile.model_dump(exclude={"user_id"}, exclude_unset=True))
    await memory_service.sync_profile(profile.user_id, saved)
    return saved
