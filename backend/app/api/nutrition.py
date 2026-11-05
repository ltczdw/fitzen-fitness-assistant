from datetime import date, datetime
from typing import Literal
from fastapi import APIRouter, Header, HTTPException, Path
from pydantic import BaseModel, Field, field_validator

from app.db.food_log_listing import list_food_logs
from app.db.food_library_mutations import update_library_food
from app.db.custom_food_repository import delete_custom_food, list_custom_foods, touch_custom_food, upsert_custom_food
from app.db.food_log_mutations import delete_food_meal, update_food_meal
from app.db.food_repository import delete_catalog_food, list_food_catalog, portion_references, resolve_food
from app.db.nutrition_repository import create_nutrition, delete_nutrition, list_nutrition, update_nutrition
from app.db.nutrition_sync import synced_nutrition_summary
from app.services.nutrition_planner import build_nutrition_plan
from app.services.agent.food_calculator import calculate_manual_nutrition
from app.services.llm.client import LLMGatewayError
from app.services.nutrition_ai import estimate_food_nutrition


router = APIRouter(tags=["nutrition"])


class NutritionLog(BaseModel):
    user_id: str = "demo-user"
    meal_type: str = Field(default="其他", max_length=50)
    description: str = Field(min_length=1, max_length=1000)
    calories: float | None = Field(default=None, ge=0, le=10000)
    protein_g: float | None = Field(default=None, ge=0, le=500)
    carbs_g: float | None = Field(default=None, ge=0, le=1000)
    fat_g: float | None = Field(default=None, ge=0, le=500)
    log_date: date | None = None


class NutritionLogUpdate(BaseModel):
    meal_type: str | None = Field(default=None, max_length=50)
    description: str | None = Field(default=None, min_length=1, max_length=1000)
    calories: float | None = Field(default=None, ge=0, le=10000)
    protein_g: float | None = Field(default=None, ge=0, le=500)
    carbs_g: float | None = Field(default=None, ge=0, le=1000)
    fat_g: float | None = Field(default=None, ge=0, le=500)
    log_date: date | None = None


class FoodNutritionPer100g(BaseModel):
    kcal: float = Field(ge=0, le=1200)
    protein_g: float = Field(ge=0, le=100)
    carbs_g: float = Field(ge=0, le=100)
    fat_g: float = Field(ge=0, le=100)
    fiber_g: float = Field(default=0, ge=0, le=100)


class FoodItemWeightUpdate(BaseModel):
    id: int | None = None
    food_name: str | None = Field(default=None, min_length=1, max_length=200)
    weight_g: float = Field(gt=0, le=100000)
    source: Literal["local", "custom", "ai"] = "local"
    nutrition_per_100g: FoodNutritionPer100g | None = None


class FoodMealUpdate(BaseModel):
    record_time: datetime | None = None
    meal_type: str | None = Field(default=None, max_length=20)
    items: list[FoodItemWeightUpdate] | None = Field(default=None, min_length=1, max_length=30)

    @field_validator("meal_type")
    @classmethod
    def normalize_meal_type(cls, value: str | None) -> str | None:
        value = value.strip() if value is not None else None
        return value or None

    @field_validator("record_time")
    @classmethod
    def normalize_record_time(cls, value: datetime | None) -> datetime | None:
        return value.replace(second=0, microsecond=0) if value is not None else None


class ManualFoodEntry(BaseModel):
    food_name: str = Field(min_length=1, max_length=200)
    weight_g: float = Field(gt=0, le=100000)
    source: Literal["local", "custom", "ai"] = "local"
    nutrition_per_100g: FoodNutritionPer100g | None = None

    @field_validator("food_name")
    @classmethod
    def normalize_food_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("食物名称不能为空")
        return value


class CustomFoodPayload(BaseModel):
    user_id: str = "demo-user"
    food_name: str = Field(min_length=1, max_length=200)
    nutrition_per_100g: FoodNutritionPer100g

    @field_validator("food_name")
    @classmethod
    def normalize_food_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("自定义食物名称不能为空")
        return value


class FoodEstimateRequest(BaseModel):
    description: str = Field(min_length=2, max_length=500)

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("请描述食物名称、品牌或烹饪方式")
        return value


class ManualFoodMeal(BaseModel):
    user_id: str = "demo-user"
    meal_type: str = Field(default="其他", min_length=1, max_length=20)
    record_time: datetime | None = None
    foods: list[ManualFoodEntry] = Field(min_length=1, max_length=30)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=150)

    @field_validator("record_time")
    @classmethod
    def normalize_record_time(cls, value: datetime | None) -> datetime | None:
        return value.replace(second=0, microsecond=0) if value is not None else None


@router.get("/nutrition")
def get_nutrition(user_id: str = "demo-user", limit: int = 30, start_date: date | None = None, end_date: date | None = None) -> list[dict]:
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="开始日期不能晚于结束日期")
    limit = min(max(limit, 1), 100)
    legacy = list_nutrition(user_id, limit, start_date, end_date)
    precise = list_food_logs(user_id, limit, start_date, end_date)
    return sorted(legacy + precise, key=lambda item: str(item.get("date", "")), reverse=True)[:limit]


@router.get("/nutrition/summary")
def get_nutrition_summary(user_id: str = "demo-user", days: int = 7) -> dict:
    return synced_nutrition_summary(user_id, min(max(days, 1), 30))


@router.get("/nutrition/plan")
def get_nutrition_plan(user_id: str = "demo-user") -> dict:
    return build_nutrition_plan(user_id)


@router.get("/nutrition/food/catalog")
def get_food_catalog(search: str = "", limit: int = 200, user_id: str = "demo-user") -> dict:
    foods = list_food_catalog(search.strip(), limit, user_id.strip() or "demo-user")
    return {"foods": foods, "count": len(foods), "source": "local_database"}


@router.get("/nutrition/custom-foods")
def get_custom_foods(user_id: str = "demo-user", search: str = "", limit: int = 30) -> dict:
    foods = list_custom_foods(user_id.strip() or "demo-user", search.strip(), min(max(limit, 1), 200))
    return {"foods": foods, "count": len(foods), "source": "user_custom_library"}


@router.post("/nutrition/custom-foods")
def post_custom_food(payload: CustomFoodPayload) -> dict:
    nutrition = payload.nutrition_per_100g
    try:
        food = upsert_custom_food(
            payload.user_id,
            payload.food_name,
            kcal=nutrition.kcal,
            protein_g=nutrition.protein_g,
            carbs_g=nutrition.carbs_g,
            fat_g=nutrition.fat_g,
            fiber_g=nutrition.fiber_g,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"saved": True, "food": food}


def _edit_library_food(food_id: int, payload: CustomFoodPayload, source: Literal["local", "custom"]) -> dict:
    try:
        food=update_library_food(food_id,payload.food_name,payload.nutrition_per_100g.model_dump(),source=source,user_id=payload.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=409,detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503,detail=str(exc)) from exc
    if food is None:
        raise HTTPException(status_code=404,detail="食物已删除或不存在，请重新加载食物库")
    return {"saved":True,"food":food}


@router.put("/nutrition/food/catalog/{food_id}")
def edit_catalog_food(payload: CustomFoodPayload, food_id: int = Path(gt=0)) -> dict:
    return _edit_library_food(food_id,payload,"local")


@router.put("/nutrition/custom-foods/{food_id}")
def edit_custom_food(payload: CustomFoodPayload, food_id: int = Path(gt=0)) -> dict:
    return _edit_library_food(food_id,payload,"custom")


@router.delete("/nutrition/food/catalog/{food_id}")
def remove_catalog_food(food_id: int = Path(gt=0)) -> dict:
    try:
        deleted = delete_catalog_food(food_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="食物不存在")
    return {"deleted": True, "id": food_id}


@router.delete("/nutrition/custom-foods/{food_id}")
def remove_custom_food(food_id: int = Path(gt=0), user_id: str = "demo-user") -> dict:
    try:
        deleted = delete_custom_food(food_id, user_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="食物不存在或不属于当前用户")
    return {"deleted": True, "id": food_id}


@router.post("/nutrition/food/estimate")
async def estimate_food(payload: FoodEstimateRequest) -> dict:
    try:
        result = await estimate_food_nutrition(payload.description)
    except LLMGatewayError as exc:
        raise HTTPException(status_code=503, detail=f"AI 营养查询暂时不可用：{exc}") from exc
    if result is None:
        raise HTTPException(status_code=503, detail="AI 未返回可验证的营养数据，请补充品牌、重量或烹饪方式后重试")
    return result


@router.get("/nutrition/food/lookup")
def lookup_food_nutrition(name: str) -> dict:
    match = resolve_food(name.strip())
    if match is None:
        raise HTTPException(status_code=404, detail="食物库暂时没有匹配结果，可手动填写每 100g 营养")
    return {
        "food_name": match.food_name,
        "category": match.category,
        "preparation_state": match.preparation_state,
        "per_100g": {
            "kcal": match.energy_kcal,
            "protein_g": match.protein_g,
            "carb_g": match.carb_g,
        "fat_g": match.fat_g,
        },
        "portions": portion_references(match.food_name),
        "match_method": match.match_method,
        "confidence": match.confidence,
        "source_version": match.source_version,
    }


@router.get("/nutrition/food/portions")
def lookup_food_portions(name: str) -> dict:
    portions = portion_references(name.strip())
    if not portions:
        raise HTTPException(status_code=404, detail="该食物暂时没有默认份量，请使用克重")
    return {"food_name": resolve_food(name.strip()).food_name, "portions": portions}


@router.post("/nutrition")
def post_nutrition(log: NutritionLog) -> dict:
    return create_nutrition(**log.model_dump())


@router.post("/nutrition/food")
def post_food_meal(payload: ManualFoodMeal, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict:
    header_token = idempotency_key if isinstance(idempotency_key, str) else None
    token = (payload.idempotency_key or header_token or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="请提供 Idempotency-Key，避免重复保存同一餐次")
    result = calculate_manual_nutrition(
        payload.user_id,
        [item.model_dump() for item in payload.foods],
        idempotency_key="manual-food-" + token,
        meal_group_id="meal-manual-" + token[:20],
        record_time=payload.record_time,
        meal_type=payload.meal_type,
    )
    if result.get("status") != "calculated":
        detail = str(result.get("message") or "缺少：可识别的食物名称，以及重量（克）或可换算的份量。")
        raise HTTPException(status_code=400, detail=detail)
    for item in payload.foods:
        if item.source == "custom" and item.nutrition_per_100g is not None:
            nutrition = item.nutrition_per_100g
            try:
                upsert_custom_food(
                    payload.user_id,
                    item.food_name,
                    kcal=nutrition.kcal,
                    protein_g=nutrition.protein_g,
                    carbs_g=nutrition.carbs_g,
                    fat_g=nutrition.fat_g,
                    fiber_g=nutrition.fiber_g,
                )
                touch_custom_food(payload.user_id, item.food_name)
            except (RuntimeError, ValueError):
                # The meal itself has already been saved. A library failure must
                # not turn a successful nutrition log into a false error.
                pass
    return result


@router.put("/nutrition/food/{meal_group_id}")
def put_food_meal(meal_group_id: str, payload: FoodMealUpdate, user_id: str = "demo-user") -> dict:
    try:
        result = update_food_meal(user_id, meal_group_id, payload.meal_type, [item.model_dump() for item in payload.items] if payload.items is not None else None, record_time=payload.record_time)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="AI 饮食餐次不存在或不属于当前用户")
    for item in payload.items or []:
        if item.source == "custom" and item.nutrition_per_100g is not None:
            nutrition = item.nutrition_per_100g
            try:
                upsert_custom_food(
                    user_id,
                    item.food_name or "",
                    kcal=nutrition.kcal,
                    protein_g=nutrition.protein_g,
                    carbs_g=nutrition.carbs_g,
                    fat_g=nutrition.fat_g,
                    fiber_g=nutrition.fiber_g,
                )
                touch_custom_food(user_id, item.food_name or "")
            except (RuntimeError, ValueError):
                pass
    return result


@router.delete("/nutrition/food/{meal_group_id}")
def remove_food_meal(meal_group_id: str, user_id: str = "demo-user") -> dict:
    result = delete_food_meal(user_id, meal_group_id)
    if result is None:
        raise HTTPException(status_code=404, detail="AI 饮食餐次不存在或不属于当前用户")
    return result


@router.put("/nutrition/{record_id}")
def put_nutrition(record_id: int, payload: NutritionLogUpdate, user_id: str = "demo-user") -> dict:
    updated = update_nutrition(user_id, record_id, payload.model_dump(exclude_unset=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="饮食记录不存在或不属于当前用户")
    return updated


@router.delete("/nutrition/{record_id}")
def remove_nutrition(record_id: int, user_id: str = "demo-user") -> dict:
    if not delete_nutrition(user_id, record_id):
        raise HTTPException(status_code=404, detail="饮食记录不存在或不属于当前用户")
    return {"deleted": True, "id": record_id}
