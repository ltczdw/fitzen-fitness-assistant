from app.db.nutrition_repository import create_nutrition
from app.db.nutrition_sync import synced_nutrition_summary
from app.db.plan_repository import get_plan
from app.db.profile_repository import get_profile
from app.db.repositories import create_workout, query_workout_read_rows
from app.schemas.workout import WorkoutRecordPayload
from app.services.agent.food_calculator import calculate_nutrition as calculate_food_nutrition, calculate_nutrition_batch as calculate_food_nutrition_batch
from app.services.agent.food_parser import FoodInput
from app.services.nutrition_planner import build_nutrition_plan


class FitnessMCPTools:
    """健身业务工具边界；Agent 通过 MCP 访问这些数据。"""

    tool_names = ("get_user_profile", "get_training_history", "get_training_plan", "get_nutrition_summary", "get_nutrition_plan", "save_workout_record", "save_nutrition_log", "calculate_nutrition", "calculate_nutrition_batch")

    def get_user_profile(self, user_id: str) -> dict:
        return get_profile(user_id)

    def get_training_history(self, user_id: str) -> list[dict]:
        rows, _ = query_workout_read_rows(user_id, limit=30)
        return rows

    def get_training_plan(self, user_id: str) -> dict:
        return get_plan(user_id)

    def get_nutrition_summary(self, user_id: str, days: int = 7) -> dict:
        return synced_nutrition_summary(user_id, days)

    def get_nutrition_plan(self, user_id: str) -> dict:
        return build_nutrition_plan(user_id)

    def save_workout_record(self, user_id: str, record: dict) -> dict:
        payload = WorkoutRecordPayload.model_validate({key: value for key, value in record.items() if key in WorkoutRecordPayload.model_fields})
        values = payload.model_dump()
        idempotency_key = values.pop("idempotency_key", None) or record.get("idempotency_key")
        saved = create_workout(user_id=user_id, **values, idempotency_key=idempotency_key)
        return {"user_id": user_id, "record": saved, "saved": True}

    def save_nutrition_log(self, user_id: str, record: dict) -> dict:
        saved = create_nutrition(user_id=user_id, meal_type=str(record.get("meal_type", "其他")), description=str(record.get("description", "")), calories=record.get("calories"), protein_g=record.get("protein_g"), carbs_g=record.get("carbs_g"), fat_g=record.get("fat_g"))
        return {"user_id": user_id, "record": saved, "saved": True}

    def calculate_nutrition(self, user_id: str, entries: list[dict], raw_message: str, idempotency_key: str | None = None, meal_group_id: str | None = None, record_time=None, nutrition_overrides: dict | None = None) -> dict:
        parsed = [FoodInput(**entry) for entry in entries]
        return calculate_food_nutrition(user_id, parsed, raw_message=raw_message, idempotency_key=idempotency_key, meal_group_id=meal_group_id, record_time=record_time, nutrition_overrides=nutrition_overrides)

    def calculate_nutrition_batch(self, user_id: str, groups: list[dict], raw_message: str, idempotency_key: str | None = None, meal_group_id: str | None = None, record_time=None, nutrition_overrides: dict | None = None) -> dict:
        return calculate_food_nutrition_batch(user_id, groups, raw_message=raw_message, idempotency_key=idempotency_key, meal_group_id=meal_group_id, record_time=record_time, nutrition_overrides=nutrition_overrides)
