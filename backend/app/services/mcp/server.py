from datetime import datetime

from fastmcp import FastMCP

from app.db.session import init_db
from app.core.config import settings, warn_unknown_env_keys
from app.db.storage import StorageUnavailableError
from app.schemas.workout import WorkoutRecordPayload
from app.services.mcp.tools import FitnessMCPTools


mcp = FastMCP("Fitness Assistant")
tools = FitnessMCPTools()


@mcp.tool()
def get_user_profile(user_id: str = "demo-user") -> dict:
    """读取用户目标、经验和身体资料。"""
    return tools.get_user_profile(user_id)


@mcp.tool()
def get_training_history(user_id: str = "demo-user") -> list[dict]:
    """读取用户最近的训练记录。"""
    return tools.get_training_history(user_id)


@mcp.tool()
def get_training_plan(user_id: str = "demo-user") -> dict:
    """读取用户当前训练计划。"""
    return tools.get_training_plan(user_id)


@mcp.tool()
def get_nutrition_summary(user_id: str = "demo-user", days: int = 1) -> dict:
    """读取用户指定天数内的饮食汇总。"""
    return tools.get_nutrition_summary(user_id, days)


@mcp.tool()
def get_nutrition_plan(user_id: str = "demo-user") -> dict:
    """读取基于用户档案和饮食记录生成的营养计划。"""
    return tools.get_nutrition_plan(user_id)


@mcp.tool()
def save_workout_record(user_id: str, record: WorkoutRecordPayload) -> dict:
    """保存一条已经通过应用层校验的训练记录。"""
    return tools.save_workout_record(user_id, record.model_dump())


@mcp.tool()
def save_nutrition_log(user_id: str, record: dict) -> dict:
    """保存一条手工营养日志。"""
    return tools.save_nutrition_log(user_id, record)


@mcp.tool()
def calculate_nutrition(user_id: str, entries: list[dict], raw_message: str, idempotency_key: str | None = None, meal_group_id: str | None = None, record_time: datetime | None = None, nutrition_overrides: dict | None = None) -> dict:
    """按食物成分表计算并保存一餐营养。"""
    return tools.calculate_nutrition(user_id, entries, raw_message, idempotency_key, meal_group_id, record_time, nutrition_overrides)


@mcp.tool()
def calculate_nutrition_batch(user_id: str, groups: list[dict], raw_message: str, idempotency_key: str | None = None, meal_group_id: str | None = None, record_time: datetime | None = None, nutrition_overrides: dict | None = None) -> dict:
    """按餐次批量计算并保存多餐营养。"""
    return tools.calculate_nutrition_batch(user_id, groups, raw_message, idempotency_key, meal_group_id, record_time, nutrition_overrides)


if __name__ == "__main__":
    warn_unknown_env_keys()
    if not init_db():
        raise StorageUnavailableError("MCP database initialization failed")
    mcp.run(transport="streamable-http", host=settings.mcp_host, port=settings.mcp_port, path="/mcp")
