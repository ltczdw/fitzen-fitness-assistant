"""阶段B只读工具 adapter：四个真实 SQL 读取 + 写工具双层拒绝。

身份（user_id）只从应用注入的请求 context 取，不信任模型参数；模型提交的
user_id/request_id/mode/time_base 一律被严格 schema 拒绝。写工具（record/calculate/
save）在此模块直接入口同样拒绝，不转到任何业务写回调。读取按需调用真实 repo，
不做无条件预加载；同请求按“工具+参数”的有界缓存由 runtime 维护，本模块保持纯读取。
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

from app.core import clock
from app.services.agent.react_tools import (
    TOOL_CATALOG,
    _schema_error,
    _strict_integers_recursive,
)

# 写/计算/保存工具在只读上下文直接拒绝，避免转到任何业务写路径。
_WRITE_TOOL_NAMES = frozenset({
    "record_workout", "record_nutrition",
    "calculate_nutrition", "calculate_nutrition_batch",
    "save_plan", "save_workout_record", "save_nutrition_log",
})


def _args_ok(name: str, args: Any) -> bool:
    spec = TOOL_CATALOG.get(name)
    if spec is None or not isinstance(args, dict):
        return False
    if _schema_error(spec.parameters, args) is not None:
        return False
    if _strict_integers_recursive(spec.parameters, args) is not None:
        return False
    return True


def _user_id(context: dict[str, Any]) -> str:
    return str((context or {}).get("user_id") or "demo-user")


def _reject_write(args: Any, context: dict[str, Any], name: str) -> dict[str, Any]:
    return {"status": "rejected", "error": "write_tool_not_available_in_readonly", "tool": name}


def read_profile(args: Any, context: dict[str, Any]) -> dict[str, Any]:
    if not _args_ok("read_profile", args):
        return {"status": "rejected", "error": "schema_invalid"}
    from app.db import profile_repository
    profile = profile_repository.get_profile(_user_id(context))
    return {"status": "completed", "result": profile}


def read_workout_history(args: Any, context: dict[str, Any]) -> dict[str, Any]:
    if not _args_ok("read_workout_history", args):
        return {"status": "rejected", "error": "schema_invalid"}
    days = args.get("days", 7)
    if not isinstance(days, int) or isinstance(days, bool) or not (1 <= days <= 365):
        return {"status": "rejected", "error": "invalid_days"}
    from app.db import repositories

    end_date = clock.today()
    start_date = end_date - timedelta(days=days - 1)
    workouts, truncated = repositories.query_workout_read_rows(
        _user_id(context), start_date=start_date, end_date=end_date, limit=30,
    )
    return {"status": "completed", "result": {"workouts": workouts, "truncated": truncated}}


def read_nutrition_summary(args: Any, context: dict[str, Any]) -> dict[str, Any]:
    if not _args_ok("read_nutrition_summary", args):
        return {"status": "rejected", "error": "schema_invalid"}
    days = args.get("days", 7)
    if not isinstance(days, int) or isinstance(days, bool) or not (1 <= days <= 365):
        return {"status": "rejected", "error": "invalid_days"}
    from app.db import nutrition_sync
    summary = nutrition_sync.synced_nutrition_summary(_user_id(context), days=days)
    return {"status": "completed", "result": summary}


def read_current_plan(args: Any, context: dict[str, Any]) -> dict[str, Any]:
    if not _args_ok("read_current_plan", args):
        return {"status": "rejected", "error": "schema_invalid"}
    from app.db import plan_repository
    plan = plan_repository.get_plan(_user_id(context))
    return {"status": "completed", "result": plan}


async def record_workout(args: Any, context: dict[str, Any]) -> dict[str, Any]:
    return _reject_write(args, context, "record_workout")


async def record_nutrition(args: Any, context: dict[str, Any]) -> dict[str, Any]:
    return _reject_write(args, context, "record_nutrition")


async def calculate_nutrition(args: Any, context: dict[str, Any]) -> dict[str, Any]:
    return _reject_write(args, context, "calculate_nutrition")


async def calculate_nutrition_batch(args: Any, context: dict[str, Any]) -> dict[str, Any]:
    return _reject_write(args, context, "calculate_nutrition_batch")


async def save_plan(args: Any, context: dict[str, Any]) -> dict[str, Any]:
    return _reject_write(args, context, "save_plan")


READONLY_HANDLERS = {
    "read_profile": read_profile,
    "read_workout_history": read_workout_history,
    "read_nutrition_summary": read_nutrition_summary,
    "read_current_plan": read_current_plan,
}


__all__ = [
    "READONLY_HANDLERS",
    "read_profile",
    "read_workout_history",
    "read_nutrition_summary",
    "read_current_plan",
    "record_workout",
    "record_nutrition",
    "calculate_nutrition",
    "calculate_nutrition_batch",
    "save_plan",
]
