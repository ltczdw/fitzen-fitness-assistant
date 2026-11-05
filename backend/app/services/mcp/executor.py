from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi.encoders import jsonable_encoder
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from app.core.config import settings
from app.db.storage import CommitUnknownError

logger = logging.getLogger(__name__)


class ToolSchemaError(ValueError):
    """Rejected before dispatch; distinct from an unknown write outcome."""

READ_TOOLS = frozenset({
    "get_user_profile", "get_training_history", "get_training_plan",
    "get_nutrition_summary", "get_nutrition_plan",
})
WRITE_TOOLS = frozenset({
    "save_workout_record", "save_nutrition_log",
    "calculate_nutrition", "calculate_nutrition_batch",
})

_USER = {"type": "string"}
_OBJECT = {"type": "object"}
_OPTIONAL_STRING = {"type": ["string", "null"]}
_NUTRITION_OVERRIDES = {
    "type": ["object", "null"], "maxProperties": 30,
    "additionalProperties": {
        "type": "object",
        "properties": {key: {"type": "number", "minimum": 0,
                             "maximum": 1200 if key == "kcal" else 100}
                       for key in ("kcal", "protein_g", "carbs_g", "fat_g", "fiber_g")},
        "required": ["kcal", "protein_g", "carbs_g", "fat_g", "fiber_g"],
        "additionalProperties": False,
    },
}
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "get_user_profile": {"type": "object", "properties": {"user_id": _USER}, "required": ["user_id"], "additionalProperties": False},
    "get_training_history": {"type": "object", "properties": {"user_id": _USER}, "required": ["user_id"], "additionalProperties": False},
    "get_training_plan": {"type": "object", "properties": {"user_id": _USER}, "required": ["user_id"], "additionalProperties": False},
    "get_nutrition_summary": {"type": "object", "properties": {"user_id": _USER, "days": {"type": "integer", "minimum": 1, "maximum": 365}}, "required": ["user_id"], "additionalProperties": False},
    "get_nutrition_plan": {"type": "object", "properties": {"user_id": _USER}, "required": ["user_id"], "additionalProperties": False},
    "save_workout_record": {"type": "object", "properties": {"user_id": _USER, "record": _OBJECT}, "required": ["user_id", "record"], "additionalProperties": False},
    "save_nutrition_log": {"type": "object", "properties": {"user_id": _USER, "record": _OBJECT}, "required": ["user_id", "record"], "additionalProperties": False},
    "calculate_nutrition": {
        "type": "object", "properties": {
            "user_id": _USER, "entries": {"type": "array", "items": _OBJECT},
            "raw_message": {"type": "string"}, "idempotency_key": _OPTIONAL_STRING,
            "meal_group_id": _OPTIONAL_STRING, "record_time": {},
            "nutrition_overrides": _NUTRITION_OVERRIDES,
        }, "required": ["user_id", "entries", "raw_message"], "additionalProperties": False,
    },
    "calculate_nutrition_batch": {
        "type": "object", "properties": {
            "user_id": _USER, "groups": {"type": "array", "items": _OBJECT},
            "raw_message": {"type": "string"}, "idempotency_key": _OPTIONAL_STRING,
            "meal_group_id": _OPTIONAL_STRING, "record_time": {},
            "nutrition_overrides": _NUTRITION_OVERRIDES,
        }, "required": ["user_id", "groups", "raw_message"], "additionalProperties": False,
    },
}


class ToolExecutionGateway:
    """Single permission, schema, timeout, identity-injection and audit boundary."""

    @staticmethod
    def _validate(name: str, arguments: dict[str, Any], schema: dict[str, Any] | None,
                  *, trusted_user_id: str | None) -> None:
        if name not in READ_TOOLS and name not in WRITE_TOOLS:
            raise PermissionError(f"未知工具默认拒绝: {name}")
        selected = schema if isinstance(schema, dict) and schema else TOOL_SCHEMAS.get(name)
        if not isinstance(selected, dict) or not selected:
            raise ToolSchemaError(f"工具缺少有效参数 Schema: {name}")
        # Discovery may narrow a remote interface, never weaken the common
        # application contract or make local/remote validation diverge.
        schemas = [TOOL_SCHEMAS[name]]
        if selected != schemas[0]:
            schemas.append(selected)
        for candidate in schemas:
            checked_schema = copy.deepcopy(candidate)
            properties = checked_schema.get("properties") or {}
            if trusted_user_id is not None and "user_id" in properties:
                required = checked_schema.get("required")
                if isinstance(required, list):
                    checked_schema["required"] = [key for key in required if key != "user_id"]
            try:
                error = best_match(Draft202012Validator(checked_schema).iter_errors(arguments))
            except Exception as exc:
                raise ToolSchemaError(f"工具参数 Schema 无效: {name}") from exc
            if error is not None:
                raise ToolSchemaError(f"工具参数校验失败 {error.json_path}: {error.message}")

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        schema: dict[str, Any] | None,
        operation: Callable[[dict[str, Any]], Any | Awaitable[Any]],
        allowed_tools: set[str] | frozenset[str],
        blocked_tools: set[str] | frozenset[str] | None = None,
        write_authorized: bool = False,
        trusted_user_id: str | None = None,
        fallback: Callable[[dict[str, Any]], Any | Awaitable[Any]] | None = None,
        request_id: str | None = None,
        task_id: str | None = None,
    ) -> Any:
        started = time.monotonic()
        permission = "write" if name in WRITE_TOOLS else "read"
        source = "mcp"
        status = "failed"
        try:
            if name in (blocked_tools or set()):
                raise PermissionError(f"该任务禁止调用工具: {name}")
            if name not in allowed_tools:
                raise PermissionError(f"该任务未授权调用工具: {name}")
            if name not in READ_TOOLS and name not in WRITE_TOOLS:
                raise PermissionError(f"未知工具默认拒绝: {name}")
            if permission == "write" and not write_authorized:
                raise PermissionError(f"写工具缺少应用层授权: {name}")
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是对象")

            wire_arguments = jsonable_encoder(dict(arguments))
            self._validate(name, wire_arguments, schema, trusted_user_id=trusted_user_id)
            if trusted_user_id is not None:
                wire_arguments["user_id"] = trusted_user_id
            local_arguments = dict(arguments)
            if trusted_user_id is not None:
                local_arguments["user_id"] = trusted_user_id

            timeout = max(0.1, float(settings.mcp_timeout_seconds))
            result = await asyncio.wait_for(self._invoke(operation, wire_arguments), timeout=timeout)
            if result is None and fallback is not None:
                source = "mcp_to_local_fallback"
                result = await asyncio.wait_for(self._invoke(fallback, local_arguments), timeout=timeout)
            if result is None:
                raise RuntimeError(f"工具没有返回结果: {name}")
            status = "success" if source == "mcp" else "local_fallback_success"
            return result
        except PermissionError:
            status = "permission_denied"
            raise
        except ValueError:
            status = "schema_rejected"
            raise
        except (asyncio.TimeoutError, TimeoutError) as exc:
            status = "outcome_unknown" if permission == "write" else "timeout"
            if permission == "write":
                raise CommitUnknownError("Tool write timed out; verify the record before retrying") from exc
            raise
        except CommitUnknownError:
            status = "outcome_unknown"
            raise
        except Exception:
            status = "failed"
            raise
        finally:
            logger.info("tool_execution", extra={"tool_execution": {
                "name": name,
                "permission": permission,
                "source": source,
                "status": status,
                "duration_ms": int((time.monotonic() - started) * 1000),
                "retry_count": 0,
                "request_id": request_id or "",
                "task_id": task_id or "",
            }})

    @staticmethod
    async def _invoke(operation: Callable[[dict[str, Any]], Any | Awaitable[Any]],
                      arguments: dict[str, Any]) -> Any:
        value = operation(arguments)
        if inspect.isawaitable(value):
            return await value
        return value
