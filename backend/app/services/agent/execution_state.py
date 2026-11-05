from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from weakref import WeakKeyDictionary, WeakValueDictionary
import asyncio
from app.services.agent.intent import IntentResult

_memory_error = ContextVar("memory_error", default=None)
_request_route = ContextVar("request_route", default=None)
_request_memory = ContextVar("request_memory", default=None)
# 已有未完成请求冻结的模式：主协程在 begin_turn 前读取、在请求内复原，绝不假定
# to_thread 内赋值会传回（阶段B模式冻结恢复）。
_resumed_frozen_mode = ContextVar("resumed_frozen_mode", default=None)
_lock_registry_guard = RLock()
_conversation_locks = WeakKeyDictionary()


def conversation_lock(loop, conversation_id):
    # Strong references exist only while a request is waiting/using the lock.
    with _lock_registry_guard:
        per_loop = _conversation_locks.get(loop)
        if per_loop is None:
            per_loop = WeakValueDictionary()
            _conversation_locks[loop] = per_loop
        lock = per_loop.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            per_loop[conversation_id] = lock
        return lock

class RequestIdConflict(ValueError):
    """An idempotency key was reused for another or deleted request payload."""


@dataclass(frozen=True)
class PlannerDecision:
    intent: IntentResult
    parsed_input: dict
    task_specs: tuple[dict, ...]
    needs_clarification: bool = False
    degraded_reason: str | None = None
