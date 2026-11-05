"""C2-3 显式写入运行时（phaseI 实施中）。

职责边界（合同 §2–§6）：
- 资格门在任何 legacy planner/shadow/领域派发之前完成（开关/compact/账本/计量）。
- 预算引导：先以真实 BudgetStore 打开/核验请求 scope 额度行（真实 SQL、可故障），
  失败即明确 execution_limited，零 C1/模型/领域写。
- C1 身份/租约/controller、真实 ReAct 图（persistence hooks）、两桥真实派发、
  protected pending/确认双 lease、restore/终态 final_reply 与 ChatResponse 规范哈希。
"""
from __future__ import annotations

import asyncio
import contextvars
import copy
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

from app.services.agent.task_models import RECORD_TASK_TYPES

logger = logging.getLogger(__name__)

RUNTIME_VERSION = 1
SUPERVISOR_STAGE = "react_supervisor"
ESTIMATE_STAGE = "react_nutrition_estimate"

# Application-owned task types that can complete without a business commit.
# Intent is derived from these completed read targets, never from message
# keywords or a model-supplied label. Record domains are included because a
# record read keeps its domain label and has no side effect.
_READ_INTENT_TASK_TYPES = RECORD_TASK_TYPES | frozenset({"training_qa", "training_plan", "nutrition_advice"})
_WRITE_INTENT_BY_PROOF = {
    "workout_name_update_v1": "workout_record",
    "workout_db_v1": "workout_record",
    "nutrition_db_v1": "nutrition_record",
}


class WriteRuntimeUnavailable(RuntimeError):
    """运行时未就绪（显式失败，不伪装成功）。"""

    code = "write_runtime_unavailable"


_write_scope = contextvars.ContextVar("c2_3_write_lifecycle", default=None)


def bind_request_scope_store(scope, factory=None):
    """请求局部：把已存在的原 request scope 的计量依赖绑定到 write 应用拥有的
    store（默认 app SessionLocal；显式传入 factory 时贯穿真实计量/C1/adapter）。

    设计（root 同批裁定）：不修改模块 import、全局 override、原 store 对象或全局
    factory；只对当前已存在的 ModelScope 做请求局部依赖选择——用原 store 的私有
    浅拷贝（保留其类型/子类覆盖/实例策略与失败行为），仅把副本 factory 指向给定
    session_factory。进入时对原 store 做一次真实 session 失败面探针：显式失计量
    注入保持原样（由运行时预算引导 fail closed，零派发），绝不被静默替换。
    返回 (restore|None, error|None)；调用方必须在 finally 调 restore。
    """
    from app.db import session as db
    from app.services.llm.budget import BudgetStore

    original = scope.store
    if not isinstance(original, BudgetStore):
        return None, "unsupported_store_type"
    target_factory = factory if factory is not None else db.SessionLocal
    if target_factory is None:
        return None, "app_session_factory_unavailable"

    def _probe() -> None:
        session = original.session()
        session.close()

    try:
        _probe()
        probe_error = None
    except Exception:  # noqa: BLE001 - 显式失计量注入必须保持 fail closed
        logger.warning("c2_3_request_scope_store_unavailable", exc_info=True)
        probe_error = "injected_store_unavailable"
    # AND 标准：显式注入失败 → 调用方 fail closed；同时应用拥有的真实计量依赖
    # （应用库 quota/ledger）仍按本次请求局部绑定打开，二者同时成立。
    local = copy.copy(original)
    local.factory = target_factory
    scope.store = local

    def restore() -> None:
        scope.store = original

    return restore, probe_error


@asynccontextmanager
async def write_lifecycle_scope():
    """请求局部私有上下文（冻结签名：无参）。进入时不触库；覆盖 run_write、
    稳定 typed response 投影/IDs/memory/diagnostics 直到 finalize_write_response，
    并在 finally reset。"""
    scope = {"runtime_version": RUNTIME_VERSION, "session": None}
    token = _write_scope.set(scope)
    try:
        yield scope
    finally:
        session = scope.get("session")
        if session is not None and getattr(session, "lease", None) is not None:
            # 请求在提交边界被取消/异常退出：同步真实释放租约（不借旧 token）；
            # 业务提交与证明保留，后续 resume/重放重新真实取得。
            session.release_lease_quiet()
        _write_scope.reset(token)


async def _budget_bootstrap(scope, frozen_limits: tuple[int, int] | None = None) -> bool:
    """真实打开/核验请求 scope 额度行；失计量/存储故障/额度异常都 fail closed。

    持久上限 AND（同事务真实 DML）：真实行最小值、冻结 controller 上限与当前
    scope 上限的收紧用**单一数据库原子表达式**（SQLite MIN/生成 PG LEAST）作用于
    行现值，绝不做“先读后写”的绝对值覆盖——并发同 rid 两个 scope 不会互相回写
    旧上限；行缺失/非唯一/非法限额一律 fail closed；只收紧 max_calls/max_tokens，
    期限与计数不刷新。"""
    from sqlalchemy import func, select, update

    from app.db.model_budget_models import ModelQuota
    from app.services.llm.budget import BudgetStore, ModelBudgetError, ModelBudgetUnavailable

    store = scope.store

    def _valid_limit(value) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ModelBudgetUnavailable("budget_scope_limit_invalid")
        return value

    def _open() -> None:
        scope_calls = _valid_limit(getattr(scope, "max_calls", None))
        scope_tokens = _valid_limit(getattr(scope, "max_tokens", None))
        with store.session() as session, session.begin():
            store.prepare(session)
            BudgetStore.open_in_session(
                session,
                name=scope.name,
                kind=scope.kind,
                max_calls=scope_calls,
                max_tokens=scope_tokens,
                ttl_seconds=scope.ttl_seconds,
            )
            root = session.scalar(
                select(ModelQuota)
                .where(ModelQuota.quota_id == scope.quota_id)
                .with_for_update()
            )
            # The root row is aggregate-only on the new policy. Persisting the
            # marker here lets cold recovery distinguish it from an old request
            # that already spent under the shared 12-call request allowance.
            if frozen_limits is not None:
                # An old controller snapshot is authoritative even when its
                # root has zero consumption. It is a frozen shared request, not
                # a new request eligible for task quotas.
                per_task = False
            elif getattr(scope, "policy", None) == "per_task":
                per_task = store.policy_in_session(session, scope, root=root)
            else:
                per_task = False
            if per_task:
                scope.policy = "per_task"
            else:
                scope.policy = "shared"
                limit_calls = min([scope_calls] + ([int(frozen_limits[0])] if frozen_limits else []))
                limit_tokens = min([scope_tokens] + ([int(frozen_limits[1])] if frozen_limits else []))
                dialect = session.get_bind().dialect.name
                if dialect == "sqlite":
                    db_min = func.min
                elif dialect == "postgresql":
                    db_min = func.least
                else:
                    raise ModelBudgetUnavailable("unsupported_budget_storage")
                result = session.execute(
                    update(ModelQuota)
                    .where(ModelQuota.quota_id == scope.quota_id)
                    .values(max_calls=db_min(ModelQuota.max_calls, limit_calls),
                            max_tokens=db_min(ModelQuota.max_tokens, limit_tokens))
                )
                if result.rowcount != 1:
                    raise ModelBudgetUnavailable("budget_scope_row_missing")

    try:
        await asyncio.to_thread(_open)
    except (ModelBudgetError, ModelBudgetUnavailable):
        return False
    except Exception:  # noqa: BLE001 - 真实存储/故障窗口一律受限
        logger.warning("c2_3_budget_bootstrap_failed", exc_info=True)
        return False
    return True


def _limited(reason: str, *, model_used: bool = False) -> dict[str, Any]:
    return {
        "final_answer": "本次写入请求未进入执行：写入运行时当前不可用（开关/记忆后端/账本或计量未就绪）。"
                        "未切换到 legacy，未执行模型或保存。",
        "worker_results": [{"worker": "react_write", "status": "limited", "answer": ""}],
        "citations": [],
        "model_used": model_used,
        "safety_level": "normal",
        "current_task": {"task_type": "nutrition_record"},
        "react_status": "limited",
        "intent": "execution_limited",
        "react_write_status": "limited",
        "mode_reason": reason,
        "receipt_refs": [],
        "pending_confirmations": [],
    }


def _canonical_hash(value: Any) -> str | None:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_object(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _estimate_number(data: dict, key: str, *, maximum: float = 100000.0) -> float | None:
    value = data.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    import math

    if not math.isfinite(value) or value < 0 or value > maximum:
        return None
    return value


def _parse_food_estimate(assistant: Any) -> dict | None:
    """Parse one bounded AI nutrition estimate.

    Accepts a plain object, a fenced code block or one surrounding prose
    segment. Numeric strings are coerced; every value still passes finite/range
    checks. This is format normalization only, never a semantic substitution.
    """
    content = assistant.get("content") if isinstance(assistant, dict) else None
    if not isinstance(content, str):
        return None
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(data, dict):
        return None
    kcal = _estimate_number(data, "kcal")
    if kcal is None:
        return None
    return {"kcal": kcal,
            "protein_g": _estimate_number(data, "protein_g", maximum=100.0) or 0.0,
            "carbs_g": _estimate_number(data, "carbs_g", maximum=100.0) or 0.0,
            "fat_g": _estimate_number(data, "fat_g", maximum=100.0) or 0.0,
            "fiber_g": _estimate_number(data, "fiber_g", maximum=100.0) or 0.0}


def _completed_read_intent(state: dict, frozen_outline: list | None = None) -> str | None:
    """Return the application task type of the last completed non-write goal.

    Candidates are source-validated application decisions; ``call_facts`` and
    ``goal_results`` are produced by the real graph. No message keyword or
    model-provided intent label is consulted.
    """
    from app.services.agent.react_tools import TOOL_CATALOG

    candidates: dict[str, dict] = {}
    for candidate in [*(state.get("intent_candidates") or []), *(frozen_outline or [])]:
        if isinstance(candidate, dict) and candidate.get("id"):
            candidates[str(candidate["id"])] = candidate
    for cid, observation in reversed(list((state.get("goal_results") or {}).items())):
        if not isinstance(observation, dict) or observation.get("status") != "completed":
            continue
        task_type = str((candidates.get(str(cid)) or {}).get("task_type") or "")
        if task_type in _READ_INTENT_TASK_TYPES:
            return task_type
    for fact in reversed(state.get("call_facts") or []):
        if not isinstance(fact, dict) or fact.get("permission") != "read":
            continue
        observation = fact.get("observation")
        if not isinstance(observation, dict) or observation.get("status") != "completed":
            continue
        spec = TOOL_CATALOG.get(str(fact.get("tool_name") or ""))
        task_type = str(getattr(spec, "task_type", "") or "")
        if task_type in _READ_INTENT_TASK_TYPES:
            return task_type
    for cid in reversed(state.get("covered_goals") or []):
        task_type = str((candidates.get(str(cid)) or {}).get("task_type") or "")
        if task_type in _READ_INTENT_TASK_TYPES:
            return task_type
    # A direct answer may complete without a read tool observation. Only a
    # single source-validated read task in the frozen outline is unambiguous;
    # multiple/mixed tasks fall back to the neutral caller default.
    outline_types = {
        str(candidate.get("task_type") or "")
        for candidate in (frozen_outline or [])
        if isinstance(candidate, dict) and candidate.get("task_type")
    }
    if len(outline_types) == 1:
        task_type = next(iter(outline_types))
        if task_type in _READ_INTENT_TASK_TYPES:
            return task_type
    return None


def _task_span(source: str, start: int, end: int) -> tuple[int, int]:
    """C1 任务 span：包含写命令的完整句子（真实 source 坐标，左闭右开）。"""
    import re

    boundaries = [match.end() for match in re.finditer(r"[。！？!?；;\n]", source[:start])]
    begin = boundaries[-1] if boundaries else 0
    after = re.search(r"[。！？!?；;\n]", source[end:])
    finish = end + after.start() if after else len(source)
    return begin, finish


def check_confirmation_selector(request) -> None:
    """缓存重放之前的 selector 一致性：同 rid 已冻结的确认任务参数必须完全一致，
    否则保持真实请求冲突（409）。不创建/不修改任何行。"""
    from app.db import session as db
    from app.services.agent import execution_ledger_store as ledger_api

    factory = db.SessionLocal
    if factory is None:
        return
    selector = getattr(request, "confirmation", None)
    if selector is None:
        return
    exact = {"pending_request_id": str(getattr(selector, "pending_request_id", "") or ""),
             "pending_op_id": str(getattr(selector, "pending_op_id", "") or ""),
             "pending_digest": str(getattr(selector, "pending_digest", "") or "")}
    try:
        snapshot = ledger_api.ExecutionLedgerStore(factory).load_request(
            str(request.user_id), str(request.request_id))
    except Exception:
        return
    if snapshot is None:
        return
    tasks = list(snapshot.get("tasks") or [])
    if not tasks:
        return
    confirmation_tasks = [task for task in tasks
                          if str(task.get("task_type")) == "nutrition_confirmation"]
    mismatch = (not confirmation_tasks) or any(
        _json_object(task.get("parameters")) != exact for task in confirmation_tasks)
    if mismatch:
        from app.services.agent.execution_state import RequestIdConflict

        raise RequestIdConflict("request_id_payload_conflict")


def _degraded_replay(payload: dict | None) -> dict | None:
    """证明读取失败/缺失时的降级重放：保留可信回执与非空非完成状态。"""
    if not isinstance(payload, dict):
        return None
    degraded = dict(payload)
    degraded["replayed"] = True
    metrics = dict(degraded.get("metrics") or {})
    metrics["react_write"] = "commit_unknown"
    degraded["metrics"] = metrics
    degraded["answer"] = "记录的提交状态尚未确认；请稍后核对是否已写入。"
    return degraded


def _privacy_refused_payload() -> dict[str, Any]:
    """隐私失效拒绝：不返回任何冻结内容、不重放、零模型零写（ChatResponse 形状）。"""
    return {
        "answer": "该会话的隐私设置已变更，本次请求不再恢复或保存；如需记录请重新发起。",
        "intent": "execution_limited",
        "citations": [],
        "safety_level": "normal",
        "model_used": False,
        "metrics": {"react_write": "limited", "react_status": "limited",
                    "mode_reason": "privacy_invalidated",
                    "receipt_refs": [], "pending_confirmations": []},
        "replayed": False,
    }


def _bridge_ops_proven(user_id: str, request_id: str, snapshot: dict, factory, *,
                       allow_reserved: bool) -> bool:
    """每个真实操作必须通过既有闭桥证明读取器核验为 committed（只读、不改桥）：

    - ``allow_reserved``（active 恢复）：仍 reserved 的 protected pending 合法——既非
      完成证明也无需桥证明；其余状态一律要求真实 committed 证明（不伪造 dispatched/
      receipts/counters）；
    - 终态重放：所有操作都必须有真实 committed 证明（reserved 视为未证明）。
    任何读取失败/形状拒绝/未证明一律 False（调用方保守拒绝或降级）。"""
    try:
        from app.services.agent.nutrition_write_bridge import NutritionWriteBridge
        from app.services.agent.workout_write_bridge import WorkoutWriteBridge
    except Exception:
        return False
    for operation in list(snapshot.get("operations") or []):
        status = str(operation.get("status") or "")
        if allow_reserved and status == "reserved":
            continue
        if allow_reserved and status == "dispatch_intent":
            # 合法未认领恢复（load_runtime 已严格核验）：无 claim/无 commit 才放行；
            # 出现 claim/commit 事件时仍必须走真实 committed 证明（不宽 outcome）。
            kind = str(operation.get("kind") or "")
            op_id = str(operation.get("op_id") or "")
            prefixes = ("c2-1-claim:v1:", "c2-1-commit:v1:") if kind == "workout" \
                else ("c2-2-claim:v1:", "c2-2-commit:v1:")
            if kind == "workout_name_correction":
                from app.services.agent.execution_ledger_store import _CORRECTION_EVENT_PREFIX
                prefixes = (_CORRECTION_EVENT_PREFIX, _CORRECTION_EVENT_PREFIX)
            keys = {str(event.get("event_key") or "") for event in snapshot.get("events") or []}
            if op_id and prefixes[0] + op_id not in keys and prefixes[1] + op_id not in keys:
                continue
        kind = str(operation.get("kind") or "")
        op_id = str(operation.get("op_id") or "")
        if not op_id:
            return False
        bridge = (WorkoutWriteBridge(factory) if kind == "workout"
                  else NutritionWriteBridge(factory) if kind == "nutrition" else None)
        if kind == "workout_name_correction":
            from app.services.agent.workout_correction import WorkoutCorrectionBridge
            bridge = WorkoutCorrectionBridge(factory)
        if bridge is None:
            return False
        try:
            truth = bridge._read_result(user_id, request_id, op_id, dispatched=False,
                                        found_outcome=status)
        except Exception:
            return False
        if str((truth or {}).get("outcome") or "") != "committed":
            return False
    return True


def _replay_proof_valid(user_id: str, request_id: str, snapshot: dict, factory) -> bool:
    """终态重放证明（只读、零 DML/零模型）：真实运行时信封 + 全部操作真实闭桥提交证明。"""
    try:
        from app.services.agent.react_checkpoint_store import ReactCheckpointStore

        runtime = ReactCheckpointStore(factory).load_runtime(user_id, request_id)
    except Exception:
        return False
    if runtime is None:
        return False
    return _bridge_ops_proven(user_id, request_id, snapshot, factory, allow_reserved=False)


def unfinished_progress(request) -> dict | None:
    """Read-only committed progress for an early stop on an unfinished request.

    This does not acquire a lease, dispatch a worker or call a model. It only
    reuses the real C1 snapshot: verified tool-result observations provide
    receipts/proofs, while any committed-looking operation without a verified
    receipt remains explicitly unconfirmed.
    """
    from app.db import session as db
    from app.services.agent import execution_ledger_store as ledger_api

    factory = getattr(db, "SessionLocal", None)
    if factory is None:
        return None
    try:
        snapshot = ledger_api.ExecutionLedgerStore(factory).load_request(
            str(request.user_id), str(request.request_id))
    except Exception:
        return None
    if not snapshot:
        return None
    if str(snapshot.get("status") or "") in {"completed", "failed", "limited"}:
        return None
    observations = []
    for event in snapshot.get("events") or []:
        if str(event.get("phase") or "") != "tool_result":
            continue
        observation = _json_object(event.get("observation"))
        if observation:
            observations.append(observation)
    proofs = {
        str((item.get("result") or {}).get("proof_type") or "")
        for item in observations
        if item.get("status") == "completed" and item.get("commit_state") == "committed"
    } & set(_WRITE_INTENT_BY_PROOF)
    receipts: list[str] = []
    for item in observations:
        if item.get("status") != "completed" or item.get("commit_state") != "committed":
            continue
        proof = str((item.get("result") or {}).get("proof_type") or "")
        if proof not in proofs:
            continue
        refs = item.get("receipt_refs")
        if isinstance(refs, list):
            for ref in refs:
                if isinstance(ref, str) and ref and ref not in receipts:
                    receipts.append(ref)
    unconfirmed = any(
        item.get("status") == "completed"
        and item.get("commit_state") in {"unknown", "committed"}
        and (item.get("commit_state") == "unknown"
             or str((item.get("result") or {}).get("proof_type") or "") not in proofs
             or not isinstance(item.get("receipt_refs"), list)
             or not any(isinstance(ref, str) and ref for ref in item.get("receipt_refs") or []))
        for item in observations
    )
    for operation in snapshot.get("operations") or []:
        status = str(operation.get("status") or "")
        if status == "committed" and not receipts:
            unconfirmed = True
        elif status in {"dispatch_intent", "reserved", "commit_unknown"}:
            unconfirmed = True
    response_ids = {}
    for task in snapshot.get("tasks") or []:
        if str(task.get("task_type") or "") == "react_controller":
            response_ids = _json_object(_json_object(task.get("parameters")).get("response_ids"))
            break
    if receipts:
        answer = ("本次请求此前已提交的记录仍保留并保留回执；后续处理未完成，"
                  "没有新增保存。请稍后核对。")
        state = "committed"
    elif unconfirmed:
        answer = ("本次请求此前记录的提交状态尚未确认；后续处理未完成，"
                  "没有新增保存。请稍后核对是否已写入。")
        state = "unknown"
    else:
        answer = "本次请求尚未确认保存任何记录；后续处理未完成，请稍后重试。"
        state = "none"
    return {"answer": answer, "receipt_refs": receipts,
            "response_ids": response_ids, "commit_state": state}


def write_replay(request, fallback: dict | None) -> dict | None:
    """react_write 重放统一入口（只读、零 DML/零模型）：

    - C1 无行：返回内存冻结首响应（``fallback``，由调用方标记 replayed）；
    - C1 终态且有响应：真实运行时信封 + 每个操作的真实闭桥提交证明 + 规范终态响应
      哈希全部核验通过后返回冻结响应；任何声明未证明/claim 未知/证明读取失败一律
      降级（非空非完成，保留可信回执）；
    - 证明读取失败：对 ``fallback`` 降级（非空非完成，保留回执）；
    - C1 未完成：返回 None（调用方进入 restore）。
    """
    from app.db import session as db
    from app.services.agent import execution_ledger_store as ledger_api

    factory = db.SessionLocal
    if factory is None:
        return None
    try:
        snapshot = ledger_api.ExecutionLedgerStore(factory).load_request(
            str(request.user_id), str(request.request_id))
    except Exception:
        return _degraded_replay(fallback)
    if snapshot is None:
        return fallback
    status = str(snapshot.get("status") or "")
    if status not in {"completed", "failed", "limited"}:
        # A source-bound draft question has no C1 operation to resume. Reuse
        # its frozen compact response after privacy validation; asking the
        # same request again must not regenerate or overwrite the proposal.
        fallback_metrics = fallback.get("metrics") if isinstance(fallback, dict) else None
        if (isinstance(fallback_metrics, dict)
                and fallback_metrics.get("react_write") == "needs_clarification"
                and not fallback_metrics.get("receipt_refs")
                and not snapshot.get("operations")):
            try:
                from app.services.agent.record_clarifications import question_replay_valid
                if not question_replay_valid(request, fallback.get("answer")):
                    return _privacy_refused_payload()
            except Exception:
                return _privacy_refused_payload()
            return fallback
        return None
    # 隐私失效（hide/delete/owner epoch 变化）不得返回任何冻结内容：明确受限零写。
    try:
        from app.services.agent.react_checkpoint_store import ReactCheckpointStore
        ReactCheckpointStore(factory).guard_privacy(str(request.user_id), str(request.request_id))
    except Exception:
        return _privacy_refused_payload()
    stored = snapshot.get("response")
    if not isinstance(stored, dict):
        return fallback
    if not _replay_proof_valid(str(request.user_id), str(request.request_id), snapshot, factory):
        return _degraded_replay(stored)
    payload = dict(stored)
    payload["replayed"] = True
    return payload


class WriteRuntimeSession:
    """单请求 C1 会话：身份/租约/controller/runtime 事件/两桥派发/终态。"""

    def __init__(self, agent, request, *, session_factory=None):
        self.agent = agent
        self.request = request
        self.user_id = str(request.user_id)
        self.request_id = str(request.request_id)
        self.conversation_id = str(request.conversation_id)
        self.source = str(request.message)
        self.source_digest = hashlib.sha256(self.source.encode("utf-8")).hexdigest()
        self._factory_override = session_factory
        self.factory = None
        self.checkpoint_owner = None
        self.store = None
        self.capability = None
        self.scope_id = ""
        self.time_base = ""
        self.response_ids: dict | None = None
        self.lease = None
        self.revision = 0
        self.checkpoint = {"supervisor_turns": 0, "tool_steps": 0,
                           "event_sequence": 0, "budget_scope_id": ""}
        self.supervisor_turns = 0
        self.tool_steps = 0
        self.event_sequence = 0
        self.event_counter = 0
        self.intent_tool_attempts: dict[str, int] = {}
        self.receipts: list[str] = []
        self.pending_target: dict | None = None
        self.ai_estimate_committed = False
        self.committed_observations: list[dict] = []
        self.terminal_request_id = self.request_id
        self.resume_assistant: dict | None = None
        self.resume_messages: list[dict] = []
        self.resume_covered: list[str] = []
        self.resume_committed: list[str] = []
        self.resume_goal_results: dict[str, dict] = {}
        self.resume_candidates: list[dict] = []
        self.resume_final_answer = ""
        self.resume_conservative = False
        self.resume_pending_native = False
        self.resume_mode = False
        self.termination_reason = ""
        # 恢复批的已持久化调用（事件已有 tool_result / 已尝试）：绝不重复写事件或重派发。
        self.restored_dispatched: set[str] = set()
        self.restored_attempted: set[str] = set()
        # 真实停止信号（预算/取消）与冻结 controller 参数/期限（恢复 AND 输入）。
        self.policy_stopped = False
        self.budget_stop_reason = ""
        self.consecutive_no_progress = 0
        self.frozen_parameters: dict = {}
        self.frozen_deadline = ""
        self.lease_original = None
        self.revision_original = 0
        self.recovered_terminal: dict | None = None
        self.recovered_journal_hash = ""

    # ------------------------------------------------------------------ setup

    def _db_now_iso(self) -> str:
        from sqlalchemy import select, func

        with self.factory() as session:
            value = self.store.ledger_statement_now(session)
        return value.isoformat()

    def db_date(self) -> str:
        """Asia/Shanghai 日期，来自 C1 首次冻结 time_base（非当前 SQL now/系统时间）。"""
        from datetime import timedelta, timezone

        value = self._frozen_now()
        return value.astimezone(timezone(timedelta(hours=8))).date().isoformat()

    def record_facts(self) -> tuple[str, str]:
        """营养记录的真实时间基准：冻结 time_base 的 Asia/Shanghai 本地时刻与日期
        （与闭桥的 frozen_date/record_date 同一口径）。"""
        from datetime import timedelta, timezone

        local = self._frozen_now().astimezone(timezone(timedelta(hours=8)))
        return local.isoformat(), local.date().isoformat()

    def _frozen_now(self):
        from datetime import datetime, timezone

        value = datetime.fromisoformat(self.time_base)
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value

    def _identity(self, *, source: str | None = None) -> Any:
        from app.services.agent import execution_ledger_store as ledger_api

        text = self.source if source is None else source
        return ledger_api.RequestIdentity(
            user_id=self.user_id, request_id=self.request_id,
            conversation_id=self.conversation_id, source=text,
            source_digest=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            requested_mode="react_write", effective_mode="react_write",
            schema_version=2, tool_version="react-tools-v2",
            time_base=self.time_base, budget_scope_id=self.scope_id,
            request_kind="tasks",
        )

    @staticmethod
    def _restore_source_understanding(snapshot, source: str) -> None:
        """Restore only the original request's persisted model decision."""
        from app.services.agent.semantics import remember_understanding
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        for task in snapshot.get("tasks") or []:
            if task.get("task_type") != "react_controller":
                continue
            params = _json_object(task.get("parameters")) or {}
            semantic = params.get("semantic_understanding")
            if params.get("input_fingerprint") == digest and isinstance(semantic, dict):
                remember_understanding(source, semantic)
            return

    def _apply_snapshot(self, snapshot) -> None:
        self._restore_source_understanding(snapshot, self.source)
        for operation in snapshot.get("operations") or []:
            if operation.get("status") == "committed" and operation.get("kind") == "nutrition":
                params = _json_object(operation.get("parameters")) or {}
                if any(food.get("source") == "ai" for group in params.get("groups") or []
                       for food in group.get("foods") or []):
                    self.ai_estimate_committed = True
        self.revision = int(snapshot["revision"])
        self.checkpoint = dict(snapshot.get("checkpoint") or self.checkpoint)
        self.supervisor_turns = int(self.checkpoint.get("supervisor_turns") or 0)
        self.tool_steps = int(self.checkpoint.get("tool_steps") or 0)
        self.event_sequence = int(self.checkpoint.get("event_sequence") or 0)
        identity = snapshot.get("identity") or {}
        # C1 首次数据库 time_base 是唯一持久基准：恢复同一个冻结值，绝不重取 now。
        frozen_time_base = str(identity.get("time_base") or "")
        if frozen_time_base:
            self.time_base = frozen_time_base
        # 冻结 typed response 身份：恢复路径使用 controller 参数里的真实冻结值。
        for task in snapshot.get("tasks") or []:
            if str(task.get("task_type")) == "react_controller" \
                    or str(task.get("task_id")) == "react-controller-v1":
                frozen_params = _json_object(task.get("parameters")) or {}
                self.frozen_parameters = frozen_params
                deadline = str(frozen_params.get("deadline_at") or "")
                if deadline:
                    self.frozen_deadline = deadline
                frozen_ids = _json_object(frozen_params.get("response_ids"))
                if isinstance(frozen_ids, dict) and all(
                        isinstance(frozen_ids.get(key), str) and frozen_ids.get(key)
                        for key in ("request_id", "message_id", "run_id")):
                    self.response_ids = dict(frozen_ids)
                break
        # 注意：不从快照继承 lease token——本次所有权只能由真实 acquire_lease/CAS 取得。

    def _response_ids(self) -> dict[str, str]:
        """稳定 typed response 身份（恢复路径优先冻结 controller 参数值）。"""
        if isinstance(self.response_ids, dict) and all(
                isinstance(self.response_ids.get(key), str) and self.response_ids.get(key)
                for key in ("request_id", "message_id", "run_id")):
            return dict(self.response_ids)
        from app.services.memory.compact import current_turn
        from app.services.telemetry import active_metrics

        turn_id = current_turn.get() or self.request_id
        metrics = active_metrics.get()
        run_id = getattr(metrics, "run_id", None) or self.request_id
        return {"request_id": self.request_id, "message_id": str(turn_id) + ":assistant",
                "run_id": str(run_id)}

    def _acquire_lease(self, worker_id: str) -> None:
        from app.services.agent import execution_ledger_store as ledger_api

        try:
            snapshot = self.store.acquire_lease(self.user_id, self.request_id, worker_id, 300)
        except ledger_api.ExecutionLedgerError as exc:
            raise WriteRuntimeUnavailable(
                "lease_busy" if exc.code == "lease_busy" else "lease_unavailable") from exc
        self._apply_snapshot(snapshot)
        lease = snapshot.get("lease")
        if not isinstance(lease, dict):
            raise WriteRuntimeUnavailable("lease_unavailable")
        self.lease = ledger_api.LeaseProof(worker_id=lease["worker_id"], token=lease["token"],
                                           epoch=lease["epoch"])

    def _frozen_limit(self, name: str, default: int) -> int:
        """冻结 controller 的逻辑上限（恢复 AND 输入）；非法值 fail closed 用最严默认。"""
        value = (self.frozen_parameters or {}).get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        return default

    def _policy_stop_check(self) -> bool:
        """统一停止入口（无参签名不变）：真实 scope 取消/已预算停止/真实请求期限，
        或冻结 controller 期限已过；任一成立即在任何模型/领域写之前阻断。

        - 真实 ModelScope 只读：``cancelled``、``policy_stops``（真实预算层写入的停止
          集合）与 ``remaining()`` 真实 TTL；不修改 scope/身份/start/caps/store，
          不清 policy_stops；期限不可判定 fail closed。
        - 上游普通失败不会写入 policy_stops，也不会被当作 quota。
        - 冻结 controller 期限（首次冻结写入、恢复沿用同一值）AND 生效：任一期限已过
          即停止，不放宽。
        """
        if self.policy_stopped or self.budget_stop_reason:
            return True
        try:
            from app.services.llm.budget import active_scope

            scope = active_scope.get()
        except Exception:  # noqa: BLE001 - 读取不可判定 → fail closed（不再派发/调用）
            return True
        if scope is None:
            # 写运行时必须有真实请求 scope；无法核验取消/停止/期限时 fail closed。
            return True
        if bool(getattr(scope, "cancelled", False)):
            return True
        if getattr(scope, "policy_stops", None):
            return True
        remaining = getattr(scope, "remaining", None)
        if callable(remaining):
            try:
                if float(remaining()) <= 0:
                    return True
            except Exception:  # noqa: BLE001 - 真实期限不可判定 → fail closed
                return True
        if not self.frozen_deadline:
            return bool(self.frozen_parameters)
        try:
            from datetime import datetime, timezone

            value = datetime.fromisoformat(str(self.frozen_deadline))
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) >= value
        except ValueError:
            # 冻结期限不可解析：fail closed（不再派发/调用）。
            return True

    def _resume_envelope_valid(self) -> bool:
        """恢复前置核验（active 与 terminal 同一证明标准）：真实运行时信封
        （来源协议/代际/序号/自身 checkpoint/终态规范哈希/声明未证明写状态/冻结隐私）
        通过，且已有操作必须通过真实闭桥提交证明；仍 reserved 的 protected pending
        合法（既非完成证明也无需桥证明，绝不误当 committed）。任一不可信 fail closed。"""
        try:
            from app.services.agent.react_checkpoint_store import ReactCheckpointStore

            runtime = ReactCheckpointStore(self.factory).load_runtime(self.user_id, self.request_id)
            if runtime is None:
                return False
            snapshot = self.store.load_request(self.user_id, self.request_id)
            if snapshot is None:
                return False
            return _bridge_ops_proven(self.user_id, self.request_id, snapshot, self.factory,
                                      allow_reserved=True)
        except Exception:  # noqa: BLE001 - 读取失败一律不恢复
            logger.warning("c2_3_resume_envelope_check_failed", exc_info=True)
            return False

    def setup(self) -> str:
        """返回 ``fresh`` / ``resume`` / ``terminal``。"""
        from app.db import session as db

        self.factory = self._factory_override or db.SessionLocal
        if self.factory is None:
            raise WriteRuntimeUnavailable("app_session_factory_unavailable")
        from app.services.agent.react_checkpoint_store import ReactCheckpointStore

        self.checkpoint_owner = ReactCheckpointStore(self.factory)
        self.store = self.checkpoint_owner._ledger
        self.capability = self.checkpoint_owner._capability
        from app.services.llm.budget import active_scope

        scope = active_scope.get()
        self.scope_id = str(getattr(scope, "quota_id", "") or "")
        if not self.scope_id:
            raise WriteRuntimeUnavailable("model_budget_scope_required")
        self.time_base = self._db_now_iso()
        existing = self.store.load_request(self.user_id, self.request_id)
        if existing is not None:
            # 已有请求行：time_base 以真实冻结值为准（跨日/重启/改配置都不更新）。
            frozen_time_base = str((existing.get("identity") or {}).get("time_base") or "")
            if frozen_time_base:
                self.time_base = frozen_time_base
        snapshot = self.store.ensure_request(self._identity())
        status = snapshot["status"]
        if status in {"completed", "failed", "limited"}:
            self._apply_snapshot(snapshot)
            return "terminal"
        self._apply_snapshot(snapshot)
        events = [event for event in snapshot.get("events") or []
                  if str(event.get("event_key", "")).startswith("c2-3-runtime:v1:")]
        if events and not self._resume_envelope_valid():
            # 来源/序号/声明未证明/隐私任一不可信：在真实租约与任何派发之前 fail closed，
            # 零新模型零领域写；绝不把不可信恢复当普通 resume 继续。只读重建真实回执/
            # 目标状态供诚实受限响应使用（不写行、不派发）。
            self.termination_reason = "runtime_provenance_untrusted"
            try:
                self._restore_from_events(events)
            except Exception:  # noqa: BLE001 - 重建失败不影响 fail closed
                logger.warning("c2_3_refused_resume_restore_failed", exc_info=True)
            return "refused"
        if events and self._journal_pending_response(events) is not None:
            # response-save 真实故障：completed journal 已唯一落盘、响应未保存；
            # 恢复原规范冻结响应（同哈希、同 lease/CAS），不重建普通 payload。
            self._acquire_lease("c2-3-runtime")
            return "terminal_recover"
        # 本次所有权：真实 acquire_lease（活动 lease 拒绝、过期按真实 DB 时间接管）。
        self._acquire_lease("c2-3-runtime")
        if events:
            self._restore_from_events(events)
            return "resume"
        return "fresh"

    # ----------------------------------------------------------------- events

    def _append(self, *, phase: str, observation: dict, tool_call_id: str, batch_id: str,
                task_status: str = "running", op_id: str | None = None,
                operation_state: str | None = None, confirmed: bool = False,
                task_id: str | None = None) -> dict:
        from app.services.agent import execution_ledger_store as ledger_api

        # 事件序号/计数以真实行当前 checkpoint 为准（桥事件也会推进序号）。
        snapshot = self.store.load_request(self.user_id, self.request_id)
        current = dict((snapshot or {}).get("checkpoint") or self.checkpoint)
        base_sequence = int(current.get("event_sequence") or 0)
        self.supervisor_turns = max(self.supervisor_turns, int(current.get("supervisor_turns") or 0))
        self.tool_steps = max(self.tool_steps, int(current.get("tool_steps") or 0))
        self.revision = int((snapshot or {}).get("revision") or self.revision)
        self.event_counter += 1
        checkpoint = ledger_api.Checkpoint(
            supervisor_turns=self.supervisor_turns, tool_steps=self.tool_steps,
            event_sequence=base_sequence + 1, budget_scope_id=self.scope_id)
        event = ledger_api.EventInput(
            event_key=f"c2-3-runtime:v1:{phase}:{self.event_counter}:{tool_call_id}",
            tool_call_id=tool_call_id, batch_id=batch_id,
            task_id=task_id or self.controller_task_id, op_id=op_id, phase=phase,
            observation=dict(observation), task_status=task_status,
            operation_state=operation_state, confirmed=confirmed)
        with self.factory() as session:
            appended = self.store.ledger_append_runtime_event(
                session, self.user_id, self.request_id, event, checkpoint=checkpoint,
                expected_revision=self.revision, lease=self.lease, capability=self.capability)
            session.commit()
        self.revision = int(appended["revision"])
        self.event_sequence = checkpoint.event_sequence
        self.checkpoint = {"supervisor_turns": self.supervisor_turns, "tool_steps": self.tool_steps,
                           "event_sequence": self.event_sequence, "budget_scope_id": self.scope_id}
        return appended

    @property
    def controller_task_id(self) -> str:
        from app.services.agent.react_checkpoint_store import CONTROLLER_TASK_ID

        return CONTROLLER_TASK_ID

    # --------------------------------------------------------------- restore

    def _restore_from_events(self, events: list[dict]) -> None:
        """从真实持久事件重建图输入：消息/计数器/已覆盖目标/待派发 native 调用。"""
        from app.services.agent.react_graph import (_WRITE_RUNTIME_ACTIVE,
                                                    _materialize_write_subgoal,
                                                    _matching_candidates, _rule_candidates)
        from app.services.agent.react_tools import TOOL_CATALOG
        from app.services.agent.react_models import SourceScope, ToolCall

        candidates, _error = _rule_candidates(self.source)
        state = {"intent_candidates": candidates, "source_text": self.source, "write_runtime": True}
        self.event_counter = len(events)
        source_scope = SourceScope(text=self.source, intent_candidates=candidates)
        self.resume_messages = []
        last_assistant: dict | None = None
        assistant_committed = False
        dispatched: set[str] = set()
        attempted: set[str] = set()
        intents: dict[str, dict] = {}
        pending_ops: dict[str, str] = {}
        last_phase = ""
        for event in events:
            phase = str(event.get("phase") or "")
            last_phase = phase
            observation = _json_object(event.get("observation"))
            observation.pop("_react_runtime_origin", None)
            tool_call_id = str(event.get("tool_call_id") or "")
            value = observation.get("consecutive_no_progress")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                self.consecutive_no_progress = value
            if phase == "tool_intent":
                attempted.add(tool_call_id)
                intent = {"tool_name": str(observation.get("tool_name") or ""),
                          "args": observation.get("args")}
                intents[tool_call_id] = intent
                spec = TOOL_CATALOG.get(str(intent.get("tool_name") or ""))
                args = intent.get("args") if isinstance(intent.get("args"), dict) \
                    else {"source_ref": None}
                if spec is not None and spec.permission == "write":
                    # Rebuild the same action subgoal identity the fresh graph
                    # materialized before dispatch, so resume can map this
                    # committed event to the exact action instead of the whole
                    # multi-action parent.
                    call = ToolCall(id=tool_call_id, name=spec.name, args=args)
                    token = _WRITE_RUNTIME_ACTIVE.set(True)
                    try:
                        _materialize_write_subgoal(call, spec, state)
                    finally:
                        _WRITE_RUNTIME_ACTIVE.reset(token)
                continue
            if phase == "model_result":
                assistant = observation.get("assistant")
                if isinstance(assistant, dict):
                    last_assistant = copy.deepcopy(assistant)
                    last_assistant.pop("_react_runtime_origin", None)
                    assistant_committed = False
            elif phase == "tool_result":
                if last_assistant is not None and not assistant_committed:
                    # 保留完整原批 assistant（供未派发调用接续），不因首个 tool_result 丢弃。
                    self.resume_messages.append(copy.deepcopy(last_assistant))
                    assistant_committed = True
                dispatched.add(tool_call_id)
                self.resume_messages.append({
                    "role": "tool", "tool_call_id": tool_call_id,
                    "content": json.dumps(observation, ensure_ascii=False, allow_nan=False)})
                intent = intents.get(tool_call_id) or {}
                spec = TOOL_CATALOG.get(str(intent.get("tool_name") or ""))
                if spec is None:
                    continue
                args = intent.get("args") if isinstance(intent.get("args"), dict) \
                    else {"source_ref": None}
                if spec.permission != "write":
                    # 已真实持久化的只读完成结果按应用目标身份恢复覆盖/答复，恢复后
                    # 不重派已完成worker；辅助读没有目标身份，不进入goal_results。
                    # 非completed的部分/拒绝/澄清结果不在此处提升为完成覆盖。
                    if not spec.task_type:
                        continue
                    if str(observation.get("status") or "") != "completed":
                        continue
                    call = ToolCall(id=tool_call_id, name=spec.name, args=args)
                    token = _WRITE_RUNTIME_ACTIVE.set(True)
                    try:
                        matched = _matching_candidates(call, spec, state)
                    finally:
                        _WRITE_RUNTIME_ACTIVE.reset(token)
                    for candidate in matched:
                        cid = str(candidate.get("id") or "")
                        if not cid:
                            continue
                        if cid not in self.resume_covered:
                            self.resume_covered.append(cid)
                        self.resume_goal_results[cid] = observation
                    continue
                result = observation.get("result")
                pending = (result.get("pending_confirmation")
                           if isinstance(result, dict) else None)
                if isinstance(pending, dict):
                    pending_op_id = str(pending.get("pending_op_id") or "")
                    if pending_op_id:
                        pending_ops[tool_call_id] = pending_op_id
                if observation.get("status") != "completed" or observation.get("commit_state") != "committed":
                    continue
                self.receipts.extend(str(item) for item in (observation.get("receipt_refs") or []))
                call = ToolCall(id=tool_call_id, name=spec.name, args=args)

                token = _WRITE_RUNTIME_ACTIVE.set(True)
                try:
                    matched = _matching_candidates(call, spec, state)
                finally:
                    _WRITE_RUNTIME_ACTIVE.reset(token)
                for candidate in matched:
                    cid = str(candidate.get("id") or "")
                    if not cid:
                        continue
                    if cid not in self.resume_covered:
                        self.resume_covered.append(cid)
                    self.resume_goal_results[cid] = observation
                    if cid not in self.resume_committed:
                        self.resume_committed.append(cid)
        self.restored_dispatched = set(dispatched)
        self.restored_attempted = set(attempted)
        # 真实确认消费后的原 pending 目标：只读核验原 op 的真实桥提交证明，映射回
        # covered/committed/goal_results/回执（未提交/未知/无证明一律保持未完成）。
        if pending_ops and self.factory is not None:
            try:
                from app.services.agent.nutrition_write_bridge import NutritionWriteBridge
                from app.services.agent.workout_write_bridge import WorkoutWriteBridge

                for call_id, pending_op_id in pending_ops.items():
                    with self.factory() as session:
                        op_row = self.store.ledger_operation_row(
                            session, self.user_id, self.request_id, pending_op_id)
                    kind = str(getattr(op_row, "kind", "") or "")
                    if op_row is None or str(getattr(op_row, "status", "")) != "committed":
                        continue
                    bridge = (WorkoutWriteBridge(self.factory) if kind == "workout"
                              else NutritionWriteBridge(self.factory) if kind == "nutrition"
                              else None)
                    if bridge is None:
                        continue
                    truth = bridge._read_result(self.user_id, self.request_id, pending_op_id,
                                                dispatched=False, found_outcome="committed")
                    if str((truth or {}).get("outcome") or "") != "committed":
                        continue
                    observation = _json_object((truth or {}).get("observation")) or {}
                    intent = intents.get(call_id) or {}
                    spec = TOOL_CATALOG.get(str(intent.get("tool_name") or ""))
                    if spec is None or spec.permission != "write":
                        continue
                    args = intent.get("args") if isinstance(intent.get("args"), dict) else {"source_ref": None}
                    call = ToolCall(id=call_id, name=spec.name, args=args)
                    token = _WRITE_RUNTIME_ACTIVE.set(True)
                    try:
                        matched = _matching_candidates(call, spec, state)
                    finally:
                        _WRITE_RUNTIME_ACTIVE.reset(token)
                    for candidate in matched:
                        cid = str(candidate.get("id") or "")
                        if not cid:
                            continue
                        if cid not in self.resume_covered:
                            self.resume_covered.append(cid)
                        self.resume_goal_results[cid] = observation
                        if cid not in self.resume_committed:
                            self.resume_committed.append(cid)
                    self.receipts.extend(str(item) for item in (observation.get("receipt_refs") or []))
            except Exception:  # noqa: BLE001 - 只读映射失败保持未完成（保守）
                logger.warning("c2_3_pending_commit_mapping_failed", exc_info=True)
        self.resume_candidates = list(state.get("intent_candidates") or [])
        if last_assistant is not None:
            tool_calls = last_assistant.get("tool_calls")
            pending_ids = [str(item.get("id")) for item in tool_calls or []
                           if isinstance(item, dict) and str(item.get("id")) not in dispatched]
            if pending_ids:
                if not assistant_committed:
                    # 尚无 tool 结果：原完整 batch 直接作为最后一条 assistant 消息
                    # （原 call ID/参数），图从原 assistant 续接。
                    self.resume_messages.append(copy.deepcopy(last_assistant))
                # 原完整 batch 保留在历史中（其后为已持久化 tool 结果）；未派发/结果未知
                # 的 cursor 由独立 persistence 判定，绝不复制第二条 assistant 或丢后续目标。
                self.resume_pending_native = True
            else:
                content = last_assistant.get("content")
                if isinstance(content, str) and content.strip():
                    self.resume_final_answer = content
        if last_phase == "model_intent":
            # 已付费但结果未知：恢复必须保守受限，绝不重复付费调用。
            self.resume_conservative = True

    # ----------------------------------------------------------- persistence

    async def model_intent(self, state: dict) -> None:
        # 尝试先于真实模型调用计数：中断/恢复后计数不丢（checkpoint 单调，恢复沿用）。
        self.supervisor_turns += 1
        self._append(phase="model_intent", observation={"source_digest": self.source_digest},
                     tool_call_id=f"model-intent-{self.supervisor_turns}",
                     batch_id="c2-3-model", task_status="running")

    async def model_result(self, assistant: dict, state: dict) -> None:
        self._append(phase="model_result", observation={"assistant": copy.deepcopy(assistant)},
                     tool_call_id=f"model-result-{self.supervisor_turns}",
                     batch_id="c2-3-model", task_status="running")

    async def tool_intent(self, call, state: dict) -> None:
        attempt = self.intent_tool_attempts.get(call.id, 0) + 1
        self.intent_tool_attempts[call.id] = attempt
        if call.id in self.restored_dispatched:
            # 恢复批中已持久化结果的调用：不重复写事件、不重复计数。
            return
        # 尝试先于危险派发持久化计数：派发被中断时该尝试仍真实计入。
        self.tool_steps += 1
        self._append(phase="tool_intent",
                     observation={"tool_name": call.name, "args": copy.deepcopy(call.args)},
                     tool_call_id=call.id, batch_id="c2-3-tools", task_status="running",
                     op_id=self._op_for_call(call))

    async def tool_result(self, call, observation, state: dict) -> None:
        if call.id in self.restored_dispatched:
            return
        persisted = observation.to_dict()
        count = state.get("consecutive_no_progress")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            # 与真实结果同一持久状态保存累计无进展（恢复读到的即实际值）。
            persisted["consecutive_no_progress"] = count
        self._append(phase="tool_result", observation=persisted,
                     tool_call_id=call.id, batch_id="c2-3-tools", task_status="running",
                     op_id=self._op_for_call(call))

    def _op_for_call(self, call) -> str | None:
        return None

    # ------------------------------------------------------------- controller

    def _semantic_snapshot(self) -> dict:
        from app.services.agent.semantics import get_understanding
        return get_understanding(self.source) or {}

    def freeze_controller(self) -> None:
        from app.core.config import settings
        from app.services.agent import execution_ledger_store as ledger_api
        from app.services.agent.react_graph import (MAX_NO_PROGRESS, MAX_SUPERVISOR_TURNS,
                                                    MAX_TOOL_STEPS)
        from app.services.memory.compact import current_turn
        from app.services.telemetry import active_metrics

        turn_id = current_turn.get() or self.request_id
        metrics = active_metrics.get()
        run_id = getattr(metrics, "run_id", None) or self.request_id
        scope = None
        from app.services.llm.budget import active_scope

        scope = active_scope.get()
        parameters = {
            "runtime_version": RUNTIME_VERSION,
            "input_fingerprint": self.source_digest,
            "turn_id": str(turn_id),
            "memory_epoch": self._memory_epoch(),
            "started_at": self.time_base,
            "deadline_at": self._deadline_iso(),
            "model_quota_id": self.scope_id,
            "model_budget_policy": ("per_task_v1"
                                    if getattr(scope, "policy", None) == "per_task"
                                    else "shared"),
            "model_task_policy": "uniform_v2",
            "model_task_max_calls": int(getattr(settings, "model_task_max_calls", 12)),
            "model_task_max_tokens": int(getattr(settings, "model_task_max_tokens", 64000)),
            "model_prepare_max_calls": int(getattr(settings, "model_prepare_max_calls", 8)),
            "max_model_calls": int(getattr(settings, "model_request_max_calls", 12)),
            "max_model_tokens": int(getattr(settings, "model_request_max_tokens", 64000)),
            "max_supervisor_turns": MAX_SUPERVISOR_TURNS,
            "max_tool_steps": MAX_TOOL_STEPS,
            "max_no_progress": MAX_NO_PROGRESS,
            "task_outline": self._semantic_snapshot().get("validated_tasks") or [],
            "semantic_understanding": self._semantic_snapshot(),
            "auto_save_ai_estimates": True,
            "response_ids": {"request_id": self.request_id,
                             "message_id": str(turn_id) + ":assistant",
                             "run_id": str(run_id)},
        }
        with self.factory() as session:
            existing = self.store.ledger_task_row(session, self.user_id, self.request_id,
                                                  self.controller_task_id)
        if existing is not None:
            frozen_params = _json_object(getattr(existing, "parameters", None)) or {}
            self.frozen_parameters = frozen_params
            deadline = str(frozen_params.get("deadline_at") or "")
            if deadline:
                self.frozen_deadline = deadline
            frozen_ids = _json_object(frozen_params.get("response_ids"))
            if isinstance(frozen_ids, dict) and all(
                    isinstance(frozen_ids.get(key), str) and frozen_ids.get(key)
                    for key in ("request_id", "message_id", "run_id")):
                self.response_ids = dict(frozen_ids)
            return
        self.response_ids = dict(parameters["response_ids"])
        task = ledger_api.TaskSpec(
            task_id=self.controller_task_id, source_start=0, source_end=len(self.source),
            source_digest=self.source_digest, task_type="react_controller", dependencies=[],
            parameters=parameters, required=False)
        snapshot = self.store.freeze_task(self.user_id, self.request_id, task,
                                          expected_revision=self.revision, lease=self.lease)
        self._apply_snapshot(snapshot)

    def _memory_epoch(self) -> int:
        try:
            from sqlalchemy import select

            from app.db.models import UserProfileModel

            with self.factory() as session:
                profile = session.execute(
                    select(UserProfileModel).where(UserProfileModel.user_id == self.user_id)
                ).scalar_one_or_none()
                return int(getattr(profile, "memory_epoch", 0) or 0)
        except Exception:
            return 0

    def _deadline_iso(self) -> str:
        from datetime import datetime

        started = datetime.fromisoformat(self.time_base)
        return (started + timedelta(seconds=90)).isoformat()

    # ------------------------------------------------------------- dispatch

    async def estimate_food(self, candidate: dict) -> dict | None:
        """有界估算：仅 AI 未知食物；格式失败最多再试一次，预算停止不重试。"""
        from app.services.llm.budget import model_stage

        entry = candidate.get("entry") or {}
        name = str(entry.get("requested_name") or "")
        weight = entry.get("weight_g")
        prompt = (
            "你是食物营养估算助手。请只输出一个 JSON 对象，字段为 "
            '{"food_name": 名称, "preparation_state": "unknown", "kcal": 每100克千卡, '
            '"protein_g": 每100克蛋白质克, "carbs_g": 每100克碳水克, "fat_g": 每100克脂肪克, '
            '"fiber_g": 每100克纤维克, "notes": "", "confidence": 0到1之间}。'
            f"食物：{name}；份量克数：{weight if weight is not None else '未知'}。"
        )
        messages = [{"role": "system", "content": "只输出严格 JSON，不要额外文字。"},
                    {"role": "user", "content": prompt}]
        attempts = 2
        with model_stage(ESTIMATE_STAGE, max_calls=attempts):
            for attempt in range(attempts):
                try:
                    assistant = await self.agent.graph.llm.complete_message(
                        messages, temperature=0.0)
                except Exception as exc:
                    from app.services.agent.react_graph import is_budget_policy_stop
                    from app.services.llm.budget import is_task_budget_stop

                    if is_task_budget_stop(exc):
                        logger.warning("c2_3_food_estimate_task_budget_stopped")
                        return None
                    if is_budget_policy_stop(exc):
                        self.policy_stopped = True
                        self.budget_stop_reason = "model_budget_exhausted"
                        logger.warning("c2_3_food_estimate_budget_stopped: %s",
                                       type(exc).__name__)
                        return None
                    logger.warning("c2_3_food_estimate_failed", exc_info=True)
                    return None
                parsed = _parse_food_estimate(assistant)
                if parsed is not None:
                    return parsed
                if attempt + 1 < attempts:
                    # Only a format/parse failure is retried. Budget stops and
                    # transport errors returned above; this never re-sends a
                    # request whose cost/authorization is unknown.
                    messages = messages[:1] + [{
                        "role": "user",
                        "content": prompt + "\n上一次输出无法解析。只输出一个 JSON 对象；"
                                             "不要代码块、不要解释；所有数值必须是数字。",
                    }]
        return None

    def _freeze_task(self, *, task_id: str, start: int, end: int, task_type: str,
                     parameters: dict) -> None:
        from app.services.agent import execution_ledger_store as ledger_api

        with self.factory() as session:
            existing = self.store.ledger_task_row(session, self.user_id, self.request_id, task_id)
        if existing is not None:
            return
        task = ledger_api.TaskSpec(
            task_id=task_id, source_start=start, source_end=end,
            source_digest=hashlib.sha256(self.source[start:end].encode("utf-8")).hexdigest(),
            task_type=task_type, dependencies=[], parameters=parameters, required=True)
        snapshot = self.store.freeze_task(self.user_id, self.request_id, task,
                                          expected_revision=self.revision, lease=self.lease)
        self._apply_snapshot(snapshot)

    def _reserve(self, *, task_id: str, position: int, kind: str, parameters: dict,
                 dispatch: bool) -> str:
        from app.services.agent import execution_ledger_store as ledger_api

        op_id, _key = ledger_api._op_identity(self.user_id, self.request_id, task_id, position, kind)
        with self.factory() as session:
            existing = self.store.ledger_operation_row(session, self.user_id, self.request_id, op_id)
        existing_status = getattr(existing, "status", None) if existing is not None else None
        if existing_status is None or existing_status == "reserved":
            snapshot = self.store.reserve_operation(
                self.user_id, self.request_id, task_id=task_id, position=position, kind=kind,
                parameters=parameters, source_digest=self.source_digest, time_base=self.time_base,
                expected_revision=self.revision, lease=self.lease)
            self._apply_snapshot(snapshot)
            if dispatch:
                snapshot = self.store.mark_dispatch_intent(
                    self.user_id, self.request_id, op_id,
                    expected_revision=self.revision, lease=self.lease)
                self._apply_snapshot(snapshot)
        # 非 reserved（dispatch_intent/committed/commit_unknown）：真实闭桥决定
        # replay/refuse，应用不重复状态转换。
        return op_id

    async def correct_workout(self, args: Any, context: dict) -> Any:
        from app.services.agent.workout_correction import WorkoutCorrectionBridge, CorrectionError, KIND
        from app.services.agent.react_models import DomainObservation
        from app.services.agent.react_tools import _candidate_task_id
        if self._policy_stop_check():
            return DomainObservation(status="unavailable",error="policy_stop")
        correction = context.get("workout_correction")
        ref = (args or {}).get("source_ref") or {}
        binding = context.get("candidate_span") or {}
        if not correction or not binding:
            return DomainObservation(status="rejected",error="correction_source_unavailable")
        task_id = _candidate_task_id(self.request_id,"correct_workout",ref["start"],ref["end"])
        bridge = WorkoutCorrectionBridge(self.factory)
        from app.services.agent import execution_ledger_store as ledger_api
        op_id, _ = ledger_api._op_identity(self.user_id,self.request_id,task_id,0,KIND)
        with self.factory() as session:
            old = self.store.ledger_operation_row(session,self.user_id,self.request_id,op_id)
            parameters = copy.deepcopy(old.parameters) if old is not None else None
        if parameters is None:
            try:
                parameters = bridge.resolve(self.user_id,self.conversation_id,correction)
            except CorrectionError as exc:
                return DomainObservation(status="needs_clarification",error=str(exc),
                    missing_fields=["correction_target"],summary=(
                        "上一轮有多条训练记录，请明确要更正哪个动作；本次没有修改。"
                        if str(exc)=="correction_target_ambiguous" else
                        "找不到可确认的原保存记录，请指定要更正的记录；本次没有修改。"))
            parameters["source_ref"] = dict(ref)
        if self._policy_stop_check():
            return DomainObservation(status="unavailable",error="policy_stop")
        self._freeze_task(task_id=task_id,start=binding["start"],end=binding["end"],
            task_type="workout_record",parameters={"correction_operation":parameters})
        op_id=self._reserve(task_id=task_id,position=0,kind=KIND,parameters=parameters,dispatch=True)
        observation=bridge.commit(self,op_id)
        self.revision=self._revision_after_bridge()
        if observation.status.value=="completed" and observation.commit_state=="committed":
            self.receipts.extend(observation.receipt_refs)
            self.committed_observations.append(observation.to_dict())
        return observation

    async def dispatch_workout(self, span: str, record: dict, context: dict) -> Any:
        from app.services.agent import execution_ledger_store as ledger_api
        from app.services.agent.react_models import DomainObservation, ToolStatus
        from app.services.agent.react_tools import _candidate_task_id
        from app.services.agent.workout_write_bridge import WorkoutWriteBridge

        if self._policy_stop_check():
            # 真实取消/预算停止/期限在 intent→派发窗口内出现：最后入口零派发、零预留。
            return DomainObservation(status=ToolStatus.UNAVAILABLE, error="policy_stop")
        ref = context.get("source_ref") or {}
        start, end = int(ref.get("start", 0)), int(ref.get("end", 0))
        candidate_span = context.get("candidate_span")
        if not isinstance(candidate_span, dict) or not isinstance(candidate_span.get("start"), int) \
                or not isinstance(candidate_span.get("end"), int) \
                or not (0 <= candidate_span["start"] <= start and end <= candidate_span["end"]
                        <= len(self.source)):
            raise WriteRuntimeUnavailable("candidate_span_required")
        task_start, task_end = candidate_span["start"], candidate_span["end"]
        task_id = _candidate_task_id(self.request_id, "record_workout", start, end)
        self._freeze_task(task_id=task_id, start=task_start, end=task_end, task_type="workout_record",
                          parameters={"records": [record]})
        op_id = self._reserve(task_id=task_id, position=0, kind="workout", parameters=record,
                              dispatch=True)
        bridge = WorkoutWriteBridge(self.factory)
        result = bridge.commit_workout_operation(
            self.user_id, self.request_id, op_id, expected_revision=self.revision, lease=self.lease,
            tool_call_id=str(context.get("tool_call_id") or "call-c2-3"), batch_id="c2-3-workout")
        self.revision = self._revision_after_bridge()
        observation = DomainObservation.from_dict(result.get("observation"))
        if observation.status.value == "completed" and observation.commit_state == "committed":
            self.receipts.extend(list(observation.receipt_refs or []))
            self.committed_observations.append(observation.to_dict())
        return observation

    async def dispatch_nutrition(self, span: str, batch: dict, context: dict,
                                *, has_ai: bool) -> Any:
        from app.services.agent import execution_ledger_store as ledger_api
        from app.services.agent.nutrition_write_bridge import NutritionWriteBridge
        from app.services.agent.react_models import DomainObservation, ToolStatus
        from app.services.agent.react_tools import _candidate_task_id

        if self._policy_stop_check():
            # 真实取消/预算停止/期限在 intent→派发窗口内出现：最后入口零派发、零预留。
            return DomainObservation(status=ToolStatus.UNAVAILABLE, error="policy_stop")
        ref = context.get("source_ref") or {}
        start, end = int(ref.get("start", 0)), int(ref.get("end", 0))
        candidate_span = context.get("candidate_span")
        if not isinstance(candidate_span, dict) or not isinstance(candidate_span.get("start"), int) \
                or not isinstance(candidate_span.get("end"), int) \
                or not (0 <= candidate_span["start"] <= start and end <= candidate_span["end"]
                        <= len(self.source)):
            raise WriteRuntimeUnavailable("candidate_span_required")
        task_start, task_end = candidate_span["start"], candidate_span["end"]
        task_id = _candidate_task_id(self.request_id, "record_nutrition", start, end)
        self._freeze_task(task_id=task_id, start=task_start, end=task_end, task_type="nutrition_record",
                          parameters={"batches": [batch]})
        auto_ai = self.frozen_parameters.get("auto_save_ai_estimates") is True
        dispatch = not has_ai or auto_ai
        op_id = self._reserve(task_id=task_id, position=0, kind="nutrition", parameters=batch,
                              dispatch=dispatch)
        bridge = NutritionWriteBridge(self.factory)
        tool_call_id = str(context.get("tool_call_id") or "call-c2-3")
        if has_ai and not auto_ai:
            result = bridge.prepare_pending_nutrition(
                self.user_id, self.request_id, op_id, expected_revision=self.revision,
                lease=self.lease, tool_call_id=tool_call_id, batch_id="c2-3-nutrition")
        else:
            result = bridge.commit_nutrition_operation(
                self.user_id, self.request_id, op_id, expected_revision=self.revision,
                lease=self.lease, tool_call_id=tool_call_id, batch_id="c2-3-nutrition",
                allow_ai_estimate=auto_ai)
        self.revision = self._revision_after_bridge()
        outcome = str(result.get("outcome") or "")
        observation = DomainObservation.from_dict(result.get("observation"))
        if observation.status.value == "completed" and observation.commit_state == "committed":
            self.ai_estimate_committed = self.ai_estimate_committed or has_ai
            # 真实提交/重放（replayed 等）下观察本身即完整提交证明：恢复原冻结回执，
            # 与 dispatch_workout 同一判据；绝不重复业务写。
            self.receipts.extend(list(observation.receipt_refs or []))
            self.committed_observations.append(observation.to_dict())
        elif outcome == "pending_ready":
            pending = result.get("pending") or {}
            self.pending_target = {
                "pending_request_id": self.request_id,
                "pending_op_id": str(pending.get("op_id") or op_id),
                "pending_digest": str(pending.get("pending_digest") or ""),
                "expires_at": str(pending.get("expires_at") or ""),
            }
            observation.result = {"pending_confirmation": dict(self.pending_target)}
            observation.missing_fields = ["pending_confirmation"]
        return observation

    def _revision_after_bridge(self) -> int:
        snapshot = self.store.load_request(self.user_id, self.request_id)
        if snapshot is None:
            return self.revision
        self.checkpoint = dict(snapshot.get("checkpoint") or self.checkpoint)
        self.event_sequence = int(self.checkpoint.get("event_sequence") or self.event_sequence)
        return int(snapshot["revision"])

    # ------------------------------------------------------------------- run

    async def run(self) -> dict[str, Any]:
        mode = self.setup()
        self.resume_mode = (mode == "resume")
        if self.request.confirmation is not None:
            # selector/身份一致性在任何终态捷径之前核验：同 rid 加 selector 或改变
            # 已冻结 selector 都是请求冲突（409），不因请求已完成而降级为普通响应。
            self._check_confirmation_identity()
        if mode == "terminal":
            return self._terminal_payload_verified()
        if mode == "terminal_recover":
            return await asyncio.to_thread(self._recover_terminal)
        if mode == "refused":
            # 恢复信封不可信（来源/序号/声明未证明/隐私）：零模型零写，诚实受限。
            return self._unfinished_payload({})
        if self.request.confirmation is not None:
            return await self._run_confirmation()
        from app.services.agent.semantics import understand, get_understanding, understanding_unavailable_reply
        semantic = get_understanding(self.source)
        if semantic is None and mode == "fresh":
            semantic = await understand(self.agent.graph.llm, self.source)
        if semantic is None:
            self.termination_reason = "semantic_understanding_unavailable"
            payload = self._unfinished_payload({})
            payload["final_answer"] = understanding_unavailable_reply(record_mode=True)
            return payload
        from app.services.agent.semantics import safety_for
        safety = safety_for(self.source)
        if safety.get("status") != "assessed":
            self.termination_reason = "safety_assessment_unavailable"
            payload = self._unfinished_payload({})
            payload["final_answer"] = understanding_unavailable_reply(record_mode=True)
            payload["safety_status"] = "unavailable"
            return payload
        if safety.get("level") == "urgent":
            from app.services.agent.react_graph import URGENT_SAFETY_ANSWER
            self.termination_reason = "urgent_safety_gate"
            payload = self._unfinished_payload({})
            payload["final_answer"] = URGENT_SAFETY_ANSWER
            payload["safety_level"] = "urgent"
            payload["safety_status"] = "assessed"
            return payload
        if semantic.get("needs_clarification"):
            from app.services.agent import record_clarifications

            self.termination_reason = "semantic_clarification"
            payload = self._unfinished_payload({})
            try:
                question = await record_clarifications.create_question(
                    self.agent.graph.llm, self.request, semantic)
            except record_clarifications.DraftError:
                question = None
                payload["final_answer"] = record_clarifications.unavailable_response().answer
            else:
                payload["final_answer"] = (question or semantic.get("clarification")
                                           or understanding_unavailable_reply(record_mode=True))
            if question:
                response = await record_clarifications.question_response(self.request, question)
                payload["clarifications"] = [item.model_dump(mode="json")
                                             for item in response.clarifications]
                payload["react_write_status"] = "needs_clarification"
                payload["react_status"] = "needs_clarification"
            return payload
        return await self._run_graph()

    def _check_confirmation_identity(self) -> None:
        selector = self.request.confirmation
        exact_selector = {
            "pending_request_id": str(getattr(selector, "pending_request_id", "") or ""),
            "pending_op_id": str(getattr(selector, "pending_op_id", "") or ""),
            "pending_digest": str(getattr(selector, "pending_digest", "") or ""),
        }
        self_tasks = self._self_tasks()
        if not self_tasks:
            return
        confirmation_tasks = [task for task in self_tasks
                              if str(task.get("task_type")) == "nutrition_confirmation"]
        mismatch = (not confirmation_tasks) or any(
            _json_object(task.get("parameters")) != exact_selector for task in confirmation_tasks)
        if mismatch:
            from app.services.agent.execution_state import RequestIdConflict

            self.release_lease_quiet()
            raise RequestIdConflict("request_id_payload_conflict")

    def _journal_pending_response(self, events: list[dict]) -> dict | None:
        """response-save 故障恢复：唯一 completed journal 携带的原规范响应。

        只读：从真实事件读 journal 的 canonical_response 并校验其规范哈希等于 journal
        response_hash；响应已保存/缺失/哈希不符/非 completed 一律 None（保守）。"""
        snapshot = self.store.load_request(self.user_id, self.request_id)
        if snapshot is None:
            return None
        journal = None
        previous_hashes: list[str] = []
        for event in events:
            if str(event.get("phase") or "") != "final_reply":
                continue
            if isinstance(journal, dict):
                previous_hashes.append(str(journal.get("response_hash") or ""))
            journal = _json_object(event.get("observation"))
        if not isinstance(journal, dict) or str(journal.get("status") or "") != "completed":
            return None
        canonical = journal.get("canonical_response")
        response_hash = str(journal.get("response_hash") or "")
        if not isinstance(canonical, dict) or not response_hash:
            return None
        if _canonical_hash(canonical) != response_hash:
            return None
        stored = snapshot.get("response")
        if isinstance(stored, dict):
            # 已保存响应只能是历史 partial 投影（与较早 journal 同哈希）：真实非终态
            # request + 新 completed journal 待 finish 窗口；否则保守拒绝。
            stored_hash = _canonical_hash(stored)
            if stored_hash is None or stored_hash not in previous_hashes:
                return None
            if str(snapshot.get("status") or "") in {"completed", "failed", "limited"}:
                return None
        self.recovered_terminal = dict(canonical)
        self.recovered_journal_hash = response_hash
        return self.recovered_terminal

    def _recover_terminal(self) -> dict[str, Any]:
        """response-save 故障恢复：以原规范响应同哈希复用唯一 journal，同 lease/CAS
        完成真实终态；零业务重写/零新模型，绝不写新假哈希或第二 final journal。"""
        canonical = dict(self.recovered_terminal or {})
        response_hash = _canonical_hash(canonical)
        if response_hash is None or response_hash != self.recovered_journal_hash:
            self.termination_reason = "terminal_journal_mismatch"
            return self._unfinished_payload({})
        self.finalize(canonical)
        metrics = _json_object(canonical.get("metrics"))
        intent = str(canonical.get("intent") or "execution_limited")
        return {
            "final_answer": str(canonical.get("answer") or ""),
            "worker_results": [{"worker": "react_write", "status": "completed", "answer": ""}],
            "citations": list(canonical.get("citations") or []),
            "model_used": bool(canonical.get("model_used")),
            "safety_level": str(canonical.get("safety_level") or "normal"),
            "current_task": {"task_type": intent},
            "react_status": "completed",
            "intent": intent,
            "react_write_status": "completed",
            "mode_reason": "recovered_terminal",
            "receipt_refs": list(metrics.get("receipt_refs") or []),
            "pending_confirmations": list(metrics.get("pending_confirmations") or []),
            "replayed": True,
            "request_id": canonical.get("request_id"),
            "message_id": canonical.get("message_id"),
            "run_id": canonical.get("run_id"),
            "response_ids": {"request_id": canonical.get("request_id"),
                             "message_id": canonical.get("message_id"),
                             "run_id": canonical.get("run_id")},
        }

    def _terminal_payload_verified(self) -> dict[str, Any]:
        """终态公开入口统一核验：真实运行时信封 + 全部操作真实桥 proof + 冻结隐私；
        任一不可信返回诚实受限（零新模型/业务写），绝不转发冻结内容。"""
        snapshot = self.store.load_request(self.user_id, self.request_id)
        if snapshot is None or not _replay_proof_valid(
                self.user_id, self.request_id, snapshot, self.factory):
            self.termination_reason = "terminal_proof_untrusted"
            return self._unfinished_payload({})
        return self._terminal_payload()

    def _terminal_payload(self) -> dict[str, Any]:
        snapshot = self.store.load_request(self.user_id, self.request_id)
        stored = dict(snapshot.get("response") or {}) if snapshot else {}
        metrics = _json_object(stored.get("metrics"))
        intent = str(stored.get("intent") or "execution_limited")
        return {
            "final_answer": str(stored.get("answer") or ""),
            "worker_results": [{"worker": "react_write", "status": "completed", "answer": ""}],
            "citations": list(stored.get("citations") or []),
            "model_used": bool(stored.get("model_used")),
            "safety_level": str(stored.get("safety_level") or "normal"),
            "current_task": {"task_type": intent},
            "react_status": "completed" if str(metrics.get("react_write") or "") == "completed"
                            else "limited",
            "intent": intent,
            "react_write_status": "completed" if stored else "limited",
            "mode_reason": "replay_terminal",
            "receipt_refs": list(metrics.get("receipt_refs") or []),
            "pending_confirmations": list(metrics.get("pending_confirmations") or []),
            "replayed": True,
            "response_ids": self._response_ids(),
            "request_id": stored.get("request_id"),
            "message_id": stored.get("message_id"),
            "run_id": stored.get("run_id"),
        }

    @staticmethod
    def _pre_dispatch_gate_reason(value: Any) -> bool:
        """True only for an application-owned, non-empty pre-dispatch gate reason.

        The terminal shortcut must never infer "not dispatched" from a missing
        field or from a reused/unknown-result marker. Ordinary gate reasons are
        kept open-ended so a new application gate remains valid, but status-only
        sentinels are not dispatch proofs.
        """
        if not isinstance(value, str) or not value.strip():
            return False
        return value.strip() not in {"restored_result_reused", "attempted_result_unknown"}

    def _rejected_before_dispatch(self, state: dict) -> bool:
        """True only when every write call was stopped by an application gate.

        This is a real execution fact (call_facts dispatched/gate_reason), not an
        empty receipt or zero-row delta. Operations/claim/commit events make the
        terminal untrusted for this shortcut; unknown keeps its existing path.
        """
        facts = [item for item in state.get("call_facts") or [] if isinstance(item, dict)]
        writes = [item for item in facts if item.get("permission") == "write"]
        if not writes:
            return False
        if any(item.get("dispatched") is not False for item in writes):
            return False
        if any(not self._pre_dispatch_gate_reason(item.get("gate_reason"))
               for item in writes):
            return False
        snapshot = self.store.load_request(self.user_id, self.request_id)
        if snapshot is None:
            return False
        if list(snapshot.get("operations") or []):
            return False
        for event in snapshot.get("events") or []:
            key = str(event.get("event_key") or "")
            if key.startswith(("c2-1-claim:v1:", "c2-2-claim:v1:",
                               "c2-1-commit:v1:", "c2-2-commit:v1:")):
                return False
        return True


    async def _run_graph(self) -> dict[str, Any]:
        from app.services.agent.react_graph import (MAX_NO_PROGRESS, MAX_SUPERVISOR_TURNS,
                                                    MAX_TOOL_STEPS, ReactDependencies,
                                                    invoke_react)
        from app.services.agent.react_write_adapters import WriteAdapters, write_catalog

        # controller 冻结必须先于任何终态/恢复捷径：final_reply 事件与完成检查
        # 都引用真实 controller 任务。
        self.freeze_controller()
        if self.resume_mode and self.resume_conservative:
            # 付费结果未知：明确受限，零新模型/零领域派发。
            self.termination_reason = "model_result_unavailable"
            return self._unfinished_payload({})
        if self.resume_mode and not self.resume_pending_native and not self.resume_committed:
            stored = self._stored_pending()
            if stored is not None:
                # 未消费的 protected pending：重放同一 pending 目标，零新模型/零新计费。
                return self._pending_replay_payload(stored)
            if self._resume_all_goals_completed():
                # 纯只读恢复：全部真实工具目标已由持久completed结果覆盖，按同一
                # 提取规则交付全部答案，零新模型/零领域派发。
                self.termination_reason = "goals_complete"
                return self._committed_payload({})
            # 无待派发 native 调用且无可信提交（已受限/仅声明未证明）：保守恢复，
            # 零新模型/零领域派发，状态真实非完成。
            self.termination_reason = self.termination_reason or "resume_no_dispatch"
            return self._unfinished_payload({})
        if self.resume_mode and not self.resume_pending_native and self.resume_committed \
                and self._no_unhandled_action_goals(self.source, set(self.resume_covered)):
            # 全部 native 调用已提交、无待派发调用：直接从真实提交证据恢复完成响应，
            # 不重发模型、不重写业务。
            self.termination_reason = "goals_complete"
            return self._committed_payload({})
        adapters = WriteAdapters(self)
        deps = ReactDependencies(
            llm=self.agent.graph.llm,
            tool_executor=adapters.execute,
            tool_catalog=write_catalog(),
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            request_id=self.request_id,
            # 真实停止信号：已置位的预算/取消停止 + 冻结 controller 期限（恢复 AND 输入）。
            policy_stop_check=self._policy_stop_check,
            persistence=self,
            stop_after_batch=True,
            max_supervisor_turns=self._frozen_limit("max_supervisor_turns", MAX_SUPERVISOR_TURNS),
            max_tool_steps=self._frozen_limit("max_tool_steps", MAX_TOOL_STEPS),
            max_no_progress=self._frozen_limit("max_no_progress", MAX_NO_PROGRESS),
        )
        initial: dict[str, Any] = {
            "user_id": self.user_id, "conversation_id": self.conversation_id,
            "request_id": self.request_id,
            # 恢复沿用真实持久逻辑计数/无进展：已消耗上限不得被恢复重置。
            "supervisor_turns": int(self.supervisor_turns or 0),
            "tool_steps": int(self.tool_steps or 0),
            "consecutive_no_progress": int(self.consecutive_no_progress or 0),
        }
        logger.info("c2_3_write_graph_start resume_messages=%d covered=%d committed=%d",
                    len(self.resume_messages), len(self.resume_covered), len(self.resume_committed))
        if self.resume_messages:
            initial["messages"] = copy.deepcopy(self.resume_messages)
        if self.resume_candidates:
            initial["intent_candidates"] = copy.deepcopy(self.resume_candidates)
        if self.resume_covered:
            initial["covered_goals"] = list(self.resume_covered)
            initial["committed_tasks"] = list(self.resume_committed)
            initial["goal_results"] = copy.deepcopy(self.resume_goal_results)
        state = await invoke_react(deps, self.source, **initial)
        self.termination_reason = str(state.get("termination_reason") or "")
        if self.pending_target is not None:
            await self.client_reply_pending()
            return self._pending_payload(state)
        if self.termination_reason in {"supervisor_turn_limit", "tool_step_limit", "no_progress",
                                       "policy_stop", "model_budget_exhausted"}:
            # 上限终止：真实持久终止原因与无进展计数（可审计），部分提交保留回执；
            # 本请求已确定的 limited 终态连同响应真实持久化，使同 rid 重放零新调用。
            await self.client_reply_limited(state)
            payload = self._limited_payload(state)
            finished = await asyncio.to_thread(self._finish_limited, payload)
            if finished:
                # 终态持久化已在服务端清除租约。
                self.lease = None
            else:
                self.release_lease_quiet()
            return payload
        if str(state.get("status") or "") == "rejected" and self._rejected_before_dispatch(state):
            # A gate rejection before any domain dispatch is a trusted terminal:
            # freeze the same honest answer/identity with an application-owned
            # rejection journal so the same rid replays read-only instead of
            # becoming a false commit_unknown.
            self.termination_reason = "rejected_before_dispatch"
            payload = self._unfinished_payload(state, release_lease=False)
            finished = await asyncio.to_thread(
                self._finish_limited, payload, terminal_kind="rejected_before_dispatch")
            if finished:
                self.lease = None
            else:
                self.release_lease_quiet()
            return payload
        # 共享完成判定：只有真实图终态 complete（目标全部由真实观察覆盖）才作为
        # 完成返回；回执只证明已提交子集，未处理目标/受限/部分一律诚实非完成。
        if str(state.get("status") or "") == "complete":
            return self._committed_payload(state)
        return self._unfinished_payload(state)

    async def client_reply_limited(self, state: dict) -> None:
        self._append(phase="client_reply",
                     observation={"termination_reason": self.termination_reason,
                                  "consecutive_no_progress": int(state.get("consecutive_no_progress") or 0),
                                  "supervisor_turns": int(state.get("supervisor_turns") or 0),
                                  "tool_steps": int(state.get("tool_steps") or 0)},
                     tool_call_id="client-reply", batch_id="c2-3-client", task_status="running")
        # 租约由调用方在真实持久 limited 终态之后释放（持久化需要当前租约）。

    def _finish_limited(self, payload: dict[str, Any], *, terminal_kind: str = "limited") -> bool:
        """上限终止是本请求的确定终态：真实持久 limited 响应（ChatResponse 形状，
        保留回执）+ protected final_reply 事件（规范哈希），使同 rid 重放走 C1 终态
        冻结路径且零新调用。``terminal_kind`` 只记录应用已核验的终态类型
        （例如 rejected_before_dispatch），不改变回执/提交证明要求。"""
        from app.services.agent import execution_ledger_store as ledger_api

        ids = self._response_ids()
        task_type = "execution_limited"
        stored = {
            "answer": str(payload.get("final_answer") or ""),
            "intent": task_type,
            "citations": list(payload.get("citations") or []),
            "safety_level": str(payload.get("safety_level") or "normal"),
            "model_used": bool(payload.get("model_used")),
            "request_id": ids["request_id"],
            "message_id": ids["message_id"],
            "run_id": ids["run_id"],
            "metrics": {"react_write": "limited",
                        "mode_reason": self.termination_reason or "limited",
                        "terminal_kind": terminal_kind,
                        "receipt_refs": list(payload.get("receipt_refs") or []),
                        "pending_confirmations": []},
        }
        try:
            response_hash = _canonical_hash(stored)
            if response_hash is None:
                return False
            snapshot = self.store.load_request(self.user_id, self.request_id)
            if snapshot is None:
                return False
            existing = None
            for event in snapshot.get("events") or []:
                if str(event.get("event_key", "")).startswith("c2-3-runtime:v1:final_reply:") \
                        and str(event.get("phase") or "") == "final_reply":
                    observation = _json_object(event.get("observation"))
                    if observation is not None and observation.get("response_hash") == response_hash:
                        existing = event
                        break
            if existing is None:
                current = dict(snapshot.get("checkpoint") or {})
                self.supervisor_turns = max(self.supervisor_turns,
                                            int(current.get("supervisor_turns") or 0))
                self.tool_steps = max(self.tool_steps, int(current.get("tool_steps") or 0))
                self.revision = int(snapshot["revision"])
                checkpoint = ledger_api.Checkpoint(
                    supervisor_turns=self.supervisor_turns, tool_steps=self.tool_steps,
                    event_sequence=int(current.get("event_sequence") or 0) + 1,
                    budget_scope_id=self.scope_id)
                event = ledger_api.EventInput(
                    event_key=f"c2-3-runtime:v1:final_reply:{self.event_counter + 1}:final",
                    tool_call_id="final-reply", batch_id="c2-3-client",
                    task_id=self.controller_task_id, op_id=None, phase="final_reply",
                    observation={"response_hash": response_hash, "status": "limited",
                                 "terminal_kind": terminal_kind},
                    task_status="completed", operation_state=None, confirmed=False)
                with self.factory() as session:
                    appended = self.store.ledger_append_runtime_event(
                        session, self.user_id, self.request_id, event, checkpoint=checkpoint,
                        expected_revision=self.revision, lease=self.lease, capability=self.capability)
                    session.commit()
                self.revision = int(appended["revision"])
                self.event_sequence = checkpoint.event_sequence
            else:
                self.revision = int(snapshot["revision"])
            snapshot = self.store.finish_request(
                self.user_id, self.request_id, status="limited", response=stored,
                termination_reason=self.termination_reason or "limited",
                expected_revision=self.revision, lease=self.lease)
            self._apply_snapshot(snapshot)
            return True
        except Exception:
            logger.warning("c2_3_limited_finish_failed", exc_info=True)
            return False

    def _limited_payload(self, state: dict) -> dict[str, Any]:
        answer, receipts = self._limited_answer(state)
        return {
            "final_answer": answer,
            "worker_results": [{"worker": "react_write", "status": "limited", "answer": answer}],
            "citations": self._citations(state),
            "model_used": True,
            "safety_level": "normal",
            "current_task": {"task_type": "nutrition_record"},
            "react_status": "limited",
            "intent": "execution_limited",
            "react_write_status": "limited",
            "mode_reason": self.termination_reason or "limited",
            "receipt_refs": receipts,
            "pending_confirmations": [],
            "response_ids": self._response_ids(),
        }

    async def client_reply_pending(self) -> None:
        target = self.pending_target or {}
        self._append(phase="client_reply",
                     observation={"pending_op_id": target.get("pending_op_id"),
                                  "pending_digest": target.get("pending_digest"),
                                  "expires_at": target.get("expires_at"),
                                  "termination_reason": "pending_confirmation"},
                     tool_call_id="client-reply", batch_id="c2-3-client", task_status="running")
        # 保持请求非终态但释放租约，允许后续确认请求重新取得原请求租约。
        if self.lease is not None:
            try:
                self.store.release_lease(self.user_id, self.request_id,
                                         expected_revision=self.revision, lease=self.lease)
                self.lease = None
            except Exception:
                logger.warning("c2_3_pending_lease_release_failed", exc_info=True)

    def _pending_payload(self, state: dict) -> dict[str, Any]:
        target = dict(self.pending_target or {})
        answer = "该记录需要你确认后才会保存。请回复“确认保存”。"
        return {
            "final_answer": answer,
            "worker_results": [{"worker": "react_write", "status": "needs_confirmation", "answer": answer}],
            "citations": [],
            "model_used": True,
            "safety_level": "normal",
            "current_task": {"task_type": "nutrition_record"},
            "react_status": "pending",
            "intent": "nutrition_record",
            "react_write_status": "pending",
            "mode_reason": "protected_pending",
            "receipt_refs": [],
            "pending_confirmations": [target] if target else [],
            "response_ids": self._response_ids(),
        }

    def _completed_state(self, state: dict) -> dict:
        """合并本次图状态与恢复/冻结outline：真实目标身份用于最终答复提取。

        ``goal_results``/``covered_goals`` 以本次图状态为准，恢复结果只补缺；
        候选按图状态原顺序，恢复outline只追加未出现的目标。只影响答复选择，
        不改变covered/committed/status判定。
        """
        merged = dict(state or {})
        candidates = [item for item in (merged.get("intent_candidates") or [])
                      if isinstance(item, dict)]
        seen = {str(item.get("id")) for item in candidates if item.get("id")}
        for item in (getattr(self, "frozen_parameters", {}) or {}).get("task_outline") or []:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            cid = str(item["id"])
            if cid in seen:
                continue
            candidates.append(item)
            seen.add(cid)
        if candidates:
            merged["intent_candidates"] = candidates
        goal_results = dict(getattr(self, "resume_goal_results", {}) or {})
        goal_results.update(merged.get("goal_results") or {})
        if goal_results:
            merged["goal_results"] = goal_results
        covered = list(dict.fromkeys([
            *(getattr(self, "resume_covered", []) or []),
            *(merged.get("covered_goals") or []),
        ]))
        if covered:
            merged["covered_goals"] = covered
        committed = list(dict.fromkeys([
            *(getattr(self, "resume_committed", []) or []),
            *(merged.get("committed_tasks") or []),
        ]))
        if committed:
            merged["committed_tasks"] = committed
        return merged

    def _state_answer(self, state: dict, *, include_completed_goals: bool = True) -> str:
        """真实终态文字：图内终态答案优先，其次最后一条已完成观察的真实回答/摘要
        （例如只读建议的真实结果），绝不把缺失输出伪造成完成。

        ``include_completed_goals`` 供完成终态/恢复使用：按应用目标身份与原请求
        顺序收齐已完成只读答案；最后assistant只答最后一问或只是工具调用说明时
        不覆盖其它目标，辅助观察不会被兜底打印。显式受限/不可信路径关闭该规则，
        保持确定性保守文案优先。
        """
        if include_completed_goals:
            from app.services.agent.react_graph import (answer_from_completed_read_goals,
                                                        completed_delivery_goals,
                                                        _final_assistant_text)

            merged = self._completed_state(state)
            goals = completed_delivery_goals(merged)
            if goals:
                final_text = str(merged.get("final_answer") or "").strip()
                if not final_text:
                    final_text = str(getattr(self, "resume_final_answer", "") or "")
                if not final_text:
                    final_text = _final_assistant_text(merged.get("messages") or [])
                answer = answer_from_completed_read_goals(goals, final_text)
                return answer if isinstance(answer, str) else ""
        answer = str(state.get("final_answer") or "").strip()
        if answer:
            return answer
        for observation in reversed(state.get("observations") or []):
            if not isinstance(observation, dict) or observation.get("status") != "completed":
                continue
            summary = observation.get("summary")
            if isinstance(summary, str) and summary.strip():
                return summary
            result = observation.get("result")
            if isinstance(result, dict):
                for key in ("answer", "summary", "text"):
                    value = result.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
        return ""

    def _commit_items(self, state: dict) -> tuple[dict, ...]:
        """所有可携带应用提交证明的规范观察来源（当前批次 + 恢复批次）。"""
        return (
            *self.committed_observations,
            *(state.get("observations") or []),
            *(state.get("goal_results") or {}).values(),
            *(getattr(self, "resume_goal_results", {}) or {}).values(),
        )

    def _commit_proofs(self, state: dict) -> set[str]:
        """已知应用证明类型（与 committed payload 同一生产者，不认模型标签）。"""
        proofs = {
            (item.get("result") or {}).get("proof_type")
            for item in self._commit_items(state)
            if isinstance(item, dict) and item.get("status") == "completed"
            and item.get("commit_state") == "committed"
        }
        return {proof for proof in proofs
                if isinstance(proof, str) and proof in _WRITE_INTENT_BY_PROOF}

    def _verified_commit_receipts(self, state: dict, proofs: set[str]) -> list[str]:
        """只返回与已知证明同源的真实回执；空/伪造回执不能支撑“已保存”。"""
        receipts: list[str] = []
        for item in self._commit_items(state):
            if not isinstance(item, dict) or item.get("status") != "completed" \
                    or item.get("commit_state") != "committed":
                continue
            proof = (item.get("result") or {}).get("proof_type")
            if not isinstance(proof, str) or proof not in proofs:
                continue
            refs = item.get("receipt_refs")
            if not isinstance(refs, list):
                continue
            for ref in refs:
                if isinstance(ref, str) and ref and ref not in receipts:
                    receipts.append(ref)
        if proofs and not receipts:
            for ref in getattr(self, "receipts", []) or []:
                if isinstance(ref, str) and ref and ref not in receipts:
                    receipts.append(ref)
        return receipts

    def _has_unconfirmed_commit_state(self, state: dict) -> bool:
        """是否存在完成/提交声明但缺少可信证明的目标（不能假称已保存或未写）。"""
        for item in self._commit_items(state):
            if not isinstance(item, dict) or item.get("status") != "completed":
                continue
            if item.get("commit_state") not in {"unknown", "committed"}:
                continue
            proof = (item.get("result") or {}).get("proof_type")
            refs = item.get("receipt_refs")
            if item.get("commit_state") == "unknown":
                return True
            if not isinstance(proof, str) or proof not in _WRITE_INTENT_BY_PROOF:
                return True
            if not isinstance(refs, list) or not any(
                    isinstance(ref, str) and ref for ref in refs):
                return True
        return False

    def _limited_answer(self, state: dict) -> tuple[str, list[str]]:
        """受限终态文案只按真实提交证明/回执声明保存状态；已完成的只读答案照常交付。"""
        proofs = self._commit_proofs(state)
        receipts = self._verified_commit_receipts(state, proofs) if proofs else []
        if receipts:
            notice = (
                "部分目标未完成：本轮执行达到上限，已提交的记录已保存并保留回执；"
                "未完成部分没有自动重试。"
            )
        elif self._has_unconfirmed_commit_state(state):
            notice = (
                "部分目标未完成：本轮执行达到上限；记录的保存状态尚未确认，"
                "未完成部分没有自动重试。"
            )
            receipts = []
        else:
            notice = (
                "部分目标未完成：本轮执行达到上限，本次没有已确认保存的记录；"
                "未完成部分没有自动重试。"
            )
        from app.services.agent.react_graph import (answer_from_completed_read_goals,
                                                    completed_delivery_goals)
        merged = self._completed_state(state)
        goals = completed_delivery_goals(merged)
        delivered = answer_from_completed_read_goals(goals, "")
        if delivered:
            # A budget stop must not erase answers that were already produced.
            # The limit notice stays explicit and is appended after them.
            return delivered + "\n\n" + notice, receipts
        return notice, receipts

    def _committed_payload(self, state: dict) -> dict[str, Any]:
        # Use application-owned commit proofs, never a model-provided label.
        proofs = self._commit_proofs(state)
        # A known real commit keeps the existing trusted classification. Mixed
        # workout+nutrition commits retain the historical nutrition projection;
        # an unknown/empty proof set never becomes a nutrition write success.
        if proofs & {"workout_db_v1", "workout_name_update_v1"} and "nutrition_db_v1" not in proofs:
            task_type = _WRITE_INTENT_BY_PROOF["workout_db_v1"]
        elif "nutrition_db_v1" in proofs:
            task_type = _WRITE_INTENT_BY_PROOF["nutrition_db_v1"]
        else:
            task_type = None
        has_write = task_type is not None
        merged = self._completed_state(state)
        if task_type is None:
            frozen_outline = list(
                (getattr(self, "frozen_parameters", {}) or {}).get("task_outline") or [])
            task_type = _completed_read_intent(merged, frozen_outline) or "training_qa"
        answer = self._state_answer(merged, include_completed_goals=True)
        if not answer:
            answer = "已完成记录。" if has_write else "本次咨询已完成，未保存任何记录。"
        if self.ai_estimate_committed and "AI估算" not in answer:
            answer += " 营养数据为AI估算，可在饮食记录中修改。"
        if self.termination_reason in {"supervisor_turn_limit", "tool_step_limit", "no_progress"}:
            answer, _receipts = self._limited_answer(state)
        return {
            "final_answer": answer,
            "worker_results": [{"worker": "react_write", "status": "completed", "answer": answer}],
            "citations": self._citations(state),
            "model_used": True,
            "safety_level": "normal",
            "current_task": {"task_type": task_type},
            "react_status": "completed",
            "intent": task_type,
            "react_write_status": "completed",
            "mode_reason": "write_executed" if has_write else "read_only_completed",
            "receipt_refs": list(self.receipts) if has_write else [],
            "pending_confirmations": [],
            "response_ids": self._response_ids(),
        }

    def _unfinished_payload(self, state: dict, *, release_lease: bool = True) -> dict[str, Any]:
        # 非终态返回：释放本次租约（resume/确认必须重新真实取得，不借旧 token）。
        if release_lease:
            self.release_lease_quiet()
        reason = self.termination_reason or str(state.get("status") or "limited")
        answer = self._state_answer(state, include_completed_goals=False) or (
            "本次写入未完成：执行状态尚未确认，未作为保存成功处理；"
            "请稍后核对是否已写入。")
        return {
            "final_answer": answer,
            "worker_results": [{"worker": "react_write", "status": "limited", "answer": ""}],
            "citations": self._citations(state),
            "model_used": bool(self.supervisor_turns or int(state.get("supervisor_turns") or 0)),
            "safety_level": "normal",
            "current_task": {"task_type": "nutrition_record"},
            "react_status": "limited",
            "intent": "execution_limited",
            "react_write_status": "limited",
            "mode_reason": reason,
            "receipt_refs": list(self.receipts),
            "pending_confirmations": [],
            "response_ids": self._response_ids(),
        }

    @staticmethod
    def _citations(state: dict) -> list:
        from app.services.agent.react_runtime import _collect_citations

        try:
            return _collect_citations(state)
        except Exception:
            return []

    # ------------------------------------------------------------ confirmation

    async def _run_confirmation(self) -> dict[str, Any]:
        """确认请求：真实新 C1 身份 + 新 lease，消费原 protected pending（单次）。"""
        from app.services.agent import execution_ledger_store as ledger_api
        from app.services.agent.nutrition_write_bridge import NutritionWriteBridge

        selector = self.request.confirmation
        pending_rid = str(getattr(selector, "pending_request_id", "") or "")
        pending_op = str(getattr(selector, "pending_op_id", "") or "")
        pending_digest = str(getattr(selector, "pending_digest", "") or "")
        # 本 rid 自身已冻结的任务：只有真实 nutrition_confirmation 任务才是确认请求；
        # 其它任务（如原 pending 的 nutrition_record）说明同一 rid 被加了 selector。
        # 已冻结确认任务时，selector 必须完全一致，否则同 rid 身份冲突（409）。
        self_tasks = self._self_tasks()
        confirmation_tasks = [task for task in self_tasks
                              if str(task.get("task_type")) == "nutrition_confirmation"]
        if self_tasks and not confirmation_tasks:
            from app.services.agent.execution_state import RequestIdConflict

            self.release_lease_quiet()
            raise RequestIdConflict("request_id_payload_conflict")
        exact_selector = {"pending_request_id": pending_rid, "pending_op_id": pending_op,
                          "pending_digest": pending_digest}
        if confirmation_tasks and any(
                _json_object(task.get("parameters")) != exact_selector
                for task in confirmation_tasks):
            from app.services.agent.execution_state import RequestIdConflict

            self.release_lease_quiet()
            raise RequestIdConflict("request_id_payload_conflict")
        original = self.store.load_request(self.user_id, pending_rid)
        if original is None:
            return self._unfinished_payload({})
        if original["status"] in {"completed", "failed", "limited"}:
            return self._unfinished_payload({})
        # 原 pending 的冻结隐私（owner epoch/turn hide/delete）必须在任何真实租约/
        # 消费之前核验；缺失/变更一律 fail closed，零消费零写。
        try:
            self.checkpoint_owner.guard_privacy(self.user_id, pending_rid)
        except Exception:  # noqa: BLE001 - 隐私/身份不可信即拒绝确认消费
            self.termination_reason = "privacy_invalidated"
            logger.warning("c2_3_confirmation_privacy_refused", exc_info=True)
            return self._unfinished_payload({})
        self.freeze_controller()
        task_id = f"confirm-{self.request_id}"
        parameters = {"pending_request_id": pending_rid, "pending_op_id": pending_op,
                      "pending_digest": pending_digest}
        self._freeze_task(task_id=task_id, start=0, end=len(self.source),
                          task_type="nutrition_confirmation", parameters=parameters)
        # 原请求租约：必须由本次真实 acquire_lease 取得（活动 lease 拒绝），不借旧 token。
        try:
            snapshot_orig = self.store.acquire_lease(self.user_id, pending_rid, "c2-3-confirm", 300)
        except ledger_api.ExecutionLedgerError:
            return self._unfinished_payload({})
        lease = snapshot_orig.get("lease")
        if not isinstance(lease, dict):
            return self._unfinished_payload({})
        lease_original = ledger_api.LeaseProof(
            worker_id=lease["worker_id"], token=lease["token"], epoch=lease["epoch"])
        original_revision = int(snapshot_orig["revision"])
        # 真实确认消费前的 dispatch_intent 状态转换（pending 阶段保持 reserved）。
        op_row = None
        with self.factory() as session:
            op_row = self.store.ledger_operation_row(session, self.user_id, pending_rid, pending_op)
        if op_row is not None and op_row.status == "reserved":
            try:
                snapshot_orig = self.store.mark_dispatch_intent(
                    self.user_id, pending_rid, pending_op,
                    expected_revision=original_revision, lease=lease_original)
                original_revision = int(snapshot_orig["revision"])
            except ledger_api.ExecutionLedgerError:
                return self._unfinished_payload({})
        bridge = NutritionWriteBridge(self.factory)
        result = bridge.commit_nutrition_operation(
            self.user_id, pending_rid, pending_op,
            expected_revision=original_revision, lease=lease_original,
            tool_call_id="call-c2-3-confirm", batch_id="c2-3-confirm",
            confirmation_request_id=self.request_id, pending_digest=pending_digest,
            confirmation_expected_revision=self.revision, confirmation_lease=self.lease)
        outcome = str(result.get("outcome") or "")
        from app.services.agent.react_models import DomainObservation

        observation = DomainObservation.from_dict(result.get("observation"))
        if outcome != "committed":
            logger.warning("c2_3_confirmation_not_committed outcome=%s reason=%s detail=%s obs_error=%s",
                           outcome, result.get("reason"), result.get("detail"),
                           (result.get("observation") or {}).get("error"))
        if outcome == "committed" and observation.commit_state == "committed":
            self.receipts.extend(list(observation.receipt_refs or []))
            self.committed_observations.append(observation.to_dict())
            # 镜像原请求：只使用本次真实 acquire 的原请求租约证明（不借行内 token）。
            self.lease_original = lease_original
            self.revision_original = original_revision
            await asyncio.to_thread(self._mirror_original, pending_rid, pending_op)
            return {
                "final_answer": "已按你的确认完成记录。",
                "worker_results": [{"worker": "react_write", "status": "completed", "answer": ""}],
                "citations": [],
                "model_used": False,
                "safety_level": "normal",
                "current_task": {"task_type": "nutrition_record"},
                "react_status": "completed",
                "intent": "nutrition_record",
                "react_write_status": "completed",
                "mode_reason": "confirmation_consumed",
                "receipt_refs": list(self.receipts),
                "pending_confirmations": [],
            }
        return self._unfinished_payload({})

    def _committed_goal_ids(self, events: list[dict], source: str,
                            *, pending_op: str | None = None) -> set[str] | None:
        """真实已覆盖目标（写 + 建议/只读）：写目标必须有可信已提交证明，非写目标按
        图同一 definite 状态规则覆盖；本次确认消费的 pending 对应写调用计为已提交。

        只读映射：intent 的 tool_name/args 经真实目录与匹配谓词映射到候选目标；
        候选生成故障返回 None（调用方保守不完成）。"""
        from app.services.agent.react_graph import (_WRITE_RUNTIME_ACTIVE,
                                                    _matching_candidates, _rule_candidates)
        from app.services.agent.react_models import ToolCall
        from app.services.agent.react_tools import TOOL_CATALOG

        definite = {"completed", "partial", "rejected", "needs_clarification",
                    "blocked_capability"}
        candidates, error = _rule_candidates(source)
        if error:
            return None
        state = {"intent_candidates": candidates, "source_text": source, "write_runtime": True}
        intents: dict[str, dict] = {}
        covered: set[str] = set()
        confirmed_calls: set[str] = set()

        def _match(call_id: str) -> None:
            intent = intents.get(call_id) or {}
            spec = TOOL_CATALOG.get(str(intent.get("tool_name") or ""))
            if spec is None:
                return
            args = intent.get("args") if isinstance(intent.get("args"), dict) else {"source_ref": None}
            call = ToolCall(id=call_id, name=spec.name, args=args)
            token = _WRITE_RUNTIME_ACTIVE.set(True)
            try:
                matched = _matching_candidates(call, spec, state)
            finally:
                _WRITE_RUNTIME_ACTIVE.reset(token)
            for candidate in matched:
                cid = str(candidate.get("id") or "")
                if cid:
                    covered.add(cid)

        for event in events:
            phase = str(event.get("phase") or "")
            observation = _json_object(event.get("observation"))
            observation.pop("_react_runtime_origin", None)
            call_id = str(event.get("tool_call_id") or "")
            if phase == "tool_intent":
                intents[call_id] = {"tool_name": str(observation.get("tool_name") or ""),
                                    "args": observation.get("args")}
                continue
            if phase != "tool_result":
                continue
            result = observation.get("result")
            pending = (result.get("pending_confirmation")
                       if isinstance(result, dict) else None)
            if pending_op and isinstance(pending, dict) \
                    and str(pending.get("pending_op_id") or "") == pending_op:
                confirmed_calls.add(call_id)
            intent = intents.get(call_id) or {}
            spec = TOOL_CATALOG.get(str(intent.get("tool_name") or ""))
            status = str(observation.get("status") or "")
            if spec is None:
                continue
            if spec.permission == "write":
                # 写目标必须有可信已提交证明（真实 commit_state）；声明/未提交不算。
                if status == "completed" and observation.get("commit_state") == "committed":
                    _match(call_id)
            elif status in definite:
                _match(call_id)
        for call_id in confirmed_calls:
            _match(call_id)
        return covered

    def _mirror_original(self, pending_rid: str, pending_op: str) -> None:
        """确认消费后镜像原请求（真实冻结身份/全部目标状态/自持租约 CAS；不借 token）：

        - 先在真实 session 内读取原 request 的 controller 参数并断开 ORM 引用，验证
          原冻结 response_ids（request/message/run）后才使用；
        - 完成条件与 fresh/resume 共用：按真实全部目标（写 + 建议/只读）状态决定——
          写目标必须有可信已提交证明，未处理的 non-write/建议目标同样不得封闭原请求；
        - 完成：写规范 final_reply journal 后真实完成；
        - 未完成：不终态化；同样写规范 final_reply journal（partial，可恢复核验），
          再冻结诚实非完成响应（状态与未处理目标保持）。"""
        from app.services.agent import execution_ledger_store as ledger_api

        try:
            snapshot = self.store.load_request(self.user_id, pending_rid)
            if snapshot is None or snapshot.get("status") in {"completed", "failed", "limited"}:
                return
            proof = self.lease_original
            if not isinstance(proof, ledger_api.LeaseProof):
                return
            lease_row = snapshot.get("lease") or {}
            if (str(lease_row.get("worker_id") or ""), str(lease_row.get("token") or ""),
                    lease_row.get("epoch")) != (proof.worker_id, proof.token, proof.epoch):
                # 行内租约与本次真实取得的证明不一致：绝不借 token，保持原样。
                return
            source = str((snapshot.get("identity") or {}).get("source") or "")
            if not source:
                return
            # 原请求真实冻结 scope：真实快照只有 identity/checkpoint 顶层键，budget_scope_id
            # 在 identity 与 checkpoint 内；两处必须一致且非空，绝不猜 rid/用确认 scope/填默认值。
            identity = snapshot.get("identity") or {}
            checkpoint_meta = snapshot.get("checkpoint") or {}
            scope_id = str(identity.get("budget_scope_id") or "")
            checkpoint_scope = str(checkpoint_meta.get("budget_scope_id") or "")
            if not scope_id or checkpoint_scope != scope_id:
                return
            events = [event for event in snapshot.get("events") or []
                      if str(event.get("event_key", "")).startswith("c2-3-runtime:v1:")]
            self._restore_source_understanding(snapshot, source)
            covered = self._committed_goal_ids(events, source, pending_op=pending_op)
            if covered is None:
                return
            complete = self._no_unhandled_action_goals(source, covered)
            # 原请求真实终态文字（建议/答复保留）：与共享终态提取同一规则。
            observations = [_json_object(event.get("observation"))
                            for event in events if str(event.get("phase") or "") == "tool_result"]
            answer = self._state_answer({"observations": observations},
                                        include_completed_goals=False)
            # 原请求冻结身份：真实 session 内读取 controller 参数并断开 ORM 引用后验证。
            with self.factory() as session:
                controller_row = self.store.ledger_task_row(
                    session, self.user_id, pending_rid, self.controller_task_id)
            frozen_ids = _json_object(
                (_json_object(getattr(controller_row, "parameters", None)) or {})
                .get("response_ids")) or {}
            message_id = str(frozen_ids.get("message_id") or "")
            run_id = str(frozen_ids.get("run_id") or "")
            frozen_request_id = str(frozen_ids.get("request_id") or "")
            if not (message_id and run_id) or frozen_request_id != pending_rid:
                # 冻结身份不完整或不属于原请求：绝不伪造/借用。
                return
            receipts = list(self.receipts)
            for event in events:
                if str(event.get("phase") or "") != "tool_result":
                    continue
                observation = _json_object(event.get("observation"))
                if observation.get("status") == "completed" \
                        and observation.get("commit_state") == "committed":
                    receipts.extend(str(item) for item in (observation.get("receipt_refs") or []))
            revision = int(snapshot["revision"])
            if complete:
                stored = {
                    "answer": answer or "已按你的确认完成记录。",
                    "intent": "nutrition_record",
                    "citations": [],
                    "safety_level": "normal",
                    "model_used": False,
                    "request_id": pending_rid,
                    "message_id": message_id,
                    "run_id": run_id,
                    "metrics": {"react_write": "completed", "mode_reason": "confirmation_consumed",
                                "receipt_refs": receipts, "pending_confirmations": []},
                }
                journal_status = "completed"
            else:
                stored = {
                    "answer": answer or ("已按你的确认保存该项记录；原请求仍有未完成目标，"
                                         "不会自动重试。"),
                    "intent": "nutrition_record",
                    "citations": [],
                    "safety_level": "normal",
                    "model_used": False,
                    "request_id": pending_rid,
                    "message_id": message_id,
                    "run_id": run_id,
                    "metrics": {"react_write": "limited",
                                "mode_reason": "confirmation_remaining_targets",
                                "receipt_refs": receipts, "pending_confirmations": []},
                }
                journal_status = "limited"
            response_hash = _canonical_hash(stored)
            if response_hash is None:
                return
            current = dict(snapshot.get("checkpoint") or {})
            checkpoint = ledger_api.Checkpoint(
                supervisor_turns=max(1, int(current.get("supervisor_turns") or 0)),
                tool_steps=max(1, int(current.get("tool_steps") or 0)),
                event_sequence=int(current.get("event_sequence") or 0) + 1,
                budget_scope_id=scope_id)
            journal_event = ledger_api.EventInput(
                event_key=f"c2-3-runtime:v1:final_reply:mirror:{pending_rid}",
                tool_call_id="final-reply", batch_id="c2-3-confirm",
                task_id=self.controller_task_id, op_id=None, phase="final_reply",
                observation={"response_hash": response_hash, "status": journal_status},
                task_status="completed", operation_state=None, confirmed=False)
            try:
                # journal 与响应同一事务（真实 ledger CAS）：任一失败整体回滚，
                # 绝不产生无规范证明的冻结响应或假 event 的部分状态。
                if complete:
                    self.store.finish_request(
                        self.user_id, pending_rid, status="completed", response=stored,
                        termination_reason="confirmation_consumed",
                        expected_revision=revision, lease=proof,
                        journal_event=journal_event, journal_checkpoint=checkpoint,
                        capability=self.capability)
                else:
                    self.store.freeze_response(
                        self.user_id, pending_rid, response=stored,
                        expected_revision=revision, lease=proof,
                        journal_event=journal_event, journal_checkpoint=checkpoint,
                        capability=self.capability)
            except Exception:
                logger.warning("c2_3_original_mirror_journal_failed", exc_info=True)
        except Exception:
            logger.warning("c2_3_original_mirror_failed", exc_info=True)

    def _stored_pending(self) -> dict | None:
        """未消费的 protected pending（真实 pending 事件 + op 仍 reserved）。"""
        snapshot = self.store.load_request(self.user_id, self.request_id)
        if snapshot is None:
            return None
        used = any(str(event.get("event_key", "")).startswith(
            ("c2-2-confirmation-use:v1:", "c2-2-commit:v1:")) for event in snapshot.get("events") or [])
        if used:
            return None
        for event in snapshot.get("events") or []:
            if str(event.get("phase") or "") != "nutrition_pending":
                continue
            observation = _json_object(event.get("observation"))
            op_id = str(observation.get("op_id") or "")
            digest = str(observation.get("pending_digest") or "")
            if not op_id or not digest:
                continue
            with self.factory() as session:
                op = self.store.ledger_operation_row(session, self.user_id, self.request_id, op_id)
            if op is None or op.status != "reserved":
                continue
            return {"pending_request_id": self.request_id, "pending_op_id": op_id,
                    "pending_digest": digest, "expires_at": str(observation.get("expires_at") or "")}
        return None

    def _pending_replay_payload(self, target: dict) -> dict[str, Any]:
        answer = "该记录需要你确认后才会保存。请回复“确认保存”。"
        return {
            "final_answer": answer,
            "worker_results": [{"worker": "react_write", "status": "needs_confirmation", "answer": answer}],
            "citations": [],
            "model_used": False,
            "safety_level": "normal",
            "current_task": {"task_type": "nutrition_record"},
            "react_status": "pending",
            "intent": "nutrition_record",
            "react_write_status": "pending",
            "mode_reason": "protected_pending_replay",
            "receipt_refs": [],
            "pending_confirmations": [target],
            "response_ids": self._response_ids(),
            "replayed": True,
        }

    def _self_tasks(self) -> list[dict]:
        snapshot = self.store.load_request(self.user_id, self.request_id)
        if snapshot is None:
            return []
        return list(snapshot.get("tasks") or [])

    def _no_unhandled_action_goals(self, source: str, covered: set[str]) -> bool:
        """共享完成条件（fresh/resume/确认镜像同一规则）：真实全部动作目标（写 + 建议/
        只读）均已覆盖才视为无未处理；候选生成故障 → False（保守不完成）。"""
        from app.services.agent.react_graph import (_action_goals, _rule_candidates,
                                                    _unhandled_goals)

        candidates = self.resume_candidates or None
        if candidates is None:
            candidates, error = _rule_candidates(source)
            if error:
                return False
        unhandled = _unhandled_goals({
            "intent_candidates": candidates,
            "covered_goals": sorted(covered),
            "committed_tasks": list(self.resume_committed or []),
            "goal_results": copy.deepcopy(self.resume_goal_results or {}),
            "source_text": source,
        })
        return not _action_goals(unhandled)

    def _resume_all_goals_completed(self) -> bool:
        """纯只读恢复的严格完成条件：每个真实工具目标都已由持久completed结果覆盖。

        恢复只映射真实completed只读结果与可信committed写；非completed读不会被
        提升。任何工具目标未覆盖或结果非completed都保持保守非完成，不用宽松
        fallback把未完成目标伪装完成。
        """
        from app.services.agent.react_graph import (_READ_GOAL_TASK_TYPES,
                                                    _WRITE_TASK_TYPE_VALUES,
                                                    _rule_candidates)

        candidates, error = _rule_candidates(self.source)
        if error or not self.resume_covered:
            return False
        tool_types = _READ_GOAL_TASK_TYPES | _WRITE_TASK_TYPE_VALUES
        covered = set(self.resume_covered)
        for candidate in candidates:
            if not isinstance(candidate, dict) or not candidate.get("task_type"):
                continue
            if candidate.get("task_type") not in tool_types:
                continue
            cid = str(candidate.get("id") or "")
            if not cid or cid not in covered:
                return False
            observation = (self.resume_goal_results or {}).get(cid)
            if not isinstance(observation, dict) or observation.get("status") != "completed":
                return False
        return True

    # -------------------------------------------------------------- finalize

    def finalize(self, response: dict) -> dict:
        """终态冻结：protected final_reply（规范 ChatResponse 哈希）+ C1 finish。"""
        from app.services.agent import execution_ledger_store as ledger_api

        request_id = self.terminal_request_id
        if request_id != self.request_id:
            snapshot = self.store.load_request(self.user_id, request_id)
            if snapshot is None:
                raise WriteRuntimeUnavailable("confirm_request_missing")
            self._apply_snapshot(snapshot)
            if self.lease is None:
                raise WriteRuntimeUnavailable("confirm_lease_missing")
        response_hash = _canonical_hash(response)
        if response_hash is None:
            raise WriteRuntimeUnavailable("response_not_canonical")
        snapshot = self.store.load_request(self.user_id, request_id)
        if snapshot is None:
            raise WriteRuntimeUnavailable("terminal_request_missing")
        # 已有同哈希终态事件（例如首次 finish 失败后恢复）：幂等复用，不重复写终态。
        final_reply_seen = False
        for event in snapshot.get("events") or []:
            if str(event.get("event_key", "")).startswith("c2-3-runtime:v1:final_reply:") \
                    and str(event.get("phase") or "") == "final_reply":
                final_reply_seen = True
                existing = _json_object(event.get("observation"))
                if existing.get("response_hash") == response_hash:
                    self.revision = int(snapshot["revision"])
                    self.store.finish_request(
                        self.user_id, request_id, status="completed", response=response,
                        termination_reason="goals_complete", expected_revision=self.revision,
                        lease=self.lease)
                    if request_id == self.request_id:
                        self.lease = None
                    return response
        if final_reply_seen:
            # 已有 final_reply 但无同哈希：仅当存在 status=completed 的规范 journal
            # （已完成终态已冻结）时拒绝；此前非终态 partial 的 limited journal 属
            # 真实生命周期（可被后续真正终态取代），不重复写、不写假哈希。
            for event in snapshot.get("events") or []:
                if str(event.get("event_key", "")).startswith("c2-3-runtime:v1:final_reply:") \
                        and str(event.get("phase") or "") == "final_reply":
                    existing = _json_object(event.get("observation"))
                    if str(existing.get("status") or "") == "completed":
                        raise WriteRuntimeUnavailable("terminal_journal_mismatch")
        current = dict(snapshot.get("checkpoint") or {})
        self.supervisor_turns = max(self.supervisor_turns, int(current.get("supervisor_turns") or 0))
        self.tool_steps = max(self.tool_steps, int(current.get("tool_steps") or 0))
        self.revision = int(snapshot["revision"])
        checkpoint = ledger_api.Checkpoint(
            supervisor_turns=self.supervisor_turns, tool_steps=self.tool_steps,
            event_sequence=int(current.get("event_sequence") or 0) + 1, budget_scope_id=self.scope_id)
        event = ledger_api.EventInput(
            event_key=f"c2-3-runtime:v1:final_reply:{self.event_counter + 1}:final",
            tool_call_id="final-reply", batch_id="c2-3-client",
            task_id=self.controller_task_id, op_id=None, phase="final_reply",
            observation={"response_hash": response_hash, "status": "completed",
                         "canonical_response": response},
            task_status="completed", operation_state=None, confirmed=False)
        with self.factory() as session:
            appended = self.store.ledger_append_runtime_event(
                session, self.user_id, request_id, event, checkpoint=checkpoint,
                expected_revision=self.revision, lease=self.lease, capability=self.capability)
            session.commit()
        self.revision = int(appended["revision"])
        self.event_sequence = checkpoint.event_sequence
        snapshot = self.store.finish_request(
            self.user_id, request_id, status="completed", response=response,
            termination_reason="goals_complete", expected_revision=self.revision, lease=self.lease)
        self._apply_snapshot(snapshot)
        if request_id == self.request_id:
            self.lease = None
        return response

    def release_lease_quiet(self) -> None:
        if self.lease is None:
            return
        try:
            self.store.release_lease(self.user_id, self.request_id,
                                     expected_revision=self.revision, lease=self.lease)
        except Exception:
            logger.warning("c2_3_lease_release_failed", exc_info=True)
        finally:
            self.lease = None


def refresh_circuit_state(agent) -> None:
    """请求级物理尝试隔离：上一请求的物理失败（含熔断计数）不得污染本请求的
    真实尝试能力；每个请求从真实可尝试状态开始，由本请求自己的上限收敛。"""
    try:
        breaker = getattr(getattr(getattr(agent, "graph", None), "llm", None), "breaker", None)
        if breaker is not None:
            breaker.record_success()
    except Exception:  # noqa: BLE001 - 复位失败不阻断请求（真实失败仍由 client 记录）
        logger.warning("c2_3_circuit_refresh_failed", exc_info=True)


def _frozen_budget_limits(factory, user_id: str, request_id: str) -> tuple[int, int] | None:
    """冻结 controller 的持久上限（恢复路径 AND 输入；无 controller 返回 None）。"""
    try:
        from app.services.agent import execution_ledger_store as ledger_api

        snapshot = ledger_api.ExecutionLedgerStore(factory).load_request(user_id, request_id)
        if not snapshot:
            return None
        for task in snapshot.get("tasks") or []:
            if str(task.get("task_type")) == "react_controller" \
                    or str(task.get("task_id")) == "react-controller-v1":
                parameters = _json_object(task.get("parameters")) or {}
                if parameters.get("model_budget_policy") == "per_task_v1":
                    # New controller snapshots own independent task rows; the
                    # old shared request ceiling is not a task ceiling.
                    return None
                calls = parameters.get("max_model_calls")
                tokens = parameters.get("max_model_tokens")
                if isinstance(calls, int) and not isinstance(calls, bool) \
                        and isinstance(tokens, int) and not isinstance(tokens, bool):
                    return calls, tokens
    except Exception:  # noqa: BLE001 - 读取失败按无冻结上限（scope 上限仍生效）
        logger.warning("c2_3_frozen_limits_read_failed", exc_info=True)
    return None


async def run_write(agent, request, *, session_factory=None) -> dict[str, Any]:
    """执行显式写入；返回 legacy graph 兼容结构（含稳定 response_ids 投影）。"""
    from app.services.llm.budget import active_scope, key, model_scope

    scope = active_scope.get()
    if scope is not None and scope.kind == "request":
        if getattr(scope, "owner_id", None) is None:
            scope.owner_id = key(str(request.user_id) + ':' + str(request.conversation_id))
        return await _run_write_in_scope(agent, request, session_factory=session_factory)
    # 真实 child/直调入口没有 middleware 请求 scope：以真实 request 身份新建同一
    # request scope（HTTP 路径始终复用已有 scope，行为不变）。
    with model_scope('request:' + str(request.request_id), kind='request',
                     policy='per_task') as budget:
        budget.owner_id = key(str(request.user_id) + ':' + str(request.conversation_id))
        return await _run_write_in_scope(agent, request, session_factory=session_factory)


async def _run_write_in_scope(agent, request, *, session_factory=None) -> dict[str, Any]:
    from app.services.llm.budget import active_scope

    scope = active_scope.get()
    if scope is None or scope.kind != "request":
        return _limited("model_budget_scope_required")
    lifecycle = _write_scope.get()
    meter_error = lifecycle.get("meter_error") if isinstance(lifecycle, dict) else None
    if meter_error is not None:
        # 显式失计量注入：fail closed 零派发；同时应用库真实 scope 额度按 AND 标准
        # 真实打开（真实 open_in_session，不是模型物理调用），不静默替换注入策略。
        await _budget_bootstrap(scope)
        return _limited("model_budget_storage_unavailable")
    restore = None
    try:
        if session_factory is not None:
            # 显式 factory 贯穿真实计量/C1/adapter：请求局部私有副本仅换 factory，
            # 保留原 store 策略/失败/同一 scope 身份；绑定之后的正常/失败/取消
            # 分支都在 finally 复原原 store。
            restore, meter_error = bind_request_scope_store(scope, session_factory)
            if meter_error is not None:
                # 显式策略失败必须保留：即使目标 factory 健康也零派发；应用库
                # 真实 scope 额度按 AND 标准真实打开（失败亦为受限）。
                await _budget_bootstrap(scope)
                return _limited("model_budget_storage_unavailable")
        from app.db import session as db

        limits_factory = session_factory if session_factory is not None else db.SessionLocal
        frozen_limits = None
        if limits_factory is not None:
            frozen_limits = _frozen_budget_limits(limits_factory, str(request.user_id),
                                                  str(request.request_id))
        if not await _budget_bootstrap(scope, frozen_limits):
            return _limited("model_budget_storage_unavailable")
        refresh_circuit_state(agent)
        session = WriteRuntimeSession(agent, request, session_factory=session_factory)
        lifecycle = _write_scope.get()
        if isinstance(lifecycle, dict):
            lifecycle["session"] = session
        try:
            return await session.run()
        except WriteRuntimeUnavailable as exc:
            logger.warning("c2_3_write_runtime_unavailable: %s", exc)
            return _limited(str(getattr(exc, "code", None) or "write_runtime_unavailable"))
        except asyncio.CancelledError:
            session.release_lease_quiet()
            raise
        except Exception as exc:
            from app.services.agent.execution_state import RequestIdConflict

            if isinstance(exc, RequestIdConflict):
                # 同 rid 身份冲突保持真实 HTTP 409 语义，不降级为普通 limited。
                raise
            logger.exception("c2_3_write_runtime_failed")
            return _limited("write_runtime_error")
    finally:
        if restore is not None:
            restore()


async def finalize_write_response(request, response):
    """终态冻结：稳定 IDs/diagnostics/memory 后写 protected final_reply 与 C1 终态。"""
    if getattr(response, "replayed", False):
        return response
    metrics = dict(getattr(response, "metrics", None) or {})
    if metrics.get("react_write") not in {"completed", "complete"}:
        return response
    lifecycle = _write_scope.get()
    session = lifecycle.get("session") if isinstance(lifecycle, dict) else None
    if session is None:
        return response
    # A sealed child may rebuild a typed projection with replayed=False even
    # though the request is already terminal. Reuse the first frozen response
    # only after the real runtime journal, bridge proofs and privacy all verify;
    # never write a second final_reply or a different hash.
    try:
        snapshot = session.store.load_request(session.user_id, session.request_id)
        terminal_seen = (snapshot is not None
                         and str(snapshot.get("status") or "") in {
                             "completed", "failed", "limited"})
        if terminal_seen:
            stored = snapshot.get("response")
            if (isinstance(stored, dict)
                    and _replay_proof_valid(session.user_id, session.request_id,
                                            snapshot, session.factory)):
                journal = None
                for event in snapshot.get("events") or []:
                    if str(event.get("phase") or "") == "final_reply":
                        journal = _json_object(event.get("observation"))
                stored_hash = _canonical_hash(stored)
                if (journal is not None and stored_hash is not None
                        and str(journal.get("response_hash") or "") == stored_hash
                        and str(journal.get("status") or "") == "completed"):
                    session.checkpoint_owner.guard_privacy(
                        session.user_id, session.request_id)
                    from app.schemas.chat import ChatResponse
                    canonical = ChatResponse.model_validate(stored)
                    return canonical.model_copy(update={"replayed": True})
            # A terminal request with missing/untrusted journal, proof or stored
            # response is never allowed to fall through to a caller-typed final
            # write. Return an honest unknown result and release the lease.
            await asyncio.to_thread(session.release_lease_quiet)
            metrics["react_write"] = "commit_unknown"
            return response.model_copy(update={"metrics": metrics})
    except Exception:
        logger.warning("c2_3_terminal_reuse_verification_failed", exc_info=True)
        if 'terminal_seen' in locals() and terminal_seen:
            await asyncio.to_thread(session.release_lease_quiet)
            metrics["react_write"] = "commit_unknown"
            return response.model_copy(update={"metrics": metrics})
    try:
        await asyncio.to_thread(session.finalize, response.model_dump())
    except Exception:
        logger.warning("c2_3_finalize_failed", exc_info=True)
        # 终态提交失败：释放本次租约，允许后续 resume 重新真实取得（不借旧 token）。
        await asyncio.to_thread(session.release_lease_quiet)
        metrics["react_write"] = "commit_unknown"
        return response.model_copy(update={"metrics": metrics})
    return response


__all__ = [
    "RUNTIME_VERSION",
    "WriteRuntimeUnavailable",
    "finalize_write_response",
    "run_write",
    "write_lifecycle_scope",
]
