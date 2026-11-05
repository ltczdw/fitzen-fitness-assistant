"""C2-3 私有检查点能力与运行时状态读取（phaseI 实施中）。

本模块是 ``c2-3-runtime:v1:`` 私有族的对象能力拥有者：能力只按真实
``ReactCheckpointStore`` 实例与真实 ``ExecutionLedgerStore`` 绑定，普通
public/session 入口无条件拒绝该前缀（见 execution_ledger_store 的保留族检查）。

当前批次按 18 固定场景连续实现：``load_runtime`` 只返回经真实持久事件行核验的
有限 JSON 状态（来源/owner/request/controller/版本/序号/摘要全部核验）；伪造行、
旧代、序号跳跃与无可信证明的伪观察一律不升级。
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

RUNTIME_PREFIX = "c2-3-runtime:v1:"
RUNTIME_ORIGIN_KEY = "_react_runtime_origin"
RUNTIME_ORIGIN_PROTOCOL = 1
RUNTIME_ORIGIN_GENERATION = "6bcac6d1-6791-4d5c-9003-ea4691cd2d30"
CONTROLLER_TASK_ID = "react-controller-v1"
RUNTIME_VERSION = 1
RUNTIME_PHASES = frozenset({
    "bootstrap", "model_intent", "model_result", "tool_intent", "tool_result",
    "checkpoint", "client_reply", "final_reply",
})


def _canonical_response_hash(response: dict) -> str | None:
    """First-frozen full C1 ChatResponse canonical SHA-256 (terminal clarification):
    real JSON object only, ensure_ascii=False/sort_keys/compact separators."""
    import hashlib
    import json as _json

    try:
        encoded = _json.dumps(response, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class RuntimePrivacyInvalidated(RuntimeError):
    """hide/delete/owner epoch 变化后的保守失效（fail closed）。"""

    code = "runtime_privacy_invalidated"


class ReactCheckpointStore:
    """真实 C1/store 依赖复用的运行时检查点读取与隐私失效入口。"""

    def __init__(self, session_factory):
        from app.services.agent.execution_ledger_store import ExecutionLedgerStore

        self._factory = session_factory
        self._ledger = ExecutionLedgerStore(session_factory)
        from app.services.agent.execution_ledger_store import (
            _bind_runtime_reserved_capability,
        )

        self._capability = _bind_runtime_reserved_capability(self._ledger, self)

    # ------------------------------------------------------------ runtime read

    def load_runtime(self, user_id: str, request_id: str) -> dict | None:
        """读取真实受保护运行时状态；任何伪造/形状错误/旧代/序号跳跃/摘要不符都返回 None。

        核验范围：真实请求行（owner/request/conversation/source_digest）、真实
        controller 任务（task_id 或 task_type）与 runtime_version、每个 runtime 事件
        的来源协议/固定代际、事件序号与自身 checkpoint.event_sequence 一致且严格
        递增、终态 final_reply 唯一且 payload.response_hash 与冻结 C1 响应规范哈希
        全等。普通快照、伪造行与伪观察都不升级。
        """
        import hashlib

        from sqlalchemy import or_, select

        from app.db.execution_ledger_models import (
            ExecutionEventModel,
            ExecutionOperationModel,
            ExecutionRequestModel,
            ExecutionTaskModel,
        )

        with self._factory() as session:
            request = session.execute(
                select(ExecutionRequestModel).where(
                    ExecutionRequestModel.user_id == user_id,
                    ExecutionRequestModel.request_id == request_id,
                )
            ).scalar_one_or_none()
            if request is None:
                return None
            controller = session.execute(
                select(ExecutionTaskModel).where(
                    ExecutionTaskModel.user_id == user_id,
                    ExecutionTaskModel.request_id == request_id,
                    or_(ExecutionTaskModel.task_id == CONTROLLER_TASK_ID,
                        ExecutionTaskModel.task_type == CONTROLLER_TASK_ID),
                )
            ).scalar_one_or_none()
            if controller is None:
                return None
            parameters = _json_object(controller.parameters) or {}
            if parameters.get("runtime_version") != RUNTIME_VERSION:
                return None
            if str(getattr(controller, "source_digest", "") or "") != str(request.source_digest or ""):
                return None
            events = session.execute(
                select(ExecutionEventModel).where(
                    ExecutionEventModel.user_id == user_id,
                    ExecutionEventModel.request_id == request_id,
                    ExecutionEventModel.event_key.like(RUNTIME_PREFIX + "%"),
                ).order_by(ExecutionEventModel.sequence)
            ).scalars().all()
            if not events:
                return None
            runtime_events: list[dict[str, Any]] = []
            previous_sequence = None
            terminal_payload = None
            terminal_payloads: list[dict] = []
            journal_pending = False
            for row in events:
                observation = _json_object(row.observation)
                if observation is None:
                    return None
                origin = observation.get(RUNTIME_ORIGIN_KEY)
                if not isinstance(origin, dict):
                    return None
                if (origin.get("protocol") != RUNTIME_ORIGIN_PROTOCOL
                        or origin.get("generation") != RUNTIME_ORIGIN_GENERATION):
                    return None
                phase = row.phase
                if phase not in RUNTIME_PHASES:
                    return None
                checkpoint = _json_object(row.checkpoint)
                if checkpoint is None or checkpoint.get("event_sequence") != row.sequence:
                    return None
                if previous_sequence is not None and not row.sequence > previous_sequence:
                    return None
                previous_sequence = row.sequence
                payload = {key: value for key, value in observation.items()
                           if key != RUNTIME_ORIGIN_KEY}
                if phase == "final_reply":
                    terminal_payloads.append(payload)
                runtime_events.append({"phase": phase, "payload": payload,
                                       "tool_call_id": str(row.tool_call_id or "")})
            terminal_payload = terminal_payloads[-1] if terminal_payloads else None
            if terminal_payload is not None:
                stored_response = _json_object(request.response)
                if stored_response is None:
                    if str(terminal_payload.get("status") or "") != "completed":
                        return None
                    # response-save 真实故障后的恢复窗口：终态 journal 已唯一落盘但
                    # completed 响应未保存。返回可恢复信封（journal_pending + 唯一
                    # journal 哈希）；调用方必须以同一哈希复用该 journal 并走真实
                    # lease/CAS 完成，绝不重复写 journal/假 hash。
                    journal_pending = True
                else:
                    expected = _canonical_response_hash(stored_response)
                    if expected is None:
                        return None
                    # 规范 journal 唯一性：与已保存响应同哈希的 final_reply 必须恰为一
                    # 条，且必须是最后一条；此前的非终态 partial（limited）journal 属
                    # 真实生命周期（历史保留，不删改、不宽证明）。
                    matching = [payload for payload in terminal_payloads
                                if payload.get("response_hash") == expected]
                    last_payload = terminal_payloads[-1]
                    if len(matching) == 1 and last_payload.get("response_hash") == expected:
                        pass
                    elif (str(request.status or "") not in {"completed", "failed", "limited"}
                          and str(last_payload.get("status") or "") == "completed"
                          and bool(matching)):
                        # 旧 partial 投影 + 新 completed journal 待 finish（真实非终态
                        # request）：已保存响应仍是与较早 journal 同哈希的历史 partial，
                        # 最后 journal 是已提交的新 canonical；恢复入口按 journal_pending
                        # 消费（同哈希复用、同 lease/CAS，不删改历史/不加假 journal）。
                        journal_pending = True
                    else:
                        return None
            # 声明但未证明的写状态不升级：claim 存在必须有真实 commit 证明；
            # committed 状态必须有其 commit 事件。
            for operation in session.execute(
                select(ExecutionOperationModel).where(
                    ExecutionOperationModel.user_id == user_id,
                    ExecutionOperationModel.request_id == request_id,
                )
            ).scalars().all():
                if operation.kind == "workout_name_correction":
                    from app.services.agent.workout_correction import WorkoutCorrectionBridge
                    from app.services.agent import execution_ledger_store as ledger_api
                    correction_event = session.scalar(select(ExecutionEventModel).where(
                        ExecutionEventModel.user_id == user_id,
                        ExecutionEventModel.request_id == request_id,
                        ExecutionEventModel.event_key == ledger_api._CORRECTION_EVENT_PREFIX + operation.op_id))
                    bridge = WorkoutCorrectionBridge(self._factory)
                    if bridge.verified(session, operation, correction_event):
                        continue
                    if (operation.status == "dispatch_intent" and correction_event is None
                            and self._unclaimed_dispatch_valid(session,request,operation,runtime_events,request_id,user_id)):
                        continue
                    return None
                prefixes = ("c2-1-claim:v1:", "c2-1-commit:v1:") if operation.kind == "workout" \
                    else ("c2-2-claim:v1:", "c2-2-commit:v1:")
                claim = session.execute(
                    select(ExecutionEventModel).where(
                        ExecutionEventModel.user_id == user_id,
                        ExecutionEventModel.request_id == request_id,
                        ExecutionEventModel.event_key == prefixes[0] + operation.op_id,
                    )
                ).scalar_one_or_none()
                commit = session.execute(
                    select(ExecutionEventModel).where(
                        ExecutionEventModel.user_id == user_id,
                        ExecutionEventModel.request_id == request_id,
                        ExecutionEventModel.event_key == prefixes[1] + operation.op_id,
                    )
                ).scalar_one_or_none()
                if commit is not None:
                    if operation.status != "committed":
                        return None
                    continue
                if claim is not None or operation.status == "commit_unknown":
                    return None
                if operation.status == "dispatch_intent":
                    # 合法未认领恢复：真实 SQL 无 claim、无 commit 的未认领
                    # dispatch_intent，仅当 owner/request/task/op/冻结 source/args/native
                    # cursor 严格有效时放行；重新派发由原闭桥再次核验 source/task/参数/
                    # 自有 lease/record proof。claimed 无 commit 或 commit_unknown 继续拒绝。
                    if not self._unclaimed_dispatch_valid(
                            session, request, operation, runtime_events, request_id, user_id):
                        return None
                    continue
            # active 恢复先核冻结 owner epoch/隐私：失败即 None（不升级、不借行）。
            try:
                self.guard_privacy(user_id, request_id)
            except RuntimePrivacyInvalidated:
                return None
            return {
                "user_id": user_id,
                "request_id": request_id,
                "conversation_id": request.conversation_id,
                "controller": {
                    "task_id": CONTROLLER_TASK_ID,
                    "parameters": parameters,
                    "status": controller.status,
                },
                "runtime_events": runtime_events,
                "journal_pending": journal_pending,
                "journal": ({"response_hash": str(terminal_payload.get("response_hash") or ""),
                             "status": str(terminal_payload.get("status") or "")}
                            if journal_pending and terminal_payload is not None else None),
            }

    # ---------------------------------------------------------------- privacy

    def _unclaimed_dispatch_valid(self, session, request, operation, runtime_events,
                                  request_id: str, user_id: str) -> bool:
        """未认领 dispatch_intent 的严格有效性（合法 unclaimed 恢复的唯一入口）。

        真实 SQL 无 claim、无 commit；身份/载荷/原生调用按真实算法逐项核验：
        op_id 必须等于真实 ``_op_identity`` 推导值（position=0）；任务存在且类型匹配、
        任务 span 摘要等于冻结原文切片；op source_digest/time_base 等于请求冻结值；
        parameters_digest 等于真实载荷摘要；operation 参数必须是任务冻结参数的唯一对应
        元素（nutrition 单 batch / workout 单 record，按真实 ``_reserve`` 调用者形状），
        nutrition refs 必须非空且完整（start/end/exact_text 与冻结原文一致、落在任务
        区间）；native 绑定必须是原 ``model_result`` assistant 的真实 tool_call
        （id/name/args 严格配对）、属于未完成 cursor 且无 tool_result、与 op 参数一一
        对应且唯一，且其 source_ref 经真实 ``_candidate_task_id`` 推导必须等于
        ``operation.task_id``（真实任务身份，不由 call.id 决定）。任一缺失/歧义/不匹配
        返回 False（fail closed）；本方法不作完成证明、不标 committed、不改计数，重新
        派发仍由原闭桥按自有 lease/CAS 与 record proof 最终核验。"""
        import hashlib
        import json

        from sqlalchemy import select

        from app.db.execution_ledger_models import ExecutionTaskModel
        from app.services.agent import execution_ledger_store as ledger_api
        from app.services.agent.react_tools import _candidate_task_id

        kind = str(getattr(operation, "kind", "") or "")
        spec = {"workout": ("workout_record", "record_workout"),
                "workout_name_correction": ("workout_record", "correct_workout"),
                "nutrition": ("nutrition_record", "record_nutrition")}.get(kind)
        if spec is None:
            return False
        expected_type, tool_name = spec
        position = getattr(operation, "position", None)
        if isinstance(position, bool) or not isinstance(position, int) or position != 0:
            return False
        task = session.execute(
            select(ExecutionTaskModel).where(
                ExecutionTaskModel.user_id == user_id,
                ExecutionTaskModel.request_id == request_id,
                ExecutionTaskModel.task_id == operation.task_id,
            )
        ).scalar_one_or_none()
        if task is None or str(task.task_type or "") != expected_type:
            return False
        expected_op_id, _idempotency_key = ledger_api._op_identity(
            user_id, request_id, str(operation.task_id or ""), position, kind)
        if str(operation.op_id or "") != expected_op_id:
            return False
        source = str(request.source or "")
        task_start, task_end = task.source_start, task.source_end
        if not source or isinstance(task_start, bool) or isinstance(task_end, bool) \
                or not isinstance(task_start, int) or not isinstance(task_end, int) \
                or not (0 <= task_start < task_end <= len(source)):
            return False
        if str(task.source_digest or "") != hashlib.sha256(
                source[task_start:task_end].encode("utf-8")).hexdigest():
            return False
        if str(operation.source_digest or "") != str(request.source_digest or ""):
            return False
        if str(operation.time_base or "") != str(request.time_base or ""):
            return False
        parameters = _json_object(operation.parameters)
        if not isinstance(parameters, dict) or not parameters:
            return False
        if str(operation.parameters_digest or "") != ledger_api._digest(parameters):
            return False
        task_parameters = _json_object(task.parameters)
        if not isinstance(task_parameters, dict):
            return False

        def _valid_ref(ref) -> bool:
            if not isinstance(ref, dict):
                return False
            ref_start, ref_end = ref.get("start"), ref.get("end")
            exact = ref.get("exact_text")
            if isinstance(ref_start, bool) or isinstance(ref_end, bool) \
                    or not isinstance(ref_start, int) or not isinstance(ref_end, int):
                return False
            if not (0 <= ref_start < ref_end <= len(source)):
                return False
            if not (task_start <= ref_start and ref_end <= task_end):
                return False
            return isinstance(exact, str) and exact == source[ref_start:ref_end]

        op_refs: list[dict] = []
        if kind == "workout_name_correction":
            if task_parameters.get("correction_operation") != parameters:
                return False
            if not _valid_ref(parameters.get("source_ref")):
                return False
            op_refs.append(parameters["source_ref"])
        elif kind == "nutrition":
            batches = task_parameters.get("batches")
            if not isinstance(batches, list) or len(batches) != 1 \
                    or not isinstance(batches[0], dict):
                return False
            if ledger_api._digest(batches[0]) != ledger_api._digest(parameters):
                return False
            route = parameters.get("route")
            timezone_name = parameters.get("timezone")
            groups = parameters.get("groups")
            if not (isinstance(route, str) and route.strip()):
                return False
            if not (isinstance(timezone_name, str) and timezone_name.strip()):
                return False
            if not isinstance(groups, list) or not groups:
                return False
            for group in groups:
                if not isinstance(group, dict):
                    return False
                group_ref = group.get("source_ref")
                if not _valid_ref(group_ref):
                    return False
                op_refs.append(group_ref)
                meal_type = group.get("meal_type")
                if not (isinstance(meal_type, str) and meal_type.strip()):
                    return False
                foods = group.get("foods")
                if isinstance(foods, list) and foods:
                    for food in foods:
                        if not isinstance(food, dict) or not isinstance(food.get("entry"), dict):
                            return False
                        food_ref = food.get("source_ref")
                        if not _valid_ref(food_ref):
                            return False
                        op_refs.append(food_ref)
                    continue
                # manual_nutrition_log 单组真实形状（_manual_group 调用者）：无 foods，
                # 必须有非空描述/日期与至少一个真实数值宏量。
                description = group.get("description")
                if not (isinstance(description, str) and description.strip()):
                    return False
                if not (isinstance(group.get("log_date"), str) and group.get("log_date").strip()):
                    return False
                numbers = [group.get(key) for key in ("calories", "protein_g", "carbs_g", "fat_g")]
                if not any(isinstance(value, (int, float)) and not isinstance(value, bool)
                           for value in numbers):
                    return False
            if not op_refs:
                return False
        else:
            records = task_parameters.get("records")
            if not isinstance(records, list) or len(records) != 1 \
                    or not isinstance(records[0], dict):
                return False
            record = records[0]
            if ledger_api._digest(record) != ledger_api._digest(parameters):
                return False
            exercise = record.get("exercise")
            sets, reps = record.get("sets"), record.get("reps")
            weight = record.get("weight_kg")
            record_date = record.get("date")
            if not (isinstance(exercise, str) and exercise.strip()):
                return False
            if isinstance(sets, bool) or not isinstance(sets, int) or sets < 1:
                return False
            if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
                return False
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
                return False
            if not (isinstance(record_date, str) and record_date.strip()):
                return False

        # native 绑定：原 assistant 的真实 tool_call ↔ tool_intent ↔ 未完成 cursor ↔ op。
        assistants: list[tuple[int, dict]] = []
        intents: dict[str, list[tuple[int, dict]]] = {}
        result_ids: set[str] = set()
        for index, event in enumerate(runtime_events):
            phase = str(event.get("phase") or "")
            payload = event.get("payload") or {}
            if phase == "model_result":
                assistant = payload.get("assistant")
                calls = assistant.get("tool_calls") if isinstance(assistant, dict) else None
                if isinstance(calls, list) and calls:
                    assistants.append((index, assistant))
            elif phase == "tool_intent":
                call_id = str(event.get("tool_call_id") or "")
                if call_id:
                    intents.setdefault(call_id, []).append((index, payload))
            elif phase == "tool_result":
                call_id = str(event.get("tool_call_id") or "")
                if call_id:
                    result_ids.add(call_id)
        if not assistants:
            return False
        last_assistant_index, last_assistant = assistants[-1]
        candidates: list[str] = []
        for call in last_assistant.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            if not call_id or call_id in result_ids:
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str(function.get("name") or call.get("name") or "")
            if name != tool_name:
                continue
            raw_arguments = function.get("arguments") if function else call.get("args")
            if isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                try:
                    arguments = json.loads(str(raw_arguments or ""))
                except (TypeError, ValueError):
                    continue
            if not isinstance(arguments, dict):
                continue
            call_ref = arguments.get("source_ref")
            if not _valid_ref(call_ref):
                continue
            # 真实任务身份绑定：业务 task_id 由 react_tools._candidate_task_id(request_id,
            # tool_name, source_ref.start, source_ref.end) 推导（与两个 dispatch 调用者
            # 完全一致）；匹配到的原 native 片段必须推导出 operation.task_id，不能用
            # call.id 或“同 span 任意片段”决定身份。
            if _candidate_task_id(request_id, tool_name, call_ref["start"], call_ref["end"]) \
                    != str(operation.task_id or ""):
                continue
            entries = intents.get(call_id) or []
            if not entries:
                continue
            intent_index, intent_payload = entries[-1]
            if intent_index <= last_assistant_index:
                continue
            if str(intent_payload.get("tool_name") or "") != tool_name:
                continue
            intent_args = intent_payload.get("args")
            if not isinstance(intent_args, dict) or intent_args != arguments:
                continue
            if kind in {"nutrition", "workout_name_correction"} and call_ref not in op_refs:
                continue
            candidates.append(call_id)
        return len(candidates) == 1

    # ---------------------------------------------------------------- privacy

    def guard_privacy(self, user_id: str, request_id: str) -> None:
        """恢复/重放前核冻结身份：按冻结 controller 的 turn_id/request/conversation
        与 owner 匹配真实记录；controller 缺失、冻结 turn 缺失/错配、隐藏/删除、
        冻结 owner epoch 变化一律 fail closed——绝不借其它 turn、绝不因缺行跳过
        冻结 epoch 检查。"""
        from sqlalchemy import or_, select

        from app.db.execution_ledger_models import ExecutionRequestModel, ExecutionTaskModel
        from app.db.memory_models import MemoryTurn
        from app.db.models import UserProfileModel

        with self._factory() as session:
            request = session.execute(
                select(ExecutionRequestModel).where(
                    ExecutionRequestModel.user_id == user_id,
                    ExecutionRequestModel.request_id == request_id,
                )
            ).scalar_one_or_none()
            if request is None:
                raise RuntimePrivacyInvalidated("request_missing")
            controller = session.execute(
                select(ExecutionTaskModel).where(
                    ExecutionTaskModel.user_id == user_id,
                    ExecutionTaskModel.request_id == request_id,
                    or_(ExecutionTaskModel.task_id == CONTROLLER_TASK_ID,
                        ExecutionTaskModel.task_type == CONTROLLER_TASK_ID),
                )
            ).scalar_one_or_none()
            if controller is None:
                raise RuntimePrivacyInvalidated("controller_identity_missing")
            parameters = _json_object(controller.parameters) or {}
            frozen_turn_id = str(parameters.get("turn_id") or "")
            frozen_epoch = parameters.get("memory_epoch")
            if not frozen_turn_id:
                raise RuntimePrivacyInvalidated("frozen_turn_missing")
            if isinstance(frozen_epoch, bool) or not isinstance(frozen_epoch, int):
                raise RuntimePrivacyInvalidated("frozen_epoch_missing")
            turn = session.execute(
                select(MemoryTurn).where(MemoryTurn.turn_id == frozen_turn_id)
            ).scalar_one_or_none()
            if turn is None:
                raise RuntimePrivacyInvalidated("frozen_turn_missing")
            if (str(turn.request_id) != request_id
                    or str(turn.conversation_id) != str(request.conversation_id or "")):
                raise RuntimePrivacyInvalidated("frozen_turn_mismatch")
            if turn.hidden_at is not None or turn.deleted_at is not None:
                raise RuntimePrivacyInvalidated("turn_hidden_or_deleted")
            turn_epoch = getattr(turn, "memory_epoch", None)
            if not isinstance(turn_epoch, bool) and isinstance(turn_epoch, int) \
                    and turn_epoch != frozen_epoch:
                raise RuntimePrivacyInvalidated("frozen_epoch_mismatch")
            profile = session.execute(
                select(UserProfileModel).where(UserProfileModel.user_id == user_id)
            ).scalar_one_or_none()
            owner_epoch = getattr(profile, "memory_epoch", None)
            if isinstance(owner_epoch, bool) or not isinstance(owner_epoch, int) \
                    or owner_epoch != frozen_epoch:
                raise RuntimePrivacyInvalidated("owner_epoch_changed")

    def invalidate_in_session(self, session, *, user_id: str, conversation_id: str | None = None,
                              purge: bool = False) -> int:
        """在调用者事务中撤销 managed unfinished write lease；purge 时按 FK 顺序清理
        managed C1 原文/参数/观察/响应并擦除 ModelResultCache，保留 MemoryTurn
        tombstone、真实业务行与非内容额度计数。返回受影响请求数。"""
        from sqlalchemy import delete, or_, select, update

        from app.db.execution_ledger_models import (
            ExecutionEventModel,
            ExecutionOperationModel,
            ExecutionRequestModel,
            ExecutionTaskModel,
        )

        conditions = [ExecutionRequestModel.user_id == user_id]
        if conversation_id is not None:
            conditions.append(ExecutionRequestModel.conversation_id == conversation_id)
        request_rows = session.execute(
            select(ExecutionRequestModel).where(*conditions)
        ).scalars().all()
        managed = [row for row in request_rows if _is_managed_write(row)]
        if not managed:
            return 0
        request_ids = [row.request_id for row in managed]
        if purge:
            # 删除顺序真实记录：events → operations → tasks → 请求行。
            for model in (ExecutionEventModel, ExecutionOperationModel, ExecutionTaskModel):
                session.execute(
                    delete(model).where(model.user_id == user_id,
                                        model.request_id.in_(request_ids))
                )
        # 逐请求真实撤销租约（同一事务；目标 request_id 精确相等，行级真实 DML）。
        for row in managed:
            session.execute(
                update(ExecutionRequestModel)
                .where(ExecutionRequestModel.user_id == user_id,
                       ExecutionRequestModel.request_id == row.request_id)
                .values(lease_worker_id=None, lease_token=None, lease_deadline=None)
            )
        if not purge:
            return len(request_ids)
        session.execute(
            delete(ExecutionRequestModel).where(
                ExecutionRequestModel.user_id == user_id,
                ExecutionRequestModel.request_id.in_(request_ids),
            )
        )
        # 真实 cache 擦除：按本次真实受影响会话（managed 请求行上的 conversation_id；
        # 单会话调用即该会话）的 owner key 精确删除，不拼空/dummy owner、不误删其它
        # owner；删除失败不吞——真实失败使调用者隐私事务整体回滚。
        from app.db.model_budget_models import ModelResultCache
        from app.services.llm.budget import key

        if conversation_id is not None:
            conversations = [str(conversation_id)]
        else:
            conversations = sorted({str(row.conversation_id) for row in managed
                                    if str(row.conversation_id or "")})
        owner_keys = [key(user_id + ":" + cid) for cid in conversations]
        if owner_keys:
            session.execute(
                delete(ModelResultCache).where(ModelResultCache.owner_id.in_(owner_keys))
            )
        return len(request_ids)


def _is_managed_write(row) -> bool:
    if getattr(row, "requested_mode", None) != "react_write":
        return False
    return getattr(row, "effective_mode", None) == "react_write"


def _json_object(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


__all__ = [
    "CONTROLLER_TASK_ID",
    "RUNTIME_ORIGIN_GENERATION",
    "RUNTIME_ORIGIN_KEY",
    "RUNTIME_ORIGIN_PROTOCOL",
    "RUNTIME_PREFIX",
    "RUNTIME_VERSION",
    "ReactCheckpointStore",
    "RuntimePrivacyInvalidated",
]
