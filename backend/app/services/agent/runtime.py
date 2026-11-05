"""One request lifecycle for JSON and SSE; transport never reruns business work."""
import asyncio
import logging
from uuid import uuid4

from app.schemas.chat import ChatResponse
from app.services.telemetry import RequestMetrics, active_metrics, mark_degraded
from app.services.memory.compact import current_turn
from app.services.agent.execution_state import _memory_error, _request_memory
from app.services.rag.retrieval import retrieval_traces

logger = logging.getLogger(__name__)


async def save_diagnostics(request, response, metrics, status="completed"):
    from app.db.quality_repository import record_run
    try:
        saved = await asyncio.to_thread(record_run, request, response, metrics, status)
        return "ready" if saved is not False else "not_used"
    except Exception as exc:
        # Diagnostics failure must not turn a successful business write into a retry.
        logger.warning("diagnostics_unavailable error_type=%s", type(exc).__name__)
        return "degraded"


async def _execute_request_inner(agent, request, parsed_input=None):
    metrics = RequestMetrics()
    token = active_metrics.set(metrics)
    error_token = _memory_error.set(None)
    try:
        async with agent._request_scope(request) as cached:
            cached_metrics = cached.get("metrics") if isinstance(cached, dict) else None
            cached_write = (isinstance(cached_metrics, dict)
                            and cached_metrics.get("execution_mode") == "react_write"
                            and cached_metrics.get("react_write") in {
                                "complete", "completed", "partial", "limited",
                                "needs_clarification"})
            write_request = ((getattr(request, "mode", None) or "legacy") == "react_write"
                             or cached_write)
            if cached and not write_request:
                return ChatResponse.model_validate(cached).model_copy(update={"replayed": True})
            if write_request:
                # Auto/read-only requests can resolve to the write runtime.
                # Their partial compact response is not a C1 terminal: verify
                # the same receipts or resume instead of freezing that failure.
                # react_write：C1 无行/证明读取失败→内存冻结首响应（必要时降级）；
                # C1 终态→真实证明核验后的冻结响应；C1 未完成→restore（None 落入下方）。
                # 内存轮未完成（如投影故障）时同样先走 C1 终态重放：零新模型/零 DML。
                from app.services.agent import react_write_runtime as write_runtime
                payload = write_runtime.write_replay(request, dict(cached) if cached else None)
                if payload is not None:
                    if "replayed" not in payload:
                        payload = {**payload, "replayed": True}
                    # 重放零新模型/零 DML（不经过 diagnostics/memory 写入）。
                    return ChatResponse.model_validate(payload)
                # An old controller snapshot is authoritative even with zero
                # consumption. Apply its frozen shared limits before pre-write
                # understanding can buy any model call under the new policy.
                from app.core.config import settings
                from app.db import session as db
                from app.services.llm.budget import active_scope
                budget = active_scope.get()
                if (budget is not None and settings.react_write_enabled
                        and settings.memory_read_backend == "compact_v1"):
                    factory = getattr(budget.store, "factory", None) or db.SessionLocal
                    frozen = (await asyncio.to_thread(
                        write_runtime._frozen_budget_limits,
                        factory, str(request.user_id), str(request.request_id))
                        if factory is not None else None)
                    if frozen is not None:
                        await write_runtime._budget_bootstrap(budget, frozen)
            try:
                # 请求 react_readonly 时不运行 legacy planner shadow：模式资格已在主管线内
                # 用规则/来源/pending 判定；默认开关关闭时原行为不变。
                if getattr(request, "mode", "legacy") != "react_readonly":
                    await agent._run_planner_shadow(request.message)
                response = await agent._respond_impl(request, parsed_input)
                message_id = (current_turn.get() or request.request_id) + ":assistant"
                write_metrics = getattr(response, "metrics", None) or {}
                # 本 write runtime 已冻结的原 request/message/run 三 ID（execution_mode
                # react_write + 三 ID 非空 + request_id 与本请求一致）可以保留；一般
                # partial 原请求 client retry 仍非 replayed，但保留冻结原 ID 继续做应做
                # 的 manual/finish。readonly/legacy 与客户端注入/伪 ID 不走此路径。
                frozen_write_ids = (
                    (getattr(request, "mode", None) or "legacy") == "react_write"
                    and (write_metrics.get("execution_mode") == "react_write"
                         or write_metrics.get("preserve_frozen_ids") is True)
                    and isinstance(response.request_id, str) and response.request_id
                    and isinstance(response.message_id, str) and response.message_id
                    and isinstance(response.run_id, str) and response.run_id
                    and response.request_id == request.request_id)
                if not ((getattr(response, "replayed", False) or frozen_write_ids)
                        and response.request_id and response.message_id and response.run_id):
                    # 非 C1 重放且无可信冻结写身份：本次请求身份。
                    response = response.model_copy(update={"request_id": request.request_id,
                        "message_id": message_id, "run_id": metrics.run_id})
                if _memory_error.get():
                    mark_degraded("memory_persistence", failure=True)
                from app.services.llm.budget import active_scope
                scope = active_scope.get()
                if scope and scope.policy_stops:
                    mark_degraded('model_policy_stop')
                detail = metrics.summary()
                detail['model_policy_stops'] = sorted(scope.policy_stops) if scope else []
                detail["retrieval"] = list(retrieval_traces.get() or [])
                recent=(_request_memory.get() or {}).get('recent_window')
                if recent:
                    detail['recent_memory']={k:recent[k] for k in ('turn_limit','token_limit','candidate_turns','tokens')}
                    detail['recent_memory'].update(loaded_turns=len(recent['loaded_turn_ids']),
                                                    omitted_turns=len(recent['omitted_turn_ids']))
                # 合并业务响应携带的模式元数据（阶段B requested/execution_mode/mode_reason/react_status）
                _mode_meta = getattr(response, "metrics", None) or {}
                for _k, _v in _mode_meta.items():
                    detail.setdefault(_k, _v)
                diagnostics_status = await save_diagnostics(request, response, detail)
                response = response.model_copy(update={"metrics": detail,
                    "memory_status": "degraded" if _memory_error.get() else "ready",
                    "diagnostics_status": diagnostics_status})
                if current_turn.get():
                    committed_response = await agent._finish_memory(current_turn.get(), response.answer, response.model_dump())
                    if isinstance(committed_response, dict):
                        response = ChatResponse.model_validate(committed_response)
                    if _memory_error.get():
                        response = response.model_copy(update={"memory_status": "degraded"})
                        if (response.metrics or {}).get("memory_update_requested"):
                            response = response.model_copy(update={"answer": "这次记忆保存结果尚未确认，请用原请求重试核对。", "intent": "error",
                                "metrics": {**response.metrics, "memory_update_status": "unconfirmed"}})
                        if (response.metrics or {}).get("state_update_requested"):
                            response = response.model_copy(update={"answer": "本次会话临时条件更新结果尚未确认，请用原请求重试核对。", "intent": "error",
                                "metrics": {**response.metrics, "state_update_status": "unconfirmed"}})
                if ((getattr(request, "mode", None) or "legacy") == "react_write"
                        and (response.metrics or {}).get("react_write") not in (None, "", "limited")):
                    from app.services.agent.react_write_runtime import finalize_write_response
                    response = await finalize_write_response(request, response)
                return response
            except BaseException as exc:
                # Carry verified accounting across the task/transport boundary;
                # a failed execution may already have made paid model calls.
                exc._request_model_used = any(c.get('capability') == 'chat' for c in metrics.calls)
                from app.services.llm.budget import ModelBudgetError
                from app.services.llm.client import LLMGatewayError, LLMErrorCategory
                policy = isinstance(exc, ModelBudgetError) or (isinstance(exc, LLMGatewayError) and exc.category == LLMErrorCategory.POLICY)
                if policy:
                    mark_degraded('model_policy_stop')
                    from app.services.agent.failure_details import exception_failure
                    detail = exception_failure(exc)
                    cause = (detail['message'] + detail['remedy']) if detail else '本次模型调用受限，已停止。'
                    response = ChatResponse(answer=cause + '已完成的记录保留，请在记录页核对；未完成部分没有自动重试。',
                        intent='execution_limited', request_id=request.request_id,
                        message_id=(current_turn.get() or request.request_id)+':assistant', run_id=metrics.run_id,
                        metrics={**metrics.summary(), 'execution_mode':'limited', 'react_status':'limited',
                                 'failure_details':[detail] if detail else []}, model_used=any(c.get('capability')=='chat' for c in metrics.calls))
                    diagnostics_status=await save_diagnostics(request,response,response.metrics,'budget_stopped')
                    response=response.model_copy(update={'diagnostics_status':diagnostics_status})
                    if current_turn.get():
                        await agent._finish_memory(current_turn.get(),response.answer,response.model_dump())
                        if _memory_error.get():
                            response=response.model_copy(update={'memory_status':'degraded'})
                    return response
                cancelled = isinstance(exc, asyncio.CancelledError)
                if not cancelled:
                    mark_degraded("execution_error", failure=True)
                failed = ChatResponse(answer="", intent="error", run_id=metrics.run_id,
                    request_id=request.request_id, message_id=(current_turn.get() or request.request_id) + ":assistant")
                try:
                    await asyncio.shield(save_diagnostics(request, failed, metrics.summary(), "cancelled" if cancelled else "failed"))
                except BaseException:
                    pass
                raise
    finally:
        _memory_error.reset(error_token)
        active_metrics.reset(token)


async def _execute_with_request_scope(agent, request, parsed_input=None):
    from app.services.llm.budget import TASK_POLICY, model_scope, active_scope, key

    current = active_scope.get()
    if current is not None and current.kind == 'request':
        current.name = 'request:' + request.request_id
        # Preserve an explicit low-level shared scope; middleware-created HTTP
        # roots are already marked per_task.
    with model_scope('request:' + request.request_id,
                     reuse=current is not None and current.kind == 'request',
                     policy=TASK_POLICY) as budget:
        budget.owner_id = key(request.user_id + ':' + request.conversation_id)
        mode = getattr(request, "mode", None) or "legacy"
        if mode in {"react_write", "react_readonly"}:
            # 请求级物理尝试隔离：上一请求的物理失败（含熔断）不得污染本请求。
            from app.services.agent import react_write_runtime as write_runtime
            write_runtime.refresh_circuit_state(agent)
        restore = None
        if (getattr(request, "mode", None) or "legacy") == "react_write":
            # 请求局部计量依赖选择：同一 scope 对象只替换 store 为应用拥有的私有
            # 副本（保留原策略/失败），finally 复原；原 store 对象/全局 factory 不改。
            from app.services.agent import react_write_runtime as write_runtime
            if getattr(request, "confirmation", None) is not None:
                # 缓存重放之前的 selector 一致性（真实 409 语义，不进入 replay 捷径）。
                write_runtime.check_confirmation_selector(request)
            restore, meter_error = write_runtime.bind_request_scope_store(budget)
            if meter_error is not None:
                lifecycle = write_runtime._write_scope.get()
                if isinstance(lifecycle, dict):
                    lifecycle["meter_error"] = meter_error
        try:
            # Resolve old/new policy after any request-local store binding, so
            # pre-write understanding and old cache keys use the frozen strategy.
            budget.store.resolve_scope_policy(budget)
            return await _execute_request_inner(agent, request, parsed_input)
        finally:
            if restore is not None:
                restore()


async def execute_request(agent, request, parsed_input=None):
    from app.services.llm.budget import model_scope, active_scope
    if (getattr(request, "mode", None) or "legacy") in {
            "auto", "react_readonly", "react_write"}:
        # Auto/readonly can adopt an authorized record continuation. This
        # context opens no database/tool by itself; a resolved writer needs it
        # through finalization and lease release just like an explicit writer.
        # write_lifecycle_scope 覆盖 run_write、typed response 投影/IDs/memory/diagnostics
        # 直到 finalize_write_response；计量沿用已存在的原 request scope（middleware
        # 建立并复用），不新建 scope、不改全局 store/factory、不绕显式 meter 注入。
        from app.services.agent import react_write_runtime as write_runtime
        async with write_runtime.write_lifecycle_scope():
            return await _execute_with_request_scope(agent, request, parsed_input)
    return await _execute_with_request_scope(agent, request, parsed_input)
