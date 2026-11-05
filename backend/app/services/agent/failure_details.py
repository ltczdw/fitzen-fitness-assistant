"""Public failure descriptions from application error codes, never raw errors.

This describes technical outcomes only; it does not classify user intent,
authorize actions, change statuses, or create missing personal facts.
"""
from __future__ import annotations

# code -> category, verified missing capability/result, next step, retryable
_ERRORS = {
    "no_matching_goal": ("tool_binding", "所选工具未匹配本次目标，处理尚未进入对应回答流程。", "需要核对任务与工具的衔接；这不代表你缺少个人资料。", False),
    "ambiguous_goal": ("tool_binding", "所选工具引用了多个目标，尚未确定要处理哪一项。", "需要重新绑定具体目标，不能将不同问题混在一起处理。", False),
    "knowledge_retrieval_failed": ("service", "知识检索服务未能返回可核实的结果。", "请稍后重试；服务维护者需检查检索服务连接和日志。", True),
    "memory_source_unavailable": ("source", "引用的记忆或当前约束已变更，缺少可验证的来源。", "请重新选择要引用的记录或重新说明当前条件。", True),
    "semantic_unit_dependency_missing": ("source", "缺少所选记忆的必要依赖，无法完整保留条件或否定。", "请重新选择完整来源。", False),
    "context_budget_exceeded": ("budget", "必要输入超出上下文容量，无法完整提交。", "请缩小引用的历史或计划范围。", False),
    "context_input_budget_exhausted": ("budget", "必要输入超出上下文容量，无法完整提交。", "请缩小引用的历史或计划范围。", False),
    "answer_model_empty": ("model", "回答模型没有返回有效内容。", "请稍后重试；无需重复补充已经提供的信息。", True),
    "information_service_unavailable": ("model", "个人条件识别没有得到通过校验的结果，尚不能确定具体缺哪项信息。", "请稍后重试；无需猜测或重复补齐所有个人资料。", True),
    "plan_worker_unavailable": ("service", "训练计划生成功能未加载。", "需要服务维护者检查后端功能初始化。", False),
    "plan_worker_failed": ("service", "训练计划生成过程失败，具体故障原因尚未确认。", "请稍后重试；服务维护者需检查该请求日志。", True),
    "plan_worker_empty": ("service", "训练计划生成功能未返回有效结果。", "请稍后重试；无需重复补充已经提供的条件。", True),
    "saved_record_read_failed": ("service", "保存记录的读取失败，无法核实当前值。", "请稍后重试或在记录页面核对。", True),
    "saved_record_query_unavailable": ("service", "缺少可核实的已保存记录读取结果。", "请稍后重试或在记录页面核对。", True),
    "saved_record_selection_unavailable": ("source", "尚未能确定本次引用的是哪条已保存记录。", "请明确记录的日期或动作。", True),
    "quoted_answer_unavailable": ("source", "引用内容未能完成可靠复述。", "请重新选择要引用的对话。", True),
    "read_result_unconfirmed_not_retried": ("source", "此前读取是否完成尚未确认，已避免重复调用。", "请核对结果后重新发起请求。", False),
}


# Finite application-owned technical outcomes. These do not interpret user text.
_ERRORS.update({
    "model_adapter_unavailable": ("configuration", "AI服务缺少当前所需的结构化识别接口。", "需要维护者检查模型适配器；不能据此判断缺少个人信息。", False),
    "model_not_configured": ("configuration", "AI服务尚未配置。", "需要维护者检查模型配置；补充个人资料无法解决。", False),
    "model_auth_failed": ("configuration", "AI服务鉴权失败。", "需要维护者检查供应商凭据和访问权限。", False),
    "model_payment_required": ("configuration", "AI供应商拒绝了计费或账户权限请求。", "需要维护者核对供应商账户状态。", False),
    "model_request_rejected": ("model", "AI接口拒绝了请求参数。", "需要维护者检查请求格式和接口合同。", False),
    "model_rate_limited": ("model", "AI服务当前请求过于频繁（HTTP 429）。", "请稍后重试。", True),
    "model_http_503": ("model", "AI服务暂时不可用（HTTP 503）。", "请稍后重试；具体供应商故障原因尚未确认。", True),
    "model_upstream_failed": ("model", "AI上游服务未能正常完成请求。", "请稍后重试；需要维护者核对该请求日志。", True),
    "model_transport_failed": ("model", "连接AI服务失败或连接中断。", "请稍后重试；需要维护者检查服务连接。", True),
    "model_timeout": ("model", "等待AI服务响应超时。", "请稍后重试。", True),
    "model_protocol_invalid": ("model", "AI返回的结构化结果未通过格式或字段校验。", "本次结果未被采用，请重试；无需猜测补充个人资料。", True),
    "model_empty_response": ("model", "AI没有返回有效回答内容。", "请稍后重试。", True),
    "model_circuit_open": ("model", "AI服务连续失败，当前暂时停止新调用。", "请稍后重试。", True),
    "model_token_budget_exhausted": ("budget", "本次处理剩余Token预算不足。", "请缩小处理范围；本次未完成部分不会自动重试。", False),
    "model_call_budget_exhausted": ("budget", "本次处理的模型调用次数达到上限。", "请核对已完成结果后重新发起未完成部分。", False),
    "model_budget_unavailable": ("service", "模型计量或预算状态无法核实，已停止调用。", "需要维护者检查计量存储；不能据此认定额度用尽。", True),
    "model_policy_stopped": ("budget", "本次处理受模型调用策略限制，已停止。", "请核对该请求的策略停止原因。", False),
    "model_request_deadline": ("budget", "本次处理已超时。", "请核对已完成结果后重试未完成部分。", False),
    "model_request_cancelled": ("budget", "本次请求已取消。", "如需继续，请核对已有结果后重新发起。", False),
    "semantic_result_reference_invalid": ("source", "引用的历史回答未通过来源校验。", "请明确要继续的回答或重新说明当前目标。", False),
})


def exception_failure(error: BaseException) -> dict | None:
    """Classify typed technical exceptions; never copy exception messages/URLs."""
    from app.services.llm.budget import ModelBudgetError, ModelBudgetUnavailable
    from app.services.llm.client import LLMGatewayError, LLMErrorCategory, LLMContextBudgetError
    from app.services.context.budget import ContextBudgetExceeded
    code = None
    if isinstance(error, (ContextBudgetExceeded, LLMContextBudgetError)):
        code = ("semantic_unit_dependency_missing" if getattr(error, "reason", None) == "missing_dependency"
                else "context_budget_exceeded")
    elif isinstance(error, ModelBudgetError) or (
            isinstance(error, LLMGatewayError) and error.category == LLMErrorCategory.POLICY):
        value = getattr(error, "code", None)
        cause = error.__cause__
        if value is None and isinstance(cause, ModelBudgetError):
            value = cause.code
        if isinstance(error, ModelBudgetUnavailable) or isinstance(cause, ModelBudgetUnavailable):
            code = "model_budget_unavailable"
        elif value == "model_request_cancelled": code = value
        elif value in {"model_request_deadline", "model_budget_expired"}: code = "model_request_deadline"
        elif isinstance(value, str) and (value in {"model_prepare_token_budget_exhausted", "model_task_token_budget_exhausted"}
                or value.startswith("model_token_budget_exhausted:")): code = "model_token_budget_exhausted"
        elif isinstance(value, str) and (value in {"model_prepare_budget_exhausted", "model_task_budget_exhausted"}
                or value.startswith("model_budget_exhausted:")): code = "model_call_budget_exhausted"
        else: code = "model_policy_stopped"
    elif isinstance(error, LLMGatewayError):
        value, status = getattr(error, "code", None), error.status_code
        if value == "semantic_result_reference_invalid": code = value
        elif value in {"model_timeout", "model_circuit_open", "model_empty_response", "model_adapter_unavailable"}: code = value
        elif error.category == LLMErrorCategory.CONFIG: code = "model_not_configured"
        elif status in {401, 403}: code = "model_auth_failed"
        elif status == 402: code = "model_payment_required"
        elif status == 429: code = "model_rate_limited"
        elif status == 503: code = "model_http_503"
        elif status in {408, 504}: code = "model_timeout"
        elif status in {400, 413, 422}: code = "model_request_rejected"
        elif error.category == LLMErrorCategory.TRANSPORT: code = "model_transport_failed"
        elif error.category == LLMErrorCategory.PROTOCOL: code = "model_protocol_invalid"
        else: code = "model_upstream_failed"
    if code is None:
        return None
    category, message, remedy, retryable = _ERRORS[code]
    detail = dict(code=code, category=category, message=message, remedy=remedy,
                  missing_fields=[], retryable=retryable)
    if code == "context_budget_exceeded" and getattr(error, "detail", {}).get("tools_alone_exceed") is True:
        detail.update(component="tools", message="必需工具及字段定义本身已超出本次上下文预算。",
                      remedy="需要维护者精简请求结构或调整上下文配置；仅缩短这句提问无法解决。此次超限调用未发送。")
    previous = getattr(error, "correction_failure_code", None)
    if previous:
        messages = {
            "structured_invalid_json": "AI返回的JSON格式不合法，需要纠错",
            "structured_invalid_arguments": "AI返回的字段未通过校验，需要纠错",
            "semantic_result_reference_invalid": "历史回答引用未通过来源校验，需要重新识别",
        }
        previous_code = previous if previous in messages else "structured_validation_failed"
        prefix = messages.get(previous, "AI返回的结构化识别结果未通过校验，需要纠错")
        detail.update(phase="structured_correction", preceding_failure={"code":previous_code,"message":prefix})
        detail["message"] = prefix + "；" + message
    return detail


def transport_failure(error: BaseException) -> dict:
    """Shared HTTP/SSE error envelope; commit uncertainty forbids blind retries."""
    from app.db.storage import StorageUnavailableError, CommitUnknownError, IdempotencyConflict
    from app.services.agent.execution_state import RequestIdConflict
    if isinstance(error, CommitUnknownError):
        return dict(code="commit_unknown", detail="保存结果尚未确认，请先查看记录；不要直接重复保存。", retryable=False, http_status=503)
    if isinstance(error, StorageUnavailableError):
        return dict(code="storage_unavailable", detail="数据存储暂时不可用，请稍后重试；已有保存结果请在记录中核对。", retryable=True, http_status=503)
    if isinstance(error, (RequestIdConflict, IdempotencyConflict)):
        return dict(code="request_id_conflict" if isinstance(error, RequestIdConflict) else "idempotency_conflict",
                    detail="同一请求标识已用于其他内容，请先核对原请求；修改内容需使用新的请求标识。", retryable=False, http_status=409)
    known = exception_failure(error)
    if known:
        return dict(code=known["code"], detail=known["message"] + known["remedy"],
                    retryable=known["retryable"], http_status=503)
    return dict(code="execution_failed", detail="本次执行失败，具体原因尚未确认；涉及保存的数据请先查看记录。",
                retryable=False, http_status=500)


def observation_failure(observation: dict) -> dict | None:
    if observation.get('status') not in {'unavailable','rejected','blocked_dependency'} and not (
            observation.get('status')=='needs_clarification'
            and observation.get('error') in {'no_matching_goal','ambiguous_goal'}):
        return None
    code = observation.get('error')
    # The executor emits a finite technical frame. Never display its suffix
    # (exception type/body) or infer a missing configuration from it.
    if isinstance(code, str) and code.partition(':')[0] == 'executor_error':
        return dict(code='tool_execution_failed', category='service',
                    message='工具执行失败，具体故障原因尚未确认。',
                    remedy='请稍后重试；服务维护者需核对该请求日志。',
                    missing_fields=[], retryable=True)
    if code == 'insufficient_evidence':
        result = observation.get('result') or {}
        quality = result.get('retrieval_quality') or {}
        reason = quality.get('reason')
        if result.get('evidence_filtered') is True:
            missing, message = 'applicable_knowledge_evidence', '缺少适用于当前问题及条件的知识依据。'
        elif reason == 'no_hits':
            missing, message = 'matching_knowledge_evidence', '本次检索未找到与问题匹配的知识依据。'
        elif reason == 'low_evidence_confidence':
            missing, message = 'reliable_knowledge_evidence', '已检索到资料，但缺少可靠性达到回答要求的知识依据。'
        else:
            missing, message = 'knowledge_evidence', '缺少支持当前回答的可靠知识依据，具体不足类型尚未确认。'
        return dict(code=code, category='evidence', message=message,
                    remedy='需要补充或重新检索相关可靠资料；补充个人信息不能替代知识依据。',
                    missing_fields=[missing], retryable=False)
    if code not in _ERRORS:
        return dict(code='unclassified_failure', category='unknown',
                    message='本次操作未返回可核实的结果，具体故障原因尚未确认。',
                    remedy='请按本次请求记录核对日志，暂不能确定需要补充哪项条件。',
                    missing_fields=[], retryable=False)
    category, message, remedy, retryable = _ERRORS[code]
    return dict(code=code, category=category, message=message, remedy=remedy,
                missing_fields=[], retryable=retryable)


def read_failure_details(state: dict) -> list[dict]:
    """Current per-goal results win; canonical auxiliary reads are fallbacks.

    A later completed result clears an earlier failure for that read goal.
    Unknown model/error text is never reflected into the public response.
    """
    from app.services.agent.task_models import is_record_write, serves_read_goal
    from app.services.agent.react_tools import TOOL_CATALOG
    candidates = state.get('intent_candidates') or []
    if (not candidates or state.get('safety_level') == 'urgent'
            or state.get('unbound_write_calls')
            or any(is_record_write(c) for c in candidates)
            or any(f.get('permission') == 'write' for f in state.get('call_facts') or [])
            or any(o.get('commit_state') or o.get('receipt_refs')
                   for o in state.get('observations') or [])):
        return []
    results = state.get('goal_results') or {}
    observations = list(results.values())
    # Keep the latest result for each actual auxiliary read tool. It may have
    # failed before it could be bound as a goal result.
    latest = {}
    for obs in state.get('observations') or []:
        source = obs.get('_source') or {}
        name = source.get('tool_name')
        spec = TOOL_CATALOG.get(name)
        if spec is not None and spec.permission == 'read':
            latest[name] = obs
    # Answer tool results have goal identities; never resurrect a stale
    # failure from their observation history after the goal succeeded.
    observations.extend(obs for name, obs in latest.items()
                        if TOOL_CATALOG[name].task_type is None
                        or obs.get('error') in {'no_matching_goal','ambiguous_goal'} or any(
                            serves_read_goal(TOOL_CATALOG[name].task_type, c.get('task_type'))
                            and c.get('id') not in (state.get('covered_goals') or [])
                            for c in candidates))
    details = []
    model_code = state.get('model_failure_code')
    if model_code in _ERRORS and state.get('status') != 'complete':
        details.append(observation_failure({'status':'unavailable','error':model_code}))
    for obs in observations:
        detail = observation_failure(obs)
        if detail and detail not in details:
            details.append(detail)
    return details
