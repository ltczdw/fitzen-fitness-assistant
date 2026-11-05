"""Full protocol preflight, whole-unit pruning, and durable correction budgets."""
import asyncio
import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import BaseModel
from sqlalchemy import select

from app.core.config import settings
from app.db.model_budget_models import ModelCallLedger, ModelQuota
from app.services.context.budget import ContextBlock, ContextBudgetExceeded, ContextBudgetManager
from app.services.context.token_counter import TokenCounter
from app.services.context.request_tokens import count_request, matching_native_model, PreparedPayload
from app.services.llm.client import LLMClient, LLMGatewayError
from app.services.llm.budget import model_prepare
from app.services.llm.structured import StructuredOutputError
from app.services.agent.failure_details import exception_failure
from tests.test_task_model_budget import make_store, request_scope


class Result(BaseModel):
    value: int


def native(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "mock-not-real")
    monkeypatch.setattr(settings, "llm_base_url", "https://api.deepseek.com/v1")
    monkeypatch.setattr(settings, "llm_model", "deepseek-flash")
    monkeypatch.setattr(settings, "llm_retry_delays", "0,0")
    monkeypatch.setattr(settings, "llm_max_output_tokens", 128)


def payload(messages, tools=None, output=64):
    result = {"model": settings.llm_model, "messages": messages, "max_tokens": output, "thinking": {"type": "disabled"}}
    if tools: result.update(tools=tools, tool_choice="required")
    return result


def tools(size=800):
    return [{"type": "function", "function": {"name": "read", "description": "D" * size,
        "parameters": {"type": "object", "properties": {"value": {"type": "string"}}}}}]


def test_official_model_counter_includes_schema_controls_and_tool_history(monkeypatch):
    native(monkeypatch)
    from app.services.agent.semantics import RequestUnderstanding
    tool = [{"type": "function", "function": {"name": "understand", "parameters": RequestUnderstanding.model_json_schema()}}]
    messages = [{"role": "system", "content": "识别条件，保留否定。"}, {"role": "user", "content": "我对什么过敏"}]
    counted = count_request(payload(messages, tool))
    assert counted.mode == "deepseek-v41-estimate"
    assert counted.input_tokens > count_request(payload(messages)).input_tokens
    assert counted.input_tokens < len(json.dumps(payload(messages, tool), ensure_ascii=False).encode()) / 2
    conversation = [*messages, {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
        "function": {"name": "read", "arguments": '{"value":"SOURCE_CONDITION"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "不得遗漏条件" * 200}]
    assert count_request(payload(conversation, tool)).input_tokens > counted.input_tokens
    literal = count_request(payload([{"role": "user", "content": "<｜Assistant｜>" * 100}]))
    assert literal.input_tokens > 200  # literal control spellings never collapse to single tokens


def test_unmatched_or_unavailable_tokenizer_uses_explicit_full_payload_bound(monkeypatch):
    native(monkeypatch)
    p = payload([{"role": "user", "content": "small"}], tools())
    monkeypatch.setattr(settings, "llm_base_url", "https://other-provider.invalid")
    counted = count_request(p)
    assert counted.mode == "conservative-utf8"
    assert counted.input_tokens == len(json.dumps(p, ensure_ascii=False, sort_keys=True, default=str).encode())
    monkeypatch.setattr(settings, "llm_base_url", "https://api.deepseek.com")
    monkeypatch.setattr('app.services.context.request_tokens.official_plain_tokenizer', lambda: (_ for _ in ()).throw(OSError("missing")))
    assert count_request(p).mode == "conservative-utf8"


def test_required_tool_schema_overflow_is_rejected_before_http(monkeypatch):
    from app.services.llm import client as module
    monkeypatch.setattr(settings, "llm_base_url", "https://unmatched.invalid")
    manager = ContextBudgetManager(window=1000, output=64, margin=0, counter=TokenCounter(tokenizer_path=""))
    monkeypatch.setattr(module, "context_budget", manager)
    monkeypatch.setattr(settings, "llm_max_output_tokens", 64)
    messages = [{"role": "user", "content": "small"}]
    assert manager.validate(messages) < manager.input_budget
    with pytest.raises(ContextBudgetExceeded) as caught:
        LLMClient._payload(messages, 0, False, tools=tools(2000))
    assert caught.value.reason == "required_request_over_budget"
    assert caught.value.detail["tools_present"] is True


def test_full_request_refits_only_whole_optional_groups_and_keeps_dependencies(monkeypatch):
    monkeypatch.setattr(settings, "llm_base_url", "https://unmatched.invalid")
    manager = ContextBudgetManager(window=3000, output=64, margin=0, counter=TokenCounter(tokenizer_path=""))
    blocks = [ContextBlock("system", "system", [{"role": "system", "content": "RULES"}], 100, required=True),
              ContextBlock("query", "query", [{"role": "user", "content": "CURRENT: no load unless condition holds"}], 100, required=True),
              ContextBlock("dep", "rag", [{"role": "system", "content": "REQUIRED_DEP"}], 90, group_id="evidence", required=True),
              ContextBlock("turn", "recent", [{"role": "user", "content": "OLD_USER" * 90},
                                                {"role": "assistant", "content": "OLD_ANSWER" * 90}], 40)]
    messages = manager.build(blocks)
    assert any("OLD_USER" in m['content'] for m in messages)
    full = manager.prepare_request(payload(messages, tools(1200)))
    text = json.dumps(full["messages"])
    assert "OLD_USER" not in text and "OLD_ANSWER" not in text
    assert "CURRENT: no load unless condition holds" in text and "REQUIRED_DEP" in text
    assert full.tokens.input_tokens <= manager.input_budget
    assert full.report["dropped_blocks"] == ["turn"]
    assert full["tools"] == tools(1200)


def test_required_content_and_output_reservation_are_never_silently_reduced(monkeypatch):
    monkeypatch.setattr(settings, "llm_base_url", "https://unmatched.invalid")
    manager = ContextBudgetManager(window=700, output=64, margin=0, counter=TokenCounter(tokenizer_path=""))
    original = payload([{"role": "user", "content": "REQUIRED_NEGATION" * 100}], tools(50))
    snapshot = copy.deepcopy(original)
    with pytest.raises(ContextBudgetExceeded): manager.prepare_request(original)
    assert original == snapshot
    with pytest.raises(ContextBudgetExceeded, match="input budget"):
        manager.prepare_request(payload([{"role": "user", "content": "short"}], output=700))


def test_semantic_history_pruning_updates_inventory_and_correction_keeps_source(monkeypatch):
    from app.services.agent.dialogue_context import semantic_messages
    from app.services.llm import client as module
    monkeypatch.setattr(settings, "llm_base_url", "https://unmatched.invalid")
    memory = {"short_memory": [], "recent_window_messages": [], "recent_window": {
        "turn_limit": 10, "token_limit": 12000, "candidate_turns": 2,
        "omitted_turn_ids": [], "loaded_turn_ids": ["t0", "t1"], "tokens": 1000}}
    for n in range(2):
        for role in ("user", "assistant"):
            message = {"role": role, "message_id": f"t{n}:{role}", "turn_id": "t" + str(n),
                       "content": ("HISTORY_" + str(n)) * 50,
                       "app_meta": {"recent_window": True}}
            memory["short_memory"].append(message)
            memory["recent_window_messages"].append(message)
    messages = semantic_messages("RULES", "CURRENT_QUERY", memory)
    original = payload(messages, tools(800))
    used = count_request(original).input_tokens
    manager = ContextBudgetManager(window=used - 100 + 64, output=64, margin=0, counter=TokenCounter(tokenizer_path=""))
    fitted = manager.prepare_request(original)
    dropped = set(memory["recent_window"]["omitted_turn_ids"])
    assert dropped
    data = json.loads(fitted["messages"][-1]["content"])
    assert data["query"] == "CURRENT_QUERY"
    assert not any(m["message_id"].split(':')[0] in dropped for m in data["history"])
    corrected = messages.clone()
    corrected[0]["content"] += "\nCORRECTION_INSTRUCTION" * 10
    again = manager.prepare_request(payload(corrected, tools(800)))
    assert "CORRECTION_INSTRUCTION" in again["messages"][0]["content"]
    data_again = json.loads(again["messages"][-1]["content"])
    assert not any(m["message_id"].split(':')[0] in dropped for m in data_again["history"])
    assert not any("[历史消息ID: " + tid + ":" in m['content'] for tid in dropped for m in again['messages'])


def test_prepared_reservation_recounts_mutated_payload(monkeypatch):
    native(monkeypatch)
    manager = ContextBudgetManager(window=65536, output=128, margin=0)
    prepared = manager.prepare_request(payload([{"role": "user", "content": "small"}]))
    old = prepared.tokens.input_tokens
    prepared["tools"] = tools(4000)
    assert prepared.counted_tokens().input_tokens > old


def provider(monkeypatch, handler):
    native(monkeypatch)
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs:
        original(transport=httpx.MockTransport(handler), **kwargs))
    client = LLMClient()
    client.admission.token = AsyncMock()
    return client


def test_native_schema_correction_uses_counted_tokens_not_bytes(tmp_path, monkeypatch):
    store, engine = make_store(tmp_path)
    seen = []
    def handle(request):
        body = json.loads(request.content)
        seen.append(body)
        args = '{"value":' if len(seen) == 1 else '{"value":7}'
        return httpx.Response(200, json={"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "return", "type": "function", "function": {"name": "structured_result", "arguments": args}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    client = provider(monkeypatch, handle)
    messages = [{"role": "system", "content": "RETAIN CONDITIONS " * 500}, {"role": "user", "content": "CURRENT"}]
    async def run():
        with request_scope(store), model_prepare():
            return await client.complete_structured(messages, Result, allow_json_retry=True)
    try:
        assert asyncio.run(run()).value == 7
        assert len(seen) == 2 and "JSON syntax is invalid" in seen[1]["messages"][0]["content"]
        with store.factory() as session:
            rows = list(session.scalars(select(ModelCallLedger).order_by(ModelCallLedger.started_at)))
            assert len(rows) == 2 and all(row.status == "success" for row in rows)
            assert [row.reserved_tokens for row in rows] == [count_request(body).input_tokens + body['max_tokens'] for body in seen]
            assert rows[0].reserved_tokens < len(json.dumps(seen[0], ensure_ascii=False).encode())
            roots = list(session.scalars(select(ModelQuota).where(ModelQuota.kind.in_(["request", "prepare"]))))
            assert roots and all(q.calls == 2 and q.tokens == 220 for q in roots)
    finally: engine.dispose()


def test_correction_budget_stop_keeps_original_format_failure_and_only_one_http(tmp_path, monkeypatch):
    store, engine = make_store(tmp_path)
    seen = []
    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "return", "type": "function", "function": {"name": "structured_result", "arguments": '{"value":'}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    client = provider(monkeypatch, handle)
    messages = [{"role": "system", "content": "KEEP SOURCE AND NEGATION " * 800}, {"role": "user", "content": "CURRENT"}]
    spec = [{"type": "function", "function": {"name": "structured_result", "description": "Return the required structured result. This function does not execute any action.", "parameters": Result.model_json_schema()}}]
    first = LLMClient._payload(messages, 0, False, tools=spec, tool_choice={'type':'function','function':{'name':'structured_result'}})
    maximum = first.tokens.input_tokens + first['max_tokens']
    monkeypatch.setattr(settings, 'model_task_max_tokens', maximum)
    async def run():
        with request_scope(store, max_tokens=maximum), model_prepare():
            with pytest.raises(LLMGatewayError) as caught:
                await client.complete_structured(messages, Result, allow_json_retry=True)
        return caught.value
    try:
        error = asyncio.run(run())
        assert len(seen) == 1
        detail = exception_failure(error)
        assert detail["code"] == "model_token_budget_exhausted"
        assert detail["preceding_failure"]["code"] == "structured_invalid_json"
        assert "JSON格式不合法" in detail["message"] and "Token" in detail["message"]
        with store.factory() as session:
            rows = list(session.scalars(select(ModelCallLedger)))
            assert len(rows) == 1 and rows[0].usage['prompt_tokens'] == 100
    finally: engine.dispose()


@pytest.mark.parametrize("transport", ["json", "sse"])
def test_api_reports_both_json_failure_and_correction_budget_stop(tmp_path, monkeypatch, transport):
    from tests.react_phase_b_codex_support import environment, sse_body
    from app.services.llm.client import LLMErrorCategory
    error = LLMGatewayError("STOP", LLMErrorCategory.POLICY, code="model_prepare_token_budget_exhausted")
    error.correction_failure_code = "structured_invalid_json"
    with environment(tmp_path, monkeypatch, "correction-stop") as env:
        env.mock_llm(semantic_error=error)
        before = env.snapshot()
        result = (env.post("correction-stop", message="synthetic") if transport == "json" else
                  sse_body(env.stream("correction-stop", message="synthetic")))
        assert "JSON格式不合法" in result['answer'] and "Token" in result['answer']
        assert result['metrics']['failure_details'][0]['phase'] == 'structured_correction'
        calls = len(env.llm.semantic_calls)
        assert env.post("correction-stop", message="synthetic")['answer'] == result['answer']
        assert len(env.llm.semantic_calls) == calls and env.snapshot() == before


@pytest.mark.parametrize("mode", ["text", "tools", "stream"])
def test_gateway_required_overflow_creates_no_http_or_ledger_row(tmp_path, monkeypatch, mode):
    from app.services.llm import client as module
    store, engine = make_store(tmp_path)
    monkeypatch.setattr(settings, "llm_api_key", "mock-not-real")
    monkeypatch.setattr(settings, "llm_base_url", "https://full-preflight.invalid")
    monkeypatch.setattr(settings, "llm_model", "mock")
    monkeypatch.setattr(settings, "llm_max_output_tokens", 64)
    manager = ContextBudgetManager(window=1000, output=64, margin=0, counter=TokenCounter(tokenizer_path=""))
    monkeypatch.setattr(module, "context_budget", manager)
    seen = []
    original = httpx.AsyncClient
    def handle(request): seen.append(request); raise AssertionError("must stop before HTTP")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    llm = LLMClient(); llm.admission.token = AsyncMock()
    async def run():
        with request_scope(store), model_prepare():
            with pytest.raises(LLMGatewayError) as caught:
                messages = [{"role": "user", "content": "MUST_KEEP" * 300 if mode != "tools" else "small"}]
                if mode == "text": await llm.complete(messages)
                elif mode == "tools": await llm.complete_message(messages, tools=tools(3000))
                else: assert [v async for v in llm.stream(messages)] == []
        return caught.value
    try:
        assert getattr(asyncio.run(run()), 'code', None) == 'context_input_budget_exhausted'
        assert seen == []
        with store.factory() as session:
            assert list(session.scalars(select(ModelCallLedger))) == []
    finally: engine.dispose()


def test_format_correction_window_stop_preserves_cause_with_no_second_http(tmp_path, monkeypatch):
    from app.services.llm import client as module
    store, engine = make_store(tmp_path)
    seen = []
    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "return", "type": "function", "function": {"name": "structured_result", "arguments": '{"value":'}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    llm = provider(monkeypatch, handle)
    messages = [{"role": "system", "content": "ORIGINAL_CONDITIONS " * 100}, {"role": "user", "content": "CURRENT"}]
    spec = [{"type": "function", "function": {"name": "structured_result", "description": "Return the required structured result. This function does not execute any action.", "parameters": Result.model_json_schema()}}]
    first = LLMClient._payload(messages, 0, False, tools=spec, tool_choice={'type':'function','function':{'name':'structured_result'}})
    manager = ContextBudgetManager(window=first.tokens.input_tokens + first['max_tokens'] + 1, output=first['max_tokens'], margin=0)
    monkeypatch.setattr(module, "context_budget", manager)
    async def run():
        with request_scope(store), model_prepare():
            with pytest.raises(LLMGatewayError) as caught:
                await llm.complete_structured(messages, Result, allow_json_retry=True)
        return caught.value
    try:
        error = asyncio.run(run())
        assert len(seen) == 1 and error.code == 'context_input_budget_exhausted'
        detail = exception_failure(error)
        assert detail['preceding_failure']['code'] == 'structured_invalid_json'
        assert detail['code'] == 'context_budget_exceeded'
    finally: engine.dispose()


def test_large_real_understanding_schema_can_correct_inside_original_prepare_budget(tmp_path, monkeypatch):
    from app.services.agent.semantics import RequestUnderstanding, PROMPT, validate_understanding
    from tests.react_phase_b_codex_support import semantic_outline, semantic_task
    store, engine = make_store(tmp_path)
    seen = []
    query = "请说明训练恢复建议"
    raw = semantic_outline(query, [dict(semantic_task('training_qa', 'question', query), history_requirement='none')])
    def handle(request):
        body = json.loads(request.content); seen.append(body)
        args = '{"tasks":' if len(seen) == 1 else json.dumps(raw, ensure_ascii=False)
        return httpx.Response(200, json={"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "return", "type": "function", "function": {"name": "return_request_understanding", "arguments": args}}]}}],
            "usage": {"prompt_tokens": 13068, "completion_tokens": 354}})
    llm = provider(monkeypatch, handle)
    monkeypatch.setattr(settings, 'llm_max_output_tokens', 4096)
    messages = [{'role':'system','content':PROMPT + '\n' + 'CONTROL_BOUNDARY ' * 800}, {'role':'user','content':query}]
    async def run():
        with request_scope(store), model_prepare():
            return await llm.complete_structured(messages, RequestUnderstanding, name='return_request_understanding',
                allow_json_retry=True, validator=lambda value: validate_understanding(query, value))
    try:
        result = asyncio.run(run())
        assert result.tasks[0].text == query and len(seen) == 2
        old_second_reserve = len(json.dumps(seen[1], ensure_ascii=False, sort_keys=True).encode()) + 4096
        assert old_second_reserve > 64000 - 13422  # the old byte reservation blocks this correction
        assert count_request(seen[1]).input_tokens + 4096 <= 64000 - 13422
        with store.factory() as session:
            rows = list(session.scalars(select(ModelCallLedger)))
            assert len(rows) == 2 and all(row.status == 'success' for row in rows)
            quotas = list(session.scalars(select(ModelQuota).where(ModelQuota.kind=='prepare')))
            assert len(quotas) == 1 and quotas[0].calls == 2 and quotas[0].tokens == 26844
            assert quotas[0].max_tokens == 64000
    finally: engine.dispose()


def test_native_503_retains_its_new_counted_reservation(tmp_path, monkeypatch):
    store, engine = make_store(tmp_path)
    seen = []
    def handle(request):
        seen.append(json.loads(request.content))
        if len(seen) == 1: return httpx.Response(503)
        return httpx.Response(200, json={"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":100,"completion_tokens":10}})
    llm = provider(monkeypatch, handle)
    async def run():
        with request_scope(store), model_prepare():
            return await llm.complete([{'role':'user','content':'synthetic'}])
    try:
        assert asyncio.run(run()) == 'ok' and len(seen) == 2
        with store.factory() as session:
            rows = list(session.scalars(select(ModelCallLedger).order_by(ModelCallLedger.started_at)))
            assert [r.status for r in rows] == ['error','success'] and not rows[0].usage
            assert rows[0].reserved_tokens == count_request(seen[0]).input_tokens + seen[0]['max_tokens']
            quotas = list(session.scalars(select(ModelQuota).where(ModelQuota.kind=='prepare')))
            assert quotas[0].calls == 2 and quotas[0].tokens == rows[0].reserved_tokens + 110
    finally: engine.dispose()


def test_invalid_asset_integrity_falls_back_before_vendor_import(monkeypatch):
    import builtins
    native(monkeypatch)
    monkeypatch.setattr('app.services.context.request_tokens.official_plain_tokenizer',
                        lambda: (_ for _ in ()).throw(ValueError('tokenizer_asset_hash_mismatch')))
    original = builtins.__import__
    attempted = []
    def guarded(name, *args, **kwargs):
        if name == 'app.vendor.deepseek_v41.encoding':
            attempted.append(name)
            raise AssertionError('must verify before import')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    result = count_request(payload([{'role':'user','content':'synthetic'}]))
    assert result.mode == 'conservative-utf8' and attempted == []


def test_preflight_estimates_do_not_become_actual_usage_or_export_text(monkeypatch):
    from app.services.telemetry import active_metrics, RequestMetrics
    native(monkeypatch)
    metrics = RequestMetrics()
    token = active_metrics.set(metrics)
    try:
        prepared = LLMClient._payload([{'role':'user','content':'SYNTHETIC_SECRET_CONTENT'}],0,False)
        summary = metrics.summary()
        assert summary['model_calls'] == 0 and summary['input_tokens'] == 0
        rows = summary['context_preflights']
        assert len(rows) == 1 and rows[0]['status'] == 'accepted'
        assert rows[0]['request_input_tokens'] == prepared.tokens.input_tokens
        assert rows[0]['counter_mode'] == 'deepseek-v41-estimate'
        assert 'SYNTHETIC_SECRET_CONTENT' not in json.dumps(summary)
    finally: active_metrics.reset(token)


def test_tool_definition_overflow_names_component_not_user_information(monkeypatch):
    from app.services.llm.client import LLMContextBudgetError
    from app.services.agent.execution import ChatExecutionMixin
    from app.schemas.chat import ChatRequest
    monkeypatch.setattr(settings, 'llm_base_url', 'https://unmatched.invalid')
    manager = ContextBudgetManager(window=1000, output=64, margin=0, counter=TokenCounter(tokenizer_path=''))
    with pytest.raises(ContextBudgetExceeded) as caught:
        manager.prepare_request(payload([{'role':'user','content':'short question'}], tools(3000)))
    detail = exception_failure(LLMContextBudgetError(caught.value))
    assert detail['component'] == 'tools' and '字段定义' in detail['message']
    result = ChatExecutionMixin._safety_unavailable_response(ChatRequest(message='short'), 'normal',
        {'semantic_failure_code':'context_input_budget_exhausted','semantic_failure_detail':detail})
    assert '仅缩短这句提问无法解决' in result.answer


def test_correction_context_stop_preserves_completed_c1_receipts(monkeypatch):
    from app.services.agent import react_write_runtime
    from app.services.agent.execution import ChatExecutionMixin
    from app.schemas.chat import ChatRequest
    from app.services.llm.client import LLMContextBudgetError
    error = LLMContextBudgetError(ContextBudgetExceeded(2000,1000))
    error.correction_failure_code = 'structured_invalid_json'
    detail = exception_failure(error)
    monkeypatch.setattr(react_write_runtime, 'unfinished_progress', lambda request: {
        'answer':'目标甲已保存，目标乙未完成。','receipt_refs':['receipt-a'],
        'response_ids':{'request_id':'frozen-request','message_id':'frozen-message','run_id':'frozen-run'}})
    result = ChatExecutionMixin._safety_unavailable_response(ChatRequest(message='synthetic',mode='react_write'), 'normal',
        {'semantic_failure_code':'context_input_budget_exhausted','semantic_failure_detail':detail})
    assert result.answer == '目标甲已保存，目标乙未完成。'
    assert result.metrics['receipt_refs'] == ['receipt-a'] and result.request_id == 'frozen-request'
    assert result.metrics['failure_details'][0]['preceding_failure']['code'] == 'structured_invalid_json'
