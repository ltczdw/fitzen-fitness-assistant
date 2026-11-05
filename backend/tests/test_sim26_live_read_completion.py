"""SIM26-LIVE-01: goal read identity, completion/reuse and truthful limited replies.

All model output is a provider-boundary MockTransport response and all SQL uses an
isolated temporary database. RAG retrieval is mocked at the external boundary; the
real ASGI entry, React graph, read workers, runtime persistence and business SQL are
exercised. No real provider, resident service or personal data is used.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.services.agent.react_graph import (
    ReactDependencies,
    execute_react_batch,
)
from tests.react_phase_a_support import (
    RecordingExecutor,
    assistant,
    batch_state,
    committed_receipt,
    ref_of,
    tool_call,
)
from tests.react_phase_c2_3_support import (
    C3HTTPHarness,
    business_keys,
    ledger_snapshot,
    mock_provider_transport,
    synthetic_evidence_conflict_response,
)

LIVE_SOURCE = (
    "我想吃得更均衡，成年人早餐和晚餐的蛋白质一般怎么分配？"
    "只讲通用原则，不保存饮食或长期记忆。"
)
LIVE_FIRST_SENTENCE = "我想吃得更均衡，成年人早餐和晚餐的蛋白质一般怎么分配？"
SAVED_LIVE_QUERY = "成年人早餐和晚餐的蛋白质一般怎么分配？只要通用原则（不保存，不作长期记忆）"
SAVED_LIVE_ANSWER = (
    "目前没有你的个人数据，只能给通用原则。蛋白质分配不必每餐严格均等，更实用的做法是"
    "早晚餐都安排一个明确的蛋白质来源，让全天总量在一天内平衡即可。一般可参考：早餐约15至25克，"
    "晚餐约25至35克（估算值，需按体重和总摄入调整）；若午餐蛋白质较少，晚餐可适当多一些。\n\n"
    "早餐可选奶或酸奶、鸡蛋、豆浆配豆制品，也可加少量坚果。晚餐可用豆类、豆腐、豆制品、"
    "鸡蛋或奶制品，搭配谷物和蔬菜；素食者应轮换不同来源，以获得较全面的氨基酸。\n\n"
    "具体克数属估算，不能替代个体化评估；如有特殊健康状况或体重目标，建议咨询专业人员。"
)


@pytest.fixture
def live_env(tmp_path, monkeypatch):
    from tests.react_phase_c1_support import isolated
    from tests.react_phase_c2_2_support import ensure_catalog

    with isolated(monkeypatch, tmp_path / "sim26-live", "sqlite") as engine:
        ensure_catalog(engine)
        yield engine


def _provider_message(message):
    return {
        "choices": [{
            "message": message,
            "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
    }


def _task(task_type, text, *, action="question", fact_text=None, history_requirement="none"):
    return {
        "task_type": task_type, "action": action, "text": text,
        "is_negated": False, "is_hypothetical": False, "is_third_party": False,
        "fact_text": fact_text, "reply_mode": "grounded", "reply_purpose": "general",
        "context_ids": [], "fact_fields": [], "history_requirement": history_requirement,
    }


def _understanding(*tasks):
    return _provider_message({
        "role": "assistant", "content": "", "tool_calls": [{
            "id": "call-understanding", "type": "function",
            "function": {
                "name": "return_request_understanding",
                "arguments": json.dumps({
                    "tasks": list(tasks), "reply_mode": "grounded",
                    "reply_purpose": "general", "relation": "new",
                    "current_is_critical": False, "critical_fact_fields": [],
                    "critical_history_ids": [], "dialogue_ids": [], "references": [],
                    "memory_requests": [], "needs_clarification": False,
                    # Provider-bound normal safety result; a missing safety field
                    # is intentionally not an application default.
                    "safety": {"level": "normal", "evidence": "",
                               "categories": [], "reason": "scripted normal"},
                }, ensure_ascii=False),
            },
        }],
    })


def _tool_message(name, arguments, *, call_id="call-read"):
    return _provider_message({
        "role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(arguments, ensure_ascii=False)},
        }],
    })


def _answer_message(content):
    return _provider_message({"role": "assistant", "content": content})


def _hit(title, content):
    return SimpleNamespace(
        title=title, source="knowledge://sim26-live", chunk_id="chunk-" + title,
        content=content, score=1.0,
        metadata={"rerank_status": "model", "rerank_score": 1.0, "page": 1})


def _install_rag(monkeypatch):
    import app.services.rag.online as rag_online

    calls = []

    async def search(pipeline, query, **kwargs):
        calls.append(str(query))
        return [
            _hit("蛋白质分配原则", "早餐和晚餐都安排蛋白质来源，全天总量平衡即可。"),
            _hit("训练计划原则", "训练计划应循序渐进，安排恢复和动作质量。"),
            _hit("营养与恢复", "均衡饮食和充足恢复有助于训练适应。"),
        ]

    monkeypatch.setattr(rag_online, "search", search, raising=False)
    return calls


def _install_provider(monkeypatch, *, understanding=None, supervisor=None,
                      answer=None, plan_info=None, seen=None):
    import httpx

    def handler(request):
        body = json.loads(request.content.decode("utf-8"))
        tools = {item.get("function", {}).get("name")
                 for item in body.get("tools") or []}
        if seen is not None:
            seen.append({"tools": sorted(tools)})
        if "return_request_understanding" in tools:
            assert understanding is not None
            return httpx.Response(200, json=understanding)
        if "return_evidence_conflict_judgement" in tools:
            return synthetic_evidence_conflict_response(request)
        if "return_plan_information" in tools:
            assert plan_info is not None
            return httpx.Response(200, json=_tool_message("return_plan_information", plan_info))
        if tools & {"record_workout", "record_nutrition", "answer_training",
                    "advise_nutrition", "plan_training"}:
            value = supervisor(body) if callable(supervisor) else supervisor
            return httpx.Response(200, json=value)
        assert answer is not None
        return httpx.Response(200, json=answer)

    return mock_provider_transport(monkeypatch, [handler])


def _deps(llm, executor=None, **options):
    return ReactDependencies(
        llm=llm, tool_executor=executor, user_id="app-user-1",
        conversation_id="conv-sim26-live", request_id="req-sim26-live", **options)


class ScriptedLLM:
    """One semantic decision plus one supervisor batch, then a final text."""

    def __init__(self, *, understanding, supervisor, final="完成"):
        self.understanding = understanding
        self.supervisor = supervisor
        self.final = final
        self.semantic_calls = 0
        self.calls = 0

    def configured(self):
        return True

    async def complete_structured(self, messages, schema, *, name=None,
                                  temperature=0.0, validator=None, **kwargs):
        self.semantic_calls += 1
        value = schema.model_validate(self.understanding)
        if validator is not None:
            validator(value)
        return value

    async def complete_message(self, messages, temperature=0.0, tools=None,
                               tool_choice=None):
        self.calls += 1
        return self.supervisor if self.calls == 1 else {
            "role": "assistant", "content": self.final}


def test_live01_original_saved_local_ref_completes_once_and_replays(live_env, monkeypatch):
    rag_calls = _install_rag(monkeypatch)
    seen = []
    supervisor_calls = []

    def supervisor(body):
        supervisor_calls.append(body)
        return _tool_message("advise_nutrition", {
            "query": SAVED_LIVE_QUERY,
            "source_ref": {"start": 0, "end": 27, "exact_text": LIVE_FIRST_SENTENCE},
        }, call_id="call-saved-143")

    _install_provider(
        monkeypatch,
        understanding=_understanding(_task("nutrition_advice", LIVE_SOURCE)),
        supervisor=supervisor,
        answer=_answer_message(SAVED_LIVE_ANSWER),
        seen=seen,
    )
    harness = C3HTTPHarness(monkeypatch, live_env)
    before = business_keys(live_env)
    rid = "sim26-live01-original"
    status, payload = harness.post_json(LIVE_SOURCE, request_id=rid)

    assert status == 200
    assert payload["intent"] == "nutrition_advice"
    assert payload["answer"] == SAVED_LIVE_ANSWER
    assert payload["metrics"]["react_write"] == "completed"
    assert payload["metrics"]["receipt_refs"] == []
    assert payload["request_id"] == rid and payload["message_id"] and payload["run_id"]
    assert len(rag_calls) == 1, rag_calls
    assert len(supervisor_calls) == 1, len(supervisor_calls)
    assert len(seen) == 4, seen  # semantic + supervisor + conflict + read answer
    assert business_keys(live_env) == before
    snapshot = ledger_snapshot(live_env, rid)
    assert snapshot["request"]["status"] == "completed"
    assert snapshot["operations"] == []
    calls_after_first = len(seen)
    ids = (payload["request_id"], payload["message_id"], payload["run_id"])

    replay_status, replay = harness.post_json(LIVE_SOURCE, request_id=rid)
    assert replay_status == 200 and replay["replayed"] is True
    assert replay["intent"] == payload["intent"]
    assert replay["answer"] == payload["answer"]
    assert (replay["request_id"], replay["message_id"], replay["run_id"]) == ids
    assert replay["metrics"]["receipt_refs"] == []
    assert len(seen) == calls_after_first and len(rag_calls) == 1
    assert business_keys(live_env) == before

    sse_status, events, _ = harness.stream(LIVE_SOURCE, request_id=rid)
    assert sse_status == 200 and events
    final = events[-1]
    assert final["intent"] == payload["intent"]
    assert "".join(str(frame.get("content") or "") for frame in events) == payload["answer"]
    assert (final["request_id"], final["message_id"], final["run_id"]) == ids
    assert final["metrics"]["receipt_refs"] == []
    assert len(seen) == calls_after_first and len(rag_calls) == 1
    assert business_keys(live_env) == before


@pytest.mark.parametrize("case", [
    {
        "name": "training-qa",
        "source": "成年人每周做几次力量训练比较合适？只讲通用原则，不保存记录。",
        "first": "成年人每周做几次力量训练比较合适？",
        "task_type": "training_qa",
        "tool": "answer_training",
        "answer": "一般建议每周安排两到三次力量训练，并留出恢复时间。",
        "plan_info": None,
        "calls": 4,
    },
    {
        "name": "nutrition-advice",
        "source": "早餐和晚餐的蛋白质该怎么分配？请只讲通用原则，不保存饮食记录。",
        "first": "早餐和晚餐的蛋白质该怎么分配？",
        "task_type": "nutrition_advice",
        "tool": "advise_nutrition",
        "answer": "早晚餐各安排一个明确的蛋白质来源，全天总量平衡即可。",
        "plan_info": None,
        "calls": 4,
    },
    {
        "name": "nutrition-advice-full-ref",
        "source": "请用通用原则说明成年人早餐蛋白质怎么安排，不保存记录。",
        "first": None,
        "task_type": "nutrition_advice",
        "tool": "advise_nutrition",
        "answer": "早餐安排一个明确的蛋白质来源，并和全天总量平衡。",
        "plan_info": None,
        "calls": 4,
    },
    {
        "name": "training-plan",
        "source": "请给一个成年人通用的力量训练计划原则，不保存为长期计划。",
        "first": "请给一个成年人通用的力量训练计划原则",
        "task_type": "training_plan",
        "tool": "plan_training",
        "answer": "通用原则是循序渐进、覆盖主要动作模式并安排恢复。",
        "plan_info": {"scope": "generic", "scope_quote": "通用",
                      "facts": [], "phase_limits": []},
        "calls": 5,
    },
], ids=lambda item: item["name"])
def test_live01_local_ref_completes_training_advice_and_plan(live_env, monkeypatch, case):
    rag_calls = _install_rag(monkeypatch)
    seen = []
    supervisor_calls = []

    def supervisor(body):
        supervisor_calls.append(body)
        anchor = case["source"] if case["first"] is None else case["first"]
        return _tool_message(case["tool"], {
            "query": case["source"],
            "source_ref": ref_of(case["source"], anchor),
        }, call_id="call-" + case["name"])

    _install_provider(
        monkeypatch,
        understanding=_understanding(_task(case["task_type"], case["source"])),
        supervisor=supervisor,
        answer=_answer_message(case["answer"]),
        plan_info=case["plan_info"],
        seen=seen,
    )
    harness = C3HTTPHarness(monkeypatch, live_env)
    before = business_keys(live_env)
    status, payload = harness.post_json(case["source"], request_id="sim26-live01-" + case["name"])

    assert status == 200
    assert payload["intent"] == case["task_type"], payload
    assert payload["answer"] == case["answer"]
    assert payload["metrics"]["react_write"] == "completed"
    assert payload["metrics"]["receipt_refs"] == []
    assert len(rag_calls) == 1, rag_calls
    assert len(supervisor_calls) == 1, len(supervisor_calls)
    assert len(seen) == case["calls"], seen
    assert business_keys(live_env) == before


# ---------------------------------------------------------------------------
# Read reference identity gates (bad/empty/ambiguous refs never dispatch)
# ---------------------------------------------------------------------------


def _read_candidates(source):
    return [{
        "id": "goal-read", "task_type": "nutrition_advice", "action": "question",
        "text": source, "source_start": 0, "source_end": len(source), "depends_on": [],
    }]


def _run_read_batch(source, ref, *, candidates=None, no_progress=0, call_id="t-read"):
    state = batch_state(
        source,
        assistant(tool_calls=[tool_call(call_id, "advise_nutrition", {
            "query": "建议", "source_ref": ref})]),
        candidates=_read_candidates(source) if candidates is None else candidates,
        consecutive_no_progress=no_progress,
    )
    executor = RecordingExecutor({"advise_nutrition": {
        "status": "completed", "summary": "不应被调用"}})
    result = asyncio.run(execute_react_batch(state, _deps(
        ScriptedLLM(understanding={"tasks": []}, supervisor={}), executor)))
    return result, executor


@pytest.mark.parametrize("ref,reason", [
    ({"start": 0, "end": 44, "exact_text": LIVE_SOURCE}, "exact_text_mismatch"),
    ({"start": 0, "end": 27, "exact_text": "伪造的原文"}, "exact_text_mismatch"),
    ({"start": 0, "end": 999, "exact_text": LIVE_SOURCE}, "source_ref_out_of_bounds"),
    ({"start": 0, "end": 0, "exact_text": ""}, "empty_read_source_ref"),
], ids=["saved-149-length-mismatch", "forged-text", "out-of-bounds", "empty"])
def test_live01_bad_read_ref_zero_handler(ref, reason):
    result, executor = _run_read_batch(LIVE_SOURCE, ref)
    assert executor.calls == [], executor.names
    assert result["covered_goals"] == [] and result["committed_tasks"] == []
    assert result["observations"][0]["status"] == "rejected"
    assert result["observations"][0]["error"] == reason
    assert result["bound_tasks"] == [] and result["unbound_write_calls"] == []


def test_live01_bool_read_ref_is_protocol_error_before_handler():
    ref = {"start": False, "end": 27, "exact_text": LIVE_FIRST_SENTENCE}
    result, executor = _run_read_batch(LIVE_SOURCE, ref)
    assert executor.calls == []
    assert result["status"] == "restricted"
    assert result["termination_reason"] == "protocol_error"
    assert result.get("covered_goals", []) == []


def test_live01_ambiguous_full_ref_over_two_independent_read_goals_zero_handler():
    source = "今天午餐吃什么好？晚餐吃什么好？"
    first = "今天午餐吃什么好？"
    candidates = [
        {"id": "goal-1", "task_type": "nutrition_advice", "action": "question",
         "text": first, "source_start": 0, "source_end": len(first), "depends_on": []},
        {"id": "goal-2", "task_type": "nutrition_advice", "action": "question",
         "text": source[len(first):], "source_start": len(first),
         "source_end": len(source), "depends_on": []},
    ]
    result, executor = _run_read_batch(
        source, {"start": 0, "end": len(source), "exact_text": source},
        candidates=candidates)
    assert executor.calls == []
    assert result["covered_goals"] == []
    assert result["observations"][0]["status"] == "needs_clarification"
    assert result["observations"][0]["error"] == "ambiguous_goal"


def test_live01_valid_ref_without_matching_read_goal_zero_handler():
    source = "你好，怎么练卧推"
    candidate = "怎么练卧推"
    candidates = [{
        "id": "goal-qa", "task_type": "training_qa", "action": "question",
        "text": candidate, "source_start": source.index(candidate),
        "source_end": source.index(candidate) + len(candidate), "depends_on": [],
    }]
    result, executor = _run_read_batch(
        source, {"start": 0, "end": 2, "exact_text": "你好"}, candidates=candidates)
    assert executor.calls == []
    assert result["covered_goals"] == []
    assert result["observations"][0]["status"] == "needs_clarification"
    assert result["observations"][0]["error"] == "no_matching_goal"


def test_live01_bad_read_ref_preserves_no_progress_for_the_correction_turn():
    """A copyable bad ref is not progress, but it must not end the turn.

    The current per-task contract keeps the real counter (no reset) and leaves
    the next bounded supervisor turn available to copy the app-owned refs;
    every tool step/turn budget still counts. A non-copyable binding failure
    still increments to the bound in the next test.
    """
    result, executor = _run_read_batch(
        LIVE_SOURCE,
        {"start": 0, "end": 44, "exact_text": LIVE_SOURCE},
        no_progress=1)
    assert executor.calls == []
    assert result["consecutive_no_progress"] == 1
    assert result["terminated"] is False and result["termination_reason"] == ""
    assert result["status"] == "running"


def test_live01_nonbinding_read_goal_still_increments_to_the_bound():
    """The opposite case: a non-copyable binding failure is real no-progress."""
    source = "你好，怎么练卧推"
    candidate = "怎么练卧推"
    candidates = [{
        "id": "goal-qa", "task_type": "training_qa", "action": "question",
        "text": candidate, "source_start": source.index(candidate),
        "source_end": source.index(candidate) + len(candidate), "depends_on": [],
    }]
    result, executor = _run_read_batch(
        source, {"start": 0, "end": 2, "exact_text": "你好"},
        candidates=candidates, no_progress=1)
    assert executor.calls == []
    assert result["observations"][0]["status"] == "needs_clarification"
    assert result["observations"][0]["error"] == "no_matching_goal"
    assert result["consecutive_no_progress"] == 2
    assert result["termination_reason"] == "no_progress"
    assert result["status"] == "limited"


def test_live01_auxiliary_reads_without_goals_still_dispatch():
    source = "读取我的档案、训练历史、营养摘要和当前计划。"
    state = batch_state(
        source,
        assistant(tool_calls=[
            tool_call("t1", "read_profile", {}),
            tool_call("t2", "read_workout_history", {"days": 30}),
            tool_call("t3", "read_nutrition_summary", {"days": 7}),
            tool_call("t4", "read_current_plan", {}),
        ]),
        candidates=[],
    )
    executor = RecordingExecutor({
        name: {"status": "completed", "result": {"source": name}}
        for name in ("read_profile", "read_workout_history",
                     "read_nutrition_summary", "read_current_plan")
    })
    result = asyncio.run(execute_react_batch(state, _deps(
        ScriptedLLM(understanding={"tasks": []}, supervisor={}), executor)))
    assert executor.names == ["read_profile", "read_workout_history",
                              "read_nutrition_summary", "read_current_plan"]
    assert [item["status"] for item in result["observations"]] == ["completed"] * 4
    assert result["covered_goals"] == []
    assert result["unbound_write_calls"] == [] and result["bound_tasks"] == []


# ---------------------------------------------------------------------------
# Same-type read goals stay distinct and completed results are reused
# ---------------------------------------------------------------------------


def test_live01_two_read_goals_do_not_share_results_and_repeat_reuses():
    source = "今天午餐吃什么好？晚餐吃什么好？"
    first = "今天午餐吃什么好？"
    candidates = [
        {"id": "read-1", "task_type": "nutrition_advice", "action": "question",
         "text": first, "source_start": 0, "source_end": len(first), "depends_on": []},
        {"id": "read-2", "task_type": "nutrition_advice", "action": "question",
         "text": source[len(first):], "source_start": len(first),
         "source_end": len(source), "depends_on": []},
    ]
    ref1 = {"start": 0, "end": 4, "exact_text": source[0:4]}
    ref2 = {"start": len(first), "end": len(first) + 3,
            "exact_text": source[len(first):len(first) + 3]}
    executor = RecordingExecutor({"advise_nutrition": lambda nth, name, args, ctx: {
        "status": "completed",
        "summary": f"建议-{nth}",
        "result": {"answer": f"建议-{nth}"},
    }})
    state = batch_state(
        source,
        assistant(tool_calls=[
            tool_call("t1", "advise_nutrition", {"query": first, "source_ref": ref1}),
            tool_call("t2", "advise_nutrition", {
                "query": source[len(first):], "source_ref": ref2}),
        ]),
        candidates=candidates,
    )
    first_result = asyncio.run(execute_react_batch(state, _deps(
        ScriptedLLM(understanding={"tasks": []}, supervisor={}), executor)))
    state.update(first_result)
    assert executor.names == ["advise_nutrition", "advise_nutrition"]
    assert state["covered_goals"] == ["read-1", "read-2"]
    assert state["goal_results"]["read-1"]["summary"] == "建议-0"
    assert state["goal_results"]["read-2"]["summary"] == "建议-1"
    assert state["goal_results"]["read-1"]["summary"] != state["goal_results"]["read-2"]["summary"]

    state["messages"].append(assistant(tool_calls=[
        tool_call("t3", "advise_nutrition", {"query": first, "source_ref": ref1})]))
    repeat = asyncio.run(execute_react_batch(state, _deps(
        ScriptedLLM(understanding={"tasks": []}, supervisor={}), executor)))
    state.update(repeat)
    assert len(executor.calls) == 2, executor.names
    assert repeat["observations"][-1]["summary"] == "建议-0"
    assert state["covered_goals"] == ["read-1", "read-2"]
    assert state["goal_results"]["read-2"]["summary"] == "建议-1"


def _dependency_candidates(source):
    write_text = "帮我记录早餐吃了燕麦粥"
    advice_text = "然后根据早餐给我营养建议"
    write_start = source.index(write_text)
    advice_start = source.index(advice_text)
    return [
        {"id": "record-goal", "task_type": "nutrition_record", "action": "record",
         "text": write_text, "source_start": write_start,
         "source_end": write_start + len(write_text), "depends_on": []},
        {"id": "advice-goal", "task_type": "nutrition_advice", "action": "question",
         "text": advice_text, "source_start": advice_start,
         "source_end": advice_start + len(advice_text), "depends_on": ["record-goal"]},
    ], write_text, advice_text


def test_live01_write_read_dependency_blocks_until_committed_then_resumes_once():
    source = "帮我记录早餐吃了燕麦粥。然后根据早餐给我营养建议"
    candidates, write_text, advice_text = _dependency_candidates(source)
    write_ref = ref_of(source, write_text)
    advice_ref = {"start": source.index(advice_text), "end": source.index(advice_text) + 4,
                  "exact_text": source[source.index(advice_text):source.index(advice_text) + 4]}
    executor = RecordingExecutor({
        "record_nutrition": committed_receipt("rcpt-live01"),
        "advise_nutrition": {"status": "completed", "summary": "基于已提交早餐的建议"},
    })
    llm = ScriptedLLM(understanding={"tasks": []}, supervisor={})
    deps = _deps(llm, executor)
    state = batch_state(
        source,
        assistant(tool_calls=[tool_call("a1", "advise_nutrition", {
            "query": advice_text, "source_ref": advice_ref})]),
        candidates=candidates,
    )
    blocked = asyncio.run(execute_react_batch(state, deps))
    state.update(blocked)
    assert executor.calls == [], executor.names
    assert state["pending_goals"] and state["pending_goals"][0]["goal_ids"] == ["advice-goal"]
    assert state["covered_goals"] == []

    state["messages"].append(assistant(tool_calls=[
        tool_call("w1", "record_nutrition", {"source_ref": write_ref})]))
    committed = asyncio.run(execute_react_batch(state, deps))
    state.update(committed)
    assert executor.names == ["record_nutrition", "advise_nutrition"], executor.names
    assert state["covered_goals"] == ["record-goal", "advice-goal"]
    assert state["committed_tasks"] == ["record-goal"]
    advice_call = [call for call in executor.calls if call["name"] == "advise_nutrition"][0]
    assert advice_call["context"]["committed_tasks"] == ["record-goal"]
    assert state["pending_goals"] == []

    state["messages"].append(assistant(tool_calls=[
        tool_call("a2", "advise_nutrition", {"query": advice_text, "source_ref": advice_ref})]))
    repeat = asyncio.run(execute_react_batch(state, deps))
    state.update(repeat)
    assert len(executor.calls) == 2, executor.names
    assert repeat["observations"][-1]["summary"] == "基于已提交早餐的建议"


def test_live01_uncommitted_write_does_not_generate_advice():
    source = "帮我记录早餐吃了燕麦粥。然后根据早餐给我营养建议"
    candidates, write_text, advice_text = _dependency_candidates(source)
    write_ref = ref_of(source, write_text)
    advice_start = source.index(advice_text)
    advice_ref = {"start": advice_start, "end": advice_start + 4,
                  "exact_text": source[advice_start:advice_start + 4]}
    executor = RecordingExecutor({
        "record_nutrition": {"status": "completed", "commit_state": "unknown"},
        "advise_nutrition": {"status": "completed", "summary": "不应基于未知结果生成"},
    })
    deps = _deps(ScriptedLLM(understanding={"tasks": []}, supervisor={}), executor)
    state = batch_state(
        source,
        assistant(tool_calls=[tool_call("a1", "advise_nutrition", {
            "query": advice_text, "source_ref": advice_ref})]),
        candidates=candidates,
    )
    state.update(asyncio.run(execute_react_batch(state, deps)))
    state["messages"].append(assistant(tool_calls=[
        tool_call("w1", "record_nutrition", {"source_ref": write_ref})]))
    state.update(asyncio.run(execute_react_batch(state, deps)))
    assert "advise_nutrition" not in executor.names, executor.names
    assert state["committed_tasks"] == []
    assert state["pending_goals"] and state["pending_goals"][0]["goal_ids"] == ["advice-goal"]


# ---------------------------------------------------------------------------
# Truthful limited payload and replay
# ---------------------------------------------------------------------------


def _runtime_without_io():
    from app.services.agent.react_write_runtime import WriteRuntimeSession

    runtime = WriteRuntimeSession.__new__(WriteRuntimeSession)
    runtime.ai_estimate_committed = False
    runtime.termination_reason = "tool_step_limit"
    runtime.receipts = []
    runtime.committed_observations = []
    runtime.resume_goal_results = {}
    runtime.resume_covered = []
    runtime.frozen_parameters = {}
    runtime.response_ids = {}
    runtime._response_ids = lambda: {"request_id": "r", "message_id": "m", "run_id": "u"}
    runtime._citations = lambda state: []
    return runtime


def test_live01_limited_without_commit_never_claims_saved():
    runtime = _runtime_without_io()
    result = runtime._limited_payload({
        "observations": [{"status": "unavailable", "error": "insufficient_evidence"}],
        "goal_results": {},
    })
    assert result["intent"] == "execution_limited"
    assert result["react_write_status"] == "limited"
    assert result["receipt_refs"] == []
    assert "已保存" not in result["final_answer"]
    assert "已完成的记录已保存" not in result["final_answer"]


def test_live01_limited_unknown_commit_is_unconfirmed_not_saved():
    runtime = _runtime_without_io()
    result = runtime._limited_payload({"observations": [{
        "status": "completed", "commit_state": "unknown", "receipt_refs": [],
    }]})
    assert result["receipt_refs"] == []
    assert "尚未确认" in result["final_answer"]
    assert "已保存" not in result["final_answer"]


def test_live01_limited_keeps_real_partial_commit_receipts():
    runtime = _runtime_without_io()
    runtime.receipts = ["rcpt-real"]
    runtime.committed_observations = [{
        "status": "completed", "commit_state": "committed",
        "result": {"proof_type": "nutrition_db_v1"}, "receipt_refs": ["rcpt-real"],
    }]
    result = runtime._limited_payload({})
    assert result["intent"] == "execution_limited"
    assert result["react_write_status"] == "limited"
    assert result["receipt_refs"] == ["rcpt-real"]
    assert "已保存" in result["final_answer"]


def test_live01_limited_replay_is_stable_and_zero_new_calls(live_env, monkeypatch):
    source = "我想知道成年人早餐蛋白质怎么分配？只讲通用原则，不保存记录。"
    rag_calls = _install_rag(monkeypatch)
    seen = []
    supervisor_calls = []
    invalid_ref = {"start": 0, "end": len(source) - 1, "exact_text": source}

    def supervisor(body):
        supervisor_calls.append(body)
        return _tool_message("advise_nutrition", {
            "query": source, "source_ref": invalid_ref,
        }, call_id=f"call-invalid-{len(supervisor_calls)}")

    _install_provider(
        monkeypatch,
        understanding=_understanding(_task("nutrition_advice", source)),
        supervisor=supervisor,
        answer=_answer_message("不应调用回答模型"),
        seen=seen,
    )
    harness = C3HTTPHarness(monkeypatch, live_env)
    before = business_keys(live_env)
    rid = "sim26-live01-limited"
    status, payload = harness.post_json(source, request_id=rid)

    assert status == 200
    assert payload["intent"] == "execution_limited"
    assert payload["metrics"]["react_write"] == "limited"
    assert payload["metrics"]["receipt_refs"] == []
    assert "已保存" not in payload["answer"]
    # A copyable bad read ref keeps the next bounded correction turn available
    # (react_graph's per-task invalid_read_source_ref contract), so the real
    # bounded stop is the supervisor-turn limit, not a no-progress reset. All
    # four turns still see the same invalid ref and dispatch zero handlers.
    assert len(supervisor_calls) == 4, len(supervisor_calls)
    assert len(seen) == 5, seen
    assert rag_calls == []
    assert business_keys(live_env) == before
    ledger = ledger_snapshot(live_env, rid)
    assert ledger["request"]["status"] == "limited"
    stored = ledger["request"]["response"]
    if isinstance(stored, str):
        stored = json.loads(stored)
    assert stored["metrics"]["mode_reason"] == "supervisor_turn_limit"
    calls_after_first = len(seen)
    ids = (payload["request_id"], payload["message_id"], payload["run_id"])

    replay_status, replay = harness.post_json(source, request_id=rid)
    assert replay_status == 200 and replay["replayed"] is True
    assert replay["intent"] == payload["intent"]
    assert replay["answer"] == payload["answer"]
    assert (replay["request_id"], replay["message_id"], replay["run_id"]) == ids
    assert replay["metrics"]["receipt_refs"] == []
    assert len(seen) == calls_after_first and rag_calls == []
    assert business_keys(live_env) == before

    sse_status, events, _ = harness.stream(source, request_id=rid)
    assert sse_status == 200 and events
    final = events[-1]
    assert final["intent"] == payload["intent"]
    assert "".join(str(frame.get("content") or "") for frame in events) == payload["answer"]
    assert (final["request_id"], final["message_id"], final["run_id"]) == ids
    assert final["metrics"]["receipt_refs"] == []
    assert len(seen) == calls_after_first and rag_calls == []
    assert business_keys(live_env) == before


def test_live01_real_write_commit_resumes_blocked_read_once(live_env, monkeypatch):
    source = ("帮我记录今天午餐：牛肉面一碗，热量520千卡，蛋白质25克，碳水60克，脂肪15克，"
              "然后根据记录给我饮食建议。")
    fact = "牛肉面一碗，热量520千卡，蛋白质25克，碳水60克，脂肪15克"
    record_end = source.index(fact) + len(fact)
    advice_text = source[record_end:]
    advice_start = source.index(advice_text)
    advice_ref = {"start": advice_start, "end": advice_start + 4,
                  "exact_text": source[advice_start:advice_start + 4]}
    advice_answer = "基于已提交的午餐记录，建议晚餐补充蔬菜和蛋白质。"
    rag_calls = _install_rag(monkeypatch)
    seen = []
    supervisor_calls = []

    def supervisor(body):
        supervisor_calls.append(body)
        if len(supervisor_calls) == 1:
            return _tool_message("advise_nutrition", {
                "query": advice_text, "source_ref": advice_ref,
            }, call_id="call-blocked-advice")
        if len(supervisor_calls) == 2:
            return _tool_message("record_nutrition", {
                "source_ref": ref_of(source, fact)}, call_id="call-real-write")
        return _answer_message("额外回答")

    _install_provider(
        monkeypatch,
        understanding=_understanding(
            _task("nutrition_record", source[:record_end], action="record", fact_text=fact),
            _task("nutrition_advice", advice_text),
        ),
        supervisor=supervisor,
        answer=_answer_message(advice_answer),
        seen=seen,
    )
    harness = C3HTTPHarness(monkeypatch, live_env)
    before = business_keys(live_env)
    rid = "sim26-live01-dependency"
    status, payload = harness.post_json(source, request_id=rid)

    assert status == 200
    assert payload["metrics"]["react_write"] == "completed"
    receipts = list(payload["metrics"]["receipt_refs"])
    assert receipts, payload["metrics"]
    assert len(supervisor_calls) == 2, supervisor_calls
    assert len(rag_calls) == 1, rag_calls
    assert advice_answer in payload["answer"], payload["answer"]
    after = business_keys(live_env)
    assert len(after["nutrition_logs"]) == len(before["nutrition_logs"]) + 1
    assert after["user_food_log"] == before["user_food_log"]
    events = ledger_snapshot(live_env, rid)["events"]
    advice_events = []
    for event in events:
        observation = event.get("observation")
        if isinstance(observation, str):
            try:
                observation = json.loads(observation)
            except json.JSONDecodeError:
                observation = {}
        if str(event.get("tool_call_id")) == "call-blocked-advice" \
                and isinstance(observation, dict) \
                and observation.get("status") == "completed":
            advice_events.append(event)
    assert len(advice_events) == 1, [event.get("phase") for event in events]
    ids = (payload["request_id"], payload["message_id"], payload["run_id"])

    replay_status, replay = harness.post_json(source, request_id=rid)
    assert replay_status == 200 and replay["replayed"] is True
    assert replay["intent"] == payload["intent"]
    assert replay["answer"] == payload["answer"]
    assert replay["metrics"]["receipt_refs"] == receipts
    assert (replay["request_id"], replay["message_id"], replay["run_id"]) == ids
    assert len(seen) == 5, seen
    assert business_keys(live_env) == after


def test_live01_partial_commit_limited_keeps_receipt_and_replays(live_env, monkeypatch):
    source = ("帮我记录今天午餐：牛肉面一碗，热量520千卡，蛋白质25克，碳水60克，脂肪15克，"
              "然后根据记录给我饮食建议。")
    fact = "牛肉面一碗，热量520千卡，蛋白质25克，碳水60克，脂肪15克"
    record_end = source.index(fact) + len(fact)
    advice_text = source[record_end:]
    invalid_ref = {"start": 0, "end": len(source) - 1, "exact_text": source}
    seen = []
    supervisor_calls = []

    def supervisor(body):
        supervisor_calls.append(body)
        if len(supervisor_calls) == 1:
            return _tool_message("record_nutrition", {
                "source_ref": ref_of(source, fact)}, call_id="call-real-write")
        return _tool_message("advise_nutrition", {
            "query": advice_text, "source_ref": invalid_ref,
        }, call_id=f"call-invalid-{len(supervisor_calls)}")

    _install_provider(
        monkeypatch,
        understanding=_understanding(
            _task("nutrition_record", source[:record_end], action="record", fact_text=fact),
            _task("nutrition_advice", advice_text),
        ),
        supervisor=supervisor,
        answer=_answer_message("不应调用回答模型"),
        seen=seen,
    )
    harness = C3HTTPHarness(monkeypatch, live_env)
    before = business_keys(live_env)
    rid = "sim26-live01-partial"
    status, payload = harness.post_json(source, request_id=rid)

    assert status == 200
    assert payload["intent"] == "execution_limited"
    assert payload["metrics"]["react_write"] == "limited"
    receipts = list(payload["metrics"]["receipt_refs"])
    assert payload["metrics"]["receipt_refs"] == receipts
    assert receipts, payload["metrics"]
    assert "已保存" in payload["answer"], payload["answer"]
    # One real committed write followed by three bounded invalid-ref correction
    # turns (same per-task contract as the limited replay above).
    assert len(supervisor_calls) == 4, supervisor_calls
    assert len(seen) == 5, seen
    after = business_keys(live_env)
    assert len(after["nutrition_logs"]) == len(before["nutrition_logs"]) + 1
    assert after["user_food_log"] == before["user_food_log"]
    ledger = ledger_snapshot(live_env, rid)
    assert ledger["request"]["status"] == "limited"
    stored = ledger["request"]["response"]
    if isinstance(stored, str):
        stored = json.loads(stored)
    assert stored["metrics"]["mode_reason"] == "supervisor_turn_limit"
    ids = (payload["request_id"], payload["message_id"], payload["run_id"])

    replay_status, replay = harness.post_json(source, request_id=rid)
    assert replay_status == 200 and replay["replayed"] is True
    assert replay["intent"] == payload["intent"]
    assert replay["answer"] == payload["answer"]
    assert replay["metrics"]["receipt_refs"] == receipts
    assert (replay["request_id"], replay["message_id"], replay["run_id"]) == ids
    assert len(seen) == 5
    assert business_keys(live_env) == after
