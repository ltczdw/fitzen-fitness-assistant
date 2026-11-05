"""Tests for react read answer delivery."""
from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.services.agent.react_graph import (
    ReactDependencies,
    URGENT_SAFETY_ANSWER,
    _extract_final_answer,
    completed_read_goals,
    finalize_node,
    invoke_react,
)
from tests.react_supervisor_fixtures import (
    FakeLLM,
    RecordingExecutor,
    assistant,
    batch_state,
    ref_of,
    tool_call,
)
from tests.react_phase_c2_3_support import (
    API_OWNER,
    C3HTTPHarness,
    C3SQLFault,
    business_keys,
    ledger_snapshot,
    mock_provider_transport,
    synthetic_evidence_conflict_response,
)


# ---------------------------------------------------------------------------
# Saved live answers (calls 155/159 and 166/170 from the 2026-10-26 run)
# ---------------------------------------------------------------------------

SAVED_TRAINING_RECOVERY_ANSWER = (
    "力量训练后的恢复，核心是给同一肌群留出足够间隔、保证睡眠和蛋白质摄入，并循序渐进地增加负荷。"
    "一般建议同一肌群的力量训练间隔至少48小时，每周安排2天以上、覆盖腿、髋、背、腹、胸、肩和手臂等主要肌群；"
    "训练时每个动作做8至12次为一组、至少2组，3组可获得更多益处，但不必每次都练到力竭。\n\n"
    "睡眠是最容易被低估的恢复手段，成年人尽量保证每晚7至9小时规律睡眠，训练日避免熬夜。"
    "饮食上把蛋白质均匀分配到各餐，训练后正常吃一餐含蛋白质和碳水的主食即可，不需要刻意追求“窗口期”。"
    "训练后轻度活动、拉伸和散步有助于缓解僵硬感，但不要用高强度有氧替代休息。\n\n"
    "负荷安排上应在数周内逐步增加，而不是突然加重量或加组数；如果连续几天同一部位仍有明显酸痛、乏力，"
    "说明恢复不足，应降低强度或延长间隔。若出现明显肿胀、剧烈疼痛、麻木、关节变形或活动受限，"
    "应停止训练并寻求专业帮助，不要自行判断或继续硬练。以上为通用原则，具体重量和频率需结合个人情况调整。"
)
SAVED_PROTEIN_DISTRIBUTION_ANSWER = (
    "蛋白质分散到三餐，核心是每餐都安排一个明确的蛋白质来源，而不是集中在晚餐。"
    "一般成年人可按每餐约20至30克蛋白质来大致分配（估算值，具体需按体重和活动量调整），"
    "早餐最容易缺口，可优先补上。\n\n"
    "具体做法：早餐加鸡蛋、牛奶或无糖酸奶、豆浆；午餐和晚餐各安排一份手掌大小的蛋白来源，"
    "如鱼、禽肉、瘦肉、豆腐或豆类。来源尽量轮换，肉、鱼、蛋、豆类都算，不必只依赖一种。"
    "红肉和加工肉不宜过多，能少吃时选瘦肉并控制频次。\n\n"
    "如果某一餐蛋白质偏少，不必强求每餐完全均衡，可在一天内整体补足，比如加餐用酸奶或豆制品。"
    "素食者更要注意每餐都有豆类、豆制品、蛋奶等来源，以覆盖不同的氨基酸。"
)
SAVED_BREAKFAST_PROTEIN_ANSWER = (
    "成年人早餐的蛋白质搭配，通用原则是选两到三种不同来源，不要只靠一种。"
    "肉、鱼、鸡蛋和豆类都算蛋白质来源，豆类、豌豆、扁豆脂肪低，还提供纤维，可以替代部分肉类。\n\n"
    "具体可以这样组合：鸡蛋加豆类或豆制品，比如鸡蛋配豆浆、豆腐；或者奶类加全谷物，"
    "再配一小把坚果或种子。如果吃肉类，尽量选瘦肉，少用培根、火腿、香肠这类加工肉。\n\n"
    "早餐不必每餐都追求完美，按一天甚至一周整体平衡即可，尽量选低脂、低盐、低糖的做法。"
    "以上是通用原则，不涉及你的个人数据。"
)
SAVED_POST_TRAINING_CARB_ANSWER = (
    "力量训练后的碳水安排，通用原则是：把碳水放在训练后这一餐，并优先选全谷物或薯类等淀粉类主食。"
    "NHS 的均衡饮食建议是淀粉类食物（土豆、面包、米饭、意面、谷物）应占日常食物的三分之一略多，"
    "尽量选全谷物版本，土豆带皮吃可增加纤维。训练后这一餐可以按这个比例来配，主食不必刻意回避。\n\n"
    "搭配上，碳水最好和蛋白质一起吃。NHS 建议餐中加入豆类、鱼、鸡蛋、瘦肉等蛋白质来源；"
    "如果是增重需求，可尝试每天增加约300至500千卡（估算值，来自 NHS 增重建议），并配合适合的力量训练。"
    "训练后一餐把主食和一份蛋白质食物组合，是比较省心的做法。\n\n"
    "时间上不必卡得太死，训练后几小时内吃一顿含碳水和蛋白质的正餐即可。"
    "力量训练本身建议每周至少2天覆盖主要肌群，碳水安排跟着训练日走就行。"
    "具体份量因人而异，需结合体重、训练量和总能量目标调整，以上不构成个体化定量方案。"
)

SAVED_TWO_READ_CASES = [
    {
        "id": "native_two_read_types",
        "source": (
            "请先讲一般成年人力量训练后如何恢复，再说明日常蛋白质怎么分散到三餐。"
            "两个问题都只讲通用原则，不保存训练、饮食或长期记忆。"
        ),
        "tasks": [
            ("training_qa", "请先讲一般成年人力量训练后如何恢复"),
            ("nutrition_advice",
             "再说明日常蛋白质怎么分散到三餐。两个问题都只讲通用原则，不保存训练、饮食或长期记忆。"),
        ],
        "refs": ["一般成年人力量训练后如何恢复", "日常蛋白质怎么分散到三餐"],
        "tools": ["answer_training", "advise_nutrition"],
        "answers": [SAVED_TRAINING_RECOVERY_ANSWER, SAVED_PROTEIN_DISTRIBUTION_ANSWER],
    },
    {
        "id": "native_two_nutrition_questions",
        "source": (
            "两个营养问题请分别回答：成年人早餐蛋白质怎样搭配？力量训练后的碳水怎么安排？"
            "只要通用原则，不保存任何记录或长期记忆。"
        ),
        "tasks": [
            ("nutrition_advice", "成年人早餐蛋白质怎样搭配？"),
            ("nutrition_advice",
             "力量训练后的碳水怎么安排？只要通用原则，不保存任何记录或长期记忆。"),
        ],
        "refs": ["成年人早餐蛋白质怎样搭配？", "力量训练后的碳水怎么安排？"],
        "tools": ["advise_nutrition", "advise_nutrition"],
        "answers": [SAVED_BREAKFAST_PROTEIN_ANSWER, SAVED_POST_TRAINING_CARB_ANSWER],
    },
]

MIXED_SOURCE = (
    "帮我记录今天午餐：牛肉面一碗，热量520千卡，蛋白质25克，碳水60克，脂肪15克。"
    "然后讲一下训练后如何恢复，再说蛋白质怎么分散到三餐。"
)
MIXED_FACT = "牛肉面一碗，热量520千卡，蛋白质25克，碳水60克，脂肪15克"
MIXED_TASKS = [
    ("nutrition_record", "帮我记录今天午餐：牛肉面一碗，热量520千卡，蛋白质25克，碳水60克，脂肪15克",
     MIXED_FACT),
    ("training_qa", "然后讲一下训练后如何恢复", None),
    ("nutrition_advice", "再说蛋白质怎么分散到三餐。", None),
]
MIXED_READ_ANSWERS = [
    "训练后的恢复重点是睡眠、同一肌群间隔和循序渐进的负荷；训练后正常吃一餐即可。",
    "蛋白质分散到三餐，每餐安排一个明确来源，全天总量平衡；早餐最容易缺口。",
]

SUPERVISOR_TOOLS = {
    "record_workout", "record_nutrition", "answer_training", "advise_nutrition",
    "plan_training", "read_profile", "read_workout_history",
    "read_nutrition_summary", "read_current_plan",
}


# ---------------------------------------------------------------------------
# Fixtures and provider/RAG boundaries
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_read_env(tmp_path, monkeypatch):
    from tests.react_phase_c1_support import isolated
    from tests.react_phase_c2_2_support import ensure_catalog

    with isolated(monkeypatch, tmp_path / "sim26-multi-read", "sqlite") as engine:
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


def _task(task_type, task_text, *, action="question", fact_text=None, history_requirement="none"):
    return {
        "task_type": task_type, "action": action, "text": task_text,
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
                    "safety": {"level": "normal", "evidence": "", "categories": [],
                               "reason": "scripted normal"},
                    "current_is_critical": False, "critical_fact_fields": [],
                    "critical_history_ids": [], "dialogue_ids": [], "references": [],
                    "memory_requests": [], "needs_clarification": False,
                }, ensure_ascii=False),
            },
        }],
    })


def _batch_message(calls, *, content=""):
    return _provider_message({
        "role": "assistant", "content": content, "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(arguments, ensure_ascii=False)},
        } for call_id, name, arguments in calls],
    })


def _answer_message(content):
    return _provider_message({"role": "assistant", "content": content})


def _hit(title, content):
    return SimpleNamespace(
        title=title, source="knowledge://sim26-multi-read", chunk_id="chunk-" + title,
        content=content, score=1.0,
        metadata={"rerank_status": "model", "rerank_score": 1.0, "page": 1})


def _install_rag(monkeypatch):
    import app.services.rag.online as rag_online

    calls = []

    async def search(pipeline, query, **kwargs):
        calls.append(str(query))
        return [
            _hit("训练恢复", "训练后注意睡眠、恢复和循序渐进，正常吃一餐即可。"),
            _hit("蛋白质分配", "每餐安排一个明确的蛋白质来源，全天总量平衡。"),
            _hit("训练计划", "训练计划应循序渐进并覆盖主要动作模式。"),
        ]

    monkeypatch.setattr(rag_online, "search", search, raising=False)
    return calls


def _install_provider(monkeypatch, *, understanding, supervisor, answers, seen=None,
                      supervisor_calls=None):
    import httpx

    state = {"answer": 0}

    def handler(request):
        body = json.loads(request.content.decode("utf-8"))
        tools = {item.get("function", {}).get("name")
                 for item in body.get("tools") or []}
        if seen is not None:
            seen.append({"tools": sorted(tools)})
        if "return_request_understanding" in tools:
            return httpx.Response(200, json=understanding)
        if "return_evidence_conflict_judgement" in tools:
            return synthetic_evidence_conflict_response(request)
        if tools & SUPERVISOR_TOOLS:
            if supervisor_calls is not None:
                supervisor_calls.append(body)
            return httpx.Response(200, json=supervisor)
        index = state["answer"]
        state["answer"] += 1
        if index >= len(answers):
            raise AssertionError("unexpected extra summary/answer model call")
        return httpx.Response(200, json=_answer_message(answers[index]))

    return mock_provider_transport(monkeypatch, [handler])


def _count_read_handlers(monkeypatch):
    """Wrap the real answer workers with a pure call counter (no behaviour change)."""
    from app.services.agent import react_readonly_workers

    counts = {"total": 0}
    for name in ("answer_training", "advise_nutrition", "plan_training"):
        original = react_readonly_workers.ANSWER_WORKERS[name]

        async def wrapper(args, context, _original=original):
            counts["total"] += 1
            value = _original(args, context)
            if inspect.isawaitable(value):
                value = await value
            return value

        monkeypatch.setitem(react_readonly_workers.ANSWER_WORKERS, name, wrapper)
    return counts


def _nine_table_counts(engine):
    tables = ("workout_records", "workout_sessions", "workout_session_exercises",
              "workout_session_sets", "user_food_log", "nutrition_logs",
              "training_plans", "user_profiles", "user_facts")
    with engine.connect() as conn:
        return {table: conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                for table in tables}


def _expected_two_answers(case):
    return case["answers"][0] + "\n\n" + case["answers"][1]


def _saved_batch(case):
    return _batch_message([
        (f"call-{index}", tool, {"query": ref,
                                 "source_ref": ref_of(case["source"], ref)})
        for index, (tool, ref) in enumerate(zip(case["tools"], case["refs"]))
    ])


def _assert_answers_in_request_order(answer, answers):
    positions = [answer.find(item) for item in answers]
    assert all(position >= 0 for position in positions), answer
    assert positions == sorted(positions), positions
    for item in answers:
        assert answer.count(item) == 1, (item, answer)


# ---------------------------------------------------------------------------
# Saved live cases through the real ASGI/graph/worker/gateway/SQL chain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", SAVED_TWO_READ_CASES, ids=lambda item: item["id"])
@pytest.mark.parametrize("first_transport", ["json", "sse"])
def test_saved_two_read_answers_complete_once_in_request_order(
        multi_read_env, monkeypatch, case, first_transport):
    rag_calls = _install_rag(monkeypatch)
    seen = []
    supervisor_calls = []
    _install_provider(
        monkeypatch,
        understanding=_understanding(*[
            _task(task_type, task_text) for task_type, task_text in case["tasks"]
        ]),
        supervisor=_saved_batch(case),
        answers=case["answers"],
        seen=seen,
        supervisor_calls=supervisor_calls,
    )
    harness = C3HTTPHarness(monkeypatch, multi_read_env)
    handler_counts = _count_read_handlers(monkeypatch)
    before_business = business_keys(multi_read_env)
    expected = _expected_two_answers(case)
    rid = f"sim26-live02-{case['id']}-{first_transport}"

    if first_transport == "json":
        status, first = harness.post_json(case["source"], request_id=rid)
        first_answer = first["answer"]
    else:
        status, events, _ = harness.stream(case["source"], request_id=rid)
        first = events[-1]
        first_answer = "".join(str(frame.get("content") or "") for frame in events)

    assert status == 200
    assert first_answer == expected
    _assert_answers_in_request_order(first_answer, case["answers"])
    assert first["metrics"]["react_write"] == "completed"
    assert first["metrics"]["receipt_refs"] == []
    ids = (first["request_id"], first["message_id"], first["run_id"])
    assert ids[0] == rid and all(ids)
    assert len(seen) == 6, seen  # understanding + supervisor + 2*(conflict + answer)
    assert supervisor_calls and len(supervisor_calls) == 1
    assert handler_counts["total"] == 2
    assert len(rag_calls) == 2
    assert business_keys(multi_read_env) == before_business
    snapshot = ledger_snapshot(multi_read_env, rid)
    assert snapshot["request"]["status"] == "completed"
    assert snapshot["operations"] == []
    nine_after_first = _nine_table_counts(multi_read_env)

    if first_transport == "json":
        replay_status, replay = harness.post_json(case["source"], request_id=rid)
        assert replay_status == 200 and replay["replayed"] is True
        assert replay["answer"] == first_answer
        assert replay["intent"] == first["intent"]
        assert (replay["request_id"], replay["message_id"], replay["run_id"]) == ids
        assert replay["metrics"]["receipt_refs"] == []
        sse_status, events, _ = harness.stream(case["source"], request_id=rid)
        sse_answer = "".join(str(frame.get("content") or "") for frame in events)
        assert sse_status == 200 and events and sse_answer == first_answer
        assert events[-1]["intent"] == first["intent"]
    else:
        sse_status, events, _ = harness.stream(case["source"], request_id=rid)
        sse_answer = "".join(str(frame.get("content") or "") for frame in events)
        assert sse_status == 200 and events and sse_answer == first_answer
        replay_status, replay = harness.post_json(case["source"], request_id=rid)
        assert replay_status == 200 and replay["replayed"] is True
        assert replay["answer"] == first_answer
        assert replay["intent"] == first["intent"]

    assert (replay["request_id"], replay["message_id"], replay["run_id"]) == ids
    assert replay["metrics"]["receipt_refs"] == []
    assert len(seen) == 6, seen
    assert len(supervisor_calls) == 1
    assert handler_counts["total"] == 2
    assert len(rag_calls) == 2
    assert business_keys(multi_read_env) == before_business
    assert _nine_table_counts(multi_read_env) == nine_after_first
    assert ledger_snapshot(multi_read_env, rid)["operations"] == []


def test_saved_two_read_answers_complete_after_real_persisted_resume(
        multi_read_env, monkeypatch):
    """Pure-read persisted recovery: the read branch has no preceding write event.

    The real final_reply insert fails after both read results are persisted; the
    same request id resumes from the real checkpoint/lease, delivers both answers
    once in request order and never redispatches either read worker.
    """
    case = SAVED_TWO_READ_CASES[0]
    rag_calls = _install_rag(monkeypatch)
    seen = []
    supervisor_calls = []
    _install_provider(
        monkeypatch,
        understanding=_understanding(*[
            _task(task_type, task_text) for task_type, task_text in case["tasks"]
        ]),
        supervisor=_saved_batch(case),
        answers=case["answers"],
        seen=seen,
        supervisor_calls=supervisor_calls,
    )
    harness = C3HTTPHarness(monkeypatch, multi_read_env)
    handler_counts = _count_read_handlers(monkeypatch)
    before_business = business_keys(multi_read_env)
    expected = _expected_two_answers(case)
    rid = f"sim26-live02-read-resume-{uuid4().hex[:8]}"

    fault = _phase_fault(multi_read_env, rid, "final_reply")
    with fault:
        first_status, first = harness.post_json(case["source"], request_id=rid)
    assert fault.raised is True and fault.reached == 1, fault.report()
    assert first_status == 200
    interrupted = ledger_snapshot(multi_read_env, rid)
    assert interrupted["request"]["status"] not in {"completed", "failed", "limited"}
    assert len([event for event in interrupted["events"]
                if event.get("phase") == "tool_result"]) == 2
    assert not [event for event in interrupted["events"]
                if event.get("phase") == "final_reply"]
    assert len(seen) == 6 and handler_counts["total"] == 2 and len(rag_calls) == 2
    assert business_keys(multi_read_env) == before_business
    calls_before_resume = len(seen)

    resumed_status, resumed = harness.post_json(case["source"], request_id=rid)
    assert resumed_status == 200
    assert resumed["metrics"]["react_write"] == "completed", resumed
    assert resumed["answer"] == expected
    assert resumed["intent"] == "nutrition_advice"
    assert resumed["metrics"]["receipt_refs"] == []
    assert len(seen) == calls_before_resume, seen
    assert handler_counts["total"] == 2
    assert len(rag_calls) == 2
    assert business_keys(multi_read_env) == before_business
    snapshot = ledger_snapshot(multi_read_env, rid)
    assert snapshot["request"]["status"] == "completed"
    assert len([event for event in snapshot["events"]
                if event.get("phase") == "tool_result"]) == 2
    assert len([event for event in snapshot["events"]
                if event.get("phase") == "final_reply"]) == 1
    assert snapshot["operations"] == []
    ids = (first["request_id"], first["message_id"], first["run_id"])
    assert ids[0] == rid and all(ids)
    assert (resumed["request_id"], resumed["message_id"], resumed["run_id"]) == ids
    nine_after = _nine_table_counts(multi_read_env)

    replay_status, replay = harness.post_json(case["source"], request_id=rid)
    assert replay_status == 200 and replay["replayed"] is True
    assert replay["answer"] == expected
    assert replay["intent"] == resumed["intent"]
    assert replay["metrics"]["receipt_refs"] == []
    assert (replay["request_id"], replay["message_id"], replay["run_id"]) == ids
    assert len(seen) == calls_before_resume and handler_counts["total"] == 2
    assert business_keys(multi_read_env) == before_business
    assert _nine_table_counts(multi_read_env) == nine_after


# ---------------------------------------------------------------------------
# Graph-level identity/order/duplicate/assistant cases
# ---------------------------------------------------------------------------

def _candidate(cid, task_type, task_text, source, *, start=None):
    begin = source.index(task_text) if start is None else start
    return {
        "id": cid, "task_type": task_type, "action": "question",
        "text": task_text, "source_start": begin, "source_end": begin + len(task_text),
        "depends_on": [],
    }


def _run_graph(source, candidates, script, executor, *, stop_after_batch=True):
    from app.services.agent.semantics import remember_safety
    remember_safety(source, {"level": "normal", "evidence": "", "categories": [],
                             "reason": "scripted normal"})
    llm = FakeLLM(script=script, semantic=False)
    deps = ReactDependencies(
        llm=llm, tool_executor=executor, user_id="app-user-1",
        conversation_id="conv-sim26-multi", request_id="req-sim26-multi",
        stop_after_batch=stop_after_batch,
    )
    state = asyncio.run(invoke_react(deps, source, intent_candidates=candidates))
    return state, llm


def test_three_goal_types_reverse_execution_and_duplicate_call_deliver_once():
    source = "先讲训练恢复，再说早餐蛋白，最后给训练计划。"
    candidates = [
        _candidate("training-goal", "training_qa", "先讲训练恢复", source),
        _candidate("nutrition-goal", "nutrition_advice", "再说早餐蛋白", source),
        _candidate("plan-goal", "training_plan", "最后给训练计划", source),
    ]
    answers = {
        "training-goal": "训练恢复答案：睡眠和间隔。",
        "nutrition-goal": "早餐蛋白答案：每餐都有来源。",
        "plan-goal": "训练计划答案：循序渐进。",
    }

    def result_for(nth, name, args, context):
        ref = args.get("source_ref") or {}
        exact = ref.get("exact_text") or ""
        if "训练恢复" in exact:
            return {"status": "completed", "summary": answers["training-goal"]}
        if "早餐蛋白" in exact:
            return {"status": "completed", "summary": answers["nutrition-goal"]}
        return {"status": "completed", "summary": answers["plan-goal"]}

    executor = RecordingExecutor({
        "answer_training": result_for,
        "advise_nutrition": result_for,
        "plan_training": result_for,
    })
    calls = [
        tool_call("call-plan", "plan_training",
                  {"query": "训练计划", "source_ref": ref_of(source, "最后给训练计划")}),
        tool_call("call-training", "answer_training",
                  {"query": "训练恢复", "source_ref": ref_of(source, "先讲训练恢复")}),
        tool_call("call-nutrition", "advise_nutrition",
                  {"query": "早餐蛋白", "source_ref": ref_of(source, "再说早餐蛋白")}),
        # duplicate dispatch of an already covered goal: reuse, never re-present.
        tool_call("call-training-again", "answer_training",
                  {"query": "训练恢复", "source_ref": ref_of(source, "先讲训练恢复")}),
    ]
    state, llm = _run_graph(
        source, candidates, [assistant("", tool_calls=calls)], executor)

    assert state["status"] == "complete", state
    assert executor.names == ["plan_training", "answer_training", "advise_nutrition"]
    assert len(llm.calls) == 1, llm.calls
    assert state["covered_goals"] == ["plan-goal", "training-goal", "nutrition-goal"]
    assert state["final_answer"] == "\n\n".join([
        answers["training-goal"], answers["nutrition-goal"], answers["plan-goal"]])


def test_same_type_same_text_distinct_goals_both_delivered():
    source = "怎么安排？怎么安排？"
    candidates = [
        _candidate("first-goal", "nutrition_advice", "怎么安排？", source, start=0),
        _candidate("second-goal", "nutrition_advice", "怎么安排？", source, start=5),
    ]
    same_answer = "统一建议：规律安排，不必追求完全均等。"
    executor = RecordingExecutor({
        "advise_nutrition": {"status": "completed", "summary": same_answer}})
    calls = [
        tool_call("call-second", "advise_nutrition", {
            "query": "第二问", "source_ref": ref_of(source, "怎么安排？", occurrence=1)}),
        tool_call("call-first", "advise_nutrition", {
            "query": "第一问", "source_ref": ref_of(source, "怎么安排？", occurrence=0)}),
    ]
    state, _ = _run_graph(source, candidates, [assistant("", tool_calls=calls)], executor)

    assert state["status"] == "complete", state
    assert executor.names == ["advise_nutrition", "advise_nutrition"]
    assert state["final_answer"] == same_answer + "\n\n" + same_answer
    assert state["final_answer"].count(same_answer) == 2


def test_last_answer_assistant_does_not_hide_earlier_completed_goal():
    source = "第一问怎么安排？第二问怎么安排？"
    candidates = [
        _candidate("first-goal", "nutrition_advice", "第一问怎么安排？", source),
        _candidate("second-goal", "nutrition_advice", "第二问怎么安排？", source),
    ]
    first = "第一问答案：先保证总量。"
    second = "第二问答案：再按餐次分配。"
    executor = RecordingExecutor({
        "advise_nutrition": lambda nth, name, args, ctx: {
            "status": "completed",
            "summary": first if "第一问" in (args.get("source_ref") or {}).get("exact_text", "")
            else second,
        }})
    calls = [
        tool_call("call-first", "advise_nutrition", {
            "query": "第一问", "source_ref": ref_of(source, "第一问怎么安排？")}),
        tool_call("call-second", "advise_nutrition", {
            "query": "第二问", "source_ref": ref_of(source, "第二问怎么安排？")}),
    ]
    state, llm = _run_graph(
        source, candidates,
        [assistant("", tool_calls=calls), assistant(second)],
        executor, stop_after_batch=False)

    assert state["status"] == "complete", state
    assert len(llm.calls) == 2
    assert state["final_answer"] == first + "\n\n" + second


def test_legitimate_aggregate_summary_is_not_duplicated():
    source = "第一问怎么安排？第二问怎么安排？"
    candidates = [
        _candidate("first-goal", "nutrition_advice", "第一问怎么安排？", source),
        _candidate("second-goal", "nutrition_advice", "第二问怎么安排？", source),
    ]
    first = "第一问答案：先保证总量。"
    second = "第二问答案：再按餐次分配。"
    summary = "完整回答如下：\n\n" + first + "\n\n" + second
    executor = RecordingExecutor({
        "advise_nutrition": lambda nth, name, args, ctx: {
            "status": "completed",
            "summary": first if "第一问" in (args.get("source_ref") or {}).get("exact_text", "")
            else second,
        }})
    calls = [
        tool_call("call-first", "advise_nutrition", {
            "query": "第一问", "source_ref": ref_of(source, "第一问怎么安排？")}),
        tool_call("call-second", "advise_nutrition", {
            "query": "第二问", "source_ref": ref_of(source, "第二问怎么安排？")}),
    ]
    state, _ = _run_graph(
        source, candidates,
        [assistant("", tool_calls=calls), assistant(summary)],
        executor, stop_after_batch=False)

    assert state["final_answer"] == summary
    assert state["final_answer"].count(first) == 1
    assert state["final_answer"].count(second) == 1


def test_tool_call_preamble_is_not_delivered_instead_of_goal_answers():
    source = "第一问怎么安排？第二问怎么安排？"
    candidates = [
        _candidate("first-goal", "nutrition_advice", "第一问怎么安排？", source),
        _candidate("second-goal", "nutrition_advice", "第二问怎么安排？", source),
    ]
    first = "第一问答案：先保证总量。"
    second = "第二问答案：再按餐次分配。"
    preamble = "我先分别查一下这两个问题。"
    executor = RecordingExecutor({
        "advise_nutrition": lambda nth, name, args, ctx: {
            "status": "completed",
            "summary": first if "第一问" in (args.get("source_ref") or {}).get("exact_text", "")
            else second,
        }})
    calls = [
        tool_call("call-first", "advise_nutrition", {
            "query": "第一问", "source_ref": ref_of(source, "第一问怎么安排？")}),
        tool_call("call-second", "advise_nutrition", {
            "query": "第二问", "source_ref": ref_of(source, "第二问怎么安排？")}),
    ]
    state, _ = _run_graph(
        source, candidates, [assistant(preamble, tool_calls=calls)], executor)

    assert state["final_answer"] == first + "\n\n" + second
    assert preamble not in state["final_answer"]


def test_auxiliary_reads_with_profile_sentinel_are_not_delivered():
    source = "第一问怎么安排？第二问怎么安排？"
    candidates = [
        _candidate("first-goal", "nutrition_advice", "第一问怎么安排？", source),
        _candidate("second-goal", "nutrition_advice", "第二问怎么安排？", source),
    ]
    first = "第一问答案：先保证总量。"
    second = "第二问答案：再按餐次分配。"
    sentinel = "SENTINEL-PROFILE-DO-NOT-LEAK-7f31"
    executor = RecordingExecutor({
        "advise_nutrition": lambda nth, name, args, ctx: {
            "status": "completed",
            "summary": first if "第一问" in (args.get("source_ref") or {}).get("exact_text", "")
            else second,
        },
        "read_profile": {
            "status": "completed", "summary": sentinel,
            "result": {"profile": {"allergy": sentinel}},
        },
        "read_workout_history": {"status": "completed", "summary": sentinel + "-HISTORY"},
        "read_nutrition_summary": {"status": "completed", "summary": sentinel + "-SUMMARY"},
        "read_current_plan": {"status": "completed", "summary": sentinel + "-PLAN"},
    })
    calls = [
        tool_call("call-first", "advise_nutrition", {
            "query": "第一问", "source_ref": ref_of(source, "第一问怎么安排？")}),
        tool_call("call-second", "advise_nutrition", {
            "query": "第二问", "source_ref": ref_of(source, "第二问怎么安排？")}),
        tool_call("call-profile", "read_profile", {}),
        tool_call("call-history", "read_workout_history", {"days": 30}),
        tool_call("call-summary", "read_nutrition_summary", {"days": 7}),
        tool_call("call-plan", "read_current_plan", {}),
    ]
    state, _ = _run_graph(
        source, candidates, [assistant("", tool_calls=calls)], executor)

    assert state["status"] == "complete", state
    assert state["final_answer"] == first + "\n\n" + second
    assert sentinel not in state["final_answer"]
    assert set(state["goal_results"]) == {"first-goal", "second-goal"}
    assert state["covered_goals"] == ["first-goal", "second-goal"]


def test_unbound_failed_and_clarification_results_are_excluded():
    source = "第一问怎么安排？第二问怎么安排？第三问怎么安排？第四问怎么安排？"
    candidates = [
        _candidate("completed-goal", "training_qa", "第一问怎么安排？", source),
        _candidate("clarify-goal", "nutrition_advice", "第二问怎么安排？", source),
        _candidate("uncovered-goal", "training_qa", "第三问怎么安排？", source),
        _candidate("failed-goal", "nutrition_advice", "第四问怎么安排？", source),
        {"id": "aux-goal", "task_type": None, "action": "read", "text": "辅助"},
    ]
    completed_answer = "已完成目标的真实答复。"
    state = {
        "intent_candidates": candidates,
        "goal_results": {
            "completed-goal": {"status": "completed", "summary": completed_answer},
            "clarify-goal": {"status": "needs_clarification",
                             "summary": "CLARIFY-SENTINEL", "missing_fields": ["meal"]},
            "uncovered-goal": {"status": "completed", "summary": "UNCOVERED-SENTINEL"},
            "failed-goal": {"status": "completed", "error": "read_failed",
                            "summary": "FAILED-SENTINEL"},
            "aux-goal": {"status": "completed", "summary": "AUX-SENTINEL"},
        },
        "covered_goals": ["completed-goal", "clarify-goal", "failed-goal", "aux-goal"],
        "observations": [
            {"status": "completed", "summary": completed_answer},
            {"status": "needs_clarification", "summary": "CLARIFY-SENTINEL"},
            {"status": "completed", "summary": "UNCOVERED-SENTINEL"},
            {"status": "completed", "error": "read_failed", "summary": "FAILED-SENTINEL"},
            {"status": "completed", "summary": "AUX-SENTINEL"},
        ],
        "messages": [],
    }
    goals = completed_read_goals(
        state["intent_candidates"], state["goal_results"], state["covered_goals"])
    assert [goal["id"] for goal in goals] == ["completed-goal"]
    answer = _extract_final_answer(state)
    assert answer == completed_answer
    for marker in ("CLARIFY-SENTINEL", "UNCOVERED-SENTINEL", "FAILED-SENTINEL", "AUX-SENTINEL"):
        assert marker not in answer


def test_single_goal_keeps_final_assistant_and_ignores_tool_preamble():
    source = "怎么安排？"
    candidate = _candidate("only-goal", "nutrition_advice", source, source)
    goal = {"only-goal": {"status": "completed", "summary": "目标真实答案。"}}
    covered = ["only-goal"]

    final_state = {
        "intent_candidates": [candidate], "goal_results": goal, "covered_goals": covered,
        "messages": [assistant("", tool_calls=[tool_call("t1", "advise_nutrition", {})]),
                     assistant("单问最终答复。")],
    }
    assert _extract_final_answer(final_state) == "单问最终答复。"

    preamble_state = {
        "intent_candidates": [candidate], "goal_results": goal, "covered_goals": covered,
        "messages": [assistant("我先查一下。", tool_calls=[
            tool_call("t1", "advise_nutrition", {})])],
    }
    assert _extract_final_answer(preamble_state) == "目标真实答案。"


# ---------------------------------------------------------------------------
# Deterministic safety/limited notices keep priority over aggregation
# ---------------------------------------------------------------------------

def _two_goal_state(**overrides):
    source = "第一问怎么安排？第二问怎么安排？"
    candidates = [
        _candidate("first-goal", "nutrition_advice", "第一问怎么安排？", source),
        _candidate("second-goal", "nutrition_advice", "第二问怎么安排？", source),
    ]
    state = {
        "intent_candidates": candidates,
        "goal_results": {
            "first-goal": {"status": "completed", "summary": "A1"},
            "second-goal": {"status": "completed", "summary": "A2"},
        },
        "covered_goals": ["first-goal", "second-goal"],
        "observations": [],
        "messages": [],
        "termination_reason": "",
        "safety_level": "normal",
        "candidate_error": "",
        "unbound_write_calls": [],
        "call_facts": [],
    }
    state.update(overrides)
    return state


@pytest.mark.parametrize("reason,safety,status,notice", [
    ("urgent_safety_blocked", "urgent", "rejected", URGENT_SAFETY_ANSWER),
    ("tool_step_limit", "normal", "limited", "已达到本轮工具执行步数上限"),
    ("policy_stop", "normal", "limited", "本次模型预算或执行策略要求停止"),
])
def test_deterministic_notice_wins_over_completed_read_answers(reason, safety, status, notice):
    state = _two_goal_state(termination_reason=reason, safety_level=safety)
    deps = ReactDependencies(llm=FakeLLM(script=[], semantic=False))
    final = finalize_node(state, deps)
    assert final["status"] == status
    if reason == "urgent_safety_blocked":
        # Safety rejection still replaces every ordinary answer.
        assert final["final_answer"] == notice
        assert "A1" not in final["final_answer"] and "A2" not in final["final_answer"]
    else:
        # A limited read stop keeps the answers already produced, then states
        # the unfinished part explicitly. It must not discard completed goals.
        assert notice in final["final_answer"]
        assert final["final_answer"].startswith("A1\n\nA2\n\n")
        assert final["final_answer"].count("A1") == 1
        assert final["final_answer"].count("A2") == 1


def test_unconfirmed_write_notice_is_not_replaced_by_read_answers():
    state = _two_goal_state(
        intent_candidates=[
            {"id": "write-goal", "task_type": "nutrition_record", "action": "record",
             "text": "帮我记录午餐", "depends_on": []},
            {"id": "read-goal", "task_type": "nutrition_advice", "action": "question",
             "text": "再给我建议", "depends_on": []},
        ],
        goal_results={
            "write-goal": {"status": "completed", "commit_state": "unknown"},
            "read-goal": {"status": "completed", "summary": "READ-ANSWER"},
        },
        covered_goals=["write-goal", "read-goal"],
    )
    deps = ReactDependencies(llm=FakeLLM(script=[], semantic=False))
    final = finalize_node(state, deps)
    assert final["status"] == "blocked"
    assert "尚未确认" in final["final_answer"]
    assert "READ-ANSWER" not in final["final_answer"]


# ---------------------------------------------------------------------------
# Runtime no-final-answer fallback / frozen resume state
# ---------------------------------------------------------------------------

def _runtime_without_io():
    from app.services.agent.react_write_runtime import WriteRuntimeSession

    runtime = WriteRuntimeSession.__new__(WriteRuntimeSession)
    runtime.ai_estimate_committed = False
    runtime.termination_reason = ""
    runtime.receipts = []
    runtime.committed_observations = []
    runtime.resume_goal_results = {}
    runtime.resume_covered = []
    runtime.resume_final_answer = ""
    runtime.frozen_parameters = {}
    runtime.response_ids = {}
    runtime._response_ids = lambda: {"request_id": "r", "message_id": "m", "run_id": "u"}
    runtime._citations = lambda state: []
    return runtime


def _resume_runtime(goals, answers, task_types):
    runtime = _runtime_without_io()
    runtime.resume_covered = [goal for goal, _ in goals]
    runtime.resume_goal_results = {
        goal: {"status": "completed", "summary": answer}
        for (goal, _), answer in zip(goals, answers)
    }
    runtime.frozen_parameters = {"task_outline": [
        {"id": goal, "task_type": task_type, "action": "question", "text": goal}
        for (goal, _), task_type in zip(goals, task_types)
    ]}
    return runtime


def test_runtime_resume_fallback_matches_graph_extraction():
    goals = [("goal-1", "training_qa"), ("goal-2", "nutrition_advice")]
    answers = ["恢复答案。", "蛋白质答案。"]
    runtime = _resume_runtime(goals, answers, [item[1] for item in goals])

    state = {
        "intent_candidates": [
            {"id": goal, "task_type": task_type, "action": "question", "text": goal}
            for goal, task_type in goals
        ],
        "goal_results": {
            goal: {"status": "completed", "summary": answer}
            for (goal, _), answer in zip(goals, answers)
        },
        "covered_goals": [goal for goal, _ in goals],
        "messages": [],
    }
    expected = _extract_final_answer(state)
    assert expected == answers[0] + "\n\n" + answers[1]
    assert runtime._state_answer({}, include_completed_goals=True) == expected
    assert runtime._state_answer(state) == expected
    assert runtime._state_answer({**state, "final_answer": answers[1]}) == expected
    payload = runtime._committed_payload({})
    assert payload["final_answer"] == expected
    assert payload["intent"] in {"training_qa", "nutrition_advice"}
    assert payload["receipt_refs"] == []


def test_runtime_resume_single_goal_fallback_is_the_goal_answer():
    goals = [("goal-1", "nutrition_advice")]
    answers = ["单条恢复答案。"]
    runtime = _resume_runtime(goals, answers, ["nutrition_advice"])
    assert runtime._state_answer({}, include_completed_goals=True) == "单条恢复答案。"
    payload = runtime._committed_payload({})
    assert payload["final_answer"] == "单条恢复答案。"
    assert payload["intent"] == "nutrition_advice"
    assert payload["receipt_refs"] == []


# ---------------------------------------------------------------------------
# Mixed write + read: receipt fidelity and a real persisted resume
# ---------------------------------------------------------------------------

def _mixed_program():
    calls = [
        ("call-record", "record_nutrition",
         {"source_ref": ref_of(MIXED_SOURCE, MIXED_FACT)}),
        ("call-training", "answer_training",
         {"query": "训练后如何恢复", "source_ref": ref_of(MIXED_SOURCE, "训练后如何恢复")}),
        ("call-advice", "advise_nutrition",
         {"query": "蛋白质怎么分散到三餐", "source_ref": ref_of(MIXED_SOURCE, "蛋白质怎么分散到三餐")}),
    ]
    return _batch_message(calls)


def _mixed_understanding():
    return _understanding(*[
        _task(task_type, task_text, action="record" if task_type == "nutrition_record" else "question",
              fact_text=fact_text)
        for task_type, task_text, fact_text in MIXED_TASKS
    ])


def _nine_business_counts(engine):
    tables = ("workout_records", "workout_sessions", "workout_session_exercises",
              "workout_session_sets", "user_food_log", "nutrition_logs",
              "training_plans")
    with engine.connect() as conn:
        return {table: conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                for table in tables}


def test_mixed_write_and_two_reads_keep_receipt_and_all_answers(multi_read_env, monkeypatch):
    rag_calls = _install_rag(monkeypatch)
    seen = []
    supervisor_calls = []
    _install_provider(
        monkeypatch, understanding=_mixed_understanding(), supervisor=_mixed_program(),
        answers=MIXED_READ_ANSWERS, seen=seen, supervisor_calls=supervisor_calls)
    harness = C3HTTPHarness(monkeypatch, multi_read_env)
    handler_counts = _count_read_handlers(monkeypatch)
    before_business = business_keys(multi_read_env)
    before_business_counts = _nine_business_counts(multi_read_env)
    rid = f"sim26-live02-mixed-{uuid4().hex[:8]}"

    status, payload = harness.post_json(MIXED_SOURCE, request_id=rid)
    assert status == 200
    assert payload["metrics"]["react_write"] == "completed"
    expected = "记录已保存，保存回执已保留。\n\n" + MIXED_READ_ANSWERS[0] + "\n\n" + MIXED_READ_ANSWERS[1]
    assert payload["answer"] == expected
    assert payload["intent"] == "nutrition_record"
    receipts = list(payload["metrics"]["receipt_refs"])
    assert receipts
    after = business_keys(multi_read_env)
    assert len(after["nutrition_logs"]) == len(before_business["nutrition_logs"]) + 1
    assert after["user_food_log"] == before_business["user_food_log"]
    business_counts = _nine_business_counts(multi_read_env)
    assert business_counts["nutrition_logs"] == before_business_counts["nutrition_logs"] + 1
    assert business_counts["user_food_log"] == before_business_counts["user_food_log"]
    assert len(seen) == 6, seen
    assert len(supervisor_calls) == 1
    assert handler_counts["total"] == 2
    assert len(rag_calls) == 2
    snapshot = ledger_snapshot(multi_read_env, rid)
    assert snapshot["request"]["status"] == "completed"
    assert [item["kind"] for item in snapshot["operations"]] == ["nutrition"]
    nine_after = _nine_table_counts(multi_read_env)
    ids = (payload["request_id"], payload["message_id"], payload["run_id"])

    replay_status, replay = harness.post_json(MIXED_SOURCE, request_id=rid)
    assert replay_status == 200 and replay["replayed"] is True
    assert replay["answer"] == expected
    assert replay["intent"] == payload["intent"]
    assert replay["metrics"]["receipt_refs"] == receipts
    assert (replay["request_id"], replay["message_id"], replay["run_id"]) == ids
    sse_status, events, _ = harness.stream(MIXED_SOURCE, request_id=rid)
    sse_answer = "".join(str(frame.get("content") or "") for frame in events)
    assert sse_status == 200 and events and sse_answer == expected
    assert events[-1]["metrics"]["receipt_refs"] == receipts
    assert (events[-1]["request_id"], events[-1]["message_id"], events[-1]["run_id"]) == ids
    assert len(seen) == 6 and handler_counts["total"] == 2 and len(rag_calls) == 2
    assert business_keys(multi_read_env) == after
    assert _nine_table_counts(multi_read_env) == nine_after


def _phase_fault(engine, rid, phase):
    def _match(statement, parameters, context):
        compiled = getattr(context, "compiled_parameters", None) or []
        for item in compiled:
            if not isinstance(item, dict):
                continue
            if str(item.get("user_id")) == API_OWNER and str(item.get("request_id")) == rid \
                    and str(item.get("phase")) == phase:
                return True
        if isinstance(parameters, dict):
            return (str(parameters.get("phase")) == phase
                    and str(parameters.get("request_id")) == rid
                    and str(parameters.get("user_id")) == API_OWNER)
        return False

    return C3SQLFault(engine, table="agent_execution_events", action="insert", occurrence=1,
                      match=_match)


def test_real_persisted_resume_delivers_all_reads_without_redispatch(multi_read_env, monkeypatch):
    rag_calls = _install_rag(monkeypatch)
    seen = []
    supervisor_calls = []
    _install_provider(
        monkeypatch, understanding=_mixed_understanding(), supervisor=_mixed_program(),
        answers=MIXED_READ_ANSWERS, seen=seen, supervisor_calls=supervisor_calls)
    harness = C3HTTPHarness(monkeypatch, multi_read_env)
    handler_counts = _count_read_handlers(monkeypatch)
    before_business = business_keys(multi_read_env)
    rid = f"sim26-live02-resume-{uuid4().hex[:8]}"
    expected = "记录已保存，保存回执已保留。\n\n" + MIXED_READ_ANSWERS[0] + "\n\n" + MIXED_READ_ANSWERS[1]

    fault = _phase_fault(multi_read_env, rid, "final_reply")
    with fault:
        first_status, first = harness.post_json(MIXED_SOURCE, request_id=rid)
    assert fault.raised is True and fault.reached == 1, fault.report()
    assert first_status == 200
    interrupted = ledger_snapshot(multi_read_env, rid)
    assert interrupted["request"]["status"] not in {"completed", "failed", "limited"}
    assert len([event for event in interrupted["events"]
                if event.get("phase") == "tool_result"]) == 3
    assert not [event for event in interrupted["events"]
                if event.get("phase") == "final_reply"]
    assert len(seen) == 6 and handler_counts["total"] == 2 and len(rag_calls) == 2
    after_first = business_keys(multi_read_env)
    assert len(after_first["nutrition_logs"]) == len(before_business["nutrition_logs"]) + 1
    calls_before_resume = len(seen)

    resumed_status, resumed = harness.post_json(MIXED_SOURCE, request_id=rid)
    assert resumed_status == 200
    assert resumed["metrics"]["react_write"] == "completed", resumed
    assert resumed["answer"] == expected
    assert resumed["intent"] == "nutrition_record"
    receipts = list(resumed["metrics"]["receipt_refs"])
    assert receipts
    assert len(seen) == calls_before_resume, seen
    assert handler_counts["total"] == 2
    assert len(rag_calls) == 2
    assert business_keys(multi_read_env) == after_first
    snapshot = ledger_snapshot(multi_read_env, rid)
    assert snapshot["request"]["status"] == "completed"
    assert len([event for event in snapshot["events"]
                if event.get("phase") == "tool_result"]) == 3
    assert len([event for event in snapshot["events"]
                if event.get("phase") == "final_reply"]) == 1
    assert [item["kind"] for item in snapshot["operations"]] == ["nutrition"]
    ids = (resumed["request_id"], resumed["message_id"], resumed["run_id"])
    assert ids[0] == rid and all(ids)
    assert (first["request_id"], first["message_id"], first["run_id"]) == ids
    nine_after = _nine_table_counts(multi_read_env)

    replay_status, replay = harness.post_json(MIXED_SOURCE, request_id=rid)
    assert replay_status == 200 and replay["replayed"] is True
    assert replay["answer"] == expected
    assert replay["intent"] == resumed["intent"]
    assert replay["metrics"]["receipt_refs"] == receipts
    assert (replay["request_id"], replay["message_id"], replay["run_id"]) == ids
    assert len(seen) == calls_before_resume and handler_counts["total"] == 2
    assert business_keys(multi_read_env) == after_first
    assert _nine_table_counts(multi_read_env) == nine_after
