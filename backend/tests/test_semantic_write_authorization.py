"""Current contract: model semantics decide intent; authorization contains no regex."""
import ast
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.services.agent.planner_validation import validate_planner_output, validate_rule_plan
from app.services.agent.react_models import SourceScope
from app.services.agent.react_tools import validate_source_ref
from app.services.agent.task_planner import plan_tasks_from_llm, plan_tasks_rule
from app.services.security.write_guard import authorize_write_span, guard_planner_tasks


def _seed(message):
    from app.services.agent.semantics import remember_safety
    remember_safety(message, {"level": "normal", "evidence": "", "categories": [], "reason": "scripted normal"})


def task(message, action="record", **flags):
    return {"task_type": "nutrition_record", "action": action, "text": message,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False, **flags}


def test_authorization_module_has_no_regex_or_text_classification_helpers():
    import app.services.security.write_guard as module
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.Import) and any(item.name == "re" for item in node.names)
                   or isinstance(node, ast.ImportFrom) and node.module == "re"
                   for node in ast.walk(tree))
    assert not any(hasattr(module, name) for name in (
        "planner_write_rejection", "has_imperative_write", "is_bare_write_command",
        "is_body_weight_only", "source_write_window", "source_task_tail"))


@pytest.mark.parametrize("message", [
    "帮我记录中午吃了一个鸡蛋", "中午一个鸡蛋，记到午餐里",
    "午饭一枚鸡蛋，晚上米饭一碗，都记上", "中午鸡蛋一枚",
    "帮我记录今天中午吃了一个鸡蛋，晚上吃了一碗米饭",
    "朋友推荐的鸡蛋，中午一个给我存上", "不要忘了给我记上中午的鸡蛋",
    "记录中午鸡蛋，备注：如果有包装就再核对热量", "记录鸡蛋？", "保存",
])
def test_model_record_decision_is_not_overruled_by_source_words(message):
    _seed(message)
    raw = {"tasks": [task(message)]}
    result = validate_planner_output(message, raw)
    assert not result.fallback_required
    assert result.tasks[0]["write_authorization"] == "authorized"
    planned = plan_tasks_from_llm(message, raw)
    assert planned[0]["payload"]["write_authorization"] == "authorized"
    assert planned[0]["side_effect"]


@pytest.mark.parametrize("flag", ["is_negated", "is_hypothetical", "is_third_party"])
def test_structured_semantic_flags_still_reject(flag):
    message = "午餐鸡蛋一枚"
    _seed(message)
    raw = {"tasks": [task(message, **{flag: True})]}
    result = validate_planner_output(message, raw)
    assert result.tasks[0]["write_authorization"] == "rejected_unauthorized"
    assert not result.tasks[0]["side_effect"]
    planned = plan_tasks_from_llm(message, raw)
    assert not planned[0]["side_effect"]
    assert authorize_write_span(message, {flag: True})[0] == "rejected_unauthorized"


@pytest.mark.parametrize("action", ["question", "answer"])
def test_model_nonwrite_action_cannot_be_upgraded_by_keywords(action):
    message = "帮我记录中午吃了一个鸡蛋是什么意思"
    _seed(message)
    raw = {"tasks": [task(message, action)]}
    validated = validate_planner_output(message, raw)
    assert all(not t.get("side_effect", False) for t in validated.tasks)
    planned = plan_tasks_from_llm(message, raw)
    assert all(not t["side_effect"] for t in planned)
    assert all(t["action"] not in {"record", "confirm"} for t in planned)


def test_source_schema_and_model_flags_cannot_be_forged():
    message = "中午鸡蛋一枚"
    _seed(message)
    for raw in ({"tasks": [task("模型编造的鸡胸肉")]}, {"tasks": [task(message, is_negated="false")]}, "{"):
        result = validate_planner_output(message, raw, rule_tasks=plan_tasks_rule(message))
        assert result.fallback_required
        assert all(not t.get("side_effect", False) for t in result.tasks)


def test_unavailable_model_cannot_authorize_rule_fallback_writes():
    message = "帮我记录中午吃了一个鸡蛋"
    raw_tasks = plan_tasks_rule(message)
    assert raw_tasks
    assert all(not t["side_effect"] for t in raw_tasks)
    result = validate_rule_plan(message, raw_tasks)
    assert all(t.get("task_type") == "training_qa" and t.get("action") == "question" for t in result.tasks)
    assert all(not t.get("side_effect", False) for t in result.tasks)


def test_authorization_does_not_scan_old_forward_or_tail_windows():
    assert authorize_write_span("中午鸡蛋一枚", write_window="不要记录", tail_window="如果朋友吃了？") == ("authorized", None)
    assert authorize_write_span("中午鸡蛋一枚", urgent=True)[0] == "rejected_unauthorized"


@pytest.mark.parametrize("message", ["中午一个鸡蛋，记到午餐里", "不要忘了给我记上鸡蛋", "记录鸡蛋？"])
def test_react_source_ref_uses_tool_choice_not_regex(message):
    ref = {"start": 0, "end": len(message), "exact_text": message}
    result = validate_source_ref(SourceScope(text=message), ref, "nutrition_record")
    assert result.ok and result.authorization == "authorized"


@pytest.mark.parametrize("ref", [
    {"start": -1, "end": 3, "exact_text": "鸡蛋"},
    {"start": True, "end": 3, "exact_text": "鸡蛋"},
    {"start": 0, "end": 999, "exact_text": "鸡蛋"},
    {"start": 0, "end": 2, "exact_text": "米饭"},
    {"start": 0, "end": 0, "exact_text": ""},
])
def test_react_source_ref_keeps_exact_bounds_and_nonempty_validation(ref):
    assert not validate_source_ref(SourceScope(text="鸡蛋一枚"), ref, "nutrition_record").ok


@pytest.mark.parametrize("session,expiry", [("another", "2099-01-01T00:00:00Z"), ("session", "2000-01-01T00:00:00Z")])
def test_rule_fallback_confirmation_keeps_session_and_expiry_checks(session, expiry):
    _seed("确认保存")
    tasks = [task("确认保存", "confirm")]
    guard_planner_tasks("确认保存", tasks, now=datetime.now(timezone.utc), semantic_available=False,
                        current_session_id="session", pending_record={"record_id": "pending",
                        "session_id": session, "task_type": "nutrition_record", "expires_at": expiry})
    assert tasks[0]["write_authorization"] == "needs_confirmation"


def test_valid_bound_pending_can_be_confirmed_without_text_regex():
    _seed("按这个存吧")
    tasks = [task("按这个存吧", "confirm")]
    guard_planner_tasks("按这个存吧", tasks, now=datetime.now(timezone.utc), semantic_available=False,
                        current_session_id="session", pending_record={"record_id": "pending",
                        "session_id": "session", "task_type": "nutrition_record", "expires_at": "2099-01-01T00:00:00Z"})
    assert tasks[0]["write_authorization"] == "authorized"


@pytest.mark.parametrize("message", ["帮我记录中午吃了一个鸡蛋", "不要忘了给我记上中午吃的一个鸡蛋"])
def test_model_record_reaches_real_agent_worker_without_extra_confirmation(monkeypatch, message):
    from test_write_scope_and_negation_scope import _RecordingAgent
    from app.services.agent.food_parser import FoodInput
    entry = FoodInput(raw_text="一个鸡蛋", requested_name="鸡蛋", quantity=1, unit="个",
                      weight_g=None, portion_name="个", meal_type="午餐", record_date=None)
    _seed(message)
    recorder = _RecordingAgent(monkeypatch, parsed_input={"tasks": [task(message, foods=[{
        "name": "鸡蛋", "quantity": 1, "unit": "个", "meal_type": "午餐", "is_negated": False,
    }])]}, food_by_message=lambda _: [entry])
    response = asyncio.run(recorder.respond(message))
    assert len(recorder.named("calculate_nutrition")) == 1, (response.answer, recorder.calls)
    assert "已记录这餐" in response.answer


@pytest.mark.parametrize("action,flags", [("question", {}), ("record", {"is_negated": True}), ("record", {"is_hypothetical": True}), ("record", {"is_third_party": True})])
def test_model_nonwrite_semantics_dispatch_no_saving_tool(monkeypatch, action, flags):
    from test_write_scope_and_negation_scope import _RecordingAgent
    message = "记录中午吃一个鸡蛋"
    _seed(message)
    recorder = _RecordingAgent(monkeypatch, parsed_input={"tasks": [task(message, action, **flags)]})
    asyncio.run(recorder.respond(message))
    assert not recorder.named("calculate_nutrition")
    assert not recorder.named("calculate_nutrition_batch")


def test_multiple_meals_use_the_same_model_authorization_without_extra_confirmation(monkeypatch):
    from test_write_scope_and_negation_scope import _RecordingAgent
    message = "帮我记录今天中午吃了一个鸡蛋，晚上吃了一碗米饭"
    foods = [{"name": "鸡蛋", "quantity": 1, "unit": "个", "meal_type": "午餐"},
             {"name": "米饭", "quantity": 1, "unit": "碗", "meal_type": "晚餐"}]
    totals = {"kcal": 246, "protein_g": 10.2, "carb_g": 39.4, "fat_g": 5.2, "fiber_g": 0}
    calculation = {"status": "calculated", "totals": totals, "log_ids": [1, 2],
                   "calculations": [{"meal_type": "午餐", "totals": {"kcal": 72}},
                                    {"meal_type": "晚餐", "totals": {"kcal": 174}}]}
    recorder = _RecordingAgent(monkeypatch, parsed_input={"tasks": [task(message, foods=foods)]},
                               calculations=[calculation])
    response = asyncio.run(recorder.respond(message))
    assert len(recorder.named("calculate_nutrition_batch")) == 1
    assert not recorder.named("calculate_nutrition")
    assert "已分别记录" in response.answer


def test_workout_model_decision_needs_no_fixed_write_verb(monkeypatch):
    from test_write_scope_and_negation_scope import _RecordingAgent
    message = "卧推60kg4组8次，收进今天的训练里"
    model_task = {**task(message), "task_type": "workout_record"}
    recorder = _RecordingAgent(monkeypatch, parsed_input={"tasks": [model_task]},
                               workouts=[{"exercise": "卧推", "weight_kg": 60, "sets": 4, "reps": 8}])
    response = asyncio.run(recorder.respond(message))
    assert len(recorder.named("save_workout_record")) == 1
    assert "已记录" in response.answer


def test_real_agent_does_not_save_when_model_is_unavailable(monkeypatch):
    from test_write_scope_and_negation_scope import _RecordingAgent
    recorder = _RecordingAgent(monkeypatch, parsed_input=None)
    asyncio.run(recorder.respond("帮我记录中午吃了一个鸡蛋"))
    assert not any(name in {"calculate_nutrition", "calculate_nutrition_batch", "save_workout_record"}
                   for name, _ in recorder.calls)


def test_shadow_diagnostics_do_not_replace_live_semantic_planning(monkeypatch):
    from test_write_scope_and_negation_scope import _RecordingAgent
    message = "帮我记录中午吃了一个鸡蛋"
    raw = {"tasks": [task(message, foods=[{"name": "鸡蛋", "quantity": 1, "unit": "个"}])]}
    recorder = _RecordingAgent(monkeypatch, parsed_input=raw, shadow=True)
    response = asyncio.run(recorder.respond(message))
    assert len(recorder.named("calculate_nutrition")) == 1
    assert "已记录这餐" in response.answer
