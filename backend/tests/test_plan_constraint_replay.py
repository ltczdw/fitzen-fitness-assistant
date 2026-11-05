"""Actual limited-run semantic output through the real task planner, no API."""
import copy
import json
from pathlib import Path

from app.services.agent import semantics, task_planner
from app.services.agent.intent import IntentResult
from evals.v2.safety import deny_network


def test_actual_postfix_understanding_keeps_constraints_on_plan_and_no_third_worker():
    # A small synthetic fixture extracted from the two-call live check; no
    # archived evidence tree is needed to run this regression later.
    fixture = Path(__file__).parent / "fixtures/plan_constraints_semantics.json"
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    source = "先回顾刚才卧推和深蹲的重量，再给一个新手每周两次、每次20分钟、只有哑铃的增肌周计划。我没有伤病，只要建议，不保存。"
    context = {"short_memory": [{"role": "user", "message_id": mid} for mid in ("bench-user", "squat-user")]}
    token = semantics.request_understanding.set({})
    try:
        with deny_network():
            validated = semantics.validate_understanding(source, copy.deepcopy(raw), context=context)
            semantics.remember_understanding(source, validated)
            tasks = task_planner.plan_tasks(source, IntentResult("plan_adjust", 1.0), {"tasks": validated["validated_tasks"]})
        assert len(tasks) == 2
        assert [t["task_type"] for t in tasks] == ["training_qa", "training_plan"]
        assert tasks[0]["payload"]["dialogue"]["reply_purpose"] == "fact_recall"
        assert tasks[1]["text"].endswith("我没有伤病，只要建议，不保存。")
        assert all(not t["side_effect"] for t in tasks)
        assert all(source[t["payload"]["source_start"]:t["payload"]["source_end"]] == t["text"] for t in tasks)
    finally:
        semantics.request_understanding.reset(token)
