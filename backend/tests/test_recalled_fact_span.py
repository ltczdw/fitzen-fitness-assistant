"""A recalled fact stays in its history source, never a current write span."""
from copy import deepcopy
import pytest
from app.services.agent.semantics import validate_understanding, SemanticValidationError
from evals.v2.safety import deny_network


@pytest.mark.parametrize("kind,question,history,fact", [
    ("nutrition_record", "先回顾刚才早餐记录里的食物和克数", "今天早餐60克燕麦", "60克燕麦"),
    ("training_qa", "先回顾刚才记录的重量", "今天卧推50公斤3组8次", "50公斤"),
])
def test_recalled_numeric_fact_is_rejected_in_current_span_and_null_keeps_history(kind, question, history, fact):
    source = question + "，再给一个建议。"
    flags = dict(is_negated=False, is_hypothetical=False, is_third_party=False)
    raw = {"tasks": [dict(task_type=kind, action="question", text=question, fact_text=fact,
        reply_mode="conversation", reply_purpose="fact_recall", context_ids=["user-source"], **flags),
        dict(task_type="nutrition_advice", action="question", text="再给一个建议。", reply_mode="grounded", **flags)]}
    context = {"short_memory": [{"role": "user", "message_id": "user-source", "content": history}]}
    with deny_network():
        with pytest.raises(SemanticValidationError) as error:
            validate_understanding(source, raw, context=context)
        assert "context_ids" in error.value.validation_errors[0]["hint"]
        fixed = deepcopy(raw)
        fixed["tasks"][0]["fact_text"] = None
        value = validate_understanding(source, fixed, context=context)
    assert len(value["validated_tasks"]) == 2
    assert value["tasks"][0]["context_ids"] == ["user-source"]
    assert not any(t.get("side_effect") for t in value["validated_tasks"])


def test_literal_current_write_fact_stays_valid():
    source = "记录卧推50公斤3组8次"
    with deny_network():
        value = validate_understanding(source, {
            "tasks": [dict(task_type="workout_record", action="record", text=source,
                           fact_text="卧推50公斤3组8次", is_negated=False,
                           is_hypothetical=False, is_third_party=False,
                           history_requirement="none")],
            "safety": {"level": "normal", "evidence": "", "categories": [],
                       "reason": "scripted normal"},
        })
    assert value["tasks"][0]["fact_text"] == "卧推50公斤3组8次"
    assert value["validated_tasks"][0]["write_authorization"] == "authorized"
