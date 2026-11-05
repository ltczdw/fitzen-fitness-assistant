import pytest
from app.services.agent.intent import classify_intent
from app.services.agent.semantics import request_understanding, validate_understanding, remember_understanding


@pytest.mark.parametrize("source,kind,action", [
    ("今天哑铃卧推四组，每组八下，十公斤", "training_qa", "question"),
    ("帮我记录，我做了哑铃卧推10kg*10,4组，哑铃弯举10kg*10,4组", "workout_record", "record"),
])
def test_numeric_notation_does_not_override_the_model_authorization(source, kind, action):
    token = request_understanding.set({})
    try:
        value = validate_understanding(source, {"tasks": [{"task_type": kind, "action": action, "text": source,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False}]})
        remember_understanding(source, value)
        result = classify_intent(source)
        assert result.name == kind and result.action == action
    finally:
        request_understanding.reset(token)
