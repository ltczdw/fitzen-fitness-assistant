from app.services.agent.intent import IntentResult
from app.services.rag.query import route_query


def test_model_safety_intent_selects_rehabilitation():
    result = route_query("脚踝扭伤后怎么康复", IntentResult("symptom_triage", .9, "caution", "question"))
    assert result.domain == "rehabilitation" and result.task == "safety_triage"
    assert result.risk_level == "caution"


def test_raw_technique_words_do_not_override_supplied_intent():
    result = route_query("深蹲正确动作怎么做", IntentResult("training_qa", .9, "normal", "question"))
    assert result.domain == "fitness_general" and result.intent == "training_qa"
