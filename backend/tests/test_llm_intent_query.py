import unittest
from unittest.mock import AsyncMock, patch

from app.services.agent import intent
from app.services.agent.semantics import RequestUnderstanding, request_understanding
from app.services.rag import query


class TestLlmIntentAndQuery(unittest.TestCase):
    def setUp(self):
        self.token = request_understanding.set({})

    def tearDown(self):
        request_understanding.reset(self.token)

    def test_intent_llm_success_is_shared_instead_of_reclassified(self):
        source = "我想把下周训练改成三分化"
        raw = RequestUnderstanding.model_validate({"tasks": [{"task_type": "training_plan", "action": "answer", "text": source,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            "history_requirement": "none"}]})
        with patch.object(intent._LLM_CLIENT, "configured", return_value=True), patch.object(intent._LLM_CLIENT, "complete_structured", new=AsyncMock(return_value=raw)) as complete:
            result = intent.classify_intent_llm(source)
            self.assertEqual(result.name, "plan_adjust")
            self.assertEqual(result.action, "answer")
            self.assertEqual(intent.classify_intent(source), result)
            complete.assert_awaited_once()

    def test_red_flag_gate_wins_before_llm(self):
        from app.services.agent.semantics import remember_safety
        source = "出现胸痛，立即停止训练"
        remember_safety(source, {"level": "urgent", "evidence": source,
                                 "categories": ["current_symptom"], "reason": "scripted"})
        with patch.object(intent._LLM_CLIENT, "complete_structured", new=AsyncMock(side_effect=AssertionError("emergency must not call model"))) as complete:
            result = intent.classify_intent(source)
        self.assertEqual(result.name, "symptom_triage")
        self.assertEqual(result.safety_level, "urgent")
        complete.assert_not_awaited()

    def test_invalid_model_never_falls_back_to_a_record_guess(self):
        with patch.object(intent._LLM_CLIENT, "configured", return_value=True), patch.object(intent._LLM_CLIENT, "complete_structured", new=AsyncMock(side_effect=ValueError("invalid"))):
            result = intent.classify_intent("记录深蹲40kg，3组，每组8次")
        self.assertEqual(result.name, "training_qa")
        self.assertEqual(result.action, "question")

    def test_query_llm_preserves_numeric_entities(self):
        route = query.QueryRoute("workout_record", "exercise_science", "training_guidance", "normal", 0.9, ("运动学",))
        with patch.object(query._QUERY_LLM_CLIENT, "configured", return_value=True), patch.object(query._QUERY_LLM_CLIENT, "complete", new=AsyncMock(return_value="深蹲动作技术与负荷控制 40kg 3组 8次")):
            result = query.rewrite_query("深蹲40kg 3组8次怎么做", route)
        self.assertIn("40kg", result)
        self.assertIn("3组", result)
        self.assertIn("8次", result)

    def test_query_invalid_llm_falls_back_to_template(self):
        route = query.QueryRoute("training_qa", "fitness_general", "fitness_qa", "normal", 0.65, ("健身运动",))
        with patch.object(query._QUERY_LLM_CLIENT, "configured", return_value=True), patch.object(query._QUERY_LLM_CLIENT, "complete", new=AsyncMock(return_value="查询词：")):
            result = query.rewrite_query("怎么安排训练", route)
        self.assertEqual(result, query.rewrite_query_template("怎么安排训练", route))


if __name__ == "__main__":
    unittest.main()
