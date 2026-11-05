import unittest
from unittest.mock import AsyncMock, patch

from app.services.agent.intent import IntentResult
from app.services.agent.multi_agent_graph_fixed import FitnessMultiAgentGraph
from app.services.rag import query
from app.services.rag.pipeline import HybridRAGPipeline


class IntentReuseTests(unittest.TestCase):
    def test_supplied_intent_skips_classifier_and_retains_metadata(self):
        with patch.object(query, "classify_intent", side_effect=AssertionError("duplicate classification")):
            route = query.route_query("蛋白质怎么补充", IntentResult("nutrition", .87, "caution"))
        self.assertEqual((route.domain, route.confidence, route.risk_level), ("nutrition", .87, "caution"))

    def test_standalone_rag_still_classifies_once(self):
        with patch.object(query, "classify_intent", return_value=IntentResult("nutrition", .9)) as classifier:
            query.route_query("晚饭怎么搭配")
        classifier.assert_called_once_with("晚饭怎么搭配")

    def test_source_keywords_do_not_override_the_supplied_model_intent(self):
        with patch.object(query, "classify_intent", side_effect=AssertionError("duplicate classification")):
            route = query.route_query("脚踝扭伤后怎么康复", IntentResult("training_qa", .9))
        self.assertEqual(route.domain, "fitness_general")
        urgent = query.route_query("脚踝扭伤后怎么康复", IntentResult("symptom_triage", .9, "caution"))
        self.assertEqual(urgent.domain, "rehabilitation")


class WorkerIntentReuseTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_read_workers_pass_current_task_intent_to_real_rag(self):
        graph = FitnessMultiAgentGraph()
        cases = [("training_qa", "training_qa", "training_qa"),
                 ("plan_adjust", "training_plan", "plan_adjust"),
                 ("nutrition", "nutrition_advice", "nutrition")]
        with patch.object(query, "classify_intent", side_effect=AssertionError("duplicate classification")), \
             patch("app.services.agent.task_information.check_plan_information", new=AsyncMock(return_value=({}, None))), \
             patch("app.services.rag.pipeline.rewrite_query", side_effect=lambda text, route: text) as rewrite, \
             patch.object(HybridRAGPipeline, "_database_bm25_search", return_value=[]), \
             patch.object(HybridRAGPipeline, "_database_vector_search", return_value=[]), \
             patch.object(graph, "_complete_or_fallback_with_mcp", new=AsyncMock(return_value=("测试回答", True))):
            for worker, task_type, expected in cases:
                with self.subTest(worker=worker):
                    await graph.workers[worker].ainvoke({
                        "user_id": "test", "raw_message": "记录晚饭，然后解释热身，制定训练计划",
                        "task_text": "解释热身的作用", "task_type": task_type, "task_action": "question",
                        "intent_result": {"name": "nutrition", "confidence": .88, "safety_level": "normal", "action": "question"},
                    })
                    self.assertEqual(rewrite.call_args.args[1].intent, expected)
            self.assertEqual(rewrite.call_count, 3)

    async def test_urgent_upstream_is_not_downgraded(self):
        result = FitnessMultiAgentGraph._retrieval_intent({"task_type": "nutrition_advice", "intent_result": {"safety_level": "urgent"}}, "nutrition")
        self.assertEqual((result.name, result.safety_level), ("symptom_triage", "urgent"))
