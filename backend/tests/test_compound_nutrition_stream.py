import json
import unittest
from unittest.mock import AsyncMock, patch

from app.core.config import settings
from app.schemas.chat import ChatRequest
from app.services.agent.agent import FitnessAgent
from app.services.agent.intent import IntentResult


def legacy_memory():
    """Drive the unified runtime without touching the compact memory database."""
    return (
        patch.object(settings, "memory_read_backend", "legacy"),
        patch("app.services.agent.execution.memory_service.load",
              new=AsyncMock(return_value={"memory_backend": "test"})),
    )


class _FakeRag:
    def search(self, *_args, **_kwargs):
        return []


class _FakeStreamLLM:
    def configured(self):
        return True

    async def complete_structured(self, messages, schema, *, name=None,
                                  validator=None, **_kwargs):
        """Scripted shared understanding with the required source-bound safety."""
        query = ""
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            try:
                payload = json.loads(str(message.get("content") or ""))
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and isinstance(payload.get("query"), str):
                query = payload["query"]
                break
        record_text, advice_text = (query.split("，", 1) + [""])[:2] \
            if "，" in query else (query, "")
        flags = {"is_negated": False, "is_hypothetical": False,
                 "is_third_party": False}
        tasks = [{"task_type": "nutrition_record", "action": "record",
                  "text": record_text, **flags}]
        if advice_text:
            tasks.append({"task_type": "nutrition_advice", "action": "question",
                          "text": advice_text, **flags})
        raw = {
            "tasks": tasks,
            "safety": {"level": "normal", "evidence": "", "categories": [],
                       "reason": "scripted normal"},
        }
        value = schema.model_validate(raw)
        if validator is not None:
            validator(value)
        return value

    async def stream(self, _messages):
        yield "晚餐建议：选择鱼肉和蔬菜。"


class _FakeGraph:
    """Unified compound graph: route produces record+advice tasks and runs both."""

    def __init__(self, record_result):
        self.record_result = record_result
        self.rag = _FakeRag()
        self.llm = _FakeStreamLLM()
        self.planned_types: list[str] = []
        self.runs = 0

    async def parse_input(self, message):
        record_text, advice_text = message.split("，", 1)
        flags = {"is_negated": False, "is_hypothetical": False, "is_third_party": False}
        return {"tasks": [
            {"task_type": "nutrition_record", "action": "record", "text": record_text, **flags},
            {"task_type": "nutrition_advice", "action": "question", "text": advice_text, **flags},
        ]}

    async def run(self, *_args, planned_tasks=None, **_kwargs):
        self.runs += 1
        planned = list(planned_tasks or [])
        self.planned_types = [task.get("task_type") for task in planned]
        results = []
        for task in planned:
            if task.get("task_type") == "nutrition_record":
                results.append({
                    "task_id": task["id"], "task_type": "nutrition_record",
                    "status": self.record_result.get("status"),
                    "answer": self.record_result.get("answer"),
                    "calculation": self.record_result.get("calculation", {}),
                })
            elif task.get("task_type") == "nutrition_advice" and self.record_result.get("status") == "complete":
                results.append({
                    "task_id": task["id"], "task_type": "nutrition_advice",
                    "status": "complete", "answer": "晚餐建议：选择鱼肉和蔬菜。",
                    "citations": [], "model_used": True,
                })
        return {
            "worker_results": results,
            "final_answer": "\n\n".join(item["answer"] for item in results if item.get("answer")),
            "citations": [],
            "model_used": any(item.get("model_used") for item in results),
            "safety_level": "normal",
        }

    async def _load_context_node(self, _state):
        return {"profile": {}, "training_history": [], "nutrition_summary": {}}

    def _planning_prompt(self, *_args):
        return [{"role": "user", "content": "晚餐建议"}]

    @staticmethod
    def _nutrition_fallback(_state):
        return "建议保持均衡饮食。"


class CompoundNutritionStreamTests(unittest.IsolatedAsyncioTestCase):
    def test_detects_record_then_advice_without_extra_model_round_trip(self):
        intent = IntentResult("nutrition", 0.98, action="record")
        planned = ({"task_type": "nutrition_record"}, {"task_type": "nutrition_advice"})
        self.assertTrue(FitnessAgent._is_compound_nutrition_request(
            "把我记录今天中午吃了一碗饭和两个鸡蛋，今晚吃什么好",
            intent,
            planned,
        ))
        self.assertFalse(FitnessAgent._is_compound_nutrition_request("今晚吃什么好", IntentResult("nutrition", 0.9, action="question"), planned))
        # A record-only plan must not trigger the legacy second advice round trip.
        self.assertFalse(FitnessAgent._is_compound_nutrition_request(
            "把我记录今天中午吃了一碗饭",
            intent,
            ({"task_type": "nutrition_record"},),
        ))

    async def test_streams_record_result_before_dinner_advice(self):
        agent = FitnessAgent.__new__(FitnessAgent)
        agent.graph = _FakeGraph({"status": "complete", "answer": "已记录午餐。", "calculation": {}})
        legacy_helper = AsyncMock(side_effect=AssertionError("legacy second advice helper must not run"))
        agent._complete_nutrition_advice = legacy_helper
        request = ChatRequest(
            message="把我记录今天中午吃了一碗饭和两个鸡蛋，今晚吃什么好",
            user_id="stream-test",
            conversation_id="compound-nutrition",
        )
        memory_backend, load_memory = legacy_memory()
        with memory_backend, load_memory, patch.object(
            agent, "_persist_turn", new=AsyncMock()
        ), patch.object(agent, "_run_planner_shadow", new=AsyncMock()):
            events = [event async for event in agent.respond_stream(request)]

        token_text = "".join(event["data"].get("content", "") for event in events if event["event"] == "token")
        self.assertIn("已记录午餐", token_text)
        self.assertIn("晚餐建议", token_text)
        self.assertLess(token_text.index("已记录午餐"), token_text.index("晚餐建议"))
        self.assertEqual(token_text, "已记录午餐。\n\n晚餐建议：选择鱼肉和蔬菜。")
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(agent.graph.planned_types, ["nutrition_record", "nutrition_advice"])
        self.assertEqual(agent.graph.runs, 1)
        legacy_helper.assert_not_awaited()

    async def test_does_not_generate_advice_when_record_needs_confirmation(self):
        agent = FitnessAgent.__new__(FitnessAgent)
        agent.graph = _FakeGraph({"status": "needs_confirmation", "answer": "请确认食物重量。"})
        agent._build_nutrition_advice_messages = AsyncMock()
        agent._complete_nutrition_advice = AsyncMock(
            side_effect=AssertionError("advice must not run before a completed record"))
        request = ChatRequest(
            message="把我记录今天中午吃了一碗饭，今晚吃什么好",
            user_id="stream-test",
            conversation_id="compound-nutrition-confirm",
        )
        memory_backend, load_memory = legacy_memory()
        with memory_backend, load_memory, patch.object(
            agent, "_persist_turn", new=AsyncMock()
        ), patch.object(agent, "_run_planner_shadow", new=AsyncMock()):
            events = [event async for event in agent.respond_stream(request)]

        text = "".join(event["data"].get("content", "") for event in events if event["event"] == "token")
        self.assertIn("请确认食物重量", text)
        self.assertNotIn("晚餐建议", text)
        agent._build_nutrition_advice_messages.assert_not_awaited()
        agent._complete_nutrition_advice.assert_not_awaited()
        self.assertEqual(events[-1]["event"], "done")

    async def test_non_stream_response_also_combines_record_and_advice(self):
        agent = FitnessAgent.__new__(FitnessAgent)
        agent.graph = _FakeGraph({"status": "complete", "answer": "已记录午餐。", "calculation": {}})
        legacy_helper = AsyncMock(side_effect=AssertionError("legacy second advice helper must not run"))
        request = ChatRequest(
            message="把我记录今天中午吃了一碗饭和两个鸡蛋，今晚吃什么好",
            user_id="sync-test",
            conversation_id="compound-nutrition-sync",
        )
        memory_backend, load_memory = legacy_memory()
        with memory_backend, load_memory, patch.object(
            agent,
            "_complete_nutrition_advice",
            new=legacy_helper,
        ), patch.object(agent, "_persist_turn", new=AsyncMock()), patch.object(
            agent, "_run_planner_shadow", new=AsyncMock()
        ):
            response = await agent.respond(request)

        self.assertEqual(response.answer, "已记录午餐。\n\n晚餐建议：选择鱼肉和蔬菜。")
        self.assertIn("已记录午餐", response.answer)
        self.assertIn("晚餐建议", response.answer)
        self.assertTrue(response.model_used)
        self.assertEqual(agent.graph.planned_types, ["nutrition_record", "nutrition_advice"])
        self.assertEqual(agent.graph.runs, 1)
        legacy_helper.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
