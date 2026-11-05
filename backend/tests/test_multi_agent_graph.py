import unittest

from app.services.agent.intent import classify_intent
from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph, _tasks_for_message
from app.services.agent.semantics import remember_understanding, request_understanding


def _task(task_type, worker, action, text, depends_on=()):
    return {"task_type": task_type, "worker": worker, "action": action, "text": text,
            "is_negated": False, "is_hypothetical": False, "is_third_party": False,
            "side_effect": action in ("record", "confirm"), "depends_on": list(depends_on)}


def _remember(source, *tasks):
    remember_understanding(source, {"validated_tasks": list(tasks)})


class MultiAgentGraphTests(unittest.TestCase):
    def setUp(self):
        # Shared model understanding is request-scoped; isolate it per test so a
        # cached decision never leaks into an unrelated routing assertion.
        self._token = request_understanding.set({})

    def tearDown(self):
        request_understanding.reset(self._token)

    def test_has_four_single_responsibility_workers(self):
        graph = FitnessMultiAgentGraph()
        self.assertEqual(set(graph.workers), {"workout_record", "plan_adjust", "nutrition", "training_qa"})

    def test_routes_food_record_to_nutrition_worker(self):
        message = "记录饮食：一碗米饭，两个鸡蛋"
        _remember(message, _task("nutrition_record", "nutrition", "record", message))
        self.assertEqual(_tasks_for_message(message), ["nutrition"])

    def test_routes_multi_intent_in_finite_order(self):
        message = "记录训练4组8次，然后给我做饮食计划"
        _remember(message,
                  _task("workout_record", "workout_record", "record", "记录训练4组8次"),
                  _task("nutrition_advice", "nutrition", "question", "给我做饮食计划"))
        self.assertEqual(_tasks_for_message(message), ["workout_record", "nutrition"])

    def test_food_capability_question_is_not_record_action(self):
        message = "为什么你这个保存不了，我吃了一碗饭你能保存"
        _remember(message, _task("nutrition_advice", "nutrition", "question", message))
        result = classify_intent(message)
        self.assertEqual(result.name, "nutrition")
        self.assertEqual(result.action, "question")

    def test_explicit_food_record_is_record_action(self):
        message = "我吃了一碗饭，帮我记录"
        _remember(message, _task("nutrition_record", "nutrition", "record", message))
        result = classify_intent(message)
        self.assertEqual(result.name, "nutrition")
        self.assertEqual(result.action, "record")

    def test_food_confirmation_is_confirm_action(self):
        message = "确认保存"
        _remember(message, _task("nutrition_record", "nutrition", "confirm", message))
        result = classify_intent(message)
        self.assertEqual(result.name, "nutrition")
        self.assertEqual(result.action, "confirm")


if __name__ == "__main__":
    unittest.main()
