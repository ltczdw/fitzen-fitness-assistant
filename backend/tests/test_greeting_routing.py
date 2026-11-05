from unittest import TestCase

from app.services.agent.intent import IntentResult, is_greeting
from app.services.agent.task_planner import plan_tasks


class GreetingRoutingTests(TestCase):
    def test_standalone_greeting_has_no_business_task(self):
        for message in ("你好", "你好呀", "您好！", "hi"):
            self.assertTrue(is_greeting(message))
            self.assertTrue(all(not t["side_effect"] for t in plan_tasks(message)))

    def test_greeting_with_business_question_is_not_collapsed(self):
        self.assertFalse(is_greeting("你好，我想问热量"))
        tasks = plan_tasks("你好，我想问热量", parsed_input={"tasks": [{"task_type": "nutrition_advice", "text": "你好，我想问热量", "action": "question"}]})
        self.assertEqual(tasks[0]["task_type"], "nutrition_advice")

    def test_food_record_still_routes_to_record(self):
        tasks = plan_tasks("记录两个鸡蛋", parsed_input={"tasks": [{"task_type": "nutrition_record", "text": "记录两个鸡蛋", "action": "record"}]})
        self.assertEqual(tasks[0]["task_type"], "nutrition_record")
        self.assertTrue(tasks[0]["side_effect"])
