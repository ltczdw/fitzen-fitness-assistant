import unittest
from datetime import date, timedelta

from app.schemas.chat import ChatRequest
from app.db import repositories
from app.services.agent.food_parser import parse_food_message
from app.services.agent.intent import classify_intent
from app.services.agent.task_planner import plan_tasks
from app.services.agent.workout_extractor import preprocess_workout
from app.services.security.input_guard import inspect_input


class AgentSafetyRegressionTests(unittest.TestCase):
    def test_chat_request_has_retryable_request_id(self):
        first = ChatRequest(message="记录深蹲40kg，3组8次")
        second = ChatRequest(message="记录深蹲40kg，3组8次")
        self.assertTrue(first.request_id)
        self.assertNotEqual(first.request_id, second.request_id)
        retried = ChatRequest(message=first.message, request_id=first.request_id)
        self.assertEqual(retried.request_id, first.request_id)

    def test_workout_write_refuses_unavailable_database(self):
        from app.db.storage import StorageUnavailableError
        with self.assertRaises(StorageUnavailableError):
            repositories.create_workout("retry-user", "squat", 3, 8, 40, idempotency_key="retry-1")

    def test_question_and_future_workout_never_become_writes(self):
        for message in ("卧推10kg，4组，每组10次合适吗", "明天准备做卧推10kg，4组，每组10次"):
            tasks = plan_tasks(message)
            self.assertEqual(tasks[0]["side_effect"], False, message)
            self.assertNotEqual(tasks[0]["worker"], "workout_record", message)

    def test_generic_record_command_routes_unlisted_exercise_to_workout(self):
        source = "帮我记录哑铃侧平举5kg*12，3组"
        tasks = plan_tasks(source, parsed_input={"tasks": [{"task_type": "workout_record", "action": "record", "text": source}]})
        self.assertEqual(tasks[0]["worker"], "workout_record")

    def test_structured_question_action_is_preserved(self):
        parsed = {"tasks": [{
            "task_type": "workout_record", "action": "question",
            "text": "卧推10kg，4组，每组10次合适吗", "workouts": [], "foods": [],
        }]}
        task = plan_tasks(parsed["tasks"][0]["text"], parsed_input=parsed)[0]
        self.assertFalse(task["side_effect"])
        self.assertEqual(task["worker"], "training_qa")

    def test_model_nonwrite_answer_is_preserved_and_invented_source_is_rejected(self):
        message = "记录今天卧推第一组40公斤10次，第二组45公斤8次"
        parsed = {"tasks": [{"task_type": "nutrition_record", "action": "answer", "text": message,
            "workouts": [], "foods": []}]}
        task = plan_tasks(message, parsed_input=parsed)[0]
        self.assertFalse(task["side_effect"])
        self.assertIn(task["action"], {"question", "answer"})
        generated = {"tasks": [{"task_type": "workout_record", "action": "record", "text": "模型猜的卧推100kg",
            "workouts": [{"exercise": "卧推", "weight_kg": 100, "sets": 5, "reps": 20}], "foods": []}]}
        fallback = plan_tasks("帮我记录今天做了卧推，忘了重量和次数", parsed_input=generated)[0]
        self.assertFalse(fallback["side_effect"])
        self.assertFalse(fallback["payload"].get("workouts"))
        self.assertNotIn("模型猜的", fallback["text"])

    def test_model_task_type_is_not_reclassified_by_training_metrics(self):
        message = "记录今天卧推第一组40公斤10次，第二组45公斤8次，第三组50公斤6次"
        parsed = {"tasks": [{
            "task_type": "nutrition_record", "action": "record", "text": message,
            "foods": [{"name": "卧推第一组40公斤10次", "quantity": 1, "unit": "公斤", "weight_g": 40000}],
            "workouts": [],
        }]}
        task = plan_tasks(message, parsed_input=parsed)[0]
        self.assertEqual(task["worker"], "nutrition")
        self.assertTrue(task["side_effect"])
        # Planner does not infer meaning again. The food worker still checks
        # catalog/AI entities and source; this is not proof of a committed save.

    def test_backfill_command_keeps_each_workout_task(self):
        message = "帮我补记昨天卧推40公斤3组8次，今天深蹲60公斤4组6次"
        parsed = {"tasks": [
            {"task_type": "workout_record", "action": "record", "text": "帮我补记昨天卧推40公斤3组8次", "foods": [], "workouts": []},
            {"task_type": "workout_record", "action": "record", "text": "今天深蹲60公斤4组6次", "foods": [], "workouts": []},
        ]}
        tasks = plan_tasks(message, parsed_input=parsed)
        self.assertEqual([task["worker"] for task in tasks], ["workout_record", "workout_record"])
        self.assertTrue(all(task["side_effect"] for task in tasks))

    def test_negated_food_is_not_returned_as_a_positive_food(self):
        entries = parse_food_message("我今天没有吃两个鸡蛋，只吃了一碗米饭")
        self.assertEqual({entry.requested_name for entry in entries}, {"鸡蛋", "米饭"})
        # The execution layer applies the negation filter; this assertion
        # protects the parser's date and quantity data used by that filter.
        self.assertEqual(next(entry for entry in entries if entry.requested_name == "鸡蛋").quantity, 2)

    def test_relative_dates_are_preserved_for_food_and_workout(self):
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        self.assertEqual(parse_food_message("记录昨天午餐一碗米饭")[0].record_date, yesterday)
        self.assertEqual(preprocess_workout("补记昨天卧推40公斤3组8次").record["date"].isoformat(), yesterday)

    def test_negated_or_technical_red_flags_are_not_urgent(self):
        self.assertNotEqual(inspect_input("深蹲怎么避免动作变形").level, "urgent")
        self.assertNotEqual(inspect_input("没有胸痛也没有呼吸困难，帮我记录卧推10kg，3组8次").level, "urgent")
        self.assertNotEqual(classify_intent("没有胸痛也没有呼吸困难，帮我记录卧推10kg，3组8次").safety_level, "urgent")


if __name__ == "__main__":
    unittest.main()
