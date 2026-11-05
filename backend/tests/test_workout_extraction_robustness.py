import asyncio
import unittest

from app.services.agent.workout_extractor import preprocess_workout, split_workout_messages
from app.services.agent.workout_graph import WorkoutExtractionWorkflow


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def configured(self):
        return True

    async def extract_workout_json(self, message, preprocessed, enhanced=False):
        self.calls += 1
        return self.outputs.pop(0) if self.outputs else None


class WorkoutExtractionRobustnessTests(unittest.TestCase):
    def test_normalizes_chinese_numbers_and_units(self):
        result = preprocess_workout("今天哑铃卧推四组，每组八下，十公斤")

        self.assertEqual(result.status, "valid")
        self.assertEqual(result.record["sets"], 4)
        self.assertEqual(result.record["reps"], 8)
        self.assertEqual(result.record["weight_kg"], 10)

    def test_supports_x_notation_and_bodyweight_default(self):
        weighted = preprocess_workout("卧推10公斤4x8")
        bodyweight = preprocess_workout("今天做俯卧撑4组每组12")

        self.assertEqual((weighted.record["sets"], weighted.record["reps"]), (4, 8))
        self.assertEqual((bodyweight.record["sets"], bodyweight.record["reps"], bodyweight.record["weight_kg"]), (4, 12, 0.0))

    def test_supports_weight_times_reps_and_multiple_exercises(self):
        message = "帮我记录，我做了哑铃卧推10kg*10,4组，哑铃弯举10kg*10,4组"
        segments = split_workout_messages(message)

        self.assertEqual(len(segments), 2)
        parsed = [preprocess_workout(segment) for segment in segments]
        self.assertEqual([item.record["exercise"] for item in parsed], ["哑铃卧推", "哑铃弯举"])
        self.assertEqual([(item.record["sets"], item.record["reps"], item.record["weight_kg"]) for item in parsed], [(4, 10, 10.0), (4, 10, 10.0)])

    def test_llm_has_only_one_enhanced_retry(self):
        llm = FakeLLM([
            {"exercise": "卧推", "weight_kg": 10, "sets": 4, "reps": None, "inferred_fields": []},
            {"exercise": "卧推", "weight_kg": 10, "sets": 4, "reps": 8, "inferred_fields": ["reps"]},
        ])
        workflow = WorkoutExtractionWorkflow(llm)
        result = asyncio.run(workflow.run("记录某个动作10kg，4组，每组8"))

        self.assertEqual(llm.calls, 2)
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(result["record"]["reps"], 8)

    def test_llm_failure_stops_after_one_retry(self):
        llm = FakeLLM([None, None, {"reps": 8}])
        workflow = WorkoutExtractionWorkflow(llm)
        result = asyncio.run(workflow.run("记录一个动作"))

        self.assertEqual(llm.calls, 2)
        self.assertEqual(result["retry_count"], 1)
        self.assertIsNone(result.get("record"))


if __name__ == "__main__":
    unittest.main()
