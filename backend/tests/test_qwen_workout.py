import json
import unittest

from app.services.agent.deepseek_workout import DeepSeekWorkoutExtractor


class _FakeLLM:
    def configured(self):
        return True

    async def complete(self, messages, temperature=0.0):
        return json.dumps({
            "records": [
                {"exercise": "哑铃卧推", "weight_kg": 10, "sets": 4, "reps": 10, "inferred_fields": []},
                {"exercise": "哑铃弯举", "weight_kg": 10, "sets": 4, "reps": 10, "inferred_fields": []},
            ]
        }, ensure_ascii=False)


class QwenWorkoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_returns_all_exercises_in_one_request(self):
        records = await DeepSeekWorkoutExtractor(_FakeLLM()).extract(
            "帮我记录哑铃卧推10kg*10,4组，哑铃弯举10kg*10,4组"
        )

        self.assertEqual([item["exercise"] for item in records], ["哑铃卧推", "哑铃弯举"])
        self.assertEqual([(item["weight_kg"], item["sets"], item["reps"]) for item in records], [(10, 4, 10), (10, 4, 10)])


if __name__ == "__main__":
    unittest.main()
