import unittest
from unittest.mock import patch

from app.services.agent.food_calculator import calculate_manual_nutrition
from app.services.nutrition_ai import estimate_food_nutrition


class FakeLLM:
    def __init__(self, response):
        self.response = response

    async def complete(self, messages, temperature=0.0):
        return self.response


class NutritionSourceTests(unittest.TestCase):
    def test_custom_and_local_foods_share_one_meal_payload(self):
        foods = [
            {"food_name": "自制鸡肉卷", "weight_g": 150, "source": "custom",
             "nutrition_per_100g": {"kcal": 200, "protein_g": 20, "carbs_g": 15, "fat_g": 6, "fiber_g": 2}},
            {"food_name": "鸡蛋", "weight_g": 50, "source": "local", "nutrition_per_100g": None},
        ]
        with patch("app.services.agent.food_calculator.log_food_items", return_value=[1, 2]) as save:
            result = calculate_manual_nutrition("test", foods, idempotency_key="mixed-meal", meal_group_id="meal-1", meal_type="午餐")
        self.assertEqual(result["status"], "calculated")
        self.assertEqual(result["items"][0]["food_id"], None)
        self.assertEqual(result["items"][0]["energy_kcal"], 300.0)
        self.assertEqual(result["items"][1]["food_name"], "鸡蛋")
        self.assertEqual(save.call_count, 1)

    def test_ai_food_is_marked_as_estimated(self):
        foods = [{"food_name": "AI 牛肉卷", "weight_g": 80, "source": "ai",
                  "nutrition_per_100g": {"kcal": 250, "protein_g": 18, "carbs_g": 20, "fat_g": 10, "fiber_g": 2}}]
        with patch("app.services.agent.food_calculator.log_food_items", return_value=[1]):
            result = calculate_manual_nutrition("test", foods, idempotency_key="ai-meal", meal_group_id="meal-ai")
        self.assertTrue(result["items"][0]["is_estimated"])
        self.assertEqual(result["items"][0]["match_method"], "ai_estimate")


class NutritionAiTests(unittest.IsolatedAsyncioTestCase):
    async def test_ai_json_is_validated_and_normalized(self):
        raw = """{"food_name":"低糖蛋白棒","preparation_state":"即食","kcal":320,"protein_g":30,"carbs_g":35,"fat_g":8,"fiber_g":10,"notes":"按同类产品估算","confidence":"medium"}"""
        result = await estimate_food_nutrition("低糖蛋白棒", FakeLLM(raw))
        self.assertEqual(result["food_name"], "低糖蛋白棒")
        self.assertEqual(result["per_100g"]["protein_g"], 30.0)
        self.assertEqual(result["source"], "ai_estimate")

    async def test_invalid_ai_payload_is_rejected(self):
        raw = """{"food_name":"异常食物","kcal":9999,"protein_g":0,"carbs_g":0,"fat_g":0}"""
        self.assertIsNone(await estimate_food_nutrition("异常食物", FakeLLM(raw)))


if __name__ == "__main__":
    unittest.main()
