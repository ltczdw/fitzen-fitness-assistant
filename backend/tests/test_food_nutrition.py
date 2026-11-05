import unittest
from unittest.mock import patch

from app.services.agent.food_calculator import calculate_nutrition, calculate_nutrition_batch
from app.services.agent.food_parser import parse_food_message


class FoodNutritionTests(unittest.TestCase):
    def test_parses_portions_and_names(self):
        entries = parse_food_message("记录饮食：一碗米饭，两个鸡蛋")
        self.assertEqual([entry.requested_name for entry in entries], ["米饭", "鸡蛋"])
        self.assertEqual([entry.portion_name for entry in entries], ["碗", "个"])

    def test_calculator_is_deterministic_and_logs(self):
        entries = parse_food_message("记录饮食：一碗米饭，两个鸡蛋")
        with patch("app.services.agent.food_calculator.log_food_items", return_value=[11, 12]):
            result = calculate_nutrition("test", entries, raw_message="记录饮食：一碗米饭，两个鸡蛋", idempotency_key="unit-food")
        self.assertEqual(result["status"], "calculated")
        self.assertEqual(result["totals"]["kcal"], 318.0)
        self.assertEqual(result["totals"]["protein_g"], 16.5)

    def test_parses_evening_phrase_and_keeps_both_foods(self):
        entries = parse_food_message("我晚上吃了一碗饭，两个鸡蛋")
        self.assertEqual([entry.requested_name for entry in entries], ["饭", "鸡蛋"])
        self.assertEqual([entry.quantity for entry in entries], [1.0, 2.0])
        self.assertEqual([entry.meal_type for entry in entries], ["晚餐", "晚餐"])

    def test_unknown_food_does_not_calculate(self):
        entries = parse_food_message("记录饮食：一份不存在的食物")
        result = calculate_nutrition("test", entries, raw_message="记录饮食：一份不存在的食物", idempotency_key="unit-unknown")
        self.assertEqual(result["status"], "needs_confirmation")
        self.assertEqual(result["log_ids"] if "log_ids" in result else [], [])

    def test_parses_two_meals_with_independent_meal_types(self):
        entries = parse_food_message("我中午吃了一碗米饭，晚上吃了两个鸡蛋")
        self.assertEqual([entry.requested_name for entry in entries], ["米饭", "鸡蛋"])
        self.assertEqual([entry.meal_type for entry in entries], ["午餐", "晚餐"])
        self.assertEqual([entry.quantity for entry in entries], [1.0, 2.0])

    def test_half_bowl_uses_half_of_default_weight(self):
        entries = parse_food_message("半碗米饭")
        self.assertEqual(entries[0].quantity, 0.5)
        with patch("app.services.agent.food_calculator.log_food_items", return_value=[1]):
            result = calculate_nutrition("test", entries, raw_message="半碗米饭", idempotency_key="unit-half")
        self.assertEqual(result["status"], "calculated")
        self.assertEqual(result["items"][0]["weight_g"], 75.0)

    def test_porridge_has_alias_and_default_bowl_weight(self):
        entries = parse_food_message("记录早餐一碗粥")
        with patch("app.services.agent.food_calculator.log_food_items", return_value=[1]):
            result = calculate_nutrition("test", entries, raw_message="记录早餐一碗粥", idempotency_key="unit-porridge")
        self.assertEqual(result["status"], "calculated")
        self.assertEqual(result["items"][0]["food_name"], "白粥")
        self.assertEqual(result["items"][0]["weight_g"], 250.0)

    def test_batch_validates_all_meals_before_writing(self):
        entries = parse_food_message("中午一碗米饭，晚上两个鸡蛋")
        groups = [{"meal_type": meal_type, "entries": [entry.__dict__ for entry in entries if entry.meal_type == meal_type]} for meal_type in ("午餐", "晚餐")]
        with patch("app.services.agent.food_calculator.log_food_items") as save:
            result = calculate_nutrition_batch("test", groups, raw_message="中午一碗米饭，晚上两个鸡蛋", idempotency_key="unit-batch")
        self.assertEqual(result["status"], "calculated")
        self.assertEqual([item["meal_type"] for item in result["calculations"]], ["午餐", "晚餐"])
        self.assertEqual(save.call_count, 2)


if __name__ == "__main__":
    unittest.main()
