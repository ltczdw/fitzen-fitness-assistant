import unittest

from app.services.agent.food_calculator import calculate_nutrition, format_food_resolution_message
from app.services.agent.food_parser import parse_food_message
from app.services.agent.workout_extractor import format_missing_workout_fields, preprocess_workout


class MissingParameterMessageTests(unittest.TestCase):
    def test_workout_missing_fields_are_explicitly_named(self):
        parsed = preprocess_workout("记录卧推")

        self.assertEqual(parsed.missing_fields, ("sets", "reps", "weight_kg"))
        message = format_missing_workout_fields(parsed.missing_fields)
        self.assertIn("组数", message)
        self.assertIn("每组次数", message)
        self.assertIn("重量", message)

    def test_empty_food_record_lists_required_fields(self):
        result = calculate_nutrition(
            "missing-params",
            parse_food_message("记录饮食"),
            raw_message="记录饮食",
            idempotency_key="missing-food-1",
        )

        self.assertEqual(result["status"], "needs_confirmation")
        self.assertIn("食物名称", result["message"])
        self.assertIn("重量", result["message"])
        self.assertIn("份量", result["message"])

    def test_unknown_food_identifies_the_name_field(self):
        result = calculate_nutrition(
            "missing-params",
            parse_food_message("记录饮食：一份不存在的食物"),
            raw_message="记录饮食：一份不存在的食物",
            idempotency_key="missing-food-2",
        )

        self.assertEqual(result["status"], "needs_confirmation")
        self.assertIn("可识别的食物名称", format_food_resolution_message(result["unresolved"]))


if __name__ == "__main__":
    unittest.main()
