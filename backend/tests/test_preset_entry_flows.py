import unittest
from datetime import date
from unittest.mock import patch

from app.api.workout_sessions import post_workout_session
from app.schemas.workout_session import WorkoutSessionPayload
from app.services.agent.food_calculator import calculate_nutrition
from app.services.agent.food_parser import FoodInput


class PresetEntryFlowTests(unittest.TestCase):
    def test_new_food_presets_calculate_and_reach_save_boundary(self):
        entries = (
            FoodInput(raw_text="熟红薯 200g", requested_name="熟红薯", quantity=1, unit="g", weight_g=200, portion_name=None),
            FoodInput(raw_text="瘦牛肉 150g", requested_name="瘦牛肉", quantity=1, unit="g", weight_g=150, portion_name=None),
        )
        with patch("app.services.agent.food_calculator.log_food_items", return_value=[101, 102]) as save:
            result = calculate_nutrition("preset-test", entries, raw_message="preset-test", idempotency_key="preset-test")

        self.assertEqual(result["status"], "calculated")
        self.assertEqual(result["totals"]["kcal"], 443.0)
        self.assertEqual(result["totals"]["protein_g"], 42.25)
        self.assertEqual(result["totals"]["carb_g"], 35.4)
        self.assertEqual(result["totals"]["fat_g"], 14.5)
        self.assertEqual(result["log_ids"], [101, 102])
        save.assert_called_once()

    def test_quick_exercise_payload_keeps_multiple_sets_and_completion(self):
        payload = WorkoutSessionPayload(
            user_id="preset-test",
            session_date=date(2026, 9, 11),
            title="快捷动作测试",
            exercises=[{
                "exercise_name": "高脚杯深蹲",
                "note": "",
                "sets": [
                    {"weight_kg": 12, "reps": 12, "completed": True, "note": ""},
                    {"weight_kg": 12, "reps": 10, "completed": False, "note": ""},
                ],
            }],
        )
        expected = {"id": 88, **payload.model_dump()}
        with patch("app.api.workout_sessions.create_session", return_value=expected) as save:
            result = post_workout_session(payload)

        self.assertEqual(result["exercises"][0]["exercise_name"], "高脚杯深蹲")
        self.assertEqual(len(result["exercises"][0]["sets"]), 2)
        self.assertFalse(result["exercises"][0]["sets"][1]["completed"])
        save.assert_called_once_with(payload.model_dump())


if __name__ == "__main__":
    unittest.main()
