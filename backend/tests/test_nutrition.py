import unittest
from unittest.mock import patch
from app.db.storage import StorageUnavailableError

from app.services.nutrition_planner import build_nutrition_plan


class NutritionPlanTests(unittest.TestCase):
    def test_plan_has_meals_and_target_contract(self):
        summary = {key: 0 for key in ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g")}
        summary.update(daily_average={}, target_ready=False, food_log={"entries": 0})
        with patch("app.services.nutrition_planner.get_profile", return_value={}), patch("app.services.nutrition_planner.synced_nutrition_summary", return_value=summary):
            plan = build_nutrition_plan("nutrition-test-user")
        self.assertEqual(plan["user_id"], "nutrition-test-user")
        self.assertEqual(len(plan["meals"]), 4)
        self.assertIn("target_ready", plan["targets"])
        self.assertTrue(plan["guidance"])

    def test_unavailable_database_refuses_plan(self):
        with self.assertRaises(StorageUnavailableError):
            build_nutrition_plan("nutrition-test-user")


if __name__ == "__main__":
    unittest.main()
