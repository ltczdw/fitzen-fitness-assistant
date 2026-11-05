import unittest
from unittest.mock import patch

from app.services.agent.task_planner import plan_tasks
from app.services.nutrition_planner import build_nutrition_plan


def _tasks(message, kind, action):
    parsed = {"tasks": [{"task_type": kind, "action": action, "text": message,
                         "is_negated": False, "is_hypothetical": False,
                         "is_third_party": False}]}
    return [task["worker"] for task in plan_tasks(message, None, parsed)]


class NutritionSyncTests(unittest.TestCase):
    def test_nutrition_planning_language_routes_to_nutrition_worker(self):
        self.assertEqual(_tasks("根据最近饮食记录给我做饮食规划", "nutrition_advice", "question"), ["nutrition"])
        self.assertEqual(_tasks("记录饮食：一碗米饭，两个鸡蛋", "nutrition_record", "record"), ["nutrition"])

    def test_plan_exposes_synced_recent_intake(self):
        summary = {
            "days": 7,
            "entries": 2,
            "calories": 318.0,
            "protein_g": 16.5,
            "carbs_g": 39.95,
            "fat_g": 9.95,
            "fiber_g": 0.45,
            "daily_average": {"calories": 45.4, "protein_g": 2.4, "carbs_g": 5.7, "fat_g": 1.4, "fiber_g": 0.1},
            "food_log": {"entries": 2},
            "sync_source": "nutrition_logs + user_food_log",
            "target_ready": False,
        }
        with patch("app.services.nutrition_planner.get_profile", return_value={"goal": "保持健康"}), patch("app.services.nutrition_planner.synced_nutrition_summary", return_value=summary):
            plan = build_nutrition_plan("sync-user")
        self.assertEqual(plan["recent_intake"]["food_entries"], 2)
        self.assertEqual(plan["recent_intake"]["source"], "nutrition_logs + user_food_log")

