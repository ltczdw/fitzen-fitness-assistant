import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from app.api.nutrition import FoodMealUpdate, ManualFoodMeal, post_food_meal, put_food_meal
from app.db.food_log_mutations import update_food_meal


class MealTimeEditTests(unittest.TestCase):
    def test_create_route_forwards_minute_precise_timestamp(self):
        payload = ManualFoodMeal(
            record_time="2026-09-23T12:45:37+08:00",
            meal_type="午餐",
            foods=[{"food_name": "米饭", "weight_g": 100}],
        )
        with patch("app.api.nutrition.calculate_manual_nutrition", return_value={"status": "calculated"}) as calculate:
            post_food_meal(payload, "test-idempotency-key")
        value = calculate.call_args.kwargs["record_time"]
        self.assertEqual(value.isoformat(), "2026-09-23T12:45:00+08:00")

    def test_route_forwards_timestamp_with_timezone(self):
        payload = FoodMealUpdate(record_time="2026-09-23T12:45:00+08:00", meal_type="午餐")
        with patch("app.api.nutrition.update_food_meal", return_value={"updated": 1}) as update:
            put_food_meal("meal-1", payload, "test-user")
        self.assertEqual(update.call_args.kwargs["record_time"].isoformat(), "2026-09-23T12:45:00+08:00")

    def test_time_update_is_scoped_and_old_clients_keep_time(self):
        row = dict(id=1, food_id=None, food_name="synthetic", match_method="user_custom", weight_g=100, meal_type="午餐", record_time=datetime.now(timezone.utc),
                   energy_kcal=100, protein_g=10, carb_g=10, fat_g=2, fiber_g=1)
        for new_time in (None, datetime(2026, 9, 23, 4, 45, tzinfo=timezone.utc)):
            with self.subTest(new_time=new_time):
                session = MagicMock()
                session.execute.return_value.mappings.return_value.all.return_value = [row]
                factory = MagicMock()
                factory.return_value.__enter__.return_value = session
                with patch("app.db.food_log_mutations.db_session.DB_READY", True), \
                     patch("app.db.food_log_mutations.db_session.SessionLocal", factory), \
                     patch("app.db.food_log_mutations.ensure_food_schema", return_value=True):
                    result = update_food_meal("test-user", "meal-1", record_time=new_time)
                self.assertEqual(result["updated"], 1)
                writes = [call for call in session.execute.call_args_list if "SET record_time=" in str(call.args[0])]
                self.assertEqual(len(writes), 0 if new_time is None else 1)
                if writes:
                    self.assertIn("user_id=:user_id AND meal_group_id=:group_id", str(writes[0].args[0]))
                    self.assertEqual(writes[0].args[1], {"record_time": new_time, "user_id": "test-user", "group_id": "meal-1"})
                session.commit.assert_called_once()

    def test_existing_meal_accepts_ai_food_with_shared_meal_time(self):
        original_time = datetime(2026, 9, 23, 4, 45, tzinfo=timezone.utc)
        row = dict(id=1, food_id=None, food_name="synthetic", match_method="user_custom", weight_g=100, meal_type="午餐", record_time=original_time,
                   energy_kcal=116, protein_g=2.6, carb_g=25.9, fat_g=0.3, fiber_g=0.3)
        session = MagicMock()
        session.execute.return_value.mappings.return_value.all.return_value = [row]
        factory = MagicMock()
        factory.return_value.__enter__.return_value = session
        items = [
            {"id": 1, "weight_g": 100},
            {"food_name": "AI 鸡肉卷", "weight_g": 80, "source": "ai",
             "nutrition_per_100g": {"kcal": 250, "protein_g": 18, "carbs_g": 20, "fat_g": 10, "fiber_g": 2}},
        ]
        with patch("app.db.food_log_mutations.db_session.DB_READY", True),              patch("app.db.food_log_mutations.db_session.SessionLocal", factory),              patch("app.db.food_log_mutations.ensure_food_schema", return_value=True):
            result = update_food_meal("test-user", "meal-1", items=items)
        inserts = [call for call in session.execute.call_args_list if "INSERT INTO user_food_log" in str(call.args[0])]
        self.assertEqual(result["updated"], 2)
        self.assertEqual(len(inserts), 1)
        params = inserts[0].args[1][0]
        self.assertEqual(params["record_time"], original_time)
        self.assertTrue(params["estimated"])
        self.assertEqual(params["food_id"], None)
        self.assertEqual(params["kcal"], 200.0)

    def test_missing_or_other_users_meal_is_not_changed(self):
        session = MagicMock()
        session.execute.return_value.mappings.return_value.all.return_value = []
        factory = MagicMock()
        factory.return_value.__enter__.return_value = session
        with patch("app.db.food_log_mutations.db_session.DB_READY", True), \
             patch("app.db.food_log_mutations.db_session.SessionLocal", factory), \
             patch("app.db.food_log_mutations.ensure_food_schema", return_value=True):
            self.assertIsNone(update_food_meal("other-user", "meal-1", record_time=datetime.now(timezone.utc)))
        session.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()

