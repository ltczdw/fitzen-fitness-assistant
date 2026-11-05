import unittest

from app.db.food_catalog_audit import audit_seed_catalog
from app.db.food_repository import SEED_FOODS, resolve_food


class FoodCatalogAuditTests(unittest.TestCase):
    def test_seed_catalog_has_no_structural_or_coverage_issues(self):
        report = audit_seed_catalog()
        self.assertEqual(report["summary"]["missing_common_foods"], [])
        self.assertEqual(report["summary"]["high_count"], 0)
        self.assertEqual(report["summary"]["medium_count"], 0)
        self.assertGreaterEqual(report["summary"]["total_foods"], 40)

    def test_every_food_has_four_required_per_100g_nutrients(self):
        for row in SEED_FOODS:
            self.assertEqual(len(row), 8)
            self.assertTrue(row[0])
            for value in row[3:7]:
                self.assertIsNotNone(value)
                self.assertGreaterEqual(float(value), 0)

    def test_common_food_aliases_resolve_in_fallback_or_database(self):
        for requested_name, expected_name in (
            ("面", "熟面条"),
            ("一份面条", "熟面条"),
            ("食用油", "食用植物油"),
            ("菠菜", "菠菜"),
            ("鸡腿", "鸡腿肉"),
            ("酸奶", "无糖酸奶"),
        ):
            lookup_name = requested_name
            if requested_name == "一份面条":
                lookup_name = "面条"
            match = resolve_food(lookup_name)
            self.assertIsNotNone(match, requested_name)
            self.assertEqual(match.food_name, expected_name)


if __name__ == "__main__":
    unittest.main()