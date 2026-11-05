import unittest

from app.services.agent.agent import FitnessAgent


class WorkoutParserInferenceTests(unittest.TestCase):
    def test_infers_reps_from_each_set_context(self):
        parsed = FitnessAgent._parse_workout("记录今天的训练，我做了4组，每组8的10kg哑铃卧推")

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["exercise"], "哑铃卧推")
        self.assertEqual(parsed["sets"], 4)
        self.assertEqual(parsed["reps"], 8)
        self.assertEqual(parsed["weight_kg"], 10)
        self.assertTrue(parsed["reps_inferred"])

    def test_accepts_explicit_reps_units(self):
        for unit in ("次", "个", "reps"):
            with self.subTest(unit=unit):
                parsed = FitnessAgent._parse_workout(f"记录卧推60kg，4组，每组8{unit}")
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed["reps"], 8)
                self.assertFalse(parsed["reps_inferred"])

    def test_does_not_treat_weight_as_reps(self):
        parsed = FitnessAgent._parse_workout("记录卧推，4组，每组8kg")

        self.assertIsNone(parsed)


if __name__ == "__main__":
    unittest.main()
