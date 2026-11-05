import unittest

from app.services.agent.agent import FitnessAgent


class WorkoutParserOrderTests(unittest.TestCase):
    def test_infers_reps_when_set_count_comes_after_exercise(self):
        parsed = FitnessAgent._parse_workout("记录今天的训练，每组8的10kg哑铃卧推，我做了4组")

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["sets"], 4)
        self.assertEqual(parsed["reps"], 8)
        self.assertEqual(parsed["weight_kg"], 10)
        self.assertTrue(parsed["reps_inferred"])


if __name__ == "__main__":
    unittest.main()
