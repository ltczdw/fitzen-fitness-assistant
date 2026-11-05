import unittest

from app.db import session as db_session
from app.db.models import WorkoutRecordModel
from app.db.repositories import create_workout
from app.services.agent.agent import FitnessAgent
from app.services.agent.intent import classify_intent
from app.services.agent.window import SlidingWindow
from app.services.rag.pipeline import RetrievedChunk
from app.services.rag.reranker import LocalReranker

class FakeSession:
    row = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def add(self, row):
        if isinstance(row, WorkoutRecordModel):
            self.row = row
        row.id = 1

    def flush(self):
        if self.row is not None and self.row.id is None:
            self.row.id = 1

    def scalar(self, statement):
        return None

    def scalars(self, statement):
        from unittest.mock import Mock
        return Mock(all=lambda: [])

    def execute(self, statement, *args):
        return None

    def commit(self):
        return None

    def refresh(self, row):
        return None


class CoreBehaviorTests(unittest.TestCase):
    def test_workout_model_mapping_does_not_pass_api_date_field(self):
        previous_ready = db_session.DB_READY
        previous_session_local = db_session.SessionLocal
        fake_session = FakeSession()
        db_session.DB_READY = True
        db_session.SessionLocal = lambda: fake_session
        try:
            from unittest.mock import patch
            with patch("pathlib.Path.write_text", side_effect=AssertionError("No fallback writes")):
                saved = create_workout("unit-user", "深蹲", 3, 8, 40, "test")
        finally:
            db_session.DB_READY = previous_ready
            db_session.SessionLocal = previous_session_local

        self.assertIsInstance(fake_session.row, WorkoutRecordModel)
        self.assertEqual(saved["exercise"], "深蹲")
        self.assertEqual(saved["sets"], 3)

    def test_intent_safety_and_workout_record(self):
        from app.services.agent.semantics import remember_understanding, request_understanding
        urgent_source = "膝盖明显肿胀并且无法活动"
        token = request_understanding.set({})
        try:
            # Shared model understanding drives business intent; without it the
            # rule path stays read-only (training_qa), never a guessed write.
            # Emergency meaning is the model's source-bound assessment, not a
            # keyword/literal check.
            remember_understanding(urgent_source, {
                "tasks": [{
                    "task_type": "safety_triage", "worker": "safety_triage",
                    "action": "question", "text": urgent_source,
                    "is_negated": False, "is_hypothetical": False, "is_third_party": False,
                    "side_effect": False, "depends_on": [],
                }],
                "safety": {"level": "urgent", "evidence": urgent_source,
                           "categories": ["current_symptom"], "reason": "scripted current emergency"},
            })
            urgent = classify_intent(urgent_source)
            assert (urgent.name, urgent.safety_level) == ("symptom_triage", "urgent")

            record_source = "记录深蹲40kg，3组，每组8次"
            remember_understanding(record_source, {
                "validated_tasks": [{
                    "task_type": "workout_record", "worker": "workout_record",
                    "action": "record", "text": record_source,
                    "is_negated": False, "is_hypothetical": False, "is_third_party": False,
                    "side_effect": True, "depends_on": [],
                }],
                "safety": {"level": "normal", "evidence": "", "categories": [],
                           "reason": "scripted normal"},
            })
            record = classify_intent(record_source)
            # No assessed safety for a different text: the old lexical urgent
            # path must not guess; the rule fallback is read-only. The model
            # boundary is forced unavailable so no real provider is reachable.
            from unittest.mock import patch
            with patch("app.services.agent.intent.classify_intent_llm", return_value=None):
                missing = classify_intent("现在胸痛并且呼吸困难")
        finally:
            request_understanding.reset(token)
        self.assertEqual(record.name, "workout_record")
        self.assertEqual((missing.name, missing.safety_level), ("training_qa", "normal"))

    def test_sliding_window_keeps_bounded_context_and_summary(self):
        window = SlidingWindow(max_messages=3)
        for index in range(5):
            window.add("user", f"message-{index}")
        self.assertEqual(len(window.messages), 3)
        self.assertTrue(window.summary)
        self.assertEqual(len(window.context()), 4)

    def test_workout_parser(self):
        parsed = FitnessAgent._parse_workout("记录卧推60kg，4组，每组8次")
        self.assertEqual(parsed["exercise"], "卧推")
        self.assertEqual(parsed["sets"], 4)
        self.assertEqual(parsed["reps"], 8)
        self.assertEqual(parsed["weight_kg"], 60)

    def test_local_reranker_prioritizes_safety_for_pain_query(self):
        chunks = [
            RetrievedChunk("普通训练", "保持规律训练。", 0.8),
            RetrievedChunk("疼痛安全分流", "出现肿胀时停止训练。", 0.5),
        ]
        result = LocalReranker().rerank("膝盖肿痛怎么办", chunks, top_n=2)
        self.assertEqual(result[0].title, "疼痛安全分流")


if __name__ == "__main__":
    unittest.main()
