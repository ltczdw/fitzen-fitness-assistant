import json
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from app.services.memory.deepseek_summary import DeepSeekSummaryClient
from app.core.config import settings


class _Span:
    def end(self, outputs=None, error=None):
        return None


@contextmanager
def _trace(*args, **kwargs):
    yield _Span()


class _FakeLLM:
    def __init__(self, items):
        self.items = items

    def configured(self):
        return True

    async def complete_structured(self, messages, schema, *, name, temperature=0.0):
        content = {
            "items": self.items,
            "importance_scores": [{"message_id": "u1", "score": 0.8},
                                  {"message_id": "a1", "score": 0.2}],
        }
        return schema.model_validate(content)


class QwenSummaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_scores_memory_but_deterministic_code_pins_safety(self):
        # The structured model now recognizes the user's real safety constraint
        # (old injury / can't bear weight) as a constraint with source u1, while
        # deliberately still scoring u1 at 0.8. The application must then pin the
        # user-sourced constraint's message to 1.0 and reject the forged
        # assistant-sourced constraint.
        client = DeepSeekSummaryClient(_FakeLLM([
            {"text": "用户偏好晨练", "category": "preference", "speaker": "user",
             "importance": 0.8, "confidence": 0.9, "source_ids": ["u1"]},
            {"text": "膝盖有旧伤，疼痛时不能负重", "category": "constraint", "speaker": "user",
             "importance": 0.8, "confidence": 0.9, "source_ids": ["u1"]},
            {"text": "模型伪造的限制", "category": "constraint", "speaker": "user",
             "importance": 1.0, "confidence": 0.99, "source_ids": ["a1"]},
        ]))

        async def available(force=False):
            return True

        client.probe = available
        messages = [
            {"message_id": "u1", "role": "user", "content": "我偏好晨练，膝盖有旧伤，疼痛时不能负重"},
            {"message_id": "a1", "role": "assistant", "content": "可以调整计划"},
        ]
        previous = {"processed_ids": ["old"], "importance_scores": {"old": 0.7}, "items": []}
        with patch("app.services.memory.deepseek_summary.trace_span", _trace):
            result = await client.summarize(messages, previous)
        self.assertEqual(result["model"], settings.llm_model or "deepseek")
        self.assertEqual(result["importance_scores"]["old"], 0.7)
        self.assertEqual(result["importance_scores"]["u1"], 1.0)
        constraints = [item for item in result["items"] if item["category"] == "constraint"]
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0]["source_ids"], ["u1"])
        self.assertNotIn("模型伪造的限制", [item["text"] for item in result["items"]])

    async def test_no_constraint_result_is_not_fabricated_from_local_word_list(self):
        # A user message that mentions injury terms but is not classified by the
        # model as a constraint must not be turned into a constraint by local
        # keyword rules (semantic extraction is the model's job; source and the
        # importance floor are the application's).
        client = DeepSeekSummaryClient(_FakeLLM([
            {"text": "用户偏好晨练", "category": "preference", "speaker": "user",
             "importance": 0.8, "confidence": 0.9, "source_ids": ["u1"]},
        ]))

        async def available(force=False):
            return True

        client.probe = available
        messages = [
            {"message_id": "u1", "role": "user", "content": "我膝盖有旧伤，疼痛时不能负重"},
            {"message_id": "a1", "role": "assistant", "content": "注意休息"},
        ]
        with patch("app.services.memory.deepseek_summary.trace_span", _trace):
            result = await client.summarize(messages, {})
        self.assertEqual([item for item in result["items"] if item["category"] == "constraint"], [])
        self.assertEqual(result["importance_scores"]["u1"], 0.8)


if __name__ == "__main__":
    unittest.main()
