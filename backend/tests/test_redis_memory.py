import unittest
import json
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from app.api.profile import ProfileUpdate
from app.services.memory.redis_memory import RedisConversationMemory


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.lists = {}

    async def ping(self):
        return True

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def get(self, key):
        return self.values.get(key)

    async def lrange(self, key, start, end):
        return list(self.lists.get(key, []))

    async def eval(self, script, numkeys, key, token):
        if self.values.get(key) == token:
            self.values.pop(key, None)
            return 1
        return 0

    async def aclose(self):
        return None


class RedisMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.memory = RedisConversationMemory(enabled=False, window=12, ttl=120)

    async def asyncTearDown(self):
        await self.memory.close()

    async def test_short_memory_keeps_last_twelve_messages_and_summary(self):
        for index in range(8):
            await self.memory.append_turn("user-a", "conversation-a", f"问题{index}", f"回答{index}")
        context = await self.memory.load("user-a", "conversation-a")
        self.assertEqual(len(context["short_memory"]), 12)
        self.assertEqual(context["short_memory"][0]["content"], "问题2")
        self.assertIn("问题0", context["memory_summary"])
        self.assertEqual(context["memory_backend"], "memory-fallback")

    async def test_display_history_is_larger_than_model_context(self):
        for index in range(8):
            await self.memory.append_turn("user-a", "history", f"问题{index}", f"回答{index}")
        context = await self.memory.load("user-a", "history")
        history = await self.memory.load_history("user-a", "history")
        self.assertEqual(len(context["short_memory"]), 12)
        self.assertEqual(len(history["messages"]), 16)

    async def test_memory_is_isolated_by_user_and_conversation(self):
        await self.memory.append_turn("user-a", "one", "A", "A1")
        await self.memory.append_turn("user-a", "two", "B", "B1")
        await self.memory.append_turn("user-b", "one", "C", "C1")
        self.assertEqual((await self.memory.load("user-a", "one"))["short_memory"][0]["content"], "A")
        self.assertEqual((await self.memory.load("user-a", "two"))["short_memory"][0]["content"], "B")
        self.assertEqual((await self.memory.load("user-b", "one"))["short_memory"][0]["content"], "C")

    async def test_only_explicit_chat_fact_enters_long_term_memory(self):
        await self.memory.remember_explicit("user-a", "我喜欢晨练")
        from app.services.agent.semantics import request_understanding, validate_understanding, remember_understanding
        message = "请记住我更喜欢晨练"
        token = request_understanding.set({})
        try:
            value = validate_understanding(message, {"tasks": [{"task_type": "chitchat", "action": "answer", "text": message,
                "is_negated": False, "is_hypothetical": False, "is_third_party": False}], "memory_requests": [{
                "action": "remember", "quote": "我更喜欢晨练", "key": "preference:morning", "confidence": 1}]})
            remember_understanding(message, value)
            await self.memory.remember_explicit("user-a", message)
        finally:
            request_understanding.reset(token)
        context = await self.memory.load("user-a", "one")
        facts = [value for key, value in context["long_memory"].items() if key.startswith("semantic:")]
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["text"], "我更喜欢晨练")

    async def test_saved_profile_is_long_term_memory(self):
        await self.memory.sync_profile("user-a", {"goal": "减脂", "target_weight_kg": 62.5})
        profile = (await self.memory.load("user-a", "one"))["long_memory"]["profile"]
        self.assertEqual(profile["facts"]["target_weight_kg"], 62.5)


class TargetWeightValidationTests(unittest.TestCase):
    def test_target_weight_accepts_decimal(self):
        profile = ProfileUpdate(target_weight_kg=62.5)
        self.assertEqual(profile.target_weight_kg, 62.5)

    def test_target_weight_rejects_unsafe_range(self):
        with self.assertRaises(ValidationError):
            ProfileUpdate(target_weight_kg=10)


class SemanticMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_pending_messages_are_summarized_once_under_redis_lease(self):
        memory = RedisConversationMemory(enabled=True, window=12, ttl=120)
        fake = FakeRedis()
        memory.client = fake
        _, pending_key, _ = memory.keys("qa-user", "qa-conversation")
        pending = [{"role": "user", "content": "请记住我膝盖旧伤", "message_id": "m1"},
                   {"role": "assistant", "content": "收到", "message_id": "m2"}]
        fake.lists[pending_key] = [json.dumps(item, ensure_ascii=False) for item in pending]
        summary = {"items": [{"text": "用户膝盖有旧伤", "category": "constraint", "speaker": "user",
                              "importance": 1.0, "confidence": 1.0, "source_ids": ["m1"]}],
                   "importance_scores": {"m1": 1.0, "m2": 0.1},
                   "processed_ids": ["m1", "m2"], "model": "deepseek"}
        summarize = AsyncMock(return_value=summary)
        from app.core.config import settings
        with patch.object(settings, "memory_background_enabled", True), patch.object(settings, "memory_summary_enabled", True), patch("app.services.memory.redis_memory.deepseek_summary.summarize", summarize):
            self.assertFalse(await memory._summarize_pending("qa-user", "qa-conversation"))
            self.assertFalse(await memory._summarize_pending("qa-user", "qa-conversation"))
        self.assertEqual(summarize.await_count, 1)
        stored = json.loads(fake.values[memory.semantic_key("qa-user", "qa-conversation")])
        self.assertEqual(stored["processed_ids"], ["m1", "m2"])
        self.assertNotIn(memory.semantic_key("qa-user", "qa-conversation") + ":lock", fake.values)
        await memory.close()


if __name__ == "__main__":
    unittest.main()
