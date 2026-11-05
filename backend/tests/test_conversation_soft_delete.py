"""User delete is a soft hide: list/history stop showing it, data stays on disk."""
from __future__ import annotations

import asyncio
import json

from app.db.memory_models import MemoryTurn
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.memory import compact
from app.services.memory.compact import CompactMemory
from app.services.memory.redis_memory import RedisConversationMemory


class _LegacyStub:
    """Compact tests never touch Redis; the stub only satisfies delegation."""

    async def delete_conversation(self, user_id, conversation_id):
        return True

    async def list_conversations(self, user_id):
        return []

    async def load_history(self, user_id, conversation_id):
        return {"messages": []}


def _memory():
    return CompactMemory(_LegacyStub())


def _complete_rollout(request, answer, run_id):
    from app.db import quality_repository as repo

    tid, _ = compact.begin_turn(request)
    response = {"answer": answer, "intent": "training_qa", "citations": [],
                "run_id": run_id, "message_id": tid + ":assistant",
                "request_id": request.request_id}
    repo.record_run(request, ChatResponse.model_validate(response), {}, status="completed")
    compact.finish_turn(tid, answer, response)
    return tid, response


def test_user_delete_hides_list_and_history_but_keeps_transcript(compact_db):
    first = ChatRequest(message="第一问", conversation_id="soft", request_id="soft-1")
    tid, response = _complete_rollout(first, "第一答", "run-soft")
    _complete_rollout(ChatRequest(message="第二问", conversation_id="soft", request_id="soft-2"),
                      "第二答", "run-soft-2")
    _complete_rollout(ChatRequest(message="其他会话", conversation_id="keep", request_id="keep-1"),
                      "其他答", "run-keep")

    memory = _memory()
    listed = asyncio.run(memory.list_conversations("demo-user"))
    assert {item["conversation_id"] for item in listed} == {"soft", "keep"}

    assert asyncio.run(memory.delete_conversation("demo-user", "soft")) is True

    listed = asyncio.run(memory.list_conversations("demo-user"))
    assert [item["conversation_id"] for item in listed] == ["keep"]
    assert asyncio.run(memory.load_history("demo-user", "soft"))["messages"] == []
    assert len(asyncio.run(memory.load_history("demo-user", "keep"))["messages"]) == 2

    with compact_db() as session:
        row = session.get(MemoryTurn, tid)
        assert row.hidden_at is not None
        assert row.deleted_at is None and row.status == "completed"
        assert [message["role"] for message in row.messages] == ["user", "assistant"]
        assert row.response == response
    assert compact.recent_rows("soft") == []


def test_physical_purge_still_works_after_hide(compact_db):
    request = ChatRequest(message="先隐藏再清理", conversation_id="purge-me", request_id="purge-1")
    tid, _ = _complete_rollout(request, "回答", "run-purge")
    assert asyncio.run(_memory().delete_conversation("demo-user", "purge-me")) is True

    assert compact.delete_memory("purge-me") == 1
    with compact_db() as session:
        row = session.get(MemoryTurn, tid)
        assert row.messages == [] and row.response is None
        assert row.deleted_at is not None


def test_delete_endpoint_hides_conversation_from_list_and_history(compact_db, monkeypatch):
    from app.api import chat as chat_api

    monkeypatch.setattr(chat_api, "memory_service", _memory())
    _complete_rollout(ChatRequest(message="接口删除", conversation_id="api-soft", request_id="api-soft-1"),
                      "接口回答", "run-api-soft")

    result = asyncio.run(chat_api.delete_chat_conversation("api-soft", user_id="demo-user"))
    assert result == {"deleted": True, "conversation_id": "api-soft"}
    conversations = asyncio.run(chat_api.chat_conversations(user_id="demo-user"))
    assert conversations["conversations"] == []
    history = asyncio.run(chat_api.chat_history(user_id="demo-user", conversation_id="api-soft"))
    assert history["messages"] == []


def test_memory_schema_migration_adds_hidden_at(tmp_path):
    from sqlalchemy import create_engine, text

    from app.db.session import _migrate_memory_schema

    engine = create_engine("sqlite:///" + str(tmp_path / "legacy-memory.db"))
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE memory_turns (turn_id VARCHAR(36) PRIMARY KEY, operations JSON)"))
        _migrate_memory_schema(connection)
        columns = {row[1] for row in connection.execute(text("PRAGMA table_info(memory_turns)"))}
    engine.dispose()
    assert {"operations", "hidden_at"} <= columns


class _RegistryRedis:
    """Minimal registry-only fake; transcript lists stay in plain dicts."""

    def __init__(self):
        self.hashes: dict[str, dict[str, str]] = {}
        self.lists: dict[str, list[str]] = {}

    async def ping(self):
        return True

    async def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    async def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value
        return 1

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def expire(self, key, ttl):
        return True

    async def aclose(self):
        return None


def test_legacy_backend_hides_without_deleting_history():
    memory = RedisConversationMemory(enabled=True, window=12, ttl=120)
    memory.client = _RegistryRedis()

    async def run():
        assert await memory.initialize()
        await memory.register_conversation("user-a", "conversation-a", "问题一", message_count=2)
        history_key = memory.history_key("user-a", "conversation-a")
        memory.client.lists[history_key] = ['{"role": "user", "content": "问题一"}']
        listed = await memory.list_conversations("user-a")
        assert [item["conversation_id"] for item in listed] == ["conversation-a"]

        assert await memory.delete_conversation("user-a", "conversation-a") is True
        assert await memory.list_conversations("user-a") == []
        assert (await memory.load_history("user-a", "conversation-a"))["messages"] == []
        # The hidden transcript stays in Redis instead of being deleted.
        assert memory.client.lists[history_key]
        registry = memory.client.hashes[memory.conversation_registry_key("user-a")]
        assert json.loads(registry["conversation-a"])["hidden_at"]

        # A genuinely new user message revives the conversation.
        await memory.register_conversation("user-a", "conversation-a", "新消息", message_count_delta=2)
        listed = await memory.list_conversations("user-a")
        assert [item["conversation_id"] for item in listed] == ["conversation-a"]
        await memory.close()

    asyncio.run(run())
