from __future__ import annotations
from app.core import clock

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import settings
from app.services.llm.budget import ModelBudgetError
from app.services.security.input_guard import redact_phi
from app.services.memory.deepseek_summary import deepseek_summary


logger = logging.getLogger(__name__)
PROFILE_FIELDS = ("goal", "experience", "available_days", "body_weight_kg", "target_weight_kg", "height_cm", "body_fat_pct")

# A single atomic operation prevents concurrent requests from losing messages
# between append, overflow archival, trim and TTL renewal.
APPEND_TURN = """
for i = 4, #ARGV do redis.call('RPUSH', KEYS[1], ARGV[i]) end
local overflow = redis.call('LLEN', KEYS[1]) - tonumber(ARGV[1])
if overflow > 0 then
  local removed = redis.call('LRANGE', KEYS[1], 0, overflow - 1)
  if #removed > 0 then redis.call('RPUSH', KEYS[2], unpack(removed)) end
  redis.call('LTRIM', KEYS[1], -tonumber(ARGV[1]), -1)
  redis.call('LTRIM', KEYS[2], -tonumber(ARGV[3]), -1)
end
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
redis.call('EXPIRE', KEYS[2], tonumber(ARGV[2]))
return redis.call('LLEN', KEYS[1])
"""

RELEASE_LOCK = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


def _identity(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _decode_messages(values: list[str]) -> list[dict[str, str]]:
    result = []
    for value in values:
        try:
            item = json.loads(value)
            if item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str):
                result.append({"role": item["role"], "content": item["content"],
                               "message_id": str(item.get("message_id") or _identity(value))})
        except (ValueError, TypeError, AttributeError):
            logger.warning("Ignored malformed conversation memory entry")
    return result


class RedisConversationMemory:
    """Redis-backed 12-message window and user-confirmed long-term facts.

    Redis failure falls back to bounded process memory, never to business JSON
    files. The health endpoint explicitly reports this non-persistent mode.
    """

    def __init__(self, *, url: str | None = None, enabled: bool | None = None,
                 window: int | None = None, ttl: int | None = None,
                 history_max: int | None = None, history_ttl: int | None = None) -> None:
        self.enabled = settings.redis_enabled if enabled is None else enabled
        self.window = max(2, min(window or settings.memory_short_window_messages, 100))
        self.ttl = max(60, ttl or settings.memory_short_ttl_seconds)
        self.history_max = max(20, min(history_max or settings.chat_history_max_messages, 1000))
        self.history_ttl = max(self.ttl, history_ttl or settings.chat_history_ttl_seconds)
        self.client = Redis.from_url(url or settings.redis_url, decode_responses=True,
                                     socket_connect_timeout=1, socket_timeout=2)
        self.connected = False
        self._next_retry = 0.0
        self._short: dict[tuple[str, str], list[dict[str, str]]] = {}
        self._summary: dict[tuple[str, str], list[dict[str, str]]] = {}
        self._history: dict[tuple[str, str], list[dict[str, str]]] = {}
        self._long: dict[str, dict[str, Any]] = {}
        self._semantic: dict[tuple[str, str], dict[str, Any]] = {}
        self._summary_tasks: dict[tuple[str, str], asyncio.Task] = {}
        # Process-local fallback for user-facing soft delete when Redis is down.
        self._hidden: set[tuple[str, str]] = set()

    def keys(self, user_id: str, conversation_id: str = "") -> tuple[str, str, str]:
        user = _identity(user_id)
        conversation = _identity(conversation_id)
        return (f"fitness:memory:short:{user}:{conversation}",
                f"fitness:memory:summary:{user}:{conversation}",
                f"fitness:memory:long:{user}")

    def history_key(self, user_id: str, conversation_id: str) -> str:
        return f"fitness:chat:history:{_identity(user_id)}:{_identity(conversation_id)}"

    def conversation_registry_key(self, user_id: str) -> str:
        return f"fitness:chat:conversations:{_identity(user_id)}"

    def semantic_key(self, user_id: str, conversation_id: str) -> str:
        return f"fitness:memory:semantic:{_identity(user_id)}:{_identity(conversation_id)}"

    def status(self) -> dict[str, Any]:
        return {"backend": "redis" if self.connected else "memory-fallback",
                "status": "connected" if self.connected else ("degraded" if self.enabled else "disabled"),
                "persistent": self.connected, "window_messages": self.window,
                "short_ttl_seconds": self.ttl, "history_messages": self.history_max,
                "history_ttl_seconds": self.history_ttl, "summary_model": deepseek_summary.status()}

    def _unavailable(self) -> None:
        if self.connected or not self._next_retry:
            logger.warning("Redis memory unavailable; using non-persistent process memory")
        self.connected = False
        self._next_retry = time.monotonic() + 10

    async def initialize(self) -> bool:
        if not self.enabled:
            return False
        if not self.connected and time.monotonic() < self._next_retry:
            return False
        try:
            await self.client.ping()
            self.connected = True
            return True
        except (RedisError, OSError):
            self._unavailable()
            return False

    async def close(self) -> None:
        tasks = list(self._summary_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.client.aclose()
        self.connected = False

    async def load(self, user_id: str, conversation_id: str) -> dict[str, Any]:
        identity = (user_id, conversation_id)
        if await self.initialize():
            short_key, summary_key, long_key = self.keys(user_id, conversation_id)
            try:
                async with self.client.pipeline(transaction=False) as pipe:
                    pipe.lrange(short_key, 0, -1)
                    pipe.lrange(summary_key, 0, -1)
                    pipe.hgetall(long_key)
                    pipe.get(self.semantic_key(user_id, conversation_id))
                    pipe.expire(short_key, self.ttl)
                    pipe.expire(summary_key, self.ttl)
                    values = await pipe.execute()
                self._short[identity] = _decode_messages(values[0])[-self.window:]
                self._summary[identity] = _decode_messages(values[1])[-32:]
                long_memory: dict[str, Any] = {}
                for field, value in values[2].items():
                    try:
                        long_memory[field] = json.loads(value)
                    except (ValueError, TypeError):
                        logger.warning("Ignored malformed long-term memory entry")
                self._long[user_id] = long_memory
                if values[3]:
                    try:
                        self._semantic[identity] = json.loads(values[3])
                    except (ValueError, TypeError):
                        logger.warning("Ignored malformed semantic memory entry")
            except (RedisError, OSError):
                self._unavailable()
        semantic = self._semantic.get(identity, {})
        summary = "\n".join(f"{item['role']}: {item['content'][:160]}" for item in self._summary.get(identity, []))[-1200:]
        result = {"short_memory": list(self._short.get(identity, [])), "memory_summary": summary,
                "semantic_memory": semantic.get("items", []),
                "message_importance": semantic.get("importance_scores", {}),
                "long_memory": dict(self._long.get(user_id, {})), "memory_backend": self.status()["backend"]}
        if self._summary.get(identity):
            self._schedule_summary(user_id, conversation_id)
        return result

    async def append_turn(self, user_id: str, conversation_id: str, user: str, assistant: str) -> None:
        identity = (user_id, conversation_id)
        turn = [{"role": "user", "content": redact_phi(user)[:4000], "message_id": uuid4().hex},
                {"role": "assistant", "content": redact_phi(assistant)[:16000], "message_id": uuid4().hex}]
        messages = self._short.setdefault(identity, []) + turn
        overflow = max(0, len(messages) - self.window)
        self._summary[identity] = (self._summary.get(identity, []) + messages[:overflow])[-8:]
        self._short[identity] = messages[-self.window:]
        self._history[identity] = (self._history.get(identity, []) + turn)[-self.history_max:]
        if await self.initialize():
            short_key, summary_key, _ = self.keys(user_id, conversation_id)
            history_key = self.history_key(user_id, conversation_id)
            try:
                serialized = [json.dumps(item, ensure_ascii=False) for item in turn]
                async with self.client.pipeline(transaction=True) as pipe:
                    pipe.eval(APPEND_TURN, 2, short_key, summary_key, self.window, self.ttl, 32, *serialized)
                    pipe.rpush(history_key, *serialized)
                    pipe.ltrim(history_key, -self.history_max, -1)
                    pipe.expire(history_key, self.history_ttl)
                    await pipe.execute()
                await self.register_conversation(user_id, conversation_id, redact_phi(user), message_count_delta=len(turn))
                if overflow:
                    self._schedule_summary(user_id, conversation_id)
            except (RedisError, OSError):
                self._unavailable()

    def _schedule_summary(self, user_id: str, conversation_id: str) -> None:
        from app.services.memory.background import background_memory_enabled
        if not background_memory_enabled():
            return
        identity = (user_id, conversation_id)
        current = self._summary_tasks.get(identity)
        if not self.enabled or current is not None and not current.done():
            return
        task = asyncio.create_task(self._summarize_pending(user_id, conversation_id))
        self._summary_tasks[identity] = task

        def completed(done: asyncio.Task) -> None:
            self._summary_tasks.pop(identity, None)
            try:
                more = bool(done.result())
            except (Exception, asyncio.CancelledError):
                more = False
            if more:
                self._schedule_summary(user_id, conversation_id)
        task.add_done_callback(completed)

    async def _summarize_pending(self, user_id: str, conversation_id: str) -> bool:
        from app.services.memory.background import background_memory_enabled
        if not background_memory_enabled():
            return False
        identity = (user_id, conversation_id)
        if not await self.initialize():
            return False
        semantic_key = self.semantic_key(user_id, conversation_id)
        lock_key = semantic_key + ":lock"
        lock_token = uuid4().hex
        lock_seconds = max(60, int(settings.memory_summary_timeout_seconds) + 30)
        try:
            # Background tasks are process-local. This Redis lease prevents two
            # API workers from summarizing and overwriting the same batch.
            if not await self.client.set(lock_key, lock_token, nx=True, ex=lock_seconds):
                return False
            pending = _decode_messages(await self.client.lrange(self.keys(user_id, conversation_id)[1], 0, -1))
            stored = await self.client.get(semantic_key)
            semantic = json.loads(stored) if stored else self._semantic.get(identity, {})
            processed = set(semantic.get("processed_ids", []))
            new_messages = [item for item in pending if item["message_id"] not in processed]
            if not new_messages:
                return False
            from app.services.llm.budget import model_scope, model_stage, key
            batch_id=key(user_id+conversation_id+json.dumps([m['message_id'] for m in new_messages[:6]]))
            with model_scope('redis-summary:'+batch_id,kind='background',owner_id=key(user_id+':'+conversation_id)):
                with model_stage('summary',2):
                    result = await deepseek_summary.summarize(new_messages, semantic)
            result["processed_ids"] = list(dict.fromkeys([*semantic.get("processed_ids", []),
                                                         *result["processed_ids"]]))[-64:]
            result["updated_at"] = clock.utc_now().isoformat()
            await self.client.set(semantic_key,
                                  json.dumps(result, ensure_ascii=False), ex=self.ttl)
            self._semantic[identity] = result
            # Messages may overflow while DeepSeek is running. Recheck Redis after
            # the write so that the completion callback schedules another batch
            # instead of waiting for the user's next request.
            latest = _decode_messages(await self.client.lrange(self.keys(user_id, conversation_id)[1], 0, -1))
            completed_ids = set(result["processed_ids"])
            return any(item["message_id"] not in completed_ids for item in latest)
        except (RedisError, OSError, httpx.HTTPError, ValueError, TypeError, RuntimeError, ModelBudgetError):
            logger.warning("Semantic memory summarization unavailable; deterministic window remains active")
            return False
        finally:
            try:
                await self.client.eval(RELEASE_LOCK, 1, lock_key, lock_token)
            except (RedisError, OSError):
                pass

    async def load_history(self, user_id: str, conversation_id: str) -> dict[str, Any]:
        identity = (user_id, conversation_id)
        if await self._conversation_hidden(user_id, conversation_id):
            return {"messages": [], "memory_backend": self.status()["backend"],
                    "max_messages": self.history_max, "ttl_seconds": self.history_ttl}
        if await self.initialize():
            history_key = self.history_key(user_id, conversation_id)
            try:
                values = await self.client.lrange(history_key, -self.history_max, -1)
                await self.client.expire(history_key, self.history_ttl)
                if values:
                    self._history[identity] = _decode_messages(values)
                    if self._history[identity]:
                        first_user = next((item["content"] for item in self._history[identity] if item["role"] == "user"), None)
                        await self.register_conversation(user_id, conversation_id, first_user, message_count=len(self._history[identity]))
            except (RedisError, OSError):
                self._unavailable()
        if not self._history.get(identity):
            # Conversations created before the transcript feature can still
            # display the currently available short-memory window.
            context = await self.load(user_id, conversation_id)
            self._history[identity] = list(context["short_memory"])
        return {"messages": list(self._history.get(identity, [])),
                "memory_backend": self.status()["backend"],
                "max_messages": self.history_max, "ttl_seconds": self.history_ttl}

    async def register_conversation(
        self,
        user_id: str,
        conversation_id: str,
        title: str | None = None,
        *,
        message_count_delta: int = 0,
        message_count: int | None = None,
    ) -> dict[str, Any]:
        conversation_id = conversation_id.strip()
        now = clock.utc_now().isoformat()
        metadata: dict[str, Any] = {
            "conversation_id": conversation_id,
            "title": (title or "新对话").strip()[:80] or "新对话",
            "created_at": now,
            "updated_at": now,
            "message_count": max(0, int(message_count or 0)),
        }
        if await self.initialize():
            key = self.conversation_registry_key(user_id)
            try:
                existing_raw = await self.client.hget(key, conversation_id)
                existing: dict[str, Any] = {}
                if existing_raw:
                    try:
                        parsed = json.loads(existing_raw)
                        if isinstance(parsed, dict):
                            existing = parsed
                    except (ValueError, TypeError):
                        existing = {}
                metadata["created_at"] = str(existing.get("created_at") or now)
                existing_title = str(existing.get("title") or "")
                if not title and existing_title:
                    metadata["title"] = existing_title
                elif title and existing_title not in {"", "新对话", "当前对话"}:
                    metadata["title"] = existing_title
                if message_count is None:
                    metadata["message_count"] = max(0, int(existing.get("message_count") or 0) + int(message_count_delta))
                else:
                    metadata["message_count"] = max(0, int(message_count))
                # New user messages revive a soft-deleted conversation; plain
                # metadata refreshes keep the hidden marker so a deleted
                # conversation cannot reappear just by being listed or read.
                if message_count_delta > 0:
                    self._hidden.discard((user_id, conversation_id))
                elif existing.get("hidden_at"):
                    metadata["hidden_at"] = existing["hidden_at"]
                await self.client.hset(key, conversation_id, json.dumps(metadata, ensure_ascii=False))
                await self.client.expire(key, self.history_ttl)
            except (RedisError, OSError):
                self._unavailable()
        return metadata

    async def _registry_entry(self, user_id: str, conversation_id: str) -> dict[str, Any] | None:
        if not await self.initialize():
            return None
        try:
            raw = await self.client.hget(self.conversation_registry_key(user_id), conversation_id)
        except (RedisError, OSError):
            self._unavailable()
            return None
        if not raw:
            return None
        try:
            item = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return item if isinstance(item, dict) else None

    async def _conversation_hidden(self, user_id: str, conversation_id: str) -> bool:
        if (user_id, conversation_id) in self._hidden:
            return True
        item = await self._registry_entry(user_id, conversation_id)
        return bool(item and item.get("hidden_at"))

    async def list_conversations(self, user_id: str) -> list[dict[str, Any]]:
        if not await self.initialize():
            return []
        key = self.conversation_registry_key(user_id)
        try:
            raw_items = await self.client.hgetall(key)
        except (RedisError, OSError):
            self._unavailable()
            return []
        conversations: list[dict[str, Any]] = []
        for conversation_id, raw in raw_items.items():
            try:
                item = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if not isinstance(item, dict):
                continue
            if item.get("hidden_at"):
                # User-facing soft delete: the transcript keys stay in Redis,
                # the conversation only disappears from the user's list.
                continue
            item["conversation_id"] = str(item.get("conversation_id") or conversation_id)
            item["title"] = str(item.get("title") or "新对话")[:80]
            item["message_count"] = max(0, int(item.get("message_count") or 0))
            conversations.append(item)
        conversations.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return conversations

    async def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        """User-facing soft delete for the legacy Redis backend.

        The registry entry is marked hidden while short/summary/history/semantic
        keys stay untouched, so the transcript is never physically removed.
        """
        identity = (user_id, conversation_id)
        self._hidden.add(identity)
        if not await self.initialize():
            return False
        key = self.conversation_registry_key(user_id)
        try:
            existing = await self._registry_entry(user_id, conversation_id) or {}
            metadata = dict(existing)
            metadata.setdefault("conversation_id", conversation_id)
            metadata.setdefault("title", "新对话")
            metadata["hidden_at"] = clock.utc_now().isoformat()
            await self.client.hset(key, conversation_id, json.dumps(metadata, ensure_ascii=False))
            await self.client.expire(key, self.history_ttl)
            return True
        except (RedisError, OSError):
            self._unavailable()
            return False

    async def restore_latest_history(self, user_id: str, conversation_id: str) -> dict[str, Any]:
        """Restore the most recently renewed history for a user into a target conversation.

        Conversation IDs are intentionally hashed in Redis, so an old client that
        lost its random ID cannot address its transcript directly.  This recovery
        path is explicit and only used by the frontend migration flow; normal new
        conversations remain empty.
        """
        current = await self.load_history(user_id, conversation_id)
        if current["messages"] or not await self.initialize():
            return {**current, "recovered": False}
        if await self._conversation_hidden(user_id, conversation_id):
            # Never refill a conversation the user has hidden.
            return {**current, "recovered": False}

        target_key = self.history_key(user_id, conversation_id)
        prefix = f"fitness:chat:history:{_identity(user_id)}:"
        hidden_history_keys: set[str] = set()
        try:
            registry = await self.client.hgetall(self.conversation_registry_key(user_id))
            for conversation, raw in registry.items():
                try:
                    item = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if isinstance(item, dict) and item.get("hidden_at"):
                    hidden_history_keys.add(self.history_key(user_id, conversation))
        except (RedisError, OSError):
            self._unavailable()
        candidate: tuple[int, int, list[str], list[dict[str, str]]] | None = None
        try:
            async for key in self.client.scan_iter(match=f"{prefix}*", count=100):
                if key == target_key or key in hidden_history_keys:
                    continue
                values = await self.client.lrange(key, -self.history_max, -1)
                if not values:
                    continue
                messages = _decode_messages(values)
                if not messages:
                    continue
                # TTL is renewed on every write, so it is the best recency signal
                # available for pre-migration records. Message count breaks ties.
                ttl = max(-1, int(await self.client.ttl(key)))
                score = (ttl, len(messages), values, messages)
                if candidate is None or score[:2] > candidate[:2]:
                    candidate = score
        except (RedisError, OSError):
            self._unavailable()
            return {**current, "recovered": False}

        if candidate is None:
            return {**current, "recovered": False}
        _, _, values, messages = candidate
        try:
            async with self.client.pipeline(transaction=True) as pipe:
                pipe.delete(target_key)
                pipe.rpush(target_key, *values)
                pipe.ltrim(target_key, -self.history_max, -1)
                pipe.expire(target_key, self.history_ttl)
                await pipe.execute()
        except (RedisError, OSError):
            self._unavailable()
            return {**current, "recovered": False}
        identity = (user_id, conversation_id)
        self._history[identity] = messages[-self.history_max:]
        self._short[identity] = self._history[identity][-self.window:]
        self._summary[identity] = []
        first_user = next((item["content"] for item in messages if item["role"] == "user"), None)
        await self.register_conversation(user_id, conversation_id, first_user, message_count=len(messages))
        return {"messages": list(self._history[identity]),
                "memory_backend": self.status()["backend"],
                "max_messages": self.history_max, "ttl_seconds": self.history_ttl,
                "recovered": True}

    async def sync_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        confirmed = {field: profile.get(field) for field in PROFILE_FIELDS if field != "body_fat_pct" or profile.get(field) is not None}
        value = {"facts": confirmed, "source": "saved_profile", "updated_at": clock.utc_now().isoformat()}
        self._long.setdefault(user_id, {})["profile"] = value
        if await self.initialize():
            try:
                await self.client.hset(self.keys(user_id)[2], "profile", json.dumps(value, ensure_ascii=False))
            except (RedisError, OSError):
                self._unavailable()

    async def remember_explicit(self, user_id: str, message: str) -> None:
        from app.services.agent.semantics import get_understanding
        semantic = get_understanding(message)
        if not semantic or semantic.get("needs_clarification"):
            return
        facts = self._long.setdefault(user_id, {})
        for item in semantic.get("memory_requests") or []:
            fact = redact_phi(item["quote"]).strip()
            if not fact or fact not in message:
                continue
            field = "semantic:" + _identity(item["key"])
            targets = item.get("target_fields") or []
            if any(target not in facts for target in targets):
                continue
            if field in facts and facts[field].get("text") != fact and item["action"] != "replace":
                continue
            if targets and item["action"] != "replace":
                continue
            value = {"text": fact, "source": "explicit_user_request", "updated_at": clock.utc_now().isoformat()}
            # One Redis transaction keeps retirement+replacement together.
            if await self.initialize():
                try:
                    async with self.client.pipeline(transaction=True) as pipe:
                        if item["action"] == "replace" and targets:
                            pipe.hdel(self.keys(user_id)[2], *targets)
                        pipe.hset(self.keys(user_id)[2], field, json.dumps(value, ensure_ascii=False))
                        await pipe.execute()
                except (RedisError, OSError):
                    self._unavailable()
                    continue
            for target in targets if item["action"] == "replace" else []:
                facts.pop(target, None)
            facts[field] = value



from app.services.memory.compact import CompactMemory
memory_service = CompactMemory(RedisConversationMemory())
