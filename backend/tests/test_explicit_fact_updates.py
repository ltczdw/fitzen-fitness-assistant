"""Exercise load corrections commit with history, sources and replay state."""
import pytest
from sqlalchemy import select

from app.db.memory_models import MemoryTurn, UserFact
from app.db.wiki_models import WikiFact
from app.schemas.chat import ChatRequest
from app.services.memory import compact
from evals.v2.safety import deny_network


@pytest.fixture(autouse=True)
def offline():
    with deny_network():
        yield


def _semantics(text):
    """Test-owned structured memory decision (the live app never guesses from text)."""
    from app.services.memory.facts import (
        _RULE_REPLACE, _RULE_REPLACE_BLOCKERS, parse_rule_load,
    )
    prefix = next((value for value in ("请帮我记住", "请记住", "记住")
                   if text.startswith(value)), None)
    if prefix is None:
        return None
    quote = text[len(prefix):]
    parsed = parse_rule_load(quote)
    replace = bool(parsed and _RULE_REPLACE.search(quote)
                   and not _RULE_REPLACE_BLOCKERS.search(quote))
    current = compact.current_facts()
    target_fields, target_sources = [], {}
    if parsed:
        for field, value in current.items():
            old = parse_rule_load(value.get("source_quote", ""))
            if field == parsed.field("demo-user") or (old and old.exercise == parsed.exercise):
                target_fields.append(field)
                target_sources[field] = {
                    "source_message_id": value.get("source_message_id"),
                    "source_quote": value.get("source_quote"),
                }
    decision = {
        "action": "replace" if replace and target_fields else "remember",
        "quote": quote,
        "key": ("load:" + parsed.exercise) if parsed else "explicit",
        "target_fields": target_fields if replace else [],
        "confidence": 1,
    }
    return {
        "memory_requests": [decision],
        "allowed_memory_fields": list(current),
        "memory_target_sources": target_sources if replace else {},
        "needs_clarification": False,
        "current_is_critical": False,
    }


def remember(text, request_id, conversation_id="facts"):
    semantic = _semantics(text)
    tid, _ = compact.begin_turn(ChatRequest(message=text, request_id=request_id,
                                              conversation_id=conversation_id))
    compact.finish_turn(tid, "已处理", {"answer": "已处理"},
                        semantic_values=[(text, semantic)] if semantic else ())
    return tid


def facts(factory):
    with factory() as s:
        return list(s.scalars(select(UserFact)))


def test_explicit_correction_retires_old_fact_and_preserves_source_audit(compact_db):
    old_tid = remember("记住我卧推60公斤，4组8次", "fact-old")
    quote = "我现在卧推65公斤，4组8次，之前的重量已经过时，请以这次为准"
    tid = remember("记住" + quote, "fact-new")
    old, new = sorted(facts(compact_db), key=lambda r: r.status == "active")
    assert old.status == "superseded" and old.valid_to is not None
    assert old.source_turn_id == old_tid and old.source_quote == "我卧推60公斤，4组8次"
    assert new.status == "active" and new.source_quote == quote
    assert new.value == quote and "4组8次" in new.value
    assert new.field == "semantic:" + compact.digest("demo-user")[:16] + ":" + compact.digest("load:卧推")[:48]
    with compact_db() as s:
        row = s.get(MemoryTurn, tid)
        update = row.messages[0]["app_meta"]["fact_update"]
        assert update["supersedes"] == [old.fact_id] and update["fact_id"] == new.fact_id
        assert row.content_hash == compact.digest(row.messages)
        assert s.get(WikiFact, old.fact_id).status == "superseded"
        assert s.get(WikiFact, old.fact_id).valid_to == old.valid_to
        assert s.get(WikiFact, new.fact_id).status == "active"
        assert s.get(MemoryTurn, old_tid).messages[0]["content"] == "记住我卧推60公斤，4组8次"
    active = " ".join(item["text"] for item in compact.current_facts().values())
    assert "60公斤" not in active and "65公斤" in active
    compact.finish_turn(tid, "重复调用", {"answer": "已处理"})
    assert len(facts(compact_db)) == 2


def test_legacy_hashed_fact_replacement_is_lazy_and_same_transaction(compact_db):
    tid = remember("记住我卧推60kg", "legacy-old")
    with compact_db() as s, s.begin():
        old = s.scalar(select(UserFact))
        old.field = "explicit:" + compact.digest(old.value)[:24]
        legacy_id = old.fact_id
    remember("记住我卧推重量改为65kg", "legacy-new")
    rows = facts(compact_db)
    old = next(r for r in rows if r.fact_id == legacy_id)
    assert old.field.startswith("explicit:") and old.status == "superseded"
    assert old.source_turn_id == tid and old.source_quote == "我卧推60kg"
    assert sum(r.status == "active" for r in rows) == 1


@pytest.mark.parametrize("correction", [
    "记住我卧推65公斤",  # no replacement instruction
    "记住我朋友卧推65公斤，以这次为准",
    "记住如果我卧推65公斤，以这次为准",
    "记住我卧推目标65公斤，以这次为准",
    "记住我卧推1RM65公斤，以这次为准",
    "记住我卧推65公斤，深蹲80公斤，以这次为准",
    "记住我卧推从60公斤改为65公斤",  # multi-value corrections need confirmation in this first phase
    "记住我卧推65公斤，但别以这次为准",
    "记住我卧推65公斤，是否改为当前重量？",
    "记住我卧推65公斤，不要改为当前重量",
    "记住我卧推65公斤，要不要以这次为准",
    "记住我不再卧推65公斤，以这次为准",
])
def test_ambiguous_or_unauthorized_update_cannot_retire_existing_load(compact_db, correction):
    remember("记住我卧推60公斤", "negative-old")
    remember(correction, "negative-new")
    old = next(r for r in facts(compact_db) if r.source_quote == "我卧推60公斤")
    assert old.status == "active" and old.valid_to is None
    assert not any(r.status == "superseded" for r in facts(compact_db))


def test_another_exercise_and_attribute_remain_active(compact_db):
    remember("记住我深蹲80公斤", "different-exercise")
    remember("记住我卧推目标100公斤", "different-attribute")
    remember("记住我哑铃卧推20公斤", "different-equipment")
    remember("记住我卧推60公斤", "same-old")
    remember("记住我卧推改为65公斤", "same-new")
    active = [r.value for r in facts(compact_db) if r.status == "active"]
    assert set(active) == {"我深蹲80公斤", "我卧推目标100公斤", "我哑铃卧推20公斤", "我卧推改为65公斤"}


@pytest.mark.parametrize("text", ["我卧推65公斤", "不要记住我卧推65公斤", "如果你记住我卧推65公斤"])
def test_only_explicit_memory_command_creates_facts(compact_db, text):
    remember(text, "not-memory-command")
    assert facts(compact_db) == []


def test_polite_explicit_command_preserves_source_and_load(compact_db):
    remember("请帮我记住我卧推60公斤，4组8次", "polite-command")
    rows = facts(compact_db)
    assert len(rows) == 1 and rows[0].status == "active"
    assert rows[0].source_quote == "我卧推60公斤，4组8次"


def test_failed_final_commit_rolls_back_replacement_and_response(compact_db, monkeypatch):
    from app.services.memory import wiki
    remember("记住我卧推60公斤", "rollback-old")
    tid, _ = compact.begin_turn(ChatRequest(message="记住我卧推改为65公斤",
                                             request_id="rollback-new", conversation_id="facts"))
    def fail(*args):
        raise RuntimeError("fixture_materialization_failure")
    monkeypatch.setattr(wiki, "materialize", fail)
    semantic = _semantics("记住我卧推改为65公斤")
    with pytest.raises(RuntimeError, match="fixture_materialization_failure"):
        compact.finish_turn(tid, "已处理", {"answer": "已处理"},
                            semantic_values=[("记住我卧推改为65公斤", semantic)])
    rows = facts(compact_db)
    assert len(rows) == 1 and rows[0].status == "active" and rows[0].valid_to is None
    with compact_db() as s:
        row = s.get(MemoryTurn, tid)
        assert row.status == "in_progress" and row.response is None
        assert "fact_update" not in row.messages[0]["app_meta"]


def test_purge_new_source_does_not_reactivate_superseded_history(compact_db):
    remember("记住我卧推60公斤", "delete-old", "old-conversation")
    remember("记住我卧推改为65公斤", "delete-new", "new-conversation")
    compact.delete_memory("new-conversation")
    rows = facts(compact_db)
    assert len(rows) == 1 and rows[0].status == "superseded"
    assert compact.current_facts() == {}


def test_serialized_corrections_have_one_active_fact_and_linked_history(compact_db):
    """Competing revisions applied under the profile lock must form a chain."""
    remember("记住我卧推60公斤", "serial-old")
    first_tid = remember("记住我卧推改为65公斤", "serial-first")
    second_tid = remember("记住我卧推改为70公斤", "serial-second")
    rows = facts(compact_db)
    assert len(rows) == 3 and sum(r.status == "active" for r in rows) == 1
    first = next(r for r in rows if r.source_turn_id == first_tid)
    with compact_db() as s:
        update = s.get(MemoryTurn, second_tid).messages[0]["app_meta"]["fact_update"]
        assert update["supersedes"] == [first.fact_id]
        assert s.get(WikiFact, first.fact_id).status == "superseded"


def test_concurrent_final_commits_serialize_before_reading_active_facts(compact_db, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread
    from app.services.memory import wiki
    remember("记住我卧推60公斤", "parallel-old")
    texts = [f"记住我卧推改为{weight}公斤" for weight in (65, 70)]
    tids = [compact.begin_turn(ChatRequest(message=text,
                request_id=f"parallel-{weight}", conversation_id="facts"))[0]
            for text, weight in zip(texts, (65, 70))]
    first_ready, second_entered, release = Event(), Event(), Event()
    materialize, lock = wiki.materialize, compact.lock_profile
    semantic_values = {tid: [] for tid in tids}
    def held_materialize(s, turn):
        if turn.turn_id == tids[0]:
            first_ready.set()
            assert release.wait(5), "test did not release first writer"
        return materialize(s, turn)
    def observed_lock(s):
        if current_thread().name.startswith("second-writer"):
            second_entered.set()
        result = lock(s)
        index = 0 if current_thread().name.startswith("first-writer") else 1
        # The model snapshot is read under the same profile lock as the commit,
        # so the second correction sees the first writer's active value.
        if not semantic_values[tids[index]]:
            text = texts[index]
            semantic_values[tids[index]].append((text, _semantics(text)))
        return result
    monkeypatch.setattr(wiki, "materialize", held_materialize)
    monkeypatch.setattr(compact, "lock_profile", observed_lock)
    with ThreadPoolExecutor(1, thread_name_prefix="first-writer") as first_pool, \
            ThreadPoolExecutor(1, thread_name_prefix="second-writer") as second_pool:
        first = first_pool.submit(compact.finish_turn, tids[0], "已处理", {"answer": "已处理"},
                                  semantic_values=semantic_values[tids[0]])
        try:
            assert first_ready.wait(5)
            second = second_pool.submit(compact.finish_turn, tids[1], "已处理", {"answer": "已处理"},
                                        semantic_values=semantic_values[tids[1]])
            assert second_entered.wait(5)
        finally:
            release.set()
        first.result(timeout=10)
        second.result(timeout=10)
    rows = facts(compact_db)
    assert len(rows) == 3 and sum(r.status == "active" for r in rows) == 1
    assert next(r for r in rows if r.status == "active").value == "我卧推改为70公斤"
    first_fact = next(r for r in rows if r.source_turn_id == tids[0])
    with compact_db() as s:
        assert s.get(MemoryTurn, tids[1]).messages[0]["app_meta"]["fact_update"]["supersedes"] == [first_fact.fact_id]
