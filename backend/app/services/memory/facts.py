"""Apply explicit memories inside the final response transaction.

The semantic model owns meaning when it is available. When no model
understanding exists (offline/unconfigured adapter, or the adapter failed),
the previous conservative load parser still commits explicit "记住…" facts so
an unavailable model never silently drops an explicit memory request.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re

from sqlalchemy import func, or_, select

from app.db.memory_models import MemoryTurn, UserFact
from app.services.agent.semantics import MemoryRequest


def _binding(fact):
    return {'fact_id': fact.fact_id, 'field': fact.field,
            'source_turn_id': fact.source_turn_id, 'source_message_id': fact.source_message_id}


def semantic_field(owner, key, digest):
    return 'semantic:' + digest(owner)[:16] + ':' + digest(key)[:48]


def store_explicit_fact(s, *, row, user, owner, epoch, decision, allowed_fields,
                        digest, timestamp, new_id, target_sources):
    """Model chooses meaning; SQL validates source, active identity and atomicity.

    No inference or paid call occurs while the profile/transaction lock is held.
    Facts remain sourced to the current user turn. Unknown/stale replacement
    targets cannot retire active data.
    """
    item = MemoryRequest.model_validate(decision)
    if item.quote not in user.get("content", "") or item.confidence < .8:
        raise ValueError("memory_quote_not_in_current_user_message")
    if not set(item.target_fields) <= set(allowed_fields):
        raise ValueError("memory_target_outside_model_snapshot")
    if row.memory_epoch != epoch or row.hidden_at or row.deleted_at:
        raise ValueError("memory_source_no_longer_visible")
    namespace = "semantic:" + digest(owner)[:16] + ":"
    field = semantic_field(owner, item.key, digest)
    targets = list(s.scalars(select(UserFact).join(
        MemoryTurn, UserFact.source_turn_id == MemoryTurn.turn_id).where(
        UserFact.memory_epoch == epoch, UserFact.status == "active",
        or_(UserFact.field.startswith(namespace), UserFact.field.in_(allowed_fields)),
        UserFact.valid_from <= func.now(),
        or_(UserFact.valid_to.is_(None), UserFact.valid_to > func.now()),
        MemoryTurn.memory_epoch == epoch, MemoryTurn.deleted_at.is_(None),
        MemoryTurn.hidden_at.is_(None),
        or_(MemoryTurn.status == "completed", MemoryTurn.turn_id == row.turn_id))))
    same_key = [old for old in targets if old.field == field]
    requested = [old for old in targets if old.field in item.target_fields]
    # The model's semantic key is not a durable content identity. Repeated
    # explicit content can reuse an owned, live source even when keys differ.
    # Do not treat an implicit/ambiguous replacement as a harmless repeat.
    existing = next((old for old in targets if old.field.startswith(namespace)
                     and old.source_quote == item.quote), None)
    if existing is not None and item.action == "remember" and not item.target_fields:
        return {"status": existing.status, "quote": existing.source_quote, 'binding': _binding(existing)}
    pending = s.scalar(select(UserFact).join(MemoryTurn, UserFact.source_turn_id == MemoryTurn.turn_id).where(UserFact.field == field,
                                      UserFact.status == "pending_confirmation",
                                      UserFact.memory_epoch == epoch,
                                      UserFact.source_quote == item.quote,
                                      MemoryTurn.memory_epoch == epoch, MemoryTurn.hidden_at.is_(None),
                                      MemoryTurn.deleted_at.is_(None),
                                      or_(MemoryTurn.status == "completed", MemoryTurn.turn_id == row.turn_id)))
    replace = item.action == "replace"
    target_fields = {old.field for old in requested}
    stale = bool(set(item.target_fields) - target_fields)
    for old in requested:
        expected = target_sources.get(old.field) or {}
        if expected.get("source_message_id") != old.source_message_id or expected.get("source_quote") != old.source_quote:
            stale = True
    # Only explicit replacement may retire values; concurrent disappearance or
    # a conflicting stable key outside the supplied targets remains pending.
    conflicting = [old for old in same_key if old.field not in item.target_fields
                   and old.source_quote != item.quote]
    status = "pending_confirmation" if stale or conflicting or (
        item.action == "remember" and requested) else "active"
    if pending is not None and status == "pending_confirmation":
        return {"status": "pending_confirmation", "quote": item.quote}
    fact_id = existing.fact_id if existing is not None and status == "active" else new_id()
    if existing is None or status != "active":
        s.add(UserFact(fact_id=fact_id, field=field, value=item.quote,
                       source_turn_id=row.turn_id, source_message_id=user["message_id"],
                       source_quote=item.quote, memory_epoch=epoch, status=status))
    superseded = []
    if pending is not None and status == "active":
        pending.status = "superseded"
        pending.valid_to = timestamp
        superseded.append(pending.fact_id)
    if replace and status == "active":
        for old in requested:
            if old.fact_id == fact_id:
                continue
            old.status = "superseded"
            old.valid_to = timestamp
            superseded.append(old.fact_id)
    first = dict(row.messages[0])
    meta = dict(first.get("app_meta") or {})
    actual_field = existing.field if existing is not None and status == "active" else field
    update = {"fact_id": fact_id, "field": actual_field, "supersedes": superseded, "status": status}
    updates = list(meta.get("fact_updates") or [])
    updates.append(update)
    meta.update(fact_update=update, fact_updates=updates)
    first["app_meta"] = meta
    row.messages = [first, *row.messages[1:]]
    return {"status": status, "quote": item.quote, 'binding': _binding(existing) if existing is not None and status == 'active'
            else {'fact_id': fact_id, 'field': actual_field, 'source_turn_id': row.turn_id, 'source_message_id': user['message_id']}}


# ---------------------------------------------------------------------------
# Deterministic fallback used only when no semantic understanding exists.
# ---------------------------------------------------------------------------

_RULE_EXERCISES = (
    "哑铃卧推", "杠铃卧推", "上斜卧推", "下斜卧推", "哑铃肩推", "杠铃肩推",
    "哑铃划船", "杠铃划船", "哑铃弯举", "杠铃弯举", "罗马尼亚硬拉",
    "卧推", "深蹲", "硬拉", "肩推", "腿举", "划船", "弯举", "飞鸟",
)
_RULE_EXERCISE = re.compile("|".join(map(re.escape, _RULE_EXERCISES)))
_RULE_LOAD = re.compile(r"(?<![\da-z.])(\d+(?:\.\d+)?)\s*(公斤|千克|kg|斤|磅|lb)(?![a-z])", re.I)
_RULE_REPLACE = re.compile(r"改为|改成|更新为|(?:之前|原来|旧).{0,12}(?:过时|作废|不再有效)|以这次为准")
_RULE_REPLACE_BLOCKERS = re.compile(
    r"不要|不用|不必|别|暂不|先不|不以|是否|能否|能不能|可不可以|要不要|该不该|"
    r"是不是|会不会|怎么|如何|吗|[？?]"
)
_RULE_UNSAFE_IDENTITY = re.compile(
    r"朋友|他|她|别人|假如|如果|假设|例如|举例|可能|打算|计划|目标|希望|争取|"
    r"1\s*rm|极限|最大|体重|去年|以前|曾经|不是|并非|没有|没做|不再(?!有效)|不适用", re.I
)


@dataclass(frozen=True)
class RuleLoadFact:
    exercise: str
    amount: str
    unit: str

    def field(self, owner):
        return f"explicit_load:{owner}:{self.exercise}:recorded_load"


def parse_rule_load(value: str) -> RuleLoadFact | None:
    """Only one self-reported exercise/load, never a goal or inferred 1RM."""
    if not re.match(r"^(?:我|我的)", value) or _RULE_UNSAFE_IDENTITY.search(value):
        return None
    exercises = list(_RULE_EXERCISE.finditer(value))
    loads = list(_RULE_LOAD.finditer(value))
    if len(exercises) != 1 or len(loads) != 1 or loads[0].start() < exercises[0].end():
        return None
    if not re.fullmatch(r"(?:的|现在|目前|当前|重量|负重|记录|是|为|用|改为|改成|更新为|\s)*",
                        value[exercises[0].end():loads[0].start()]):
        return None
    prefix = value[:exercises[0].start()]
    if not re.fullmatch(r"(?:我|我的)(?:现在|目前|当前|的|训练|记录|重量|负重|是|用)*", prefix):
        return None
    number = Decimal(loads[0][1])
    if number <= 0:
        return None
    amount = format(number.normalize(), "f")
    unit = loads[0][2].lower()
    unit = {"kg": "公斤", "千克": "公斤", "lb": "磅"}.get(unit, unit)
    return RuleLoadFact(exercises[0][0], amount, unit)


def store_rule_explicit_fact(s, *, row, user, owner, epoch, value, digest, timestamp, new_id):
    """Conservative fallback under the profile lock; never infers entities."""
    parsed = parse_rule_load(value)
    has_cue = bool(_RULE_REPLACE.search(value))
    replace = has_cue and not _RULE_REPLACE_BLOCKERS.search(value)
    field = parsed.field(owner) if parsed else "explicit:" + digest(value)[:24]
    if s.scalar(select(UserFact).where(UserFact.field == field,
                                      UserFact.status == "pending_confirmation",
                                      UserFact.source_quote == value)):
        return
    active = list(s.scalars(select(UserFact).where(UserFact.status == "active")))
    matching = []
    if parsed:
        for old in active:
            old_load = parse_rule_load(old.source_quote)
            if old.field == field or (
                old.field.startswith("explicit:") and old_load
                and old_load.exercise == parsed.exercise
            ):
                matching.append(old)
        if any(parse_rule_load(old.source_quote) == parsed for old in matching) and not replace:
            return
    elif any(old.field == field for old in active):
        return

    # An ambiguous correction or a conflicting value without replacement
    # authorization stays pending; it cannot retire or contradict active facts.
    status = "active"
    if (has_cue and not (parsed and replace)) or (matching and not replace):
        status = "pending_confirmation"
    fact_id = new_id()
    s.add(UserFact(fact_id=fact_id, field=field, value=value,
                   source_turn_id=row.turn_id, source_message_id=user["message_id"],
                   source_quote=value, memory_epoch=epoch, status=status))
    superseded = []
    if parsed and replace:
        for old in matching:
            old.status = "superseded"
            old.valid_to = timestamp
            superseded.append(old.fact_id)
    first = dict(row.messages[0])
    meta = dict(first.get("app_meta") or {})
    meta["fact_update"] = {"fact_id": fact_id, "field": field,
                           "supersedes": superseded, "status": status}
    meta["fact_updates"] = [*(meta.get("fact_updates") or []), meta["fact_update"]]
    first["app_meta"] = meta
    row.messages = [first, *row.messages[1:]]
