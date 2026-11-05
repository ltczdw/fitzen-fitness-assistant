"""C2-2 nutrition transaction bridge fixtures only. No bridge implementation.

The helper prepares requests/tasks/operations with the real C1
``ExecutionLedgerStore`` API and uses the real isolated database/isolation
helpers from the frozen C1 support module. The nutrition bridge import stays
lazy so a missing implementation is a genuine failure, never a skip.

Food reference rows for the isolated synthetic databases are created with raw
SQL in this helper only: PostgreSQL gets the real migrated catalog (seeded by
``init_db``/``migrate_food_schema``), SQLite gets a test-local compatible
catalog. These rows are fixtures, never business writes.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from tests.react_phase_c1_support import (
    fail_after_insert,
    fail_sql,
    isolated,
    network_fence,
    race,
    rows,
    sync,
)
from tests.react_phase_c2_1_support import (
    CommitCountingSession,
    delay_on_insert,
    fail_on_event_key,
)

C2_2_MODULE = "app.services.agent.nutrition_write_bridge"
BRIDGE_CLASS = "NutritionWriteBridge"

PENDING_PREFIX = "c2-2-pending:v1:"
USE_PREFIX = "c2-2-confirmation-use:v1:"
CLAIM_PREFIX = "c2-2-claim:v1:"
COMMIT_PREFIX = "c2-2-commit:v1:"
RESERVED_PREFIXES = (PENDING_PREFIX, USE_PREFIX, CLAIM_PREFIX, COMMIT_PREFIX)

NUTRITION_ORIGIN_KEY = "_nutrition_bridge_origin"
NUTRITION_ORIGIN = {"protocol": 2, "generation": "ea0c7b67-2ec9-4db9-a3a8-81a3475e4cda"}
TRAINING_ORIGIN = {"protocol": 2, "generation": "60459400-f1c8-4ca3-875c-54d03b755834"}

NUTRITION_TABLES = ("nutrition_logs", "user_food_log")

FROZEN_DATE = "2026-10-07"
FROZEN_TIME_BASE = "2026-10-07T03:30:00+00:00"  # 11:30 Asia/Shanghai on 2026-10-07
FROZEN_RECORD_TIME = "2026-10-07T12:00:00+08:00"

MANUAL_SOURCE = "帮我记录今天午餐：牛肉面一碗，热量520千卡，蛋白质25克，碳水60克，脂肪15克。谢谢"
MANUAL_SPAN = "牛肉面一碗，热量520千卡，蛋白质25克，碳水60克，脂肪15克"

FOOD_SOURCE = "帮我记录今天午餐：一碗米饭，两个鸡蛋。谢谢"
FOOD_SPAN = "一碗米饭，两个鸡蛋"
AI_SOURCE = "帮我记录今天午餐：300克神秘盖饭。谢谢"
AI_SPAN = "300克神秘盖饭"
CUSTOM_SOURCE = ("帮我记录今天午餐：200克自制沙拉，每100克热量120千卡，蛋白质5克，碳水10克，"
                 "脂肪7克，纤维2克。谢谢")
CUSTOM_ITEM_SPAN = "200克自制沙拉"
CUSTOM_SPAN = "200克自制沙拉，每100克热量120千卡，蛋白质5克，碳水10克，脂肪7克，纤维2克"


def emit(case, **values):
    print("C2_2_EVIDENCE=" + json.dumps({"case": case, **values}, ensure_ascii=False, allow_nan=False))


def check_all(case, checks):
    emit(case, checks=checks)
    assert all(checks.values()), f"{case}: {[k for k, v in checks.items() if not v]}"


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def op_id_for(user_id: str, request_id: str, task_id: str, position: int, kind: str) -> str:
    material = ["ledger-v2", user_id, request_id, task_id, position, kind]
    return "op2_" + hashlib.sha256(canonical(material)).hexdigest()[:32]


def event_by_key(snapshot: dict, event_key: str) -> dict | None:
    if not snapshot:
        return None
    for item in snapshot["events"]:
        if item["event_key"] == event_key:
            return item
    return None


def bridge_module():
    return importlib.import_module(C2_2_MODULE)


def bridge_outcome(function):
    """Capture a raised bridge error without echoing arbitrary exception text."""
    try:
        return {"ok": True, "value": function(), "raised": None}
    except Exception as exc:  # pragma: no cover - genuine protocol failure
        return {"ok": False, "value": None, "raised": type(exc).__name__}


def span_of(source: str, text: str) -> dict:
    start = source.index(text)
    return {"start": start, "end": start + len(text), "exact_text": text}


def span_after(source: str, text: str, offset: int = 0) -> dict:
    start = source.index(text, offset)
    return {"start": start, "end": start + len(text), "exact_text": text}


def nutrition_rows(engine) -> dict:
    return rows(engine, NUTRITION_TABLES)


def food_rows_for_key(engine, user_id: str, idempotency_key: str) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT id, idempotency_key, food_name, weight_g, meal_group_id, total_kcal "
                 "FROM user_food_log WHERE user_id=:user_id AND idempotency_key LIKE :prefix ORDER BY id"),
            {"user_id": user_id, "prefix": idempotency_key + ":%"},
        )
        return [dict(row._mapping) for row in result]


# ---------------------------------------------------------------- catalog


SQLITE_CATALOG_DDL = (
    """CREATE TABLE IF NOT EXISTS food_nutrition (
        id INTEGER PRIMARY KEY AUTOINCREMENT, food_name VARCHAR(200) NOT NULL,
        category VARCHAR(100) NOT NULL DEFAULT '其他', preparation_state VARCHAR(50) NOT NULL DEFAULT '熟',
        energy_kcal NUMERIC(10,2) NOT NULL, protein_g NUMERIC(10,2) NOT NULL,
        carb_g NUMERIC(10,2) NOT NULL, fat_g NUMERIC(10,2) NOT NULL, fiber_g NUMERIC(10,2) NOT NULL DEFAULT 0,
        source_name VARCHAR(200) NOT NULL DEFAULT 'public-summary', source_version VARCHAR(100) NOT NULL DEFAULT 'public-summary-v1',
        is_active BOOLEAN NOT NULL DEFAULT 1, UNIQUE(food_name, preparation_state)
    )""",
    """CREATE TABLE IF NOT EXISTS food_alias_index (
        id INTEGER PRIMARY KEY AUTOINCREMENT, food_id INTEGER NOT NULL,
        alias_text VARCHAR(200) NOT NULL, normalized_alias VARCHAR(200) NOT NULL,
        alias_type VARCHAR(50) NOT NULL DEFAULT 'manual', confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
        is_active BOOLEAN NOT NULL DEFAULT 1, UNIQUE(food_id, normalized_alias)
    )""",
    """CREATE TABLE IF NOT EXISTS food_portion_reference (
        id INTEGER PRIMARY KEY AUTOINCREMENT, food_id INTEGER NOT NULL,
        portion_name VARCHAR(50) NOT NULL, weight_g NUMERIC(10,2) NOT NULL,
        source VARCHAR(200) NOT NULL DEFAULT 'public-portion-estimate', UNIQUE(food_id, portion_name)
    )""",
)

CATALOG_FOODS = (
    # name, category, state, kcal, protein, carb, fat, fiber
    ("熟大米", "谷薯类", "熟", 116, 2.60, 25.90, 0.30, 0.30),
    ("鸡蛋", "蛋类", "熟", 144, 12.60, 1.10, 9.50, 0),
)

CATALOG_ALIASES = (
    ("米饭", "熟大米"), ("饭", "熟大米"), ("蛋", "鸡蛋"),
)

CATALOG_PORTIONS = (
    ("熟大米", "碗", 150), ("熟大米", "份", 150), ("鸡蛋", "个", 50), ("鸡蛋", "枚", 50),
)


def ensure_catalog(engine) -> None:
    """Create (SQLite) and seed the real reference tables in the isolated DB."""
    from app.db.food_repository import normalize_food_name

    dialect = engine.dialect.name
    with engine.begin() as conn:
        if dialect == "sqlite":
            for ddl in SQLITE_CATALOG_DDL:
                conn.execute(text(ddl))
        for name, category, state, kcal, protein, carb, fat, fiber in CATALOG_FOODS:
            conn.execute(
                text("""INSERT INTO food_nutrition (food_name, category, preparation_state, energy_kcal, protein_g, carb_g, fat_g, fiber_g)
                        VALUES (:name,:category,:state,:kcal,:protein,:carb,:fat,:fiber)
                        ON CONFLICT DO NOTHING"""),
                {"name": name, "category": category, "state": state, "kcal": kcal,
                 "protein": protein, "carb": carb, "fat": fat, "fiber": fiber},
            )
        ids = {}
        for row in conn.execute(text("SELECT id, food_name FROM food_nutrition WHERE is_active=TRUE")):
            ids[row[1]] = int(row[0])
        for alias, standard in CATALOG_ALIASES:
            food_id = ids.get(standard)
            if food_id is None:
                continue
            conn.execute(
                text("""INSERT INTO food_alias_index (food_id, alias_text, normalized_alias)
                        VALUES (:food_id,:alias,:normalized) ON CONFLICT DO NOTHING"""),
                {"food_id": food_id, "alias": alias, "normalized": normalize_food_name(alias)},
            )
        for standard, portion, weight in CATALOG_PORTIONS:
            food_id = ids.get(standard)
            if food_id is None:
                continue
            conn.execute(
                text("""INSERT INTO food_portion_reference (food_id, portion_name, weight_g)
                        VALUES (:food_id,:portion,:weight) ON CONFLICT DO NOTHING"""),
                {"food_id": food_id, "portion": portion, "weight": weight},
            )


# ---------------------------------------------------------------- batches


def manual_group(source: str, *, text: str | None = None, meal_type: str = "午餐",
                 log_date: str = FROZEN_DATE, description: str = "牛肉面一碗",
                 calories: float | None = 520.0, protein_g: float | None = 25.0,
                 carbs_g: float | None = 60.0, fat_g: float | None = 15.0) -> dict:
    return {
        "source_ref": span_of(source, text or MANUAL_SPAN),
        "meal_type": meal_type,
        "log_date": log_date,
        "description": description,
        "calories": calories,
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
    }


def manual_batch(source: str = MANUAL_SOURCE, *, groups=None, timezone: str = "Asia/Shanghai") -> dict:
    return {"route": "manual_nutrition_log", "timezone": timezone,
            "groups": list(groups if groups is not None else [manual_group(source)])}


def food_entry(source: str, *, raw_text: str, requested_name: str, quantity: float, unit: str,
               weight_g: float | None = None, portion_name: str | None = None,
               meal_type: str | None = "午餐", record_date: str | None = FROZEN_DATE,
               is_negated: bool = False) -> dict:
    return {
        "raw_text": raw_text, "requested_name": requested_name, "quantity": quantity, "unit": unit,
        "weight_g": weight_g, "portion_name": portion_name, "meal_type": meal_type,
        "record_date": record_date, "is_negated": is_negated,
    }


def food_item(source: str, *, text: str, entry: dict, origin: str = "local",
              nutrition_per_100g: dict | None = None) -> dict:
    return {"source_ref": span_of(source, text), "entry": dict(entry), "source": origin,
            "nutrition_per_100g": copy.deepcopy(nutrition_per_100g)}


def food_group(source: str, *, text: str | None = None, meal_type: str = "午餐",
               record_time: str = FROZEN_RECORD_TIME, items=None) -> dict:
    return {"source_ref": span_of(source, text or FOOD_SPAN),
            "meal_type": meal_type, "record_time": record_time, "foods": list(items or [])}


def food_batch(source: str = FOOD_SOURCE, *, groups=None, timezone: str = "Asia/Shanghai") -> dict:
    return {"route": "food_items", "timezone": timezone,
            "groups": list(groups if groups is not None else [])}


def local_rice_item(source: str = FOOD_SOURCE) -> dict:
    return food_item(source, text="一碗米饭",
                     entry=food_entry(source, raw_text="一碗米饭", requested_name="米饭", quantity=1.0,
                                      unit="碗", portion_name="碗"))


def local_egg_item(source: str = FOOD_SOURCE) -> dict:
    return food_item(source, text="两个鸡蛋",
                     entry=food_entry(source, raw_text="两个鸡蛋", requested_name="鸡蛋", quantity=2.0,
                                      unit="个", portion_name="个"))


def ai_item(source: str = AI_SOURCE, *, kcal: float = 180.0, protein: float = 7.0,
            carb: float = 25.0, fat: float = 6.0, fiber: float = 1.0) -> dict:
    return food_item(source, text="300克神秘盖饭",
                     entry=food_entry(source, raw_text="300克神秘盖饭", requested_name="神秘盖饭",
                                      quantity=1.0, unit="克", weight_g=300.0, portion_name=None),
                     origin="ai",
                     nutrition_per_100g={"kcal": kcal, "protein_g": protein, "carbs_g": carb,
                                         "fat_g": fat, "fiber_g": fiber})


def custom_item(source: str = CUSTOM_SOURCE, *, kcal: float = 120.0, protein: float = 5.0,
                carb: float = 10.0, fat: float = 7.0, fiber: float = 2.0) -> dict:
    return food_item(source, text=CUSTOM_ITEM_SPAN,
                     entry=food_entry(source, raw_text=CUSTOM_ITEM_SPAN, requested_name="自制沙拉",
                                      quantity=1.0, unit="克", weight_g=200.0, portion_name=None),
                     origin="custom",
                     nutrition_per_100g={"kcal": kcal, "protein_g": protein, "carbs_g": carb,
                                         "fat_g": fat, "fiber_g": fiber})


# ---------------------------------------------------------------- harness


NORMAL_SAFETY = {"level": "normal", "evidence": "", "categories": [],
                 "reason": "c2-2 synthetic request source"}


@contextmanager
def assessed_safety(*sources):
    """Bind normal source-bound safety to this test request.

    The production bridge requires the request-local model SafetyAssessment
    before any write. These tests script the semantic layer away, so the
    fixture supplies the same assessed state a real request would have.
    """
    from app.services.agent.semantics import request_understanding

    current = dict(request_understanding.get() or {})
    for source in sources:
        if isinstance(source, str) and source:
            current[source] = {"safety": dict(NORMAL_SAFETY),
                               **(current.get(source) or {})}
    token = request_understanding.set(current)
    try:
        yield
    finally:
        request_understanding.reset(token)


class NutritionBridgeHarness:
    """Real C1 ledger preparation plus the real bridge over one isolated engine."""

    def __init__(self, engine, *, batches=None, source=None, task_type="nutrition_record",
                 kind="nutrition", ttl=300, user_id="c2-2-owner", request_id="c2-2-request",
                 conversation_id="c2-2-conversation", task_id="task-1", position=0,
                 time_base=FROZEN_TIME_BASE, source_digest=None, auto_freeze=True,
                 auto_reserve=True, dispatch=True, session_class=None, task_span=None):
        from app.services.agent import execution_ledger_store as ledger_api

        self.ledger = ledger_api
        self.engine = engine
        self.factory = sessionmaker(bind=engine, expire_on_commit=False)
        self.store = ledger_api.ExecutionLedgerStore(self.factory)
        if session_class is not None:
            self.bridge_factory = sessionmaker(bind=engine, class_=session_class, expire_on_commit=False)
        else:
            self.bridge_factory = self.factory
        self.source = source if source is not None else MANUAL_SOURCE
        self.task_type = task_type
        self.parent_task_span = task_span
        self.kind = kind
        self.task_id = task_id
        self.position = position
        self.batches = [copy.deepcopy(item) for item in (batches or [manual_batch(self.source)])]
        self.identity = {
            "user_id": user_id,
            "request_id": request_id,
            "conversation_id": conversation_id,
            "source": self.source,
            "source_digest": source_digest or hashlib.sha256(self.source.encode("utf-8")).hexdigest(),
            "requested_mode": "react_readonly",
            "effective_mode": "react_readonly",
            "schema_version": 2,
            "tool_version": "react-tools-v2",
            "time_base": time_base,
            "budget_scope_id": "scope:synthetic-c2-2",
            "request_kind": "tasks",
        }
        snapshot = sync(self.store.ensure_request(ledger_api.RequestIdentity(**self.identity)))
        snapshot = sync(self.store.acquire_lease(self.identity["user_id"], self.identity["request_id"],
                                                 "c2-2-worker", ttl))
        self.snapshot = snapshot
        if auto_freeze:
            self.freeze_task()
        if auto_reserve:
            self.record_op(self.position, dispatch=dispatch)
        self.proof = self.lease_proof()

    # -------------------------------------------------------------- C1 side

    @property
    def ids(self):
        return self.identity["user_id"], self.identity["request_id"]

    def lease_proof(self):
        lease = self.snapshot["lease"]
        return self.ledger.LeaseProof(worker_id=lease["worker_id"], token=lease["token"], epoch=lease["epoch"])

    def call_store(self, name, *args, **kwargs):
        kwargs.setdefault("expected_revision", self.snapshot["revision"])
        if "lease" not in kwargs:
            kwargs["lease"] = self.lease_proof()
        self.snapshot = sync(getattr(self.store, name)(*self.ids, *args, **kwargs))
        return self.snapshot

    def freeze_task(self, *, task_id=None, task_type=None, parameters=None):
        task_id = task_id or self.task_id
        parameters = parameters if parameters is not None else {"batches": copy.deepcopy(self.batches)}
        if self.parent_task_span is not None:
            start, end = self.parent_task_span
        else:
            start, end = 0, len(self.source)
        source_digest = hashlib.sha256(self.source[start:end].encode("utf-8")).hexdigest()
        task = self.ledger.TaskSpec(
            task_id=task_id, source_start=start, source_end=end,
            source_digest=source_digest, task_type=task_type or self.task_type,
            dependencies=[], parameters=copy.deepcopy(parameters), required=True)
        self.snapshot = sync(self.call_store("freeze_task", task))
        return task

    def record_op(self, position=0, *, kind=None, parameters=None, task_id=None, dispatch=True):
        task_id = task_id or self.task_id
        self.snapshot = sync(self.call_store(
            "reserve_operation",
            task_id=task_id, position=position, kind=kind or self.kind,
            parameters=copy.deepcopy(parameters if parameters is not None else self.batches[position]),
            source_digest=self.identity["source_digest"], time_base=self.identity["time_base"]))
        op_id = self.op_id(position, task_id=task_id)
        if dispatch:
            self.snapshot = sync(self.call_store("mark_dispatch_intent", op_id))
        return op_id

    def op_id(self, position=0, *, task_id=None):
        return op_id_for(self.identity["user_id"], self.identity["request_id"],
                         task_id or self.task_id, position, self.kind)

    def op_key(self, position=0, *, task_id=None):
        digest = hashlib.sha256(canonical([
            "ledger-v2", self.identity["user_id"], self.identity["request_id"],
            task_id or self.task_id, position, self.kind])).hexdigest()
        return "ledger-v2:" + digest

    def reload(self):
        self.snapshot = sync(self.ledger.ExecutionLedgerStore(self.factory).load_request(*self.ids))
        return self.snapshot

    def event(self, event_key):
        return event_by_key(self.snapshot, event_key)

    def operation(self, op_id=None):
        target = op_id or self.op_id(self.position)
        return next(item for item in self.snapshot["operations"] if item["op_id"] == target)

    def task(self, task_id=None):
        target = task_id or self.task_id
        return next(item for item in self.snapshot["tasks"] if item["task_id"] == target)

    def pending_event_key(self, op_id=None):
        return PENDING_PREFIX + (op_id or self.op_id(self.position))

    def claim_event_key(self, op_id=None):
        return CLAIM_PREFIX + (op_id or self.op_id(self.position))

    def confirmation_event_key(self, op_id=None):
        return COMMIT_PREFIX + (op_id or self.op_id(self.position))

    def use_event_key(self, confirmation_request_id):
        return USE_PREFIX + confirmation_request_id

    # --------------------------------------------------------------- bridge

    def bridge(self):
        return getattr(bridge_module(), BRIDGE_CLASS)(self.bridge_factory)

    def prepare(self, *, op_id=None, expected_revision=None, lease=None, tool_call_id="call-c2-2-0001",
                batch_id="batch-c2-2"):
        bridge = self.bridge()
        with assessed_safety(self.source):
            result = bridge.prepare_pending_nutrition(
                self.identity["user_id"], self.identity["request_id"], op_id or self.op_id(self.position),
                expected_revision=self.snapshot["revision"] if expected_revision is None else expected_revision,
                lease=self.lease_proof() if lease is None else lease,
                tool_call_id=tool_call_id, batch_id=batch_id)
        self.reload()
        return result

    def commit(self, *, op_id=None, expected_revision=None, lease=None, tool_call_id="call-c2-2-0001",
               batch_id="batch-c2-2", confirmation=None, **overrides):
        bridge = self.bridge()
        kwargs = dict(
            expected_revision=self.snapshot["revision"] if expected_revision is None else expected_revision,
            lease=self.lease_proof() if lease is None else lease,
            tool_call_id=tool_call_id, batch_id=batch_id)
        if confirmation:
            kwargs.update(confirmation)
        kwargs.update(overrides)
        with assessed_safety(self.source, "确认保存"):
            result = bridge.commit_nutrition_operation(
                self.identity["user_id"], self.identity["request_id"], op_id or self.op_id(self.position), **kwargs)
        self.reload()
        return result

    def recover(self, *, op_id=None):
        bridge = self.bridge()
        result = bridge.recover_nutrition_operation(
            self.identity["user_id"], self.identity["request_id"], op_id or self.op_id(self.position))
        self.reload()
        return result


class ConfirmationHarness:
    """A real C1 confirmation request bound to one protected pending."""

    def __init__(self, engine, pending_harness, *, confirmation_source="确认保存",
                 user_id=None, conversation_id=None, request_id="c2-2-confirm",
                 task_id="confirm-task-1", time_base=None, parameters=None):
        from app.services.agent import execution_ledger_store as ledger_api

        self.ledger = ledger_api
        self.engine = engine
        self.factory = sessionmaker(bind=engine, expire_on_commit=False)
        self.store = ledger_api.ExecutionLedgerStore(self.factory)
        self.pending = pending_harness
        self.source = confirmation_source

        # Default confirmation time: the real isolated-database statement clock
        # (PG clock_timestamp() / SQLite database clock) taken on a real Session
        # after the pending prepare succeeded — the same clock as the pending
        # window, timezone and full precision preserved. An explicitly supplied
        # time_base (including an invalid one) is kept unchanged.
        if time_base is None:
            session = self.factory()
            try:
                time_base = self.store.ledger_statement_now(session).isoformat()
            finally:
                session.close()
        self.taken_time_base = time_base
        self.identity = {
            "user_id": user_id or pending_harness.identity["user_id"],
            "request_id": request_id,
            "conversation_id": conversation_id or pending_harness.identity["conversation_id"],
            "source": self.source,
            "source_digest": hashlib.sha256(self.source.encode("utf-8")).hexdigest(),
            "requested_mode": "react_readonly",
            "effective_mode": "react_readonly",
            "schema_version": 2,
            "tool_version": "react-tools-v2",
            "time_base": time_base,
            "budget_scope_id": "scope:synthetic-c2-2",
            "request_kind": "tasks",
        }
        snapshot = sync(self.store.ensure_request(ledger_api.RequestIdentity(**self.identity)))
        snapshot = sync(self.store.acquire_lease(self.identity["user_id"], self.identity["request_id"],
                                                 "c2-2-confirmer", 300))
        self.snapshot = snapshot
        self.task_id = task_id
        self.parameters = parameters

    @property
    def ids(self):
        return self.identity["user_id"], self.identity["request_id"]

    def lease_proof(self):
        lease = self.snapshot["lease"]
        return self.ledger.LeaseProof(worker_id=lease["worker_id"], token=lease["token"], epoch=lease["epoch"])

    def call_store(self, name, *args, **kwargs):
        kwargs.setdefault("expected_revision", self.snapshot["revision"])
        if "lease" not in kwargs:
            kwargs["lease"] = self.lease_proof()
        self.snapshot = sync(getattr(self.store, name)(*self.ids, *args, **kwargs))
        return self.snapshot

    def freeze(self, *, parameters=None, task_type="nutrition_confirmation", task_id=None):
        task = self.ledger.TaskSpec(
            task_id=task_id or self.task_id, source_start=0, source_end=len(self.source),
            source_digest=hashlib.sha256(self.source.encode("utf-8")).hexdigest(),
            task_type=task_type, dependencies=[],
            parameters=copy.deepcopy(parameters if parameters is not None else self.parameters),
            required=True)
        self.snapshot = sync(self.call_store("freeze_task", task))
        return task

    def reload(self):
        self.snapshot = sync(self.ledger.ExecutionLedgerStore(self.factory).load_request(*self.ids))
        return self.snapshot

    def commit_args(self, *, pending_digest, confirmation_source=None, expected_revision=None, lease=None):
        """The four application-injected confirmation selectors/lease."""
        return {
            "confirmation_request_id": self.identity["request_id"],
            "pending_digest": pending_digest,
            "confirmation_expected_revision": self.snapshot["revision"] if expected_revision is None else expected_revision,
            "confirmation_lease": self.lease_proof() if lease is None else lease,
        }


def confirmation_harness(engine, pending_harness, *, pending_digest, confirmation_source="确认保存",
                         user_id=None, conversation_id=None, request_id="c2-2-confirm",
                         time_base=None, parameters=None):
    """Build the real confirmation request/task for one prepared pending."""
    harness = ConfirmationHarness(
        engine, pending_harness, confirmation_source=confirmation_source, user_id=user_id,
        conversation_id=conversation_id, request_id=request_id, time_base=time_base, parameters=parameters)
    params = parameters if parameters is not None else {
        "pending_request_id": pending_harness.identity["request_id"],
        "pending_op_id": pending_harness.op_id(pending_harness.position),
        "pending_digest": pending_digest,
    }
    harness.freeze(parameters=params)
    return harness


# ---------------------------------------------------------------- statements


class StatementRecorder:
    """Records (verb, statement, physical connection identity) per statement."""

    def __init__(self, engine):
        self.engine = engine
        self.records = []
        self.executions = []
        self._listener = None
        self._rowcount_listener = None

    def __enter__(self):
        from sqlalchemy import event as _event

        def record(conn, cursor, statement, parameters, context, executemany):
            verb = statement.lstrip().split(" ", 1)[0].upper()
            self.records.append((verb, statement, id(conn.connection.driver_connection)))

        def record_rowcount(conn, cursor, statement, parameters, context, executemany):
            verb = statement.lstrip().split(" ", 1)[0].upper()
            self.executions.append((verb, statement, cursor.rowcount))

        self._listener = record
        self._rowcount_listener = record_rowcount
        _event.listen(self.engine, "before_cursor_execute", record)
        _event.listen(self.engine, "after_cursor_execute", record_rowcount)
        return self

    def __exit__(self, *args):
        from sqlalchemy import event as _event

        _event.remove(self.engine, "before_cursor_execute", self._listener)
        _event.remove(self.engine, "after_cursor_execute", self._rowcount_listener)

    def writes(self):
        return [item for item in self.records if item[0] in {"INSERT", "UPDATE", "DELETE"}]

    def connections_for(self, verbs=None):
        return {item[2] for item in self.records if verbs is None or item[0] in verbs}

    def last_write_rowcount(self, table=None):
        for verb, statement, rowcount in reversed(self.executions):
            if verb in {"INSERT", "UPDATE", "DELETE"} and (table is None or table.lower() in statement.lower()):
                return rowcount, statement
        return None, None

    def last_write(self):
        """The last real INSERT/UPDATE/DELETE, never a trailing SELECT."""
        for verb, statement, rowcount in reversed(self.executions):
            if verb in {"INSERT", "UPDATE", "DELETE"}:
                return verb, statement, rowcount
        return None, None, None


@contextmanager
def fail_after_n_inserts(engine, table: str, n: int):
    """Raise after the (n+1)-th real INSERT into ``table`` inside the open TX."""
    from sqlalchemy import event as _event

    evidence = {"reached": False, "count": 0}

    def fail(conn, cursor, statement, parameters, context, executemany):
        normalized = statement.lower().replace('"', "").strip()
        if normalized.startswith("insert into " + table.lower()):
            evidence["count"] += 1
            if evidence["count"] == n + 1:
                evidence["reached"] = True
                raise RuntimeError("synthetic_insert_failure")

    _event.listen(engine, "after_cursor_execute", fail)
    try:
        yield evidence
    finally:
        _event.remove(engine, "after_cursor_execute", fail)


@contextmanager
def fail_on_sql_containing(engine, needle: str):
    """Raise on any SQL statement whose text contains ``needle``."""
    from sqlalchemy import event as _event

    evidence = {"reached": False}

    def fail(conn, cursor, statement, parameters, context, executemany):
        if needle.lower() in statement.lower():
            evidence["reached"] = True
            raise RuntimeError("synthetic_sql_failure")

    _event.listen(engine, "before_cursor_execute", fail)
    try:
        yield evidence
    finally:
        _event.remove(engine, "before_cursor_execute", fail)


def bound_capability(bridge):
    """Test hook for real-capability negative controls: read the local capability
    object from a real bridge instance (never a fabricated value)."""
    capability = getattr(bridge, "_reserved_capability", None)
    if capability is None:
        store = getattr(bridge, "_ledger", None)
        capability = getattr(store, "_nutrition_event_capability", None)
    return capability


# ---------------------------------------------------------------- children


def _runner_path():
    return Path(__file__).resolve().parents[2] / "docs/react-phase-c2-2-evidence/run.py"


def subprocess_nutrition_call(url, backend, user_id, request_id, op_id, mode, **extra):
    """Launch the C2-2 runner child-nutrition entry (real separate process)."""
    import subprocess
    import sys

    packet = {"url": url, "backend": backend, "user_id": user_id, "request_id": request_id,
              "op_id": op_id, "mode": mode, **extra}
    result = subprocess.run(
        [sys.executable, str(_runner_path()), "child-nutrition"], input=json.dumps(packet),
        capture_output=True, text=True, encoding="utf-8", timeout=30,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, (
        "C2-2 child nutrition call failed: " + (result.stderr or result.stdout or "")[-1200:])
    return json.loads(result.stdout)


def c2_2_subprocess_workout_call(url, backend, user_id, request_id, op_id, expected_revision, lease,
                                 tool_call_id, batch_id):
    """Redirect for the old C2-1 helper: fresh-process workout replay via the new runner."""
    import subprocess
    import sys

    packet = {"url": url, "backend": backend, "user_id": user_id, "request_id": request_id,
              "op_id": op_id, "expected_revision": expected_revision,
              "lease": {"worker_id": lease.worker_id, "token": lease.token, "epoch": lease.epoch},
              "tool_call_id": tool_call_id, "batch_id": batch_id}
    result = subprocess.run(
        [sys.executable, str(_runner_path()), "child-workout"], input=json.dumps(packet),
        capture_output=True, text=True, encoding="utf-8", timeout=30,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, "C2-2 child workout call failed (see stable child code)"
    return json.loads(result.stdout)


def c2_2_subprocess_recovery(url, backend, user_id, request_id):
    """Redirect for the old C1 helper: fresh-process C1 recovery via the new runner."""
    import subprocess
    import sys

    packet = {"url": url, "backend": backend, "user_id": user_id, "request_id": request_id}
    result = subprocess.run(
        [sys.executable, str(_runner_path()), "child-read"], input=json.dumps(packet),
        capture_output=True, text=True, encoding="utf-8", timeout=30,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, "C2-2 child C1-recovery call failed (see stable child code)"
    return json.loads(result.stdout)


def _race_with_request_context(original):
    """Run C1's two real connection workers inside a copy of this request's
    context so scripted safety/semantic bindings are visible in both threads,
    exactly like the application's request context."""
    import contextvars

    def race(engine, function):
        captured = contextvars.copy_context()

        def bound(*args, **kwargs):
            return captured.copy().run(function, *args, **kwargs)

        return original(engine, bound)

    return race


def redirect_old_children():
    """Only in this C2-2 plugin: point the old child launchers at the new runner."""
    import tests.react_phase_c1_support as c1_support
    import tests.react_phase_c2_1_support as c2_1_support

    c1_support.subprocess_recovery = c2_2_subprocess_recovery
    c2_1_support.subprocess_bridge_call = c2_2_subprocess_workout_call
    c1_support.race = _race_with_request_context(c1_support.race)


def install_pg_bootstrap_adaptation():
    """Nutrition-PG adaptation layer: the legacy nutrition fixtures import the
    top-level ``quality_isolation.isolated_database``; route their PostgreSQL
    path through the frozen C1 bootstrap/registration logic (UUID database
    allow-list, admin create/drop window, checked_engine/current_database and
    the pinned hostaddr) while the network fence itself stays installed. The
    package instance used by the C1 support is left untouched (no recursion)."""
    import sys
    from pathlib import Path as _Path

    import tests.react_phase_c1_support as c1_support

    tests_dir = str(_Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    try:
        import quality_isolation as legacy_isolation
    except Exception:  # pragma: no cover - legacy fixtures absent
        return
    original = legacy_isolation.isolated_database

    def adapted_isolated_database(monkeypatch, tmp_path, backend):
        if backend == "postgres":
            return c1_support.isolated(monkeypatch, tmp_path, backend)
        return original(monkeypatch, tmp_path, backend)

    legacy_isolation.isolated_database = adapted_isolated_database


# ---------------------------------------------------------------- pytest hooks

_audit_state = {"context": None, "audit": None}


def pytest_sessionstart(session):
    redirect_old_children()
    install_pg_bootstrap_adaptation()
    _audit_state["context"] = network_fence()
    _audit_state["audit"] = _audit_state["context"].__enter__()


def pytest_collection_finish(session):
    import os

    path = os.environ.get("FITZEN_C2_2_COLLECTION_MANIFEST")
    if path:
        Path(path).write_text(json.dumps([item.nodeid for item in session.items], indent=2), encoding="utf-8")


def pytest_sessionfinish(session, exitstatus):
    audit = _audit_state["audit"]
    _audit_state["context"].__exit__(None, None, None)
    _audit_state["context"] = None
    _audit_state["audit"] = None
    emit(
        "NETWORK",
        blocked_connections=audit["denied"]["blocked_connections"],
        blocked_dns=audit["denied"]["blocked_dns"],
        allowed_pg_socket_connections=audit["allowed_pg_socket_connections"],
        allowed_pg_dns=audit["allowed_pg_dns"],
        verified_db_checkouts=audit["verified_db_checkouts"],
    )
    if any(audit["denied"].values()):
        session.exitstatus = 1
