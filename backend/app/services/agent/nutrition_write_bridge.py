"""C2-2 nutrition transaction bridge.

Synchronous, session-factory scoped. One operation is exactly one frozen batch
of meals of a single route. The protected claim (and, when required, the
confirmation consumption) commits first and alone; the business rows plus the
ledger confirmation then commit in one physical Session whose last write is the
real SQL CAS (lease deadline, and the protected pending deadline when a
confirmation was consumed). A claimed operation is never re-dispatched.

The module is not wired into any chat/API/MCP entry point. The nutrition
reserved namespace is reachable only through the store's nutrition capability,
bound to this real bridge instance by object identity; the origin protocol and
generation are stamped by code, never by input.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from datetime import date as date_type, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy.exc import IntegrityError

from app.db import food_repository, nutrition_repository
from app.services.agent import execution_ledger_store as ledger_api
from app.services.agent.execution_ledger_store import ExecutionLedgerError
from app.services.agent.food_parser import (
    FOOD_NAME_ALIASES,
    _canonical_portion_unit,
    _quantity_evidence,
    _weight_evidence,
    normalize_food_name,
)
from app.services.agent.react_models import CommitState, DomainObservation, SourceScope, ToolStatus
from app.services.agent.react_tools import validate_source_ref
from app.services.security.write_guard import source_safety_blocked

_BRIDGE_VERSION = 1
_PENDING_PREFIX = "c2-2-pending:v1:"
_USE_PREFIX = "c2-2-confirmation-use:v1:"
_CLAIM_PREFIX = "c2-2-claim:v1:"
_COMMIT_PREFIX = "c2-2-commit:v1:"
_PROOF_TYPE = "nutrition_db_v1"
_ROUTES = ("manual_nutrition_log", "food_items")
_SOURCES = ("local", "custom", "ai")
_CONFIRMATION_SOURCES = ("确认保存", "确认记录")
_TTL_SECONDS = 900
_BATCH_KEYS = frozenset({"route", "timezone", "groups"})
_MANUAL_GROUP_KEYS = frozenset({"source_ref", "meal_type", "log_date", "description",
                                "calories", "protein_g", "carbs_g", "fat_g"})
_FOOD_GROUP_KEYS = frozenset({"source_ref", "meal_type", "record_time", "foods"})
_FOOD_ITEM_KEYS = frozenset({"source_ref", "entry", "source", "nutrition_per_100g"})
_ENTRY_KEYS = frozenset({"raw_text", "requested_name", "quantity", "unit", "weight_g",
                         "portion_name", "meal_type", "record_date", "is_negated"})
_PER100_KEYS = frozenset({"kcal", "protein_g", "carbs_g", "fat_g", "fiber_g"})
_PENDING_FIELDS = ("pending_id", "user_id", "request_id", "conversation_id", "task_id", "op_id",
                   "source_digest", "parameters_digest", "preview_digest", "created_at", "expires_at",
                   "preview", "pending_digest")
_MACRO_RANGES = {"calories": (0.0, 10000.0), "protein_g": (0.0, 500.0),
                 "carbs_g": (0.0, 1000.0), "fat_g": (0.0, 500.0)}
_PER100_RANGES = {"kcal": (0.0, 1200.0), "protein_g": (0.0, 100.0), "carbs_g": (0.0, 100.0),
                  "fat_g": (0.0, 100.0), "fiber_g": (0.0, 100.0)}
_PER100_TO_MACRO = {"kcal": "calories", "protein_g": "protein_g", "carbs_g": "carbs_g",
                    "fat_g": "fat_g", "fiber_g": "fiber_g"}
_DATE_RE = None
_TERMINAL = frozenset({"limited", "failed", "completed"})

try:  # Asia/Shanghai has had no DST since 1991; a fixed +08:00 fallback never guesses.
    from zoneinfo import ZoneInfo

    _TZINFO = ZoneInfo("Asia/Shanghai")
except Exception:  # pragma: no cover - platform tzdata absence
    _TZINFO = timezone(timedelta(hours=8), "Asia/Shanghai")


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_iso(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _as_utc(parsed)


def _event_identity(user_id: str, request_id: str, event_key: str) -> str:
    return "ev2:" + _digest(["ledger-v2-event", user_id, request_id, event_key])


def _origin_matches(observation) -> bool:
    if not isinstance(observation, dict):
        return False
    origin = observation.get(ledger_api._NUTRITION_ORIGIN_KEY)
    return (isinstance(origin, dict)
            and origin.get("protocol") == ledger_api._NUTRITION_ORIGIN_PROTOCOL
            and origin.get("generation") == ledger_api._NUTRITION_ORIGIN_GENERATION)


def _round2(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


_SENTENCE_BOUNDARY = "。！？\n"
_DATE_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DATE_CN_RE = re.compile(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})[日号]")
_MACRO_LABELS = {
    "calories": (("热量", "千卡"), ("热量", "大卡"), ("能量", "千卡"), ("能量", "大卡"),
                 ("卡路里", "千卡"), ("kcal", "kcal")),
    "protein_g": (("蛋白质", "克"), ("蛋白", "克")),
    "carbs_g": (("碳水化合物", "克"), ("碳水", "克")),
    "fat_g": (("脂肪", "克"),),
    "fiber_g": (("膳食纤维", "克"), ("纤维", "克")),
}


def _sentence_window(source: str, start: int, end: int) -> str:
    """Local clause window: bounded by 。！？\n so a date fact in another
    sentence never binds this record."""
    left = 0
    for position in range(start - 1, -1, -1):
        if source[position] in _SENTENCE_BOUNDARY:
            left = position + 1
            break
    right = len(source)
    for position in range(end, len(source)):
        if source[position] in _SENTENCE_BOUNDARY:
            right = position
            break
    return source[left:right]


def _date_facts(window: str, frozen_date: date_type) -> set[date_type]:
    facts: set[date_type] = set()
    for match in _DATE_ISO_RE.finditer(window):
        try:
            facts.add(date_type(int(match.group(1)), int(match.group(2)), int(match.group(3))))
        except ValueError:
            raise _Refuse("needs_confirmation", "source", "illegal_explicit_date") from None
    for match in _DATE_CN_RE.finditer(window):
        year = int(match.group(1)) if match.group(1) else frozen_date.year
        try:
            facts.add(date_type(year, int(match.group(2)), int(match.group(3))))
        except ValueError:
            raise _Refuse("needs_confirmation", "source", "illegal_explicit_date") from None
    for word, delta in (("今天", 0), ("昨天", 1), ("前天", 2)):
        if word in window:
            facts.add(frozen_date - timedelta(days=delta))
    return facts


def _number_token_in(window: str, value: float) -> bool:
    candidates = {f"{value:g}"}
    if float(value).is_integer():
        candidates.add(str(int(value)))
    for candidate in candidates:
        if re.search(r"(?<![\d.])" + re.escape(candidate) + r"(?![\d.])", window):
            return True
    return False


def _macro_evidence(field: str, value: float, window: str) -> bool:
    """A macro number must bind to this macro's own label and unit inside the
    window; a bare numeric substring or another item's number never counts."""
    for label, unit in _MACRO_LABELS[field]:
        position = window.find(label)
        while position >= 0:
            after = window[position:position + len(label) + 16]
            before = window[max(0, position - 16):position + len(label)]
            for piece in (after, before):
                if unit in piece and _number_token_in(piece, value):
                    return True
            position = window.find(label, position + 1)
    return False


def _strip_private(value):
    """Public results never carry the private code-stamped origin key."""
    if isinstance(value, dict):
        return {key: _strip_private(item) for key, item in value.items()
                if key != ledger_api._NUTRITION_ORIGIN_KEY}
    if isinstance(value, list):
        return [_strip_private(item) for item in value]
    return value


class _Refuse(Exception):
    def __init__(self, outcome, reason, detail=None, dispatched=False):
        self.outcome = outcome
        self.reason = reason
        self.detail = detail
        self.dispatched = dispatched
        super().__init__(outcome)


class _Prepared:
    def __init__(self, route, groups, unresolved, has_ai):
        self.route = route
        self.groups = groups
        self.unresolved = unresolved
        self.has_ai = has_ai


class NutritionWriteBridge:
    def __init__(self, factory):
        self._factory = factory
        self._ledger = ledger_api.ExecutionLedgerStore(factory)
        # Local, non-serializable nutrition-family capability bound to this real
        # bridge instance; never derived from a flag, string, dict or event field.
        self._reserved_capability = ledger_api._bind_nutrition_reserved_capability(self._ledger, self)

    # ------------------------------------------------------------------ public

    def prepare_pending_nutrition(self, user_id, request_id, op_id, *, expected_revision, lease,
                                  tool_call_id, batch_id) -> dict:
        try:
            self._validate_call_inputs(user_id, request_id, op_id, expected_revision, lease,
                                       tool_call_id, batch_id)
            return self._prepare(user_id, request_id, op_id, expected_revision, lease, tool_call_id, batch_id)
        except _Refuse as refuse:
            return self._limited(refuse, user_id, request_id, op_id)
        except ExecutionLedgerError as exc:
            return self._limited(self._ledger_refusal(exc), user_id, request_id, op_id)
        except Exception:
            return self._limited(_Refuse("not_committed", "storage", "storage_failure"), user_id, request_id, op_id)

    def commit_nutrition_operation(self, user_id, request_id, op_id, *, expected_revision, lease,
                                   tool_call_id, batch_id, confirmation_request_id=None,
                                   pending_digest=None, confirmation_expected_revision=None,
                                   confirmation_lease=None, allow_ai_estimate=False) -> dict:
        try:
            self._validate_call_inputs(user_id, request_id, op_id, expected_revision, lease,
                                       tool_call_id, batch_id)
            confirmation = self._confirmation_args(
                confirmation_request_id, pending_digest, confirmation_expected_revision, confirmation_lease)
            return self._commit(user_id, request_id, op_id, expected_revision, lease,
                                tool_call_id, batch_id, confirmation, allow_ai_estimate)
        except _Refuse as refuse:
            return self._limited(refuse, user_id, request_id, op_id)
        except ExecutionLedgerError as exc:
            return self._limited(self._ledger_refusal(exc), user_id, request_id, op_id)
        except Exception:
            return self._limited(_Refuse("not_committed", "storage", "storage_failure"), user_id, request_id, op_id)

    def recover_nutrition_operation(self, user_id, request_id, op_id) -> dict:
        try:
            for name, value in (("user_id", user_id), ("request_id", request_id), ("op_id", op_id)):
                if not isinstance(value, str) or not value:
                    raise _Refuse("rejected", "schema", f"{name}_invalid")
            return self._read_result(user_id, request_id, op_id, dispatched=False, found_outcome="recovered")
        except _Refuse as refuse:
            return self._limited(refuse, user_id, request_id, op_id)
        except Exception:
            return self._limited(_Refuse("commit_unknown", "commit_unknown", "recovery_read_failed"),
                                 user_id, request_id, op_id)

    # ------------------------------------------------------------- preparation

    def _prepare(self, user_id, request_id, op_id, expected_revision, lease, tool_call_id, batch_id) -> dict:
        session = self._factory()
        try:
            row = self._ledger.ledger_request_row(session, user_id, request_id, for_update=True)
            if row is None:
                raise _Refuse("rejected", "identity", "unknown_owner")
            op = self._ledger.ledger_operation_row(session, user_id, request_id, op_id)
            if op is None:
                raise _Refuse("rejected", "identity", "unknown_operation")
            if op.kind != "nutrition":
                raise _Refuse("rejected", "identity", "wrong_kind")
            existing = self._ledger.ledger_event_row(session, user_id, request_id, _PENDING_PREFIX + op_id)
            if existing is not None:
                if not self._trusted_pending(existing.observation, row, op, user_id, request_id):
                    raise _Refuse("commit_unknown", "operation_unverified", "pending_untrusted")
                observation = existing.observation
                snapshot = {key: copy.deepcopy(observation.get(key)) for key in _PENDING_FIELDS}
                return self._result("pending_ready", dispatched=False,
                                    observation=self._draft_observation(op.task_id, "pending_ready"),
                                    pending=snapshot, user_id=user_id, request_id=request_id, op_id=op_id,
                                    task_id=op.task_id, position=op.position)
            if op.status != "reserved":
                raise _Refuse("rejected", "identity", "prepare_requires_reserved")
            self._ledger.ledger_require_lease(row, lease, self._ledger.ledger_statement_now(session))
            if row.revision != expected_revision:
                raise _Refuse("rejected", "revision", "stale_revision")
            if row.status in _TERMINAL:
                raise _Refuse("rejected", "identity", "terminal_request")
            task = self._ledger.ledger_task_row(session, user_id, request_id, op.task_id)
            if task is None:
                raise _Refuse("rejected", "identity", "unknown_task")
            task_span = self._validate_source_and_task(row, task)
            batch = self._frozen_batch(task, op)
            prepared = self._prepare_batch(session, row, op, batch, task_span=task_span)
            if prepared.unresolved:
                raise _Refuse("needs_confirmation", "source", "unresolved_items")
            now = self._ledger.ledger_statement_now(session)
            payload = self._pending_payload(row, op, prepared, now)
            pending_digest = _digest(payload)
            observation = {**payload, "pending_digest": pending_digest}
            event = ledger_api.EventInput(
                event_key=_PENDING_PREFIX + op_id, tool_call_id=tool_call_id, batch_id=batch_id,
                task_id=op.task_id, op_id=op_id, phase="nutrition_pending", observation=observation,
                task_status="needs_confirmation", operation_state=None, confirmed=False)
            self._ledger.ledger_append_nutrition_event(
                session, user_id, request_id, event, checkpoint=self._next_checkpoint(row),
                expected_revision=row.revision, lease=lease, capability=self._reserved_capability)
            try:
                session.commit()
            except Exception:
                self._rollback_quiet(session)
                raise _Refuse("commit_unknown", "commit_unknown", "pending_commit_unknown")
            return self._result("pending_ready", dispatched=False,
                                observation=self._draft_observation(op.task_id, "pending_ready"),
                                pending=copy.deepcopy(observation), user_id=user_id, request_id=request_id,
                                op_id=op_id, task_id=op.task_id, position=op.position)
        except IntegrityError:
            self._rollback_quiet(session)
            raise _Refuse("commit_unknown", "operation_unverified", "pending_race")
        except ExecutionLedgerError as exc:
            self._rollback_quiet(session)
            raise self._ledger_refusal(exc)
        except _Refuse:
            self._rollback_quiet(session)
            raise
        except Exception:
            self._rollback_quiet(session)
            raise _Refuse("not_committed", "storage", "storage_failure")
        finally:
            self._close_quiet(session)

    def _pending_payload(self, row, op, prepared: _Prepared, now: datetime) -> dict:
        return {
            "pending_id": _event_identity(row.user_id, row.request_id, _PENDING_PREFIX + op.op_id),
            "user_id": row.user_id, "request_id": row.request_id, "conversation_id": row.conversation_id,
            "task_id": op.task_id, "op_id": op.op_id,
            "source_digest": row.source_digest,
            "parameters_digest": _digest(op.parameters),
            "preview_digest": _digest({"route": prepared.route, "groups": prepared.groups}),
            "created_at": _iso_utc(now), "expires_at": _iso_utc(now + timedelta(seconds=_TTL_SECONDS)),
            "preview": {"route": prepared.route, "groups": copy.deepcopy(prepared.groups)},
        }

    @staticmethod
    def _trusted_pending(observation, row, op, user_id, request_id) -> bool:
        """A stored pending is reusable only when the protected origin, the full
        identity set and the payload/preview digests are self-consistent; a
        stripped or rewritten pending is never re-stamped."""
        if not isinstance(observation, dict) or not _origin_matches(observation):
            return False
        if (observation.get("op_id") != op.op_id or observation.get("task_id") != op.task_id
                or observation.get("user_id") != user_id or observation.get("request_id") != request_id
                or observation.get("source_digest") != row.source_digest
                or observation.get("parameters_digest") != _digest(op.parameters)):
            return False
        payload = {key: value for key, value in observation.items()
                   if key not in ("pending_digest", ledger_api._NUTRITION_ORIGIN_KEY)}
        if _digest(payload) != observation.get("pending_digest"):
            return False
        preview = observation.get("preview")
        if (not isinstance(preview, dict) or preview.get("route") not in _ROUTES
                or not isinstance(preview.get("groups"), list) or not preview["groups"]):
            return False
        if _digest({"route": preview.get("route"), "groups": preview.get("groups")}) != observation.get("preview_digest"):
            return False
        return True

    # ------------------------------------------------------------------ commit

    def _commit(self, user_id, request_id, op_id, expected_revision, lease, tool_call_id, batch_id,
                confirmation, allow_ai_estimate=False) -> dict:
        session = self._factory()
        try:
            pairs = [(user_id, request_id)]
            if confirmation is not None:
                pairs.append((user_id, confirmation["request_id"]))
            locked = {}
            for owner, req in sorted(set(pairs)):
                locked[(owner, req)] = self._ledger.ledger_request_row(session, owner, req, for_update=True)
            row = locked[(user_id, request_id)]
            if row is None:
                raise _Refuse("rejected", "identity", "unknown_owner")
            op = self._ledger.ledger_operation_row(session, user_id, request_id, op_id)
            if op is None:
                raise _Refuse("rejected", "identity", "unknown_operation")
            if op.kind != "nutrition":
                raise _Refuse("rejected", "identity", "wrong_kind")
            existing_confirmation = self._ledger.ledger_event_row(session, user_id, request_id, _COMMIT_PREFIX + op_id)
            if existing_confirmation is not None:
                return self._replay(session, user_id, request_id, op, existing_confirmation)
            if self._ledger.ledger_event_row(session, user_id, request_id, _CLAIM_PREFIX + op_id) is not None:
                raise _Refuse("commit_unknown", "operation_unverified", "claim_without_confirmation")
            self._ledger.ledger_require_lease(row, lease, self._ledger.ledger_statement_now(session))
            if row.revision != expected_revision:
                raise _Refuse("rejected", "revision", "stale_revision")
            if row.status in _TERMINAL:
                raise _Refuse("rejected", "identity", "terminal_request")
            task = self._ledger.ledger_task_row(session, user_id, request_id, op.task_id)
            if task is None:
                raise _Refuse("rejected", "identity", "unknown_task")
            task_span = self._validate_source_and_task(row, task)
            batch = self._frozen_batch(task, op)
            pending_link = None
            if confirmation is not None:
                pending_event, pending_link = self._validate_confirmation(
                    session, user_id, request_id, op, confirmation, locked)
                preview = pending_event.observation["preview"]
                route = preview["route"]
                groups = preview["groups"]
                parameters_digest = pending_event.observation["parameters_digest"]
                preview_digest = pending_event.observation["preview_digest"]
            else:
                prepared = self._prepare_batch(session, row, op, batch, task_span=task_span)
                if prepared.unresolved:
                    raise _Refuse("needs_confirmation", "source", "unresolved_items")
                if prepared.has_ai and allow_ai_estimate is not True:
                    raise _Refuse("needs_confirmation", "pending", "protected_pending_required")
                route = prepared.route
                groups = prepared.groups
                parameters_digest = _digest(op.parameters)
                preview_digest = _digest({"route": route, "groups": groups})
            if route == "food_items":
                for group in groups:
                    prefix = op.idempotency_key + ":g" + str(group["group_position"])
                    if food_repository.find_food_items_in_session(session, user_id, prefix):
                        raise _Refuse("commit_unknown", "operation_unverified", "foreign_food_rows")
            if confirmation is not None:
                conf_row = locked[(user_id, confirmation["request_id"])]
                use_event = ledger_api.EventInput(
                    event_key=_USE_PREFIX + confirmation["request_id"],
                    tool_call_id=tool_call_id, batch_id=batch_id, task_id=confirmation["task_id"],
                    op_id=None, phase="nutrition_confirmation_use",
                    observation={"pending_request_id": request_id, "pending_op_id": op_id,
                                 "pending_digest": pending_link["pending_digest"],
                                 "target_task_id": confirmation["task_id"]},
                    task_status="completed", operation_state=None, confirmed=False)
                self._ledger.ledger_append_nutrition_event(
                    session, user_id, confirmation["request_id"], use_event,
                    checkpoint=self._next_checkpoint(conf_row),
                    expected_revision=conf_row.revision, lease=confirmation["lease"],
                    capability=self._reserved_capability)
            claim_payload = {
                "claim_state": "dispatched",
                "user_id": user_id, "request_id": request_id, "task_id": op.task_id, "op_id": op_id,
                "position": op.position, "idempotency_key": op.idempotency_key,
                "parameters_digest": parameters_digest, "preview_digest": preview_digest,
                "route": route, "groups": copy.deepcopy(groups), "pending": copy.deepcopy(pending_link),
            }
            claim = ledger_api.EventInput(
                event_key=_CLAIM_PREFIX + op_id, tool_call_id=tool_call_id, batch_id=batch_id,
                task_id=op.task_id, op_id=op_id, phase="write_claim", observation=claim_payload,
                task_status="running", operation_state=None, confirmed=False)
            try:
                self._ledger.ledger_append_nutrition_event(
                    session, user_id, request_id, claim, checkpoint=self._next_checkpoint(row),
                    expected_revision=row.revision, lease=lease, capability=self._reserved_capability)
            except ExecutionLedgerError as exc:
                self._rollback_quiet(session)
                raise self._ledger_refusal(exc)
            except IntegrityError:
                self._rollback_quiet(session)
                raise _Refuse("commit_unknown", "operation_unverified", "claim_race")
            except Exception:
                self._rollback_quiet(session)
                raise _Refuse("not_committed", "storage", "claim_write_failed")
            try:
                session.commit()
            except Exception:
                self._rollback_quiet(session)
                raise _Refuse("commit_unknown", "commit_unknown", "claim_commit_unknown")
        except _Refuse:
            self._rollback_quiet(session)
            raise
        except ExecutionLedgerError as exc:
            self._rollback_quiet(session)
            raise self._ledger_refusal(exc)
        except IntegrityError:
            self._rollback_quiet(session)
            raise _Refuse("commit_unknown", "operation_unverified", "claim_race")
        except Exception:
            self._rollback_quiet(session)
            raise _Refuse("not_committed", "storage", "storage_failure")
        finally:
            self._close_quiet(session)
        return self._business_transaction(user_id, request_id, op_id, lease, pending_link,
                                          tool_call_id, batch_id)

    def _validate_confirmation(self, session, user_id, request_id, op, confirmation, locked):
        conf_row = locked[(user_id, confirmation["request_id"])]
        if conf_row is None:
            raise _Refuse("rejected", "identity", "unknown_confirmation_request")
        if conf_row.conversation_id != locked[(user_id, request_id)].conversation_id:
            raise _Refuse("rejected", "identity", "confirmation_conversation_mismatch")
        source = conf_row.source if isinstance(conf_row.source, str) else ""
        if source.strip() not in _CONFIRMATION_SOURCES:
            raise _Refuse("rejected", "source", "confirmation_source_rejected")
        pending_event = self._ledger.ledger_event_row(session, user_id, request_id, _PENDING_PREFIX + op.op_id)
        if pending_event is None or not isinstance(pending_event.observation, dict):
            raise _Refuse("rejected", "pending", "pending_missing")
        observation = pending_event.observation
        if not self._trusted_pending(observation, locked[(user_id, request_id)], op, user_id, request_id):
            raise _Refuse("rejected", "pending", "pending_identity_mismatch")
        if confirmation["pending_digest"] != observation.get("pending_digest"):
            raise _Refuse("rejected", "pending", "pending_digest_mismatch")
        created_at = _parse_iso(observation.get("created_at"))
        expires_at = _parse_iso(observation.get("expires_at"))
        time_base = _parse_iso(conf_row.time_base)
        if created_at is None or expires_at is None or time_base is None:
            raise _Refuse("rejected", "pending", "pending_window_invalid")
        if not (created_at <= time_base < expires_at):
            raise _Refuse("rejected", "pending", "confirmation_time_outside_pending")
        tasks = []
        for task_row in self._request_tasks(session, user_id, confirmation["request_id"]):
            if task_row.task_type == "nutrition_confirmation":
                tasks.append(task_row)
        if len(tasks) != 1:
            raise _Refuse("rejected", "pending", "confirmation_task_missing")
        task = tasks[0]
        parameters = task.parameters
        exact = {"pending_request_id": request_id, "pending_op_id": op.op_id,
                 "pending_digest": observation.get("pending_digest")}
        if not isinstance(parameters, dict) or parameters != exact:
            raise _Refuse("rejected", "pending", "confirmation_parameters_mismatch")
        if not (task.source_start == 0 and task.source_end == len(source)):
            raise _Refuse("rejected", "pending", "confirmation_span_invalid")
        self._ledger.ledger_require_lease(conf_row, confirmation["lease"],
                                          self._ledger.ledger_statement_now(session))
        if conf_row.revision != confirmation["expected_revision"]:
            raise _Refuse("rejected", "revision", "confirmation_stale_revision")
        if self._ledger.ledger_event_row(session, user_id, confirmation["request_id"],
                                         _USE_PREFIX + confirmation["request_id"]) is not None:
            raise _Refuse("rejected", "pending", "confirmation_already_consumed")
        now = self._ledger.ledger_statement_now(session)
        if not (now < expires_at):
            raise _Refuse("rejected", "pending", "pending_expired")
        pending_link = {"pending_digest": observation.get("pending_digest"),
                        "confirmation_request_id": confirmation["request_id"],
                        "pending_deadline": _iso_utc(expires_at)}
        confirmation["task_id"] = task.task_id
        return pending_event, pending_link

    def _request_tasks(self, session, user_id, request_id):
        from sqlalchemy import select
        from app.db.execution_ledger_models import ExecutionTaskModel

        return list(session.execute(
            select(ExecutionTaskModel).where(
                ExecutionTaskModel.user_id == user_id,
                ExecutionTaskModel.request_id == request_id,
            )).scalars())

    # ------------------------------------------------------------ transaction 2

    def _business_transaction(self, user_id, request_id, op_id, lease, pending_link,
                              tool_call_id, batch_id) -> dict:
        session = self._factory()
        ctx = {"dispatched": False}
        try:
            row = self._ledger.ledger_request_row(session, user_id, request_id, for_update=True)
            if row is None:
                raise _Refuse("commit_unknown", "commit_unknown", "request_vanished")
            op = self._ledger.ledger_operation_row(session, user_id, request_id, op_id)
            if op is None or op.status not in {"reserved", "dispatch_intent"}:
                raise _Refuse("commit_unknown", "commit_unknown", "operation_changed")
            claim = self._ledger.ledger_event_row(session, user_id, request_id, _CLAIM_PREFIX + op_id)
            if claim is None or not isinstance(claim.observation, dict):
                raise _Refuse("commit_unknown", "commit_unknown", "claim_missing")
            self._ledger.ledger_require_lease(row, lease, self._ledger.ledger_statement_now(session))
            task = self._ledger.ledger_task_row(session, user_id, request_id, op.task_id)
            if task is None:
                raise _Refuse("commit_unknown", "commit_unknown", "task_vanished")
            claim_observation = claim.observation
            route = claim_observation.get("route")
            groups = claim_observation.get("groups")
            if route not in _ROUTES or not isinstance(groups, list) or not groups:
                raise _Refuse("commit_unknown", "operation_unverified", "claim_payload_invalid")
            deadline = None
            if isinstance(pending_link, dict):
                deadline = _parse_iso(pending_link.get("pending_deadline"))
                if deadline is None:
                    raise _Refuse("commit_unknown", "operation_unverified", "pending_deadline_missing")
                if not (self._ledger.ledger_statement_now(session) < deadline):
                    self._rollback_quiet(session)
                    raise _Refuse("not_committed", "pending", "pending_expired", False)
            ctx["dispatched"] = True
            record_ids: list[list[int]] = []
            if route == "manual_nutrition_log":
                for group in groups:
                    record = group.get("record")
                    if not isinstance(record, dict):
                        raise _Refuse("commit_unknown", "operation_unverified", "claim_payload_invalid", True)
                    created = nutrition_repository.create_nutrition_in_session(session, user_id, record)
                    if created.get("created") is not True or created.get("record_id") is None:
                        self._rollback_quiet(session)
                        raise _Refuse("commit_unknown", "operation_unverified", "manual_create_unverified", True)
                    record_ids.append([int(created["record_id"])])
            else:
                for group in groups:
                    items = group.get("items")
                    if not isinstance(items, list) or not items:
                        raise _Refuse("commit_unknown", "operation_unverified", "claim_payload_invalid", True)
                    result = food_repository.log_food_items_in_session(
                        session, user_id, items, group.get("totals") or {}, group.get("meal_group_id"),
                        op.idempotency_key + ":g" + str(group.get("group_position")),
                        group.get("record_time"), group.get("meal_type"))
                    if result.get("created") is not True or len(result.get("log_ids") or []) != len(items):
                        self._rollback_quiet(session)
                        raise _Refuse("commit_unknown", "operation_unverified", "food_create_unverified", True)
                    record_ids.append([int(item) for item in result["log_ids"]])
            proof_groups = [{"group_position": group["group_position"],
                             "meal_group_id": group.get("meal_group_id"),
                             "record_ids": record_ids[index]} for index, group in enumerate(groups)]
            receipts: list[str] = []
            for index, group in enumerate(groups):
                receipts += [f"nutrition:c2-2:{op_id}:g{index}:{row_id}" for row_id in record_ids[index]]
            proof = {
                "proof_type": _PROOF_TYPE, "user_id": user_id, "request_id": request_id,
                "task_id": op.task_id, "op_id": op_id, "position": op.position,
                "idempotency_key": op.idempotency_key,
                "parameters_digest": claim_observation.get("parameters_digest"),
                "preview_digest": claim_observation.get("preview_digest"),
                "route": route, "groups": proof_groups,
                "confirmation_request_id": (pending_link or {}).get("confirmation_request_id", "none"),
            }
            observation = DomainObservation(
                status=ToolStatus.COMPLETED, task_id=op.task_id, commit_state=CommitState.COMMITTED,
                result=proof, receipt_refs=receipts).to_dict()
            confirmation = ledger_api.EventInput(
                event_key=_COMMIT_PREFIX + op_id, tool_call_id=tool_call_id, batch_id=batch_id,
                task_id=op.task_id, op_id=op_id, phase="confirmation", observation=observation,
                task_status="completed", operation_state="committed", confirmed=True)
            # The final CAS below is the enforcement point for the lease (and the
            # pending deadline): no Python pre-check may short-circuit it, or the
            # required real-SQL rowcount evidence would never be produced.
            self._ledger.ledger_append_nutrition_event(
                session, user_id, request_id, confirmation, checkpoint=self._next_checkpoint(row),
                expected_revision=row.revision, lease=lease, capability=self._reserved_capability,
                pending_deadline=deadline, final_fence=True)
            try:
                session.commit()
            except Exception:
                self._rollback_quiet(session)
                return self._resolve_after_commit_fault(user_id, request_id, op_id)
            snapshot = self._safe_snapshot(user_id, request_id)
            return self._result("committed", observation=observation, dispatched=True, ledger_snapshot=snapshot,
                                user_id=user_id, request_id=request_id, op_id=op_id,
                                task_id=op.task_id, position=op.position)
        except _Refuse as refuse:
            self._rollback_quiet(session)
            refuse.dispatched = refuse.dispatched or ctx["dispatched"]
            raise
        except ExecutionLedgerError as exc:
            self._rollback_quiet(session)
            raise self._business_refusal(exc, ctx["dispatched"])
        except Exception:
            self._rollback_quiet(session)
            raise _Refuse("not_committed", "storage", "business_transaction_failed", ctx["dispatched"])
        finally:
            self._close_quiet(session)

    def _resolve_after_commit_fault(self, user_id, request_id, op_id) -> dict:
        try:
            return self._read_result(user_id, request_id, op_id, dispatched=True, found_outcome="recovered")
        except Exception:
            return self._limited(_Refuse("commit_unknown", "commit_unknown", "post_commit_read_failed", True),
                                 user_id, request_id, op_id)

    # ------------------------------------------------------------ read results

    def _read_result(self, user_id, request_id, op_id, *, dispatched, found_outcome) -> dict:
        def limited(outcome, reason, detail):
            return self._limited(_Refuse(outcome, reason, detail, dispatched), user_id, request_id, op_id)

        session = self._factory()
        try:
            op = self._ledger.ledger_operation_row(session, user_id, request_id, op_id)
            if op is None:
                return limited("rejected", "identity", "unknown_operation")
            if op.kind != "nutrition":
                return limited("rejected", "identity", "wrong_kind")
            confirmation = self._ledger.ledger_event_row(session, user_id, request_id, _COMMIT_PREFIX + op_id)
            if confirmation is not None:
                if self._trusted_proof(session, confirmation, op) is not None:
                    observation = DomainObservation.from_dict(confirmation.observation).to_dict()
                    return self._result(found_outcome, observation=observation, dispatched=dispatched,
                                        ledger_snapshot=self._safe_snapshot(user_id, request_id),
                                        user_id=user_id, request_id=request_id, op_id=op.op_id,
                                        task_id=op.task_id, position=op.position)
                return limited("commit_unknown", "operation_unverified", "confirmation_untrusted")
            if op.status in {"committed", "commit_unknown"}:
                return limited("commit_unknown", "operation_unverified", "operation_state_untrusted")
            if self._ledger.ledger_event_row(session, user_id, request_id, _CLAIM_PREFIX + op_id) is not None:
                return limited("commit_unknown", "operation_unverified", "claim_without_confirmation")
            return limited("not_committed", "operation_unverified", "no_bridge_attempt")
        except ExecutionLedgerError as exc:
            code = getattr(exc, "code", None)
            if code == "input_invalid":
                return limited("rejected", "schema", code)
            if code in {"identity_conflict", "terminal_frozen"}:
                return limited("rejected", "identity", code)
            return limited("commit_unknown", "storage", code or "ledger_error")
        except Exception:
            return limited("commit_unknown", "commit_unknown", "recovery_read_failed")
        finally:
            self._close_quiet(session)

    def _replay(self, session, user_id, request_id, op, confirmation) -> dict:
        if self._trusted_proof(session, confirmation, op) is None:
            raise _Refuse("commit_unknown", "operation_unverified", "confirmation_untrusted")
        observation = DomainObservation.from_dict(confirmation.observation).to_dict()
        return self._result("replayed", observation=observation, dispatched=False,
                            ledger_snapshot=self._safe_snapshot(user_id, request_id),
                            user_id=user_id, request_id=request_id, op_id=op.op_id,
                            task_id=op.task_id, position=op.position)

    def _trusted_proof(self, session, confirmation, op) -> dict | None:
        """Canonical claim before confirmation, same current nutrition generation,
        identity/parameters/preview/route/ordered IDs and receipts, and the
        pending + confirmation-use association when one was required."""
        try:
            if not confirmation.confirmed or confirmation.operation_state != "committed":
                return None
            if confirmation.phase != "confirmation" or confirmation.event_key != _COMMIT_PREFIX + op.op_id:
                return None
            if confirmation.op_id != op.op_id or confirmation.task_id != op.task_id:
                return None
            request_row = self._ledger.ledger_request_row(session, op.user_id, op.request_id)
            if request_row is None:
                return None
            task = self._ledger.ledger_task_row(session, op.user_id, op.request_id, op.task_id)
            if task is None:
                return None
            try:
                # Reading an already committed proof is not a new write and must
                # not require a fresh model safety call in the replay process.
                task_span = self._validate_source_and_task(request_row, task, check_safety=False)
                self._prepare_batch(None, request_row, op, self._frozen_batch(task, op),
                                    task_span=task_span, source_only=True)
            except _Refuse:
                return None
            observation = confirmation.observation
            if not isinstance(observation, dict) or observation.get("commit_state") != "committed":
                return None
            if not _origin_matches(observation):
                return None
            result = observation.get("result")
            if not isinstance(result, dict) or result.get("proof_type") != _PROOF_TYPE:
                return None
            if (result.get("user_id") != op.user_id or result.get("request_id") != op.request_id
                    or result.get("task_id") != op.task_id or result.get("op_id") != op.op_id):
                return None
            if isinstance(op.position, bool) or not isinstance(op.position, int):
                return None
            if result.get("position") != op.position or result.get("idempotency_key") != op.idempotency_key:
                return None
            route = result.get("route")
            if route not in _ROUTES:
                return None
            groups = result.get("groups")
            if not isinstance(groups, list) or not groups:
                return None
            receipts = observation.get("receipt_refs")
            if (not isinstance(receipts, list) or not receipts
                    or not all(isinstance(item, str) and item.strip() for item in receipts)):
                return None
            expected_receipts: list[str] = []
            for index, group in enumerate(groups):
                if not isinstance(group, dict) or group.get("group_position") != index:
                    return None
                record_ids = group.get("record_ids")
                if not isinstance(record_ids, list) or not record_ids:
                    return None
                for row_id in record_ids:
                    if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id <= 0:
                        return None
                    expected_receipts.append(f"nutrition:c2-2:{op.op_id}:g{index}:{row_id}")
            if receipts != expected_receipts:
                return None
            claim = self._ledger.ledger_event_row(session, op.user_id, op.request_id, _CLAIM_PREFIX + op.op_id)
            if claim is None or claim.phase != "write_claim" or claim.event_key != _CLAIM_PREFIX + op.op_id:
                return None
            if claim.op_id != op.op_id or claim.task_id != op.task_id:
                return None
            claim_sequence, confirmation_sequence = claim.sequence, confirmation.sequence
            if (isinstance(claim_sequence, bool) or not isinstance(claim_sequence, int)
                    or isinstance(confirmation_sequence, bool) or not isinstance(confirmation_sequence, int)
                    or claim_sequence >= confirmation_sequence):
                return None
            claim_observation = claim.observation
            if not _origin_matches(claim_observation):
                return None
            claim_groups = claim_observation.get("groups")
            if not isinstance(claim_groups, list) or not claim_groups:
                return None
            if claim_observation.get("parameters_digest") != _digest(op.parameters):
                return None
            if claim_observation.get("parameters_digest") != result.get("parameters_digest"):
                return None
            if claim_observation.get("preview_digest") != result.get("preview_digest"):
                return None
            if claim_observation.get("route") != route:
                return None
            if claim_observation.get("idempotency_key") != op.idempotency_key:
                return None
            if _digest({"route": claim_observation.get("route"), "groups": claim_groups}) != result.get("preview_digest"):
                return None
            if len(claim_groups) != len(groups):
                return None
            for index, (claim_group, proof_group) in enumerate(zip(claim_groups, groups)):
                if not isinstance(claim_group, dict) or claim_group.get("group_position") != index:
                    return None
                if route == "food_items":
                    claim_items = claim_group.get("items")
                    if not isinstance(claim_items, list) or len(claim_items) != len(proof_group["record_ids"]):
                        return None
                elif len(proof_group["record_ids"]) != 1:
                    return None
            confirmation_request_id = result.get("confirmation_request_id")
            claim_pending = claim_observation.get("pending")
            if confirmation_request_id in (None, "none"):
                if claim_pending is not None:
                    return None
                return result
            if not isinstance(claim_pending, dict):
                return None
            if claim_pending.get("confirmation_request_id") != confirmation_request_id:
                return None
            pending = self._ledger.ledger_event_row(session, op.user_id, op.request_id,
                                                    _PENDING_PREFIX + op.op_id)
            if pending is None or pending.phase != "nutrition_pending":
                return None
            if not self._trusted_pending(pending.observation, request_row, op, op.user_id, op.request_id):
                return None
            pending_sequence = pending.sequence
            if (isinstance(pending_sequence, bool) or not isinstance(pending_sequence, int)
                    or pending_sequence >= claim_sequence):
                return None
            use = self._ledger.ledger_event_row(session, op.user_id, confirmation_request_id,
                                                _USE_PREFIX + confirmation_request_id)
            if use is None or use.phase != "nutrition_confirmation_use":
                return None
            if not _origin_matches(use.observation):
                return None
            if not isinstance(use.observation, dict):
                return None
            if use.observation.get("pending_op_id") != op.op_id:
                return None
            if use.observation.get("pending_digest") != claim_pending.get("pending_digest"):
                return None
            return result
        except Exception:
            return None

    # ------------------------------------------------------------- validation

    def _validate_call_inputs(self, user_id, request_id, op_id, expected_revision, lease,
                              tool_call_id, batch_id) -> None:
        for name, value in (("user_id", user_id), ("request_id", request_id), ("op_id", op_id),
                            ("tool_call_id", tool_call_id), ("batch_id", batch_id)):
            if not isinstance(value, str) or not value:
                raise _Refuse("rejected", "schema", f"{name}_invalid")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise _Refuse("rejected", "schema", "expected_revision_invalid")
        if not isinstance(lease, ledger_api.LeaseProof):
            raise _Refuse("rejected", "schema", "lease_type_invalid")

    def _confirmation_args(self, confirmation_request_id, pending_digest,
                           confirmation_expected_revision, confirmation_lease):
        provided = [value is not None for value in
                    (confirmation_request_id, pending_digest, confirmation_expected_revision, confirmation_lease)]
        if not any(provided):
            return None
        if not all(provided):
            raise _Refuse("rejected", "schema", "confirmation_args_partial")
        if not isinstance(confirmation_request_id, str) or not confirmation_request_id:
            raise _Refuse("rejected", "schema", "confirmation_request_id_invalid")
        if not isinstance(pending_digest, str) or not pending_digest:
            raise _Refuse("rejected", "schema", "pending_digest_invalid")
        if (isinstance(confirmation_expected_revision, bool)
                or not isinstance(confirmation_expected_revision, int) or confirmation_expected_revision < 0):
            raise _Refuse("rejected", "schema", "confirmation_expected_revision_invalid")
        if not isinstance(confirmation_lease, ledger_api.LeaseProof):
            raise _Refuse("rejected", "schema", "confirmation_lease_invalid")
        return {"request_id": confirmation_request_id, "pending_digest": pending_digest,
                "expected_revision": confirmation_expected_revision, "lease": confirmation_lease,
                "task_id": None}

    def _validate_source_and_task(self, row, task, *, check_safety: bool = True) -> tuple[int, int]:
        if task.task_type != "nutrition_record":
            raise _Refuse("rejected", "identity", "wrong_task_type")
        start, end = task.source_start, task.source_end
        if (isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool)
                or not isinstance(end, int) or start < 0 or end < start or end > len(row.source)):
            raise _Refuse("rejected", "identity", "source_range_invalid")
        digest = hashlib.sha256(row.source[start:end].encode("utf-8")).hexdigest()
        if digest != task.source_digest:
            raise _Refuse("rejected", "identity", "task_source_digest_mismatch")
        if check_safety:
            safety_block = source_safety_blocked(row.source)
            if safety_block:
                raise _Refuse("rejected", "source", safety_block)
        validation = validate_source_ref(
            SourceScope(text=row.source),
            {"start": start, "end": end, "exact_text": row.source[start:end]},
            "nutrition_record")
        if not validation.ok:
            raise _Refuse("rejected", "source", "source_ref_invalid")
        if validation.authorization != "authorized":
            raise _Refuse("needs_confirmation" if validation.authorization == "needs_confirmation" else "rejected",
                          "source", validation.reason or "source_rejected")
        return start, end

    def _frozen_batch(self, task, op) -> dict:
        parameters = task.parameters
        if not isinstance(parameters, dict) or set(parameters) != {"batches"}:
            raise _Refuse("rejected", "identity", "task_parameters_invalid")
        batches = parameters.get("batches")
        if not isinstance(batches, list) or not batches:
            raise _Refuse("rejected", "identity", "task_batches_missing")
        if isinstance(op.position, bool) or not isinstance(op.position, int) or not (0 <= op.position < len(batches)):
            raise _Refuse("rejected", "identity", "position_out_of_range")
        if not isinstance(op.parameters, dict) or op.parameters != batches[op.position]:
            raise _Refuse("rejected", "identity", "position_batch_mismatch")
        return copy.deepcopy(op.parameters)

    def _prepare_batch(self, session, row, op, batch: dict, *, task_span: tuple[int, int],
                       source_only: bool = False) -> _Prepared:
        if not isinstance(batch, dict) or set(batch) != _BATCH_KEYS:
            raise _Refuse("rejected", "schema", "batch_keys_invalid")
        route = batch.get("route")
        if route not in _ROUTES:
            raise _Refuse("rejected", "schema", "route_invalid")
        if batch.get("timezone") != "Asia/Shanghai":
            raise _Refuse("rejected", "schema", "timezone_invalid")
        groups_in = batch.get("groups")
        if not isinstance(groups_in, list) or not (1 <= len(groups_in) <= 20):
            raise _Refuse("rejected", "schema", "groups_count_invalid")
        frozen_date = self._frozen_date(row.time_base)
        task_text = row.source[task_span[0]:task_span[1]]
        prepared: list[dict] = []
        unresolved: list[str] = []
        has_ai = False
        batch_dates: set[str] = set()
        used_spans: set[tuple[int, int]] = set()
        for index, group in enumerate(groups_in):
            if route == "manual_nutrition_log":
                item = self._prepare_manual_group(row, op, group, index, frozen_date, task_span,
                                                  task_text, used_spans)
            else:
                item, item_unresolved, item_ai = self._prepare_food_group(
                    session, row, op, group, index, frozen_date, task_span, task_text, used_spans,
                    source_only=source_only)
                unresolved.extend(item_unresolved)
                has_ai = has_ai or item_ai
            prepared.append(item)
            batch_dates.add(item["_date"])
        if len(batch_dates) != 1:
            raise _Refuse("needs_confirmation", "source", "multi_day_batch")
        for item in prepared:
            item.pop("_date", None)
        return _Prepared(route, prepared, unresolved, has_ai)

    def _frozen_date(self, time_base) -> date_type:
        parsed = _parse_iso(time_base)
        if parsed is None:
            raise _Refuse("rejected", "identity", "time_base_invalid")
        return parsed.astimezone(_TZINFO).date()

    def _span(self, source: str, ref, *, container: tuple[int, int] | None = None) -> tuple[int, int, str]:
        if not isinstance(ref, dict) or set(ref) != {"start", "end", "exact_text"}:
            raise _Refuse("rejected", "schema", "source_ref_keys_invalid")
        start, end, exact = ref.get("start"), ref.get("end"), ref.get("exact_text")
        if isinstance(start, bool) or not isinstance(start, int):
            raise _Refuse("rejected", "schema", "start_invalid")
        if isinstance(end, bool) or not isinstance(end, int):
            raise _Refuse("rejected", "schema", "end_invalid")
        if not isinstance(exact, str) or not exact:
            raise _Refuse("rejected", "schema", "exact_text_invalid")
        if start < 0 or end < start or end > len(source):
            raise _Refuse("rejected", "schema", "source_ref_out_of_bounds")
        if source[start:end] != exact:
            raise _Refuse("rejected", "schema", "exact_text_mismatch")
        if container is not None and not (container[0] <= start and end <= container[1]):
            raise _Refuse("rejected", "source", "source_ref_not_nested")
        return start, end, exact

    def _strict_number(self, value, *, minimum=None, maximum=None):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        number = float(value)
        if minimum is not None and number < minimum:
            return None
        if maximum is not None and number > maximum:
            return None
        return number

    def _prepare_manual_group(self, row, op, group, index, frozen_date, task_span, task_text,
                              used_spans) -> dict:
        if not isinstance(group, dict) or set(group) != _MANUAL_GROUP_KEYS:
            raise _Refuse("rejected", "schema", "manual_group_keys_invalid")
        start, end, span = self._span(row.source, group.get("source_ref"), container=task_span)
        if (start, end) in used_spans:
            raise _Refuse("rejected", "source", "duplicate_source_occurrence")
        used_spans.add((start, end))
        meal_type = group.get("meal_type")
        if not isinstance(meal_type, str) or not (1 <= len(meal_type) <= 50):
            raise _Refuse("rejected", "schema", "meal_type_invalid")
        if meal_type not in task_text:
            raise _Refuse("rejected", "source", "meal_type_evidence_missing")
        log_date = group.get("log_date")
        if not isinstance(log_date, str):
            raise _Refuse("rejected", "schema", "log_date_invalid")
        try:
            parsed_date = date_type.fromisoformat(log_date)
        except ValueError:
            raise _Refuse("rejected", "schema", "log_date_invalid") from None
        description = group.get("description")
        if not isinstance(description, str) or not (1 <= len(description) <= 1000):
            raise _Refuse("rejected", "schema", "description_invalid")
        if description not in span:
            raise _Refuse("rejected", "source", "description_evidence_missing")
        facts = _date_facts(_sentence_window(row.source, start, end), frozen_date)
        if facts and parsed_date not in facts:
            raise _Refuse("needs_confirmation", "source", "log_date_mismatch")
        if parsed_date > frozen_date:
            raise _Refuse("rejected", "source", "future_date_rejected")
        macros = {}
        for field, (minimum, maximum) in _MACRO_RANGES.items():
            value = group.get(field)
            if value is None:
                macros[field] = None
                continue
            number = self._strict_number(value, minimum=minimum, maximum=maximum)
            if number is None:
                raise _Refuse("rejected", "schema", "macro_invalid")
            if not _macro_evidence(field, number, span):
                raise _Refuse("rejected", "source", "macro_evidence_missing")
            macros[field] = number
        group_id = self._meal_group_id(op, index)
        return {
            "group_position": index, "meal_group_id": group_id, "meal_type": meal_type,
            "log_date": log_date, "_date": log_date,
            "record": {"meal_type": meal_type, "description": description, "log_date": log_date,
                       "calories": macros["calories"], "protein_g": macros["protein_g"],
                       "carbs_g": macros["carbs_g"], "fat_g": macros["fat_g"]},
            "totals": {"calories": macros["calories"], "protein_g": macros["protein_g"],
                       "carbs_g": macros["carbs_g"], "fat_g": macros["fat_g"]},
        }

    def _meal_group_id(self, op, index: int) -> str:
        return "meal-c2-2-" + hashlib.sha256(op.op_id.encode("utf-8")).hexdigest()[:32] + "-" + str(index)

    def _prepare_food_group(self, session, row, op, group, index, frozen_date, task_span, task_text,
                            used_spans, *, source_only: bool = False):
        if not isinstance(group, dict) or set(group) != _FOOD_GROUP_KEYS:
            raise _Refuse("rejected", "schema", "food_group_keys_invalid")
        group_start, group_end, _group_span = self._span(row.source, group.get("source_ref"),
                                                         container=task_span)
        group_text = row.source[group_start:group_end]
        meal_type = group.get("meal_type")
        if not isinstance(meal_type, str) or not (1 <= len(meal_type) <= 20):
            raise _Refuse("rejected", "schema", "meal_type_invalid")
        if meal_type not in task_text:
            raise _Refuse("rejected", "source", "meal_type_evidence_missing")
        record_time_raw = group.get("record_time")
        if not isinstance(record_time_raw, str):
            raise _Refuse("rejected", "schema", "record_time_invalid")
        record_time = _parse_iso(record_time_raw)
        if record_time is None:
            raise _Refuse("rejected", "schema", "record_time_invalid")
        try:
            original = datetime.fromisoformat(record_time_raw.replace("Z", "+00:00"))
        except ValueError:
            raise _Refuse("rejected", "schema", "record_time_invalid") from None
        if original.tzinfo is None:
            raise _Refuse("rejected", "schema", "record_time_not_timezone_aware")
        foods = group.get("foods")
        if not isinstance(foods, list) or not (1 <= len(foods) <= 30):
            raise _Refuse("rejected", "schema", "foods_count_invalid")
        record_date = record_time.astimezone(_TZINFO).date()
        facts = _date_facts(_sentence_window(row.source, group_start, group_end), frozen_date)
        if facts and record_date not in facts:
            raise _Refuse("needs_confirmation", "source", "record_time_mismatch")
        if record_date > frozen_date:
            raise _Refuse("rejected", "source", "future_date_rejected")
        items: list[dict] = []
        unresolved: list[str] = []
        has_ai = False
        for item in foods:
            if not isinstance(item, dict) or set(item) != _FOOD_ITEM_KEYS:
                raise _Refuse("rejected", "schema", "food_item_keys_invalid")
            item_start, item_end, _text = self._span(row.source, item.get("source_ref"),
                                                     container=(group_start, group_end))
            if (item_start, item_end) in used_spans:
                raise _Refuse("rejected", "source", "duplicate_source_occurrence")
            used_spans.add((item_start, item_end))
            source_kind = item.get("source")
            if source_kind not in _SOURCES:
                raise _Refuse("rejected", "schema", "food_source_invalid")
            entry = item.get("entry")
            if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
                raise _Refuse("rejected", "schema", "entry_keys_invalid")
            raw_text = entry.get("raw_text")
            if not isinstance(raw_text, str) or raw_text != item["source_ref"]["exact_text"]:
                raise _Refuse("rejected", "schema", "raw_text_mismatch")
            requested_name = entry.get("requested_name")
            if not isinstance(requested_name, str) or not requested_name.strip():
                raise _Refuse("rejected", "schema", "requested_name_invalid")
            if entry.get("is_negated") is not False:
                raise _Refuse("rejected", "source", "negated_entry_rejected")
            quantity = self._strict_number(entry.get("quantity"), minimum=0.000001)
            if quantity is None:
                raise _Refuse("rejected", "schema", "quantity_invalid")
            weight = entry.get("weight_g")
            if weight is not None:
                weight = self._strict_number(weight, minimum=0.000001)
                if weight is None:
                    raise _Refuse("rejected", "schema", "weight_invalid")
            unit = entry.get("unit")
            if unit is not None and (not isinstance(unit, str) or not unit.strip()):
                raise _Refuse("rejected", "schema", "unit_invalid")
            portion_name = entry.get("portion_name")
            if portion_name is not None and (not isinstance(portion_name, str) or not portion_name.strip()):
                raise _Refuse("rejected", "schema", "portion_name_invalid")
            item_meal_type = entry.get("meal_type")
            if item_meal_type is not None and item_meal_type != meal_type:
                raise _Refuse("rejected", "source", "item_meal_type_mismatch")
            record_date_entry = entry.get("record_date")
            if record_date_entry is not None and record_date_entry != record_date.isoformat():
                raise _Refuse("rejected", "source", "entry_date_mismatch")
            self._validate_entry_evidence(raw_text, requested_name, quantity, weight, unit, portion_name)
            if source_kind == "local":
                if source_only:
                    continue
                match = food_repository.resolve_food_in_session(session, requested_name)
                if match is None:
                    unresolved.append(requested_name)
                    continue
                if weight is not None:
                    resolved_weight, weight_source, estimated = weight, "user_input", False
                elif portion_name:
                    portion = food_repository.portion_weight_in_session(session, match.food_id, portion_name)
                    if portion is None:
                        unresolved.append(requested_name)
                        continue
                    resolved_weight = _round2(Decimal(str(portion)) * Decimal(str(quantity)))
                    weight_source, estimated = "portion_estimate", True
                else:
                    unresolved.append(requested_name)
                    continue
                factor = Decimal(str(resolved_weight)) / Decimal(100)
                items.append(self._food_row(
                    food_id=match.food_id, raw_food_name=raw_text, food_name=match.food_name,
                    weight_g=resolved_weight,
                    energy_kcal=_round2(Decimal(str(match.energy_kcal)) * factor),
                    protein_g=_round2(Decimal(str(match.protein_g)) * factor),
                    carb_g=_round2(Decimal(str(match.carb_g)) * factor),
                    fat_g=_round2(Decimal(str(match.fat_g)) * factor),
                    fiber_g=_round2(Decimal(str(match.fiber_g)) * factor),
                    is_estimated=estimated, weight_source=weight_source,
                    match_method=match.match_method, match_confidence=match.confidence))
            else:
                per100 = self._per100(item.get("nutrition_per_100g"))
                if per100 is None:
                    if source_kind == "ai":
                        raise _Refuse("rejected", "schema", "ai_per100_invalid")
                    unresolved.append(requested_name)
                    continue
                if weight is None:
                    if source_kind == "ai":
                        raise _Refuse("rejected", "schema", "ai_weight_required")
                    unresolved.append(requested_name)
                    continue
                if source_kind == "custom" and not self._custom_evidence(group_text, per100):
                    raise _Refuse("rejected", "source", "custom_evidence_missing")
                if source_only:
                    continue
                factor = Decimal(str(weight)) / Decimal(100)
                items.append(self._food_row(
                    food_id=None, raw_food_name=raw_text, food_name=requested_name,
                    weight_g=weight, energy_kcal=_round2(Decimal(str(per100["kcal"])) * factor),
                    protein_g=_round2(Decimal(str(per100["protein_g"])) * factor),
                    carb_g=_round2(Decimal(str(per100["carbs_g"])) * factor),
                    fat_g=_round2(Decimal(str(per100["fat_g"])) * factor),
                    fiber_g=_round2(Decimal(str(per100["fiber_g"])) * factor),
                    is_estimated=source_kind == "ai",
                    weight_source="user_input",
                    match_method="ai_estimate" if source_kind == "ai" else "user_custom",
                    match_confidence=0.65 if source_kind == "ai" else 1.0))
                has_ai = has_ai or source_kind == "ai"
        if items:
            totals = {
                "kcal": _round2(sum(Decimal(str(item["energy_kcal"])) for item in items)),
                "protein_g": _round2(sum(Decimal(str(item["protein_g"])) for item in items)),
                "carbs_g": _round2(sum(Decimal(str(item["carb_g"])) for item in items)),
                "fat_g": _round2(sum(Decimal(str(item["fat_g"])) for item in items)),
                "fiber_g": _round2(sum(Decimal(str(item["fiber_g"])) for item in items)),
            }
        else:
            totals = {"kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0}
        prepared = {
            "group_position": index, "meal_group_id": self._meal_group_id(op, index),
            "meal_type": meal_type, "record_time": _iso_utc(record_time), "_date": record_date.isoformat(),
            "items": items, "totals": totals,
        }
        return prepared, unresolved, has_ai

    @staticmethod
    def _validate_entry_evidence(raw_text: str, requested_name: str, quantity: float,
                                 weight: float | None, unit: str | None, portion_name: str | None) -> None:
        """Name/quantity/weight/unit/portion must have evidence in the item's own
        source fragment; a dictionary-known unit never substitutes an absent
        source unit (the conflicting pair clarifies instead), while aliases,
        Chinese numerals and canonical portion aliases stay legal (the same pure
        policy as the R003 source filter)."""
        normalized = normalize_food_name(raw_text)
        name = normalize_food_name(requested_name)
        name_ok = bool(name) and name in normalized
        if not name_ok:
            name_ok = any(normalize_food_name(surface) in normalized
                          for surface, canonical in FOOD_NAME_ALIASES.items()
                          if normalize_food_name(canonical) == name)
        if not name_ok:
            raise _Refuse("rejected", "source", "requested_name_evidence_missing")
        if weight is not None:
            if not _weight_evidence(float(weight), raw_text):
                raise _Refuse("rejected", "source", "weight_evidence_missing")
        elif not _quantity_evidence(float(quantity), raw_text):
            raise _Refuse("rejected", "source", "quantity_evidence_missing")
        if unit and not NutritionWriteBridge._portion_token_evidence(unit, raw_text):
            raise _Refuse("needs_confirmation", "source", "unit_evidence_missing")
        if portion_name and not NutritionWriteBridge._portion_token_evidence(portion_name, raw_text):
            raise _Refuse("needs_confirmation", "source", "portion_evidence_missing")

    @staticmethod
    def _portion_token_evidence(token: str, raw_text: str) -> bool:
        """The unit/portion token itself must appear in this item's own source
        fragment, allowing the pure canonical portion aliases (e.g. 大碗->碗)."""
        if token in raw_text:
            return True
        canonical = _canonical_portion_unit(token)
        return canonical != token and canonical in raw_text

    @staticmethod
    def _food_row(*, food_id, raw_food_name, food_name, weight_g, energy_kcal, protein_g, carb_g,
                  fat_g, fiber_g, is_estimated, weight_source, match_method, match_confidence) -> dict:
        return {"food_id": food_id, "raw_food_name": raw_food_name, "food_name": food_name,
                "weight_g": weight_g, "energy_kcal": energy_kcal, "protein_g": protein_g,
                "carb_g": carb_g, "fat_g": fat_g, "fiber_g": fiber_g, "is_estimated": is_estimated,
                "weight_source": weight_source, "match_method": match_method,
                "match_confidence": match_confidence}

    def _per100(self, value):
        if value is None or not isinstance(value, dict) or set(value) != _PER100_KEYS:
            return None
        result = {}
        for field, (minimum, maximum) in _PER100_RANGES.items():
            number = self._strict_number(value.get(field), minimum=minimum, maximum=maximum)
            if number is None:
                return None
            result[field] = number
        return result

    @staticmethod
    def _custom_evidence(group_text: str, per100: dict) -> bool:
        if not isinstance(group_text, str):
            return False
        if not any(marker in group_text for marker in ("每100克", "每100g", "每百克")):
            return False
        for field, value in per100.items():
            if not _macro_evidence(_PER100_TO_MACRO[field], value, group_text):
                return False
        return True

    # ---------------------------------------------------------------- helpers

    def _next_checkpoint(self, row) -> ledger_api.Checkpoint:
        current = row.checkpoint
        if not isinstance(current, dict):
            raise _Refuse("not_committed", "storage", "checkpoint_corrupt")
        values = []
        for field in ("supervisor_turns", "tool_steps", "event_sequence"):
            item = current.get(field)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise _Refuse("not_committed", "storage", "checkpoint_corrupt")
            values.append(item)
        return ledger_api.Checkpoint(supervisor_turns=max(1, values[0]), tool_steps=max(1, values[1]),
                                     event_sequence=values[2] + 1, budget_scope_id=row.budget_scope_id)

    @staticmethod
    def _ledger_refusal(exc, dispatched=False) -> _Refuse:
        code = getattr(exc, "code", None)
        if code == "input_invalid":
            return _Refuse("rejected", "schema", code, dispatched)
        if code in {"identity_conflict", "terminal_frozen"}:
            return _Refuse("rejected", "identity", code, dispatched)
        if code in {"lease_busy", "lease_lost"}:
            return _Refuse("rejected", "lease", code, dispatched)
        if code == "revision_conflict":
            return _Refuse("rejected", "revision", code, dispatched)
        return _Refuse("not_committed", "storage", code or "ledger_error", dispatched)

    @staticmethod
    def _business_refusal(exc, dispatched=True) -> _Refuse:
        code = getattr(exc, "code", None)
        if code == "lease_lost":
            return _Refuse("not_committed", "lease", code, dispatched)
        if code == "revision_conflict":
            return _Refuse("not_committed", "revision", code, dispatched)
        if code == "input_invalid":
            return _Refuse("not_committed", "schema", code, dispatched)
        if code in {"identity_conflict", "terminal_frozen"}:
            return _Refuse("not_committed", "identity", code, dispatched)
        return _Refuse("not_committed", "storage", code or "ledger_error", dispatched)

    @staticmethod
    def _draft_observation(task_id, outcome) -> dict:
        return DomainObservation(status=ToolStatus.NEEDS_CLARIFICATION, task_id=task_id,
                                 commit_state=CommitState.PENDING, receipt_refs=[]).to_dict()

    def _limited(self, refuse, user_id=None, request_id=None, op_id=None) -> dict:
        if refuse.outcome in {"needs_confirmation", "pending_ready"}:
            observation = DomainObservation(status=ToolStatus.NEEDS_CLARIFICATION,
                                            commit_state=CommitState.PENDING,
                                            missing_fields=[refuse.detail] if refuse.detail else [],
                                            error=refuse.detail if refuse.outcome != "pending_ready" else None,
                                            receipt_refs=[]).to_dict()
        else:
            status = {"rejected": ToolStatus.REJECTED, "not_committed": ToolStatus.PARTIAL,
                      "commit_unknown": ToolStatus.COMMIT_UNKNOWN}.get(refuse.outcome, ToolStatus.UNAVAILABLE)
            observation = DomainObservation(status=status, task_id=None, commit_state=None,
                                            error=refuse.detail, receipt_refs=[]).to_dict()
        return self._result(refuse.outcome, reason=refuse.reason, detail=refuse.detail,
                            dispatched=refuse.dispatched, observation=observation,
                            ledger_snapshot=self._safe_snapshot(user_id, request_id),
                            user_id=user_id, request_id=request_id, op_id=op_id, pending=None)

    @staticmethod
    def _result(outcome, *, reason=None, detail=None, observation=None, dispatched=False,
                ledger_snapshot=None, user_id=None, request_id=None, op_id=None, task_id=None,
                position=None, pending=None) -> dict:
        if observation is None:
            status = {"rejected": ToolStatus.REJECTED, "not_committed": ToolStatus.PARTIAL,
                      "commit_unknown": ToolStatus.COMMIT_UNKNOWN}.get(outcome, ToolStatus.UNAVAILABLE)
            observation = DomainObservation(status=status, task_id=task_id, error=detail).to_dict()
        return {"request_id": request_id, "op_id": op_id, "task_id": task_id, "position": position,
                "outcome": outcome, "dispatched": bool(dispatched), "reason": reason,
                "observation": observation, "ledger_snapshot": ledger_snapshot, "pending": pending}

    def _safe_snapshot(self, user_id, request_id):
        try:
            if user_id is None or request_id is None:
                return None
            return _strip_private(self._ledger.load_request(user_id, request_id))
        except Exception:
            return None

    @staticmethod
    def _rollback_quiet(session) -> None:
        try:
            session.rollback()
        except Exception:
            pass

    @staticmethod
    def _close_quiet(session) -> None:
        try:
            session.close()
        except Exception:
            pass
