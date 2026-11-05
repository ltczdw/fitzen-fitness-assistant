"""Write authorization from model semantics plus application-owned source scope.

Authorization follows the 2026-10-18 semantic contract: a validated model
task/action/flags owns the record intent, and the application separately
checks that entities and spans come from the user's original words. This
module intentionally contains no keyword, regular-expression or text
classification logic; tests freeze that.

When no model decision exists (``semantic_available`` false), the guarded rule
fallback stays read-only: write candidates are marked
``needs_confirmation`` / ``semantic_intent_unavailable`` and can never execute.
"""
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from app.services.agent.task_models import is_record_write


def source_safety_blocked(source: str) -> str | None:
    """Shared pre-write safety gate.

    The authoritative decision is the request-local source-bound model
    SafetyAssessment. A missing or invalid decision is uniformly unknown and
    blocks new domain operations; no input wording is classified here.
    """
    from app.services.agent.semantics import safety_for

    decision = safety_for(source)
    if decision.get("status") == "assessed":
        return "urgent_safety_gate" if decision.get("level") == "urgent" else None
    return "safety_assessment_unavailable"


def _pending_context_matches(
    pending: Mapping[str, Any] | Any | None,
    task_type: str,
    session_id: str | None,
    now: datetime,
) -> bool:
    if pending is None:
        return False
    get = pending.get if isinstance(pending, Mapping) else lambda name, default=None: getattr(pending, name, default)
    expiry = get("expires_at")
    if isinstance(expiry, str):
        try:
            expiry = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        except ValueError:
            return False
    if not isinstance(expiry, datetime) or not get("record_id"):
        return False
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return bool(
        get("task_type") == task_type
        and session_id
        and get("session_id") == session_id
        and expiry > current
    )


def authorize_write_span(text: str, flags: Mapping[str, Any] | None = None, *,
                         write_window: str = "", tail_window: str = "",
                         pending_bound: bool = False,
                         urgent: bool = False,
                         semantic_available: bool | None = None) -> tuple[str, str | None]:
    """Return ``(authorization, reason)`` for one proposed write task.

    The model's validated task/action/structured flags decide the intent. The
    application only adds independent gates it owns: the urgent-safety gate,
    an explicitly bound pending record, and the presence of a model decision at
    all. ``write_window``/``tail_window`` are accepted for caller compatibility
    but never re-classified; source wording outside the cited span cannot
    override the model. Without a model decision the rule fallback is read-only.
    """
    flags = flags or {}
    if urgent:
        return "rejected_unauthorized", "urgent_safety_gate"
    for name, reason in (
        ("is_negated", "planner_flag_negated"),
        ("is_hypothetical", "planner_flag_hypothetical"),
        ("is_third_party", "planner_flag_third_party"),
    ):
        if flags.get(name) is True:
            return "rejected_unauthorized", reason
    if pending_bound:
        return "authorized", None
    if flags.get("action") == "confirm":
        return "needs_confirmation", "pending_record_not_found"
    if semantic_available is None:
        semantic_available = flags.get("semantic_available") is not False
    if not semantic_available:
        # No model result: never guess a record intent from the text.
        return "needs_confirmation", "semantic_intent_unavailable"
    return "authorized", None


def _source_urgent(source: str) -> bool:
    """Only an assessed model judgment can mark a request urgent."""
    from app.services.agent.semantics import safety_for

    decision = safety_for(source)
    return (decision.get("status") == "assessed"
            and decision.get("level") == "urgent")


def guard_planner_tasks(
    source: str,
    tasks: list[dict[str, Any]],
    *,
    now: datetime,
    pending_record: Mapping[str, Any] | Any | None = None,
    current_session_id: str | None = None,
    semantic_available: bool = False,
    urgent: bool | None = None,
    safety_assessed: bool | None = None,
) -> list[dict[str, Any]]:
    """Honor model task/action and flags without reclassifying its source words."""
    if urgent is None:
        urgent = _source_urgent(source)
    if safety_assessed is None:
        from app.services.agent.semantics import safety_for
        safety_assessed = safety_for(source).get("status") == "assessed"
    events: list[dict[str, Any]] = []
    for task in tasks:
        valid_write = is_record_write(task)
        if not valid_write and not task.get("_v1_write_candidate"):
            task.pop("_v1_write_candidate", None)
            continue
        pending_bound = task.get("action") == "confirm" and _pending_context_matches(
            pending_record, str(task.get("task_type") or ""), current_session_id, now
        )
        if valid_write and not semantic_available and not pending_bound and not urgent:
            # No model understanding at all: keep the established read-only
            # fallback reason. Safety is necessarily unassessed here too, but
            # the missing intent decision is the primary explanation. An
            # explicitly bound pending confirmation is the existing exception,
            # and an urgent tripwire always outranks a missing-intent notice.
            authorization, reason = "needs_confirmation", "semantic_intent_unavailable"
        elif valid_write and not safety_assessed and not urgent:
            authorization, reason = "needs_confirmation", "safety_assessment_unavailable"
        else:
            authorization, reason = authorize_write_span(
                str(task.get("text") or ""),
                {**task, "semantic_available": semantic_available},
                pending_bound=pending_bound, urgent=urgent,
                semantic_available=semantic_available,
            )
        task["status"] = authorization
        task["write_authorization"] = authorization
        task["execution_status"] = "pending"
        task["write_rejection_reason"] = reason
        task["side_effect"] = valid_write and authorization == "authorized"
        payload = task.setdefault("payload", {})
        payload["write_authorization"] = authorization
        payload["write_rejection_reason"] = reason
        if authorization == "needs_confirmation":
            events.append({"code": "v3_write_needs_confirmation", "task_id": task.get("id"), "reason": reason})
        elif reason:
            events.append({"code": "v3_write_rejected", "task_id": task.get("id"), "reason": reason})
        task.pop("_v1_write_candidate", None)
    return events


__all__ = [
    "authorize_write_span",
    "guard_planner_tasks",
]
