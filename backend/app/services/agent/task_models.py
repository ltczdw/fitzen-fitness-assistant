from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping
from typing import Any


class TaskType(StrEnum):
    WORKOUT_RECORD = "workout_record"
    NUTRITION_RECORD = "nutrition_record"
    NUTRITION_ADVICE = "nutrition_advice"
    TRAINING_PLAN = "training_plan"
    TRAINING_QA = "training_qa"
    SAFETY_TRIAGE = "safety_triage"
    CHITCHAT = "chitchat"


class TaskAction(StrEnum):
    RECORD = "record"
    CONFIRM = "confirm"
    QUESTION = "question"
    ANSWER = "answer"


# ``task_type`` names the domain; ``action`` names the operation. Only a write
# action on a record domain has a side effect. A record domain with a read
# action is a business-record read and must keep its domain label: consumers ask
# these helpers instead of re-deriving the operation from the domain.
RECORD_TASK_TYPES = frozenset({TaskType.WORKOUT_RECORD.value, TaskType.NUTRITION_RECORD.value})
WRITE_TASK_ACTIONS = frozenset({TaskAction.RECORD.value, TaskAction.CONFIRM.value})
READ_TASK_ACTIONS = frozenset({TaskAction.QUESTION.value, TaskAction.ANSWER.value})


def _task_field(task, name):
    if isinstance(task, Mapping):
        return task.get(name)
    return getattr(task, name, None)


def is_record_task(task) -> bool:
    return str(_task_field(task, "task_type") or "") in RECORD_TASK_TYPES


def is_write_action(task) -> bool:
    return str(_task_field(task, "action") or "") in WRITE_TASK_ACTIONS


def is_record_write(task) -> bool:
    return is_record_task(task) and is_write_action(task)


def is_record_read(task) -> bool:
    return (is_record_task(task)
            and str(_task_field(task, "action") or "") in READ_TASK_ACTIONS)


def read_spec_of(task):
    return _task_field(task, "read_spec")


def is_advice_task(task) -> bool:
    return str(_task_field(task, "delivery_type") or "") == "advice"


def task_scope_value(task, name):
    """Read one semantic selector from the task or its payload scopes.

    Planner outlines, validated semantic tasks and React candidates carry the
    same fields at slightly different levels. Consumers must use one lookup so a
    merged/validated task cannot silently lose its delivery identity.
    """
    if isinstance(task, Mapping):
        if name in task and task.get(name) is not None:
            return task.get(name)
        payload = task.get("payload")
    else:
        value = getattr(task, name, None)
        if value is not None:
            return value
        payload = getattr(task, "payload", None)
    if isinstance(payload, Mapping):
        for key in ("semantic", "dialogue"):
            scope = payload.get(key)
            if isinstance(scope, Mapping) and name in scope:
                return scope.get(name)
    return None


def is_preview_task(task) -> bool:
    """A result that modifies an existing preview, not a background condition."""
    return (str(task_scope_value(task, "task_type") or "") == TaskType.TRAINING_PLAN.value
            and str(task_scope_value(task, "delivery_type") or "") == "preview"
            and str(task_scope_value(task, "goal_role") or "result") == "result")


def is_plan_save_capability_task(task) -> bool:
    """A request to save/confirm a plan through chat, or an explicit capability ask.

    The chat has no plan-write tool. A plan task carrying a write action is
    therefore a save-boundary request, not a plan-generation instruction. This
    is a structural contract: it never inspects user wording.
    """
    task_type = str(task_scope_value(task, "task_type") or "")
    action = str(task_scope_value(task, "action") or "")
    delivery = str(task_scope_value(task, "delivery_type") or "")
    if task_type != TaskType.TRAINING_PLAN.value:
        return False
    return action in WRITE_TASK_ACTIONS or delivery == "capability"


def is_capability_task(task) -> bool:
    """True for an explicit application-capability target (any domain)."""
    return str(task_scope_value(task, "delivery_type") or "") == "capability"


def answer_identity(task) -> tuple:
    """Structured identity for merging equivalent answer targets.

    This is deliberately composed from model-provided semantic fields and the
    presence of application-owned scopes; it never inspects wording.
    """
    return (
        str(task_scope_value(task, "task_type") or ""),
        str(task_scope_value(task, "action") or ""),
        str(task_scope_value(task, "reply_purpose") or ""),
        str(task_scope_value(task, "delivery_type") or ""),
        str(task_scope_value(task, "goal_role") or "result"),
        bool(task_scope_value(task, "quoted")),
        bool(task_scope_value(task, "is_hypothetical")),
        bool(task_scope_value(task, "is_third_party")),
        bool(task_scope_value(task, "side_effect")),
        bool(task_scope_value(task, "record_query")),
        bool(task_scope_value(task, "read_spec")),
        bool(task_scope_value(task, "plan_contract")),
    )



def is_plan_read_task(task) -> bool:
    """A read of the stored plan, not a background condition or a new plan."""
    return (str(_task_field(task, "task_type") or "") == TaskType.TRAINING_PLAN.value
            and str(_task_field(task, "action") or "") in READ_TASK_ACTIONS
            and str(_task_field(task, "reply_purpose") or "") == "fact_recall"
            and str(_task_field(task, "goal_role") or "result") == "result"
            and (_task_field(task, "plan_result") is False or read_spec_of(task) is not None))


def is_pure_read_spec_task(task) -> bool:
    spec = read_spec_of(task)
    if not isinstance(spec, Mapping):
        return False
    return (spec.get("resource") in {"profile", "training_plan"}
            and str(_task_field(task, "action") or "") in READ_TASK_ACTIONS
            and str(_task_field(task, "reply_purpose") or "") == "fact_recall")


# A read goal is served by the answer capability of its domain, not by an exact
# type label: a saved-record read and a knowledge answer in the same domain use
# the same read tool. Every TaskType must state its answer domain here, so a new
# type cannot silently inherit an unrelated branch.
TASK_TYPE_TO_ANSWER_DOMAIN: Mapping[TaskType, str | None] = MappingProxyType({
    TaskType.NUTRITION_RECORD: "nutrition_answer",
    TaskType.NUTRITION_ADVICE: "nutrition_answer",
    TaskType.WORKOUT_RECORD: "training_answer",
    TaskType.TRAINING_QA: "training_answer",
    TaskType.TRAINING_PLAN: "training_plan_answer",
    TaskType.SAFETY_TRIAGE: "safety_answer",
    TaskType.CHITCHAT: None,
})


def answer_domain(task_type) -> str | None:
    try:
        return TASK_TYPE_TO_ANSWER_DOMAIN[TaskType(str(task_type or ""))]
    except ValueError:
        return None


def serves_read_goal(tool_task_type, goal_task_type) -> bool:
    # The existing answer tool also carries non-urgent symptom education.
    # Keep the safety goal's own domain/type; its worker selects the medical
    # evidence contract instead of relabeling it as ordinary training advice.
    if (tool_task_type == TaskType.TRAINING_QA and goal_task_type == TaskType.SAFETY_TRIAGE):
        return True
    domain = answer_domain(tool_task_type)
    return domain is not None and domain == answer_domain(goal_task_type)


# Canonical planning definitions. CHITCHAT is handled by the pre-planner
# short-circuit and intentionally has no LangGraph worker.
TASK_TYPE_TO_WORKER: Mapping[TaskType, str | None] = MappingProxyType({
    TaskType.WORKOUT_RECORD: "workout_record",
    TaskType.NUTRITION_RECORD: "nutrition",
    TaskType.NUTRITION_ADVICE: "nutrition",
    TaskType.TRAINING_PLAN: "plan_adjust",
    TaskType.TRAINING_QA: "training_qa",
    TaskType.SAFETY_TRIAGE: "training_qa",
    TaskType.CHITCHAT: None,
})


# A record write is implemented by the record worker; a record read in the same
# domain is answered by the domain's read implementation. Only the workout
# record domain needs a different implementation for the two operations.
TASK_TYPE_READ_WORKER: Mapping[TaskType, str | None] = MappingProxyType({
    **TASK_TYPE_TO_WORKER,
    TaskType.WORKOUT_RECORD: "training_qa",
})


def task_worker(task_type, action) -> str | None:
    try:
        kind = TaskType(str(task_type or ""))
    except ValueError:
        return None
    if kind in RECORD_TASK_TYPES and str(action or "") in READ_TASK_ACTIONS:
        return TASK_TYPE_READ_WORKER[kind]
    return TASK_TYPE_TO_WORKER[kind]

TASK_TYPE_TO_INTENT: Mapping[TaskType, str] = MappingProxyType({
    TaskType.WORKOUT_RECORD: "workout_record",
    TaskType.NUTRITION_RECORD: "nutrition",
    TaskType.NUTRITION_ADVICE: "nutrition",
    TaskType.TRAINING_PLAN: "plan_adjust",
    TaskType.TRAINING_QA: "training_qa",
    TaskType.SAFETY_TRIAGE: "symptom_triage",
    TaskType.CHITCHAT: "training_qa",
})

INTENT_TO_TASK_TYPE: Mapping[str, TaskType] = MappingProxyType({
    "workout_record": TaskType.WORKOUT_RECORD,
    "nutrition": TaskType.NUTRITION_ADVICE,
    "plan_adjust": TaskType.TRAINING_PLAN,
    "training_qa": TaskType.TRAINING_QA,
    "symptom_triage": TaskType.SAFETY_TRIAGE,
})

# The route tuple is (domain, retrieval task, deterministic query expansions).
TASK_TYPE_TO_RAG_ROUTE: Mapping[TaskType, tuple[str, str, tuple[str, ...]]] = MappingProxyType({
    TaskType.SAFETY_TRIAGE: ("rehabilitation", "safety_triage", ("康复", "损伤恢复", "危险信号", "就医建议")),
    TaskType.NUTRITION_ADVICE: ("nutrition", "nutrition_qa", ("营养学", "膳食安排", "能量", "蛋白质", "运动饮食")),
    TaskType.NUTRITION_RECORD: ("nutrition", "nutrition_qa", ("营养学", "膳食安排", "能量", "蛋白质", "运动饮食")),
    TaskType.WORKOUT_RECORD: ("exercise_science", "training_guidance", ("运动学", "训练方法", "动作技术", "负荷与恢复")),
    TaskType.TRAINING_PLAN: ("exercise_science", "training_guidance", ("运动学", "训练方法", "动作技术", "负荷与恢复")),
    TaskType.TRAINING_QA: ("fitness_general", "fitness_qa", ("健身运动", "运动科学", "动作安全", "训练恢复")),
    TaskType.CHITCHAT: ("fitness_general", "fitness_qa", ("健身运动", "运动科学", "动作安全", "训练恢复")),
})


@dataclass
class TaskSpec:
    """One executable intent instance inside a user request.

    ``worker`` is the implementation boundary, while ``task_type`` describes
    the business operation. They are deliberately separate: two tasks may use
    the same worker but must still keep different ids, text, and results.
    """

    id: str
    task_type: str
    worker: str
    action: str
    text: str
    payload: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    side_effect: bool = False
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
