"""Durable, conversation-bound continuation of read-only planning questions."""
from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime, timedelta
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from app.core import clock
from app.db.memory_models import MemoryTurn
from app.services.memory import compact
from app.services.security.safety import SafetyAssessment, URGENT_SAFETY_ANSWER

KEY = "__planning_question_v1__"
BINDING = "__planning_reply_v1__"
TTL = timedelta(minutes=30)


class ContinuationError(ValueError):
    pass


class PlanningReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    draft_id: str
    action: Literal["update", "cancel", "unrelated", "unclear", "explain"]
    quote: str = Field(default="", max_length=3000)
    explanation: str = Field(default="", max_length=1200,
        description="仅explain使用：直接回答用户对术语/缺项必要性的疑问，不把助手解释变成用户条件或授权。")
    safety: SafetyAssessment | None = None


def meta(row):
    return copy.deepcopy(row.operations or {})


def put(row, name, value):
    state = meta(row)
    state.setdefault("tasks", {})[name] = value
    row.operations = state


def live(s, request, epoch):
    row = s.scalar(select(MemoryTurn).where(MemoryTurn.request_id == request.request_id).with_for_update())
    if not row or row.conversation_id != request.conversation_id or row.memory_epoch != epoch or row.deleted_at or row.hidden_at:
        raise ContinuationError("continuation owner unavailable")
    return row


def valid(row, draft, request, epoch):
    return bool(row and draft and row.conversation_id == request.conversation_id and row.memory_epoch == epoch
                and row.status == "completed" and not row.deleted_at and not row.hidden_at
                and timedelta(0) <= clock.utc_now() - datetime.fromisoformat(draft["created_at"]) < TTL)


def _remember(request, result, tasks, sources):
    questions = {item["task_id"]: item["clarification"] for item in result.get("worker_results", [])
                 if item.get("task_type") == "training_plan" and (item.get("clarification") or {}).get("category") == "missing_information"}
    # ReAct carries the actual observed clarification separately from its
    # aggregate worker result. Match only an explicitly addressed plan task.
    for item in result.get("clarifications", []):
        if item.get("task_type") == "training_plan" and item.get("category") == "missing_information":
            questions.setdefault(item.get("task_id"), item)
    pending = [copy.deepcopy(task) for task in tasks if task.get("id") in questions]
    if not pending:
        return
    # Never adopt writes or replay previously completed tasks in a continuation.
    for task in pending:
        task.update(action="question", side_effect=False, depends_on=[], payload={})
    with compact.session() as s, s.begin():
        profile = compact.lock_profile(s)
        row = live(s, request, profile.memory_epoch)
        if meta(row).get("tasks", {}).get(KEY):
            return  # retry must not create a different question identity
        put(row, KEY, {"id": str(uuid4()), "origin": row.turn_id, "created_at": clock.utc_now().isoformat(),
                       "state": "waiting", "claimed_by": None, "tasks": pending, "questions": questions,
                       "reference_date": clock.today().isoformat(),
                       "source": request.message, "sources": [*sources, request.message]})


async def remember(request, result, tasks, sources=()):
    if not compact.current_turn.get():
        return
    await asyncio.to_thread(_remember, request, result, tasks, sources)


def _load(request):
    with compact.session() as s, s.begin():
        profile = compact.lock_profile(s)
        current = live(s, request, profile.memory_epoch)
        bound = meta(current).get("tasks", {}).get(BINDING)
        if bound:
            origin = s.get(MemoryTurn, bound["origin"])
            draft = meta(origin).get("tasks", {}).get(KEY) if origin else None
            if valid(origin, draft, request, profile.memory_epoch) and draft.get("claimed_by") == request.request_id:
                return {"binding": bound}
            raise ContinuationError("stale planning reply")
        rows = s.scalars(select(MemoryTurn).where(MemoryTurn.conversation_id == request.conversation_id,
                MemoryTurn.memory_epoch == profile.memory_epoch, MemoryTurn.status == "completed",
                MemoryTurn.deleted_at.is_(None), MemoryTurn.hidden_at.is_(None))
                .order_by(MemoryTurn.completed_at.desc(), MemoryTurn.turn_id.desc()).limit(20))
        for row in rows:
            draft = meta(row).get("tasks", {}).get(KEY)
            if draft:
                # A newer terminal/expired draft must never expose an older one.
                return draft if valid(row, draft, request, profile.memory_epoch) and draft["state"] == "waiting" else None
    return None


def _transition(request, expected, decision):
    with compact.session() as s, s.begin():
        profile = compact.lock_profile(s)
        current = live(s, request, profile.memory_epoch)
        origin = s.get(MemoryTurn, expected["origin"])
        draft = meta(origin).get("tasks", {}).get(KEY) if origin else None
        if not valid(origin, draft, request, profile.memory_epoch) or draft["id"] != expected["id"] or draft["state"] != "waiting":
            raise ContinuationError("planning question already handled")
        if not decision.quote or decision.quote not in request.message:
            raise ContinuationError("unbound planning supplement")
        if decision.action == "cancel":
            draft["state"] = "cancelled"
            put(origin, KEY, draft)
            return {"answer": "已取消待安排的训练计划，已保存的记录保持不变。"}
        source = draft["source"] + "\n用户补充：" + decision.quote
        if len(source) > 3800:
            raise ContinuationError("planning continuation too long")
        tasks = []
        for task in draft["tasks"]:
            text = task["text"]
            if draft.get("reference_date"):
                text += "\n原计划请求日期：" + draft["reference_date"] + "（原任务中的相对日期以此为准，用户本轮可明确更正）。"
            text += "\n用户补充：" + decision.quote
            tasks.append({"task_type": "training_plan", "action": "question", "text": text,
                          "is_negated": False, "is_hypothetical": False, "is_third_party": False})
        # The trusted source contains only pending read tasks and the user's own
        # reply. Earlier record text is context, never replanned as a new write.
        command_source = "\n".join(task["text"] for task in tasks)
        binding = {"origin": origin.turn_id, "source": command_source, "parsed_input": {"tasks": tasks},
                   "information_sources": [*draft.get("sources", [draft["source"]]), decision.quote]}
        if request.mode != "legacy" and decision.safety is not None:
            binding["reply_safety_level"] = decision.safety.level
        draft.update(state="claimed", claimed_by=request.request_id)
        put(origin, KEY, draft)
        put(current, BINDING, binding)
        return binding


def current_questions(questions):
    """Project old typed missing fields onto current requirements without rewriting history."""
    from app.services.agent.task_information import question
    current={}
    for task_id,item in questions.items():
        if item.get("category") != "missing_information":
            current[task_id]=item
            continue
        names=[field["name"] for field in item.get("fields") or []
               if field["name"] not in {"equipment","session_time"}]
        if names:
            current[task_id]=question("training_plan",names,task_id=task_id).model_dump()
    return current


async def continue_request(llm, request):
    if not compact.current_turn.get():
        return None
    draft = await asyncio.to_thread(_load, request)
    if draft:
        draft={**draft,"questions":current_questions(draft.get("questions") or {})}
    if not draft:
        return None
    if "binding" in draft:
        return draft["binding"]
    require_safety = request.mode != "legacy"
    def validate_reply(value):
        from app.services.llm.structured import StructuredOutputError
        if require_safety and (value.safety is None or not value.safety.bound_to(request.message)):
            raise StructuredOutputError("planning_reply_safety_invalid", [{
                "loc": ["safety"], "type": "planning_reply_safety_invalid",
                "hint": "必须独立判断当前reply的safety；evidence只能逐字引用当前reply，不得省略或引用历史。"}])
        if require_safety and value.safety.level == "urgent":
            return  # Current danger precedes malformed business explanations.
        if value.action == "explain" and (not value.explanation.strip() or not value.quote.strip()
                                         or value.quote not in request.message):
            raise StructuredOutputError("planning_reply_explanation_invalid", [{
                "loc": ["explanation"], "type": "planning_reply_explanation_invalid",
                "hint": "explain必须直接回答当前疑问，quote逐字绑定当前reply；不能生成新用户条件。"}])
    try:
        raw = await llm.complete_structured([
            {"role": "system", "content": "判断用户新回复是否补充唯一绑定的训练计划问题。应用现在默认未指定器械为健身房，不要求时长；questions已按当前规则更新，不能重新要求器械或时长。没有缺项且用户明确希望继续原计划时选update，即使没有新增条件，也引用其当前继续请求。新话题选unrelated；明确取消选cancel；仅没有实质信息的含糊应答选unclear。用户询问术语或质疑缺项必要性时选explain，在explanation中结合原目标直接回答疑问，不得只重复缺项。说明拆分方式描述训练内容分配，周频率用于周日历；未要求周日历可以先给循环模板，不能声称三分化必然是每周3次或6次。explain的quote逐字引用当前疑问，解释不写入用户事实、不消费计划问题。用户明确补充或修改范围时选update，quote必须逐字引用当前回复中完整的条件和限制。新目标/话题不强行续接。不执行保存，不根据历史助手内容猜测事实。同时独立判断当前reply的safety，证据逐字来自当前reply，历史症状不能当成当前急症。"},
            {"role": "user", "content": json.dumps({"draft_id": draft["id"], "questions": draft["questions"],
                "pending_tasks": draft["tasks"], "reply": request.message}, ensure_ascii=False)},
        ], PlanningReply, name="return_planning_reply", temperature=0,
            **({"validator": validate_reply} if require_safety else {}))
        decision = raw if isinstance(raw, PlanningReply) else PlanningReply.model_validate(raw)
        validate_reply(decision)
        if require_safety:
            if decision.safety.level == "urgent":
                return {"answer": URGENT_SAFETY_ANSWER, "intent": "symptom_triage",
                        "safety_level": "urgent", "safety_status": "assessed"}
        if decision.draft_id != draft["id"]:
            raise ContinuationError("wrong planning question")
    except Exception as exc:
        from app.services.llm.budget import ModelBudgetError
        from app.services.llm.client import LLMGatewayError, LLMErrorCategory
        if isinstance(exc, ModelBudgetError) or (isinstance(exc, LLMGatewayError) and exc.category == LLMErrorCategory.POLICY):
            raise
        from app.services.agent.task_information import question
        answer = "暂时无法识别这次补充，待安排的计划已保留；已完成记录不会重复保存。请稍后重试。"
        return {"answer": answer, "clarifications": [question("training_plan", [],
            category="service_unavailable", text=answer).model_dump()],
            **({"safety_status": "unavailable"} if require_safety else {})}
    if decision.action == "unrelated":
        return None
    if decision.action == "explain":
        from app.services.security.output_guard import apply_output_guardrail
        return {"answer": apply_output_guardrail(decision.explanation).text,
                "clarifications": list(draft["questions"].values()),
                **({"safety_level": decision.safety.level, "safety_status": "assessed"} if require_safety else {})}
    if decision.action == "unclear":
        return {"answer": "\n".join(item["question"] for item in draft["questions"].values()) or "请明确本次是继续原计划，还是另有需求。",
                "clarifications": list(draft["questions"].values()),
                **({"safety_level": decision.safety.level, "safety_status": "assessed"} if require_safety else {})}
    result = await asyncio.to_thread(_transition, request, draft, decision)
    if require_safety and "answer" in result:
        result.update(safety_level=decision.safety.level, safety_status="assessed")
    return result
