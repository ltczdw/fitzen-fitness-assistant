import json
from contextlib import nullcontext
from uuid import uuid4

from typing import Annotated

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.db import session as db
from app.db.storage import require_ready
from app.services.agent.agent import agent
from app.services.memory.redis_memory import memory_service


def _database_ready() -> bool:
    return bool(db.DB_READY and db.SessionLocal is not None)


def _chat_model_used(metrics) -> bool:
    return bool(metrics and any(
        call.get("capability") == "chat" for call in metrics.calls))


async def _urgent_without_storage(request: ChatRequest) -> ChatResponse | None:
    """Assess emergency safety before the storage readiness gate.

    Only used while the database is unavailable. The judgment still comes from
    the same source-bound structured model contract; no keyword rule is added.
    A normal/unassessed request then takes the existing 503 readiness path.

    The preflight is outside the normal request lifecycle, so it must create
    the same request-local semantics and telemetry scopes itself. It never
    reuses a previous request's understanding/safety cache, and an unavailable
    model budget fails closed before any provider call.
    """
    if _database_ready():
        return None
    from app.services.agent.semantics import (
        request_understanding, safety_for, understand,
    )
    from app.services.llm.budget import TASK_POLICY, active_scope, key, model_scope
    from app.services.security.safety import URGENT_SAFETY_ANSWER
    from app.services.telemetry import RequestMetrics, active_metrics

    metrics = active_metrics.get()
    metrics_token = None
    if metrics is None:
        metrics = RequestMetrics()
        metrics_token = active_metrics.set(metrics)
    understanding_token = request_understanding.set({})
    current_scope = active_scope.get()
    scope_context = (nullcontext(current_scope) if current_scope is not None
                     else model_scope("request:" + request.request_id,
                                      kind="request", policy=TASK_POLICY))
    try:
        with scope_context as budget:
            if getattr(budget, "owner_id", None) is None:
                budget.owner_id = key(
                    str(request.user_id) + ":" + str(request.conversation_id))
            await understand(agent.graph.llm, request.message)
            safety = safety_for(request.message)
            if safety.get("status") != "assessed" or safety.get("level") != "urgent":
                return None
            summary = metrics.summary()
            mode = getattr(request, "mode", None) or "legacy"
            return ChatResponse(
                answer=URGENT_SAFETY_ANSWER,
                intent="symptom_triage",
                citations=[],
                safety_level="urgent",
                model_used=_chat_model_used(metrics),
                request_id=request.request_id,
                metrics={
                    **summary,
                    "safety_status": "assessed",
                    "requested_mode": mode,
                    "execution_mode": "urgent",
                    "mode_reason": "urgent_safety_assessed",
                },
            )
    except Exception:
        # Budget/storage/model failure is an explicit 503 at the caller; it is
        # never converted into normal safety or an urgent claim.
        return None
    finally:
        request_understanding.reset(understanding_token)
        if metrics_token is not None:
            active_metrics.reset(metrics_token)


async def _urgent_stream_events(response: ChatResponse):
    payload = response.model_dump()
    events = [{"event": "status", "data": {"message": "正在处理请求"}},
              {"event": "meta", "data": {k: v for k, v in payload.items() if k != "answer"}}]
    for index in range(0, len(response.answer), 32):
        events.append({"event": "token",
                       "data": {"content": response.answer[index:index + 32]}})
    events.append({"event": "done",
                   "data": {k: v for k, v in payload.items() if k != "answer"}})
    for event in events:
        yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"


router = APIRouter(tags=["agent"])


@router.get("/chat/conversations")
async def chat_conversations(
    user_id: Annotated[str, Query(min_length=1, max_length=100)] = "demo-user",
) -> dict:
    return {"conversations": await memory_service.list_conversations(user_id)}


@router.post("/chat/conversations")
async def create_chat_conversation(
    user_id: Annotated[str, Query(min_length=1, max_length=100)] = "demo-user",
    conversation_id: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
    title: Annotated[str | None, Query(max_length=80)] = None,
) -> dict:
    conversation = conversation_id or f"conversation-{uuid4().hex}"
    result = await memory_service.register_conversation(user_id, conversation, title)
    return {"conversation": result}


@router.delete("/chat/conversations/{conversation_id}")
async def delete_chat_conversation(
    conversation_id: str,
    user_id: Annotated[str, Query(min_length=1, max_length=100)] = "demo-user",
) -> dict:
    deleted = await memory_service.delete_conversation(user_id, conversation_id)
    return {"deleted": deleted, "conversation_id": conversation_id}


@router.get("/chat/history")
async def chat_history(
    user_id: Annotated[str, Query(min_length=1, max_length=100)] = "demo-user",
    conversation_id: Annotated[str, Query(min_length=1, max_length=100)] = "demo-conversation",
    recover_latest: bool = False,
) -> dict:
    if recover_latest:
        return await memory_service.restore_latest_history(user_id, conversation_id)
    return await memory_service.load_history(user_id, conversation_id)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    urgent = await _urgent_without_storage(request)
    if urgent is not None:
        return urgent
    require_ready()
    return await agent.respond(request)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    urgent = await _urgent_without_storage(request)
    if urgent is not None:
        return StreamingResponse(
            _urgent_stream_events(urgent),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                     "X-Accel-Buffering": "no"},
        )
    require_ready()
    async def event_stream():
        async for event in agent.respond_stream(request):
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
