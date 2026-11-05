import asyncio
import logging
from sqlalchemy import text
from fastapi import APIRouter, Response

from app.core.config import settings
from app.db import session as db_session
from app.services.memory.redis_memory import memory_service
from app.services.context.budget import context_budget
from app.services.observability import tracing_status


router = APIRouter(tags=["system"])


logger = logging.getLogger(__name__)


def _database_probe() -> bool:
    if db_session.SessionLocal is None:
        return False
    try:
        with db_session.SessionLocal() as session:
            session.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("health_database_probe_failed")
        return False


@router.get("/health")
async def health(response: Response) -> dict[str, object]:
    # Health is read-only: it never initializes schema, imports data, or calls LLM inference.
    from app.api.chat import agent
    try:
        connected = await asyncio.wait_for(asyncio.to_thread(_database_probe), timeout=4)
    except TimeoutError:
        connected = False
        logger.warning("health_database_probe_timeout")
    try:
        redis_connected = await asyncio.wait_for(memory_service.initialize(), timeout=2)
    except Exception:
        redis_connected = False
        logger.exception("health_redis_probe_failed")
    llm = agent.graph.llm.status()
    from app.services.context.request_tokens import native_plain_tokenizer
    request_counter_mode = "deepseek-v41-estimate" if native_plain_tokenizer() is not None else "conservative-utf8"
    llm_status = "unconfigured" if not llm["configured"] else ("circuit_open" if llm["circuit"]["state"] == "open" else "configured_unverified")
    ready = bool(db_session.DB_READY and connected)
    from app.services.agent.capabilities import chat_record_capability
    try:
        chat_record = await asyncio.wait_for(asyncio.to_thread(
            chat_record_capability, llm_configured=bool(llm["configured"])), timeout=4)
    except Exception:
        chat_record = {"available": False, "reason": "write_storage_unavailable"}
    response.status_code = 200 if ready else 503
    return {
        "status": "ok" if ready and (redis_connected or not settings.redis_enabled) and llm_status == "configured_unverified" else "degraded",
        "ready": ready,
        "capabilities": {"chat_record": chat_record},
        "service": "fitness-assistant",
        "database": "postgresql+pgvector" if ready else "unavailable",
        "components": {
            "db": {"initialized": db_session.DB_READY, "connected": connected, "ready": ready},
            "redis": {"status": "disabled" if not settings.redis_enabled else ("connected" if redis_connected else "unavailable")},
            "llm": {"status": llm_status, "configured": llm["configured"], "circuit": llm["circuit"], "probe": "configuration_only"},
        },
        "embedding": settings.retrieval_embedding_model,
        "embedding_dimensions": settings.retrieval_embedding_dimensions,
        "embedding_index_version": settings.retrieval_embedding_index_version,
        "embedding_configured": bool(settings.retrieval_embedding_base_url and settings.retrieval_embedding_api_key),
        "intent_router": "planner_llm_with_plan_tasks_rule_fallback",
        "reranker": settings.rerank_model,
        "reranker_enabled": settings.rerank_enabled,
        "wiki_enabled": settings.wiki_enabled,
        "memory": memory_service.status(),
        "langsmith": tracing_status(),
        "context_budget": {"context_window_tokens": context_budget.window, "input_budget_tokens": context_budget.input_budget, "max_output_tokens": context_budget.output, "counter_mode": context_budget.counter.mode,
                           "request_counter_mode": request_counter_mode},
    }
