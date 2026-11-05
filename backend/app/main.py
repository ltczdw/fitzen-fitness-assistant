from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.wiki import router as wiki_router
from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.dashboard import router as dashboard_router
from app.api.nutrition import router as nutrition_router
from app.api.plans import router as plans_router
from app.api.profile import router as profile_router
from app.api.recovery import router as recovery_router
from app.api.workouts import router as workouts_router
from app.api.workout_sessions_api import router as workout_sessions_router
from app.core.config import settings, warn_unknown_env_keys
from app.db.session import init_db
from app.db import workout_session_models  # noqa: F401 - register normalized session tables before create_all
from app.services.memory.redis_memory import memory_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    warn_unknown_env_keys()
    database_ready = init_db()

    if not database_ready:
        raise StorageUnavailableError("Database initialization failed; service startup refused")
    await memory_service.initialize()
    await memory_service.start()
    try:
        yield
    finally:
        from app.services.evaluation_jobs import runner
        from app.services.rag.async_http import close
        try:
            await runner.close()
            await memory_service.close()
        finally:await close()


app = FastAPI(title="Fitness Assistant API", version="0.4.0", lifespan=lifespan)
from app.services.llm.middleware import ModelBudgetMiddleware
app.add_middleware(ModelBudgetMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(health_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(workouts_router, prefix="/api")
app.include_router(workout_sessions_router, prefix="/api")
app.include_router(profile_router, prefix="/api")
app.include_router(nutrition_router, prefix="/api")
app.include_router(plans_router, prefix="/api")
app.include_router(recovery_router, prefix="/api")

app.include_router(wiki_router,prefix="/api")
from app.api.quality import router as quality_router
app.include_router(quality_router, prefix="/api")
from app.api.evaluation import router as evaluation_router
app.include_router(evaluation_router,prefix='/api')


from fastapi.responses import JSONResponse
from app.db.storage import StorageUnavailableError, CommitUnknownError, IdempotencyConflict

from app.services.agent.execution_state import RequestIdConflict

async def _failure_response(exc):
    from app.services.agent.failure_details import transport_failure
    result = transport_failure(exc)
    status = result.pop("http_status")
    result["model_used"] = getattr(exc, "_request_model_used", None)
    return JSONResponse(status_code=status, content=result)


@app.exception_handler(IdempotencyConflict)
async def idempotency_conflict_handler(request, exc):
    return await _failure_response(exc)


@app.exception_handler(RequestIdConflict)
async def request_id_conflict_handler(request, exc):
    return await _failure_response(exc)


@app.exception_handler(StorageUnavailableError)
async def storage_unavailable_handler(request, exc):
    return await _failure_response(exc)


@app.exception_handler(CommitUnknownError)
async def commit_unknown_handler(request, exc):
    return await _failure_response(exc)


@app.exception_handler(Exception)
async def execution_failure_handler(request, exc):
    return await _failure_response(exc)
