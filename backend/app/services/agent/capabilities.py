"""Read-only feature readiness; request gates remain authoritative."""
from sqlalchemy import inspect, select, text

from app.core.config import settings
from app.db import session as db


def probe_chat_record_storage():
    from app.db.execution_ledger_schema import verify_execution_ledger_schema, ExecutionLedgerSchemaError
    from app.db.model_budget_models import ModelQuota, ModelCallLedger, ModelResultCache
    if db.engine is None or db.SessionLocal is None:
        return False, "database_unavailable"
    try:
        with db.engine.connect() as connection, connection.begin():
            if connection.dialect.name == "postgresql":
                connection.execute(text("SET TRANSACTION READ ONLY"))
                connection.execute(text("SET LOCAL lock_timeout='1000ms'"))
                connection.execute(text("SET LOCAL statement_timeout='2000ms'"))
            verify_execution_ledger_schema(connection)
            inspector = inspect(connection)
            for model in (ModelQuota, ModelCallLedger, ModelResultCache):
                if not inspector.has_table(model.__tablename__):
                    return False, "model_budget_storage_unavailable"
                connection.execute(select(model).limit(0))
        return True, "available"
    except ExecutionLedgerSchemaError:
        return False, "execution_ledger_unavailable"
    except Exception:
        return False, "write_storage_unavailable"


def chat_record_capability(*, llm_configured):
    if not settings.react_write_enabled:
        return {"available": False, "reason": "react_write_disabled"}
    if settings.memory_read_backend != "compact_v1":
        return {"available": False, "reason": "non_compact_memory_backend"}
    if not db.DB_READY:
        return {"available": False, "reason": "database_unavailable"}
    if not llm_configured:
        return {"available": False, "reason": "llm_not_configured"}
    available, reason = probe_chat_record_storage()
    return {"available": available, "reason": reason}
