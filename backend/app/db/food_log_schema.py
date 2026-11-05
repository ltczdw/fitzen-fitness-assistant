from __future__ import annotations

from app.db import session as db_session


def ensure_food_log_idempotency_schema() -> bool:
    """Compatibility guard; schema migration runs once during startup."""
    return bool(db_session.DB_READY and db_session.engine is not None and db_session.SessionLocal is not None)
