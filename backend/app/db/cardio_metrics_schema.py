"""Nullable cardio metrics upgrade; callers own the transaction.

No data reinterpretation: old weight/reps values remain unchanged. This module
does not connect to or migrate any database at import time.
"""
from sqlalchemy import inspect, text


def ensure_cardio_metrics_schema(connection):
    inspector = inspect(connection)
    if not inspector.has_table("workout_session_sets"):
        return
    columns = {column["name"] for column in inspector.get_columns("workout_session_sets")}
    for name in ("distance_km", "duration_minutes"):
        if name not in columns:
            connection.execute(text(f"ALTER TABLE workout_session_sets ADD COLUMN {name} DOUBLE PRECISION"))
