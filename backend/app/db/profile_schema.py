"""Nullable profile body-fat upgrade; no connections or writes on import."""
from sqlalchemy import inspect, text


def ensure_profile_schema(connection):
    inspector = inspect(connection)
    if not inspector.has_table("user_profiles"):
        return
    columns = {column["name"] for column in inspector.get_columns("user_profiles")}
    if "body_fat_pct" not in columns:
        connection.execute(text("ALTER TABLE user_profiles ADD COLUMN body_fat_pct DOUBLE PRECISION"))
