"""Explicit schema bootstrap; imports all application model registrations."""
from app.main import app  # noqa: F401
from app.core.config import warn_unknown_env_keys
from app.db.session import init_db

if __name__ == "__main__":
    warn_unknown_env_keys()
    if not init_db():
        raise SystemExit("Database initialization failed; see logged exception")
    print("Database schema is ready (no personal JSON imported).")
