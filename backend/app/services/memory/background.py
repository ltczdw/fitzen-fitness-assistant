"""One opt-in gate for autonomous, potentially paid memory processing."""
from app.core.config import settings


def background_memory_enabled() -> bool:
    return bool(settings.memory_background_enabled and settings.memory_summary_enabled)
