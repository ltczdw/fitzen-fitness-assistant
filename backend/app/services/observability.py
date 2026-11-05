"""Opt-in LangSmith spans containing counts/status only, never health text."""
from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from typing import Any

from langsmith import Client, trace, tracing_context

from app.core.config import settings

logger = logging.getLogger(__name__)
_client: Client | None = None


def tracing_status() -> str:
    if not settings.langsmith_tracing:
        return "disabled"
    return "configured" if settings.langsmith_api_key else "blocked-missing-api-key"


class SafeSpan:
    def __init__(self, run=None) -> None:
        self.run = run

    def end(self, outputs: dict[str, Any] | None = None, error: str | None = None) -> None:
        if self.run is not None:
            try:
                self.run.end(outputs=outputs, error=error)
            except Exception:
                logger.warning("LangSmith span update failed; business request continues")


@contextmanager
def trace_span(name: str, *, inputs: dict[str, Any], metadata: dict[str, Any] | None = None,
               run_type: str = "chain", tags: list[str] | None = None):
    global _client
    if tracing_status() != "configured":
        yield SafeSpan()
        return
    stack = ExitStack()
    try:
        if _client is None:
            _client = Client(api_url=settings.langsmith_endpoint, api_key=settings.langsmith_api_key)
        stack.enter_context(tracing_context(enabled=True, client=_client, project_name=settings.langsmith_project))
        run = stack.enter_context(trace(name, run_type=run_type, inputs=inputs, metadata=metadata,
                                        tags=["fitness-assistant", settings.app_env, *(tags or [])]))
    except Exception:
        try:
            stack.close()
        except Exception:
            pass
        logger.warning("LangSmith setup failed; business request continues")
        yield SafeSpan()
        return
    span = SafeSpan(run)
    try:
        yield span
    except BaseException as exc:
        # Do not upload exception messages, which can contain URLs or PHI.
        span.end(error=type(exc).__name__)
        raise
    finally:
        try:
            stack.close()
        except Exception:
            logger.warning("LangSmith upload failed; business request continues")
