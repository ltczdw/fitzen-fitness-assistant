import asyncio
import json
import math
import threading
import contextvars
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

from app.services.agent.workout_extractor import normalize_workout_text
from app.services.llm.client import LLMClient


@dataclass(frozen=True)
class IntentResult:
    name: str
    confidence: float
    safety_level: str = "normal"
    action: Literal["record", "question", "confirm", "answer"] = "answer"


GREETING_WORDS = ("你好", "您好", "嗨", "哈喽", "hi", "hello", "hey", "yo", "在吗", "在不在", "在么", "早上好", "下午好", "晚上好")
_GREETING_SUFFIXES = "呀啊哦呢喽！!。.~～"


def is_greeting(message: str) -> bool:
    """Return true only for a standalone greeting, never for a business request."""
    text = "".join(str(message or "").split()).strip(_GREETING_SUFFIXES)
    if not text:
        return False
    lowered = text.lower()
    return lowered in {word.lower() for word in GREETING_WORDS}


def food_action_for_text(message: str) -> str:
    return classify_intent(message).action


def classify_intent_rule(message: str) -> IntentResult:
    """Compatibility fallback: only an assessed emergency gates business intent.

    Ordinary text cannot become a business intent without model understanding.
    """
    from app.services.agent.semantics import get_understanding, safety_for
    if safety_for(message).get("level") == "urgent":
        return IntentResult("symptom_triage", 1.0, "urgent", "question")
    result = get_understanding(message)
    if result:
        from app.services.agent.task_planner import intent_from_planned_tasks
        return intent_from_planned_tasks(result.get("validated_tasks") or [], IntentResult("training_qa", 0.0, action="question"))
    return IntentResult("training_qa", 0.0, action="question")


_T = TypeVar("_T")
_LLM_CLIENT = LLMClient()
def _run_async(factory: Callable[[], Awaitable[_T]]) -> _T:
    """Run an async LLM call from the legacy synchronous routing API."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    result: list[_T] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(asyncio.run(factory()))
        except BaseException as exc:  # propagate to the caller for silent fallback
            errors.append(exc)

    context = contextvars.copy_context()
    thread = threading.Thread(target=lambda: context.run(worker), daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    if not result:
        raise RuntimeError("异步 LLM 调用未返回结果")
    return result[0]


def classify_intent_llm(message: str) -> IntentResult | None:
    """Compatibility API; reuse the same structured understanding model."""
    if not message.strip() or not _LLM_CLIENT.configured():
        return None
    from app.services.agent.semantics import understand, remember_understanding
    result = _run_async(lambda: understand(_LLM_CLIENT, message))
    if result is None:
        return None
    remember_understanding(message, result)
    return classify_intent_rule(message)


def classify_intent(message: str) -> IntentResult:
    from app.services.agent.semantics import get_understanding, safety_for
    if safety_for(message).get("level") == "urgent":
        return classify_intent_rule(message)
    if get_understanding(message) is not None:
        return classify_intent_rule(message)
    return classify_intent_llm(message) or classify_intent_rule(message)
