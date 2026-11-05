from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.agent.intent import IntentResult, _run_async, classify_intent
from app.services.agent.task_models import INTENT_TO_TASK_TYPE, TASK_TYPE_TO_RAG_ROUTE, TaskType
from app.services.llm.client import LLMClient


@dataclass(frozen=True)
class QueryRoute:
    """RAG 的正式路由结果，供召回、审计和后续多路索引使用。"""

    intent: str
    domain: str
    task: str
    risk_level: str
    confidence: float
    rewrite_terms: tuple[str, ...]


def route_query(query: str, intent_result: IntentResult | None = None) -> QueryRoute:
    result = intent_result if intent_result is not None else classify_intent(query)
    task_type = INTENT_TO_TASK_TYPE.get(result.name, TaskType.TRAINING_QA)
    domain, task, terms = TASK_TYPE_TO_RAG_ROUTE[task_type]
    return QueryRoute(result.name, domain, task, result.safety_level, result.confidence, terms)


def rewrite_query_template(query: str, route: QueryRoute | None = None) -> str:
    """将口语问题转成稳定的检索表达，同时保留原问题用于关键词命中。"""
    normalized = " ".join((query or "").strip().split())
    if not normalized:
        return ""
    route = route or route_query(normalized)
    return f"{normalized} {' '.join(route.rewrite_terms)} 官方指南 证据 安全边界"


_QUERY_LLM_CLIENT = LLMClient()
_QUERY_REWRITE_PROMPT = """你是健身知识库的检索词改写器，不是聊天助手。
把用户口语问题改写成适合运动科学、营养学和康复知识库检索的规范化学术检索词。
保留所有关键实体和限定条件，包括动作名、身体部位、重量、组数、次数、时间、症状和目标。
可以补充必要的同义术语和安全边界，但不要回答问题、不要添加原文没有的事实。
只输出一行改写后的查询字符串，不要 Markdown、解释、标签或引号。"""


def rewrite_query_llm(query: str, route: QueryRoute | None = None) -> str | None:
    """Rewrite a query with DeepSeek and silently return ``None`` on failure."""
    normalized = " ".join((query or "").strip().split())
    if not normalized or route is None or not _QUERY_LLM_CLIENT.configured():
        return None
    route_context = (
        f"意图={route.intent}; 领域={route.domain}; 任务={route.task}; "
        f"风险={route.risk_level}; 路由词={', '.join(route.rewrite_terms)}"
    )
    try:
        raw = _run_async(lambda: _QUERY_LLM_CLIENT.complete(
            [
                {"role": "system", "content": _QUERY_REWRITE_PROMPT},
                {"role": "user", "content": f"路由上下文：{route_context}\n原始问题：{normalized[:4000]}"},
            ],
            temperature=0.1,
        ))
    except Exception:
        return None
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:text)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    text = " ".join(text.split())
    if not text or len(text) > 800 or text.startswith(("{", "[")):
        return None
    if re.match(r"^(?:改写后的查询|查询词|query)\s*[:：]", text, re.IGNORECASE):
        return None
    # Numeric constraints are safety-relevant retrieval entities and must survive.
    for number in re.findall(r"\d+(?:\.\d+)?", normalized):
        if number not in text:
            return None
    return text


def rewrite_query(query: str, route: QueryRoute | None = None) -> str:
    """Prefer LLM query rewriting and fall back to the deterministic template."""
    normalized = " ".join((query or "").strip().split())
    if not normalized:
        return ""
    route = route or route_query(normalized)
    from app.core.config import settings
    rewritten = rewrite_query_llm(normalized, route) if settings.rag_query_rewrite_enabled else None
    return rewritten or rewrite_query_template(normalized, route)

