from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.services.rag.pipeline import RetrievedChunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]: ...


class LocalReranker:
    """无外部服务依赖的轻量重排器；模型名用于明确当前实际后端。"""

    def __init__(self, model_name: str = "local-reranker") -> None:
        self.requested_model = model_name
        self.active_model = "local-reranker"

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int = 5) -> list[RetrievedChunk]:
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        safety_query = any(term in query_lower for term in ("痛", "肿", "麻木", "扭伤", "拉伤", "变形", "无法活动", "无法承重", "受伤", "康复"))
        ranked = []
        for chunk in chunks:
            text = f"{chunk.title} {chunk.content}".lower()
            phrase_bonus = 0.01 if query_lower and query_lower in text else 0.0
            term_bonus = 0.001 * sum(term in text for term in query_terms if term)
            safety_document = any(term in text for term in ("疼痛", "扭伤", "拉伤", "急救", "停止训练", "危险信号", "就医"))
            # A safety document is a hard priority for a pain/injury query.
            # A small numeric bonus is not stable when a generic chunk has a
            # higher vector score, so keep the priority as a separate sort key.
            safety_priority = 1 if safety_query and safety_document else 0
            ranked.append((safety_priority, chunk.score + phrase_bonus + term_bonus, chunk))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [chunk for _, _, chunk in ranked[:top_n]]


PassthroughReranker = LocalReranker
