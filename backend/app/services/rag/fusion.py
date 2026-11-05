from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.rag.pipeline import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for index, chunk in enumerate(ranked, start=1):
            key = getattr(chunk, "chunk_id", None) or chunk.title + "\n" + chunk.content
            chunks[key] = chunk
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + index)
    return [replace(chunk, score=scores[key]) for key, chunk in sorted(chunks.items(), key=lambda item: scores[item[0]], reverse=True)]
