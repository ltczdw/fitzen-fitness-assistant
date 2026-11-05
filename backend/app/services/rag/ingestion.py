from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import html
import json
import math
import re
import unicodedata

from sqlalchemy import select, delete, or_
from app.services.rag.chunking import split_document, CHUNKER_VERSION

from app.core.config import settings
from app.db.models import KnowledgeChunkModel, KnowledgeDocumentModel
from app.services.rag.embedding_client import embedding_client, EmbeddingUnavailable


@dataclass(frozen=True)
class SourceDocument:
    title: str
    content: str
    source: str = "manual"
    metadata: dict = field(default_factory=dict)


@dataclass
class IngestionResult:
    documents_added: int = 0
    chunks_added: int = 0
    skipped_exact: int = 0
    skipped_similar: int = 0
    # Vector readiness is tracked separately from content/chunker readiness:
    # a document can be fully stored with a chunker version while its vectors
    # are still missing or built by an older embedding version.
    vectors_backfilled: int = 0
    vectors_pending: int = 0


@dataclass
class VectorBackfillResult:
    documents: int = 0
    candidates: int = 0
    backfilled: int = 0
    skipped_ready: int = 0
    skipped_source_changed: int = 0
    failed: int = 0

    @property
    def pending(self) -> int:
        return max(0, self.candidates - self.backfilled - self.skipped_ready)

    def as_dict(self) -> dict[str, int]:
        return {
            "documents": self.documents,
            "candidates": self.candidates,
            "backfilled": self.backfilled,
            "skipped_ready": self.skipped_ready,
            "skipped_source_changed": self.skipped_source_changed,
            "failed": self.failed,
            "pending": self.pending,
        }


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = html.unescape(value)
    value = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", value)
    # 只去除真正的 HTML 标签，保留“< 8”“> 3”等健身/营养比较符号。
    value = re.sub(r"</?[A-Za-z][^>]*>", " ", value)
    value = value.replace("\u200b", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in value.split("\n")]
    return "\n".join(line for line in lines if line)


def content_hash(title: str, content: str) -> str:
    normalized = f"{clean_text(title)}\n{clean_text(content)}".strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _token_list(text: str) -> list[str]:
    normalized = text.lower()
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
    bigrams = [normalized[index : index + 2] for index in range(len(normalized) - 1) if "\u4e00" <= normalized[index] <= "\u9fff" and "\u4e00" <= normalized[index + 1] <= "\u9fff"]
    return list(dict.fromkeys(words + bigrams))


def _embedding(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    for token in _token_list(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] % 2 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    return [piece.content for piece in split_document(clean_text(text),
        chunk_size=settings.rag_chunk_size if chunk_size is None else chunk_size,
        overlap=settings.rag_chunk_overlap if overlap is None else overlap)]


def _chunk_metadata(chunk) -> dict:
    try:
        metadata = json.loads(chunk.metadata_json or "{}")
    except (TypeError, ValueError):
        metadata = {}
    return metadata if isinstance(metadata, dict) else {}


def _chunk_embedding_text(chunk) -> str:
    section_path = str(_chunk_metadata(chunk).get("section_path") or "").strip()
    return f"{section_path}\n{chunk.content}" if section_path else chunk.content


def plan_vector_backfill(session, document_ids: list[str] | None = None) -> list[dict]:
    """Read-only snapshot of chunks whose vectors are missing or outdated.

    ``chunker_version=sections-v2`` in metadata says nothing about vectors, so
    readiness is decided by ``semantic_embedding`` plus ``embedding_version``.
    """
    version = settings.retrieval_embedding_index_version
    query = select(KnowledgeChunkModel).where(
        or_(KnowledgeChunkModel.semantic_embedding.is_(None),
            KnowledgeChunkModel.embedding_version.is_(None),
            KnowledgeChunkModel.embedding_version != version)
    )
    if document_ids:
        query = query.where(KnowledgeChunkModel.document_id.in_(list(document_ids)))
    candidates = []
    for chunk in session.scalars(query.order_by(KnowledgeChunkModel.id)):
        candidates.append({
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "content_hash": chunk.content_hash,
            "content": chunk.content,
            "title": chunk.title,
            "text": _chunk_embedding_text(chunk),
        })
    return candidates


def apply_vector_backfill(session, prepared: list[tuple[dict, list[float]]]) -> VectorBackfillResult:
    """Write only the prepared vectors after re-validating the source chunk.

    Never inserts parents/chunks and never recomputes a healthy vector. A chunk
    changed while the vectors were computed is skipped, not overwritten.
    """
    result = VectorBackfillResult()
    version = settings.retrieval_embedding_index_version
    result.documents = len({item[0].get("document_id") for item in prepared if item[0].get("document_id")})
    for candidate, vector in prepared:
        result.candidates += 1
        current = session.execute(
            select(KnowledgeChunkModel).where(KnowledgeChunkModel.id == candidate["chunk_id"])
            .execution_options(populate_existing=True)
        ).scalar_one_or_none()
        if current is None:
            result.failed += 1
            continue
        if ((current.content_hash or "") != (candidate["content_hash"] or "")
                or current.content != candidate["content"]):
            result.skipped_source_changed += 1
            continue
        if current.semantic_embedding is not None and current.embedding_version == version:
            result.skipped_ready += 1
            continue
        current.semantic_embedding = vector
        current.embedding_version = version
        result.backfilled += 1
    return result


def _embed_backfill_candidates(candidates: list[dict]) -> list[list[float]]:
    vectors = embedding_client.embed([candidate["text"] for candidate in candidates])
    if len(vectors) != len(candidates):
        raise EmbeddingUnavailable("missing_vectors")
    dimensions = settings.retrieval_embedding_dimensions
    for vector in vectors:
        if (len(vector) != dimensions
                or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                           and math.isfinite(float(value)) for value in vector)
                or not any(float(value) for value in vector)):
            raise EmbeddingUnavailable("invalid_vector")
    return vectors


def backfill_vectors(session, document_ids: list[str] | None = None) -> VectorBackfillResult:
    """Fill missing/outdated vectors using the caller's session.

    The embedding request is issued before any update in this call; on failure
    the lexical index stays usable and the chunks remain queued for retry.
    """
    candidates = plan_vector_backfill(session, document_ids)
    result = VectorBackfillResult(candidates=len(candidates))
    if not candidates:
        return result
    try:
        vectors = _embed_backfill_candidates(candidates)
    except EmbeddingUnavailable:
        result.failed = len(candidates)
        return result
    prepared = list(zip(candidates, vectors, strict=True))
    return apply_vector_backfill(session, prepared)


def backfill_missing_vectors(factory=None, document_ids: list[str] | None = None) -> VectorBackfillResult:
    """Retryable standalone entry point: read, embed outside the write
    transaction, then commit after re-validating every chunk."""
    from app.db import session as db

    session_factory = factory or db.SessionLocal
    if session_factory is None:
        raise RuntimeError("database_unavailable")
    with session_factory() as session:
        candidates = plan_vector_backfill(session, document_ids)
    result = VectorBackfillResult(documents=len({c["document_id"] for c in candidates if c["document_id"]}),
                                  candidates=len(candidates))
    if not candidates:
        return result
    try:
        vectors = _embed_backfill_candidates(candidates)
    except EmbeddingUnavailable:
        result.failed = len(candidates)
        return result
    with session_factory() as session, session.begin():
        applied = apply_vector_backfill(session, list(zip(candidates, vectors, strict=True)))
    return applied


class KnowledgeIngestionService:
    def __init__(self, similarity_threshold: float | None = None) -> None:
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else settings.rag_dedupe_similarity_threshold

    def ingest_documents(self, session, documents: list[SourceDocument], *, embed: bool = True) -> IngestionResult:
        result = IngestionResult()
        for source_document in documents:
            title = clean_text(source_document.title)
            parent_content = clean_text(source_document.content)
            if not title or not parent_content:
                continue
            parent_hash = content_hash(title, parent_content)
            existing_parent = session.scalar(select(KnowledgeDocumentModel).where(KnowledgeDocumentModel.content_hash == parent_hash))
            if existing_parent:
                result.skipped_exact += 1
                # The content is already stored; only fill the vectors that were
                # missing when the first import ran without an embedding service.
                if existing_parent.document_id and embed:
                    backfilled = backfill_vectors(session, [existing_parent.document_id])
                    result.vectors_backfilled += backfilled.backfilled
                    result.vectors_pending += backfilled.pending
                continue

            pieces = split_document(parent_content, title=title, chunk_size=settings.rag_chunk_size,
                                    overlap=settings.rag_chunk_overlap)
            chunks = [piece.content for piece in pieces]
            accepted: list[tuple[int, str, str, list[float]]] = []
            pending_vectors: list[list[float]] = []
            for chunk_index, chunk in enumerate(chunks):
                chunk_hash = content_hash(title + "\n" + pieces[chunk_index].section_path, chunk)
                exact_chunk = session.scalar(select(KnowledgeChunkModel).where(KnowledgeChunkModel.content_hash == chunk_hash))
                if exact_chunk:
                    result.skipped_exact += 1
                    continue
                try:
                    vector = embedding_client.embed([f"{pieces[chunk_index].section_path}\n{chunk}"])[0] if embed else None
                except EmbeddingUnavailable:
                    vector = None
                nearest = self._nearest_similarity(session, vector) if vector else -1.0
                pending_nearest = max((_cosine(vector, other) for other in pending_vectors), default=-1.0) if vector else -1.0
                if settings.rag_enable_semantic_dedupe and max(nearest, pending_nearest) >= self.similarity_threshold:
                    result.skipped_similar += 1
                    continue
                accepted.append((chunk_index, chunk, chunk_hash, vector))
                if vector:
                    pending_vectors.append(vector)

            if not accepted:
                continue

            document_id = f"doc-{parent_hash[:24]}"
            metadata_json = json.dumps({**(source_document.metadata or {}), "chunker_version": CHUNKER_VERSION}, ensure_ascii=False)
            session.add(KnowledgeDocumentModel(document_id=document_id, title=title, content=parent_content, source=source_document.source, content_hash=parent_hash, metadata_json=metadata_json))
            session.flush()
            chunk_count = len(accepted)
            for chunk_index, chunk, chunk_hash, vector in accepted:
                session.add(
                    KnowledgeChunkModel(
                        title=title,
                        content=chunk,
                        source=source_document.source,
                        document_id=document_id,
                        parent_id=document_id,
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                        content_hash=chunk_hash,
                        metadata_json=json.dumps({**(source_document.metadata or {}),
                            "section_path": pieces[chunk_index].section_path,
                            "content_type": pieces[chunk_index].content_type,
                            "chunker_version": CHUNKER_VERSION}, ensure_ascii=False),
                        semantic_embedding=vector,
                        embedding_version=settings.retrieval_embedding_index_version if vector else None,
                        embedding_json=None,
                    )
                )
                result.chunks_added += 1
            result.vectors_pending += sum(1 for _, _, _, vector in accepted if vector is None)
            result.documents_added += 1
        return result

    def _nearest_similarity(self, session, vector: list[float]) -> float:
        nearest = session.execute(
            select(
                KnowledgeChunkModel,
                KnowledgeChunkModel.semantic_embedding.cosine_distance(vector).label("distance"),
            )
            .where(KnowledgeChunkModel.semantic_embedding.is_not(None), KnowledgeChunkModel.embedding_version == settings.retrieval_embedding_index_version)
            .order_by(KnowledgeChunkModel.semantic_embedding.cosine_distance(vector))
            .limit(1)
        ).first()
        if not nearest:
            return -1.0
        return 1.0 - float(nearest[1])
