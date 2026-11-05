from dataclasses import dataclass, field, replace
import hashlib
import copy
import logging
from threading import Lock
import json
import math
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import load_only

from app.core.config import settings
from app.db import session as db_session
from app.db.models import KnowledgeChunkModel, KnowledgeDocumentModel, KnowledgeIndexVersionModel
from app.services.rag.bm25 import BM25Index
from app.services.rag.fusion import reciprocal_rank_fusion
from app.services.rag.ingestion import IngestionResult, KnowledgeIngestionService, SourceDocument
from app.services.rag.official_documents import OFFICIAL_DOCUMENTS
from app.services.rag.query import QueryRoute, rewrite_query, route_query
from app.services.agent.intent import IntentResult
from app.services.rag.reranker import LocalReranker


from app.services.rag.retrieval import RetrievedChunk, retrieval_service


def _tokens(text: str) -> list[str]:
    normalized = text.lower()
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
    bigrams = [normalized[index : index + 2] for index in range(len(normalized) - 1) if "\u4e00" <= normalized[index] <= "\u9fff" and "\u4e00" <= normalized[index + 1] <= "\u9fff"]
    return list(dict.fromkeys(words + bigrams))


def _embedding(text: str, dimensions: int = 256) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] % 2 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=False))


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnowledgeIndexSnapshot:
    key: tuple
    index: BM25Index
    chunks: tuple[RetrievedChunk, ...]


class HybridRAGPipeline:
    """BM25 + pgvector 混合召回，并支持父子文档入库和相似度去重。"""

    def __init__(self) -> None:
        self.embedding_model = settings.embedding_model
        self.documents = [
            RetrievedChunk(
                title=document.title,
                content=document.content,
                score=0.5,
                source=document.source,
                metadata=document.metadata,
                chunk_id=f"official:{document.source}",
            )
            for document in OFFICIAL_DOCUMENTS
        ]
        self.bm25 = BM25Index([f"{doc.title} {doc.content}" for doc in self.documents])
        self.ingestion = KnowledgeIngestionService()
        self.reranker = LocalReranker(settings.rerank_model)
        self.corpus = "knowledge"
        self._index_lock = Lock()
        self._index_snapshot: KnowledgeIndexSnapshot | None = None

    def search(self, query: str, top_k: int = 5, *, intent_result=None):
        if not query.strip():
            return []
        route = route_query(query, intent_result=intent_result)
        prepared = rewrite_query(query, route)
        if not settings.rag_query_rewrite_enabled: prepared = query
        return retrieval_service.search(prepared, self, final_k=top_k)

    def lexical(self, query, k):
        return self._database_bm25_search(query, k) or [
            replace(self.documents[i], score=score, chunk_id=f"seed:{i}")
            for i, score in self.bm25.rank(query, k) if score > 0]

    async def asearch(self,query,top_k=5,*,intent_result=None,strategy='hybrid_rerank',skip_rewrite=False):
        if not query.strip():return []
        # Legacy rewriting may itself call a model. Isolate that compatibility
        # path; embedding/reranking below use genuine async HTTP.
        prepared=query
        if not skip_rewrite:
            from app.services.rag.blocking import run_retrieval
            prepared=await run_retrieval(lambda:rewrite_query(query,route_query(query,intent_result)))
            if not settings.rag_query_rewrite_enabled:prepared=query
        return await retrieval_service.asearch(prepared,self,final_k=top_k,strategy=strategy)

    def vector(self, vector, k):
        return self._database_vector_search(vector, k)

    def resolve(self, chunks):
        return chunks

    def ingest_documents(self, documents: list[SourceDocument]) -> IngestionResult:
        if not db_session.DB_READY or db_session.SessionLocal is None:
            return IngestionResult()
        with db_session.SessionLocal() as session:
            result = self.ingestion.ingest_documents(session, documents)
            session.commit()
            return result

    def _database_vector_search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        if not db_session.DB_READY or db_session.SessionLocal is None or not hasattr(KnowledgeChunkModel, "embedding"):
            return []
        try:
            with db_session.SessionLocal() as session:
                statement = select(KnowledgeChunkModel, KnowledgeChunkModel.semantic_embedding.cosine_distance(query_vector).label("distance")).where(KnowledgeChunkModel.semantic_embedding.is_not(None), KnowledgeChunkModel.embedding_version == settings.retrieval_embedding_index_version, KnowledgeChunkModel.source.notin_(self._LEGACY_SOURCES)).order_by(KnowledgeChunkModel.semantic_embedding.cosine_distance(query_vector)).limit(top_k)
                rows = session.execute(statement).all()
                return [self._to_retrieved(session, row, index, max(-1.0, min(1.0, 1.0 - float(distance)))) for index, (row, distance) in enumerate(rows)]
        except Exception:
            return []

    def _database_bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not db_session.DB_READY or db_session.SessionLocal is None:
            return []
        try:
            snapshot = self._knowledge_index()
            return [replace(snapshot.chunks[i], score=score,
                            metadata=copy.deepcopy(snapshot.chunks[i].metadata))
                    for i, score in snapshot.index.rank(query, top_k) if score > 0]
        except Exception:
            # Never serve stale cached knowledge when the revision check fails.
            logger.warning("knowledge_bm25_unavailable", exc_info=True)
            return []

    def _knowledge_index(self) -> KnowledgeIndexSnapshot:
        # Single-flight rebuild: readers only see a complete detached snapshot.
        # A warm query reads one scalar revision, not every knowledge row.
        with self._index_lock, db_session.SessionLocal() as session:
            for _ in range(3):
                revision = session.scalar(select(KnowledgeIndexVersionModel.revision).where(
                    KnowledgeIndexVersionModel.corpus == "knowledge"))
                if revision is None:
                    raise RuntimeError("knowledge_version_missing; run init_db")
                key = (session.get_bind(), revision, settings.rag_parent_context_chars,
                       tuple(sorted(self._LEGACY_SOURCES)))
                if self._index_snapshot is not None and self._index_snapshot.key == key:
                    return self._index_snapshot
                rows = session.scalars(select(KnowledgeChunkModel).options(load_only(
                    KnowledgeChunkModel.id, KnowledgeChunkModel.title,
                    KnowledgeChunkModel.content, KnowledgeChunkModel.source,
                    KnowledgeChunkModel.document_id, KnowledgeChunkModel.metadata_json,
                )).where(KnowledgeChunkModel.source.notin_(self._LEGACY_SOURCES))
                    .order_by(KnowledgeChunkModel.id)).all()
                parents = {p.document_id: p for p in session.scalars(
                    select(KnowledgeDocumentModel).options(load_only(
                        KnowledgeDocumentModel.document_id, KnowledgeDocumentModel.content,
                        KnowledgeDocumentModel.metadata_json,
                    )).where(KnowledgeDocumentModel.document_id.in_(
                        {r.document_id for r in rows if r.document_id}))) }
                chunks = tuple(self._to_retrieved(session, row, i, parents=parents)
                               for i, row in enumerate(rows))
                index = BM25Index([f"{row.title} {chunks[i].metadata.get('section_path', '')} {row.content}" for i, row in enumerate(rows)])
                # A writer may commit between our SELECTs under READ COMMITTED.
                # Retry instead of publishing mixed versions of chunks/parents.
                latest = session.scalar(select(KnowledgeIndexVersionModel.revision).where(
                    KnowledgeIndexVersionModel.corpus == "knowledge"))
                if latest == revision:
                    snapshot = KnowledgeIndexSnapshot(key, index, chunks)
                    self._index_snapshot = snapshot
                    return snapshot
                session.rollback()
                session.expunge_all()
            raise RuntimeError("knowledge_changed_during_index_build")

    def _to_retrieved(self, session, row: KnowledgeChunkModel, index: int,
                      relevance_score: float | None = None, *, parents=None) -> RetrievedChunk:
        content = row.content
        metadata: dict[str, Any] = {}
        if row.metadata_json:
            try:
                parsed = json.loads(row.metadata_json)
                if isinstance(parsed, dict):
                    metadata = parsed
            except (TypeError, json.JSONDecodeError):
                metadata = {}
        if row.document_id:
            parent = (parents.get(row.document_id) if parents is not None else
                      session.scalar(select(KnowledgeDocumentModel).where(KnowledgeDocumentModel.document_id == row.document_id)))
            if parent and parent.content != row.content:
                parent_context = parent.content[: max(0, settings.rag_parent_context_chars)]
                content = f"{row.content}\n\n父文档摘要：{parent_context}"
            if parent and parent.metadata_json and not metadata:
                try:
                    parsed = json.loads(parent.metadata_json)
                    if isinstance(parsed, dict):
                        metadata = parsed
                except (TypeError, json.JSONDecodeError):
                    metadata = {}
        metadata = {**metadata}
        return RetrievedChunk(row.title, content, relevance_score if relevance_score is not None else 1.0 / (index + 1), row.source, metadata, str(row.id))

    def _seed_database(self, session) -> None:
        self.ingestion.ingest_documents(session, OFFICIAL_DOCUMENTS)

    @staticmethod
    def _is_relevant(query: str, chunk: RetrievedChunk) -> bool:
        query_terms = {term for term in _tokens(query) if len(term) >= 2}
        content_terms = set(_tokens(f"{chunk.title} {chunk.metadata.get('section_path', '')} {chunk.content}"))
        lexical_match = bool(query_terms & content_terms)
        vector_score = float(chunk.metadata.get("vector_score", 0.0) or 0.0)
        return lexical_match or vector_score >= 0.62
    _LEGACY_SOURCES = {"training-basics", "safety", "nutrition", "recovery", "programming"}

# Stable reference for injected synchronous compatibility adapters.
SYNC_SEARCH=HybridRAGPipeline.search


