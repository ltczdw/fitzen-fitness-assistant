"""P2-3: a repeated import must fill missing/outdated vectors, not skip them."""
from __future__ import annotations

import json

from sqlalchemy import func, select, text

from app.core.config import settings
from app.db.models import KnowledgeChunkModel, KnowledgeDocumentModel
from app.services.rag import ingestion as ingestion_module
from app.services.rag.ingestion import (
    KnowledgeIngestionService,
    SourceDocument,
    backfill_missing_vectors,
    backfill_vectors,
    plan_vector_backfill,
)

VECTOR = [1.0] + [0.0] * 1023


def _document() -> SourceDocument:
    return SourceDocument(
        title="训练指南",
        content="# 热身\n小重量开始。\n\n# 负荷\n逐步增加负荷。",
        source="manual",
        metadata={"url": "https://example.org/guide"},
    )


def _counts(factory):
    with factory() as session:
        return (
            session.scalar(select(func.count()).select_from(KnowledgeDocumentModel)),
            session.scalar(select(func.count()).select_from(KnowledgeChunkModel)),
        )


def _chunks(factory):
    with factory() as session:
        return list(session.scalars(select(KnowledgeChunkModel).order_by(KnowledgeChunkModel.id)))


def test_failed_embedding_then_reimport_backfills_only_missing_vectors(compact_db, monkeypatch):
    service = KnowledgeIngestionService()

    # First import: content and lexical chunks are stored, vectors are not.
    with compact_db() as session, session.begin():
        first = service.ingest_documents(session, [_document()])
    assert first.documents_added == 1 and first.chunks_added >= 1
    assert all(chunk.semantic_embedding is None for chunk in _chunks(compact_db))
    assert all(json.loads(chunk.metadata_json)["chunker_version"] for chunk in _chunks(compact_db))

    calls: list[list[str]] = []

    def embed(texts):
        calls.append(list(texts))
        return [list(VECTOR) for _ in texts]

    monkeypatch.setattr(ingestion_module.embedding_client, "embed", embed)
    documents_before, chunks_before = _counts(compact_db)
    with compact_db() as session, session.begin():
        second = service.ingest_documents(session, [_document()])
    documents_after, chunks_after = _counts(compact_db)

    assert second.skipped_exact == 1
    assert second.vectors_backfilled == chunks_before
    assert second.vectors_pending == 0
    assert second.documents_added == 0 and second.chunks_added == 0
    assert (documents_after, chunks_after) == (documents_before, chunks_before)
    assert len(calls) == 1  # one batch, only for the missing vectors
    assert all(chunk.semantic_embedding is not None for chunk in _chunks(compact_db))
    assert all(chunk.embedding_version == settings.retrieval_embedding_index_version
               for chunk in _chunks(compact_db))

    # A third import finds every vector ready and must not call the model.
    calls.clear()
    with compact_db() as session, session.begin():
        third = service.ingest_documents(session, [_document()])
    assert third.vectors_backfilled == 0 and third.vectors_pending == 0
    assert calls == []


def test_outdated_embedding_version_is_rebuilt(compact_db, monkeypatch):
    service = KnowledgeIngestionService()
    with compact_db() as session, session.begin():
        service.ingest_documents(session, [_document()])
    with compact_db() as session, session.begin():
        for chunk in session.scalars(select(KnowledgeChunkModel)):
            chunk.semantic_embedding = list(VECTOR)
            chunk.embedding_version = "qwen-old-version"

    monkeypatch.setattr(ingestion_module.embedding_client, "embed",
                        lambda texts: [list(VECTOR) for _ in texts])
    with compact_db() as session:
        pending = plan_vector_backfill(session)
    assert len(pending) == len(_chunks(compact_db))

    result = backfill_missing_vectors(factory=compact_db)
    assert result.backfilled == len(pending) and result.pending == 0
    assert all(chunk.embedding_version == settings.retrieval_embedding_index_version
               for chunk in _chunks(compact_db))


def test_source_change_during_embedding_is_never_overwritten(compact_db, monkeypatch):
    service = KnowledgeIngestionService()
    with compact_db() as session, session.begin():
        service.ingest_documents(session, [_document()])
    target_id = _chunks(compact_db)[0].id
    original_content = _chunks(compact_db)[0].content

    def embed(texts):
        # A concurrent import replaces the chunk while the vectors are computed.
        with compact_db() as session, session.begin():
            row = session.get(KnowledgeChunkModel, target_id)
            row.content = original_content + "（已更新）"
            row.content_hash = "changed-hash"
        return [list(VECTOR) for _ in texts]

    monkeypatch.setattr(ingestion_module.embedding_client, "embed", embed)
    with compact_db() as session:
        result = backfill_vectors(session)
        session.commit()
    assert result.skipped_source_changed == 1
    assert result.backfilled == result.candidates - 1
    with compact_db() as session:
        changed = session.get(KnowledgeChunkModel, target_id)
        assert changed.semantic_embedding is None
        assert changed.content.endswith("（已更新）")


def test_invalid_or_missing_vectors_are_rejected_without_writes(compact_db, monkeypatch):
    service = KnowledgeIngestionService()
    with compact_db() as session, session.begin():
        service.ingest_documents(session, [_document()])

    monkeypatch.setattr(ingestion_module.embedding_client, "embed",
                        lambda texts: [[1.0, 0.0, 0.0] for _ in texts])
    with compact_db() as session:
        result = backfill_vectors(session)
    assert result.failed == result.candidates
    assert all(chunk.semantic_embedding is None for chunk in _chunks(compact_db))

    monkeypatch.setattr(ingestion_module.embedding_client, "embed", lambda texts: [])
    with compact_db() as session:
        result = backfill_vectors(session)
    assert result.failed == result.candidates
    assert all(chunk.semantic_embedding is None for chunk in _chunks(compact_db))


def test_plan_excludes_healthy_vectors(compact_db):
    service = KnowledgeIngestionService()
    with compact_db() as session, session.begin():
        service.ingest_documents(session, [_document()])
    with compact_db() as session, session.begin():
        for chunk in session.scalars(select(KnowledgeChunkModel)):
            chunk.semantic_embedding = list(VECTOR)
            chunk.embedding_version = settings.retrieval_embedding_index_version
    with compact_db() as session:
        assert plan_vector_backfill(session) == []
        result = backfill_vectors(session)
    assert result.candidates == 0 and result.backfilled == 0
