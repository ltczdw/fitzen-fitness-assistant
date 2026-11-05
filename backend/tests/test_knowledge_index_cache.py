import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from sqlalchemy import event, select, text
from quality_isolation import checked_engine as create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db import session as db
from app.db.models import KnowledgeChunkModel as Chunk
from app.db.models import KnowledgeDocumentModel as Document
from app.db.models import KnowledgeIndexVersionModel as Version
from app.db.knowledge_version import install_knowledge_versioning
from app.services.rag.pipeline import HybridRAGPipeline


def create_schema(engine):
    with engine.begin() as connection:
        for model in (Document, Chunk, Version):
            model.__table__.create(connection)
        install_knowledge_versioning(connection)


@pytest.fixture
def knowledge_db(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///" + str(tmp_path / "knowledge.db"))
    create_schema(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db, "SessionLocal", factory)
    monkeypatch.setattr(db, "DB_READY", True)
    yield engine, factory
    engine.dispose()


def add_chunk(factory, content="squat strength", document_id=None):
    with factory.begin() as session:
        row = Chunk(title="guide", content=content, source="fixture",
                    document_id=document_id, metadata_json='{"tags": ["original"]}')
        session.add(row)
        session.flush()
        return row.id


def revision(factory):
    with factory() as session:
        return session.scalar(select(Version.revision).where(Version.corpus == "knowledge"))


def count_chunk_loads(engine):
    calls = []
    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT") and "FROM knowledge_chunks" in statement:
            calls.append(statement)
    event.listen(engine, "before_cursor_execute", record)
    return calls


def test_warm_queries_reuse_index_without_loading_vectors_or_all_chunks(knowledge_db):
    engine, factory = knowledge_db
    add_chunk(factory)
    calls = count_chunk_loads(engine)
    rag = HybridRAGPipeline()
    first = rag._database_bm25_search("squat", 3)
    snapshot = rag._index_snapshot
    first[0].metadata["tags"].append("request-specific")
    second = rag._database_bm25_search("squat", 3)
    assert rag._index_snapshot is snapshot
    assert len(calls) == 1
    assert "semantic_embedding" not in calls[0]
    assert second[0].metadata == {"tags": ["original"]}
    assert first[0].score == second[0].score > 0


def test_insert_update_delete_refresh_all_pipeline_instances(knowledge_db):
    _, factory = knowledge_db
    first, second = HybridRAGPipeline(), HybridRAGPipeline()
    for rag in (first, second):
        assert rag._database_bm25_search("squat", 3) == []
    chunk_id = add_chunk(factory)
    for rag in (first, second):
        assert rag._database_bm25_search("squat", 3)[0].chunk_id == str(chunk_id)
    with factory.begin() as session:
        session.execute(text("UPDATE knowledge_chunks SET content='cycling cardio' WHERE id=:id"), {"id": chunk_id})
    for rag in (first, second):
        assert rag._database_bm25_search("squat", 3) == []
        assert rag._database_bm25_search("cycling", 3)[0].content == "cycling cardio"
    with factory.begin() as session:
        session.execute(text("DELETE FROM knowledge_chunks WHERE id=:id"), {"id": chunk_id})
    for rag in (first, second):
        assert rag._database_bm25_search("cycling", 3) == []


def test_rollback_does_not_invalidate_committed_snapshot(knowledge_db):
    _, factory = knowledge_db
    chunk_id = add_chunk(factory)
    rag = HybridRAGPipeline()
    rag._database_bm25_search("squat", 3)
    old = rag._index_snapshot
    before = revision(factory)
    with factory() as session:
        session.execute(text("UPDATE knowledge_chunks SET content='uncommitted' WHERE id=:id"), {"id": chunk_id})
        session.rollback()
    assert revision(factory) == before
    assert rag._database_bm25_search("squat", 3)[0].content == "squat strength"
    assert rag._index_snapshot is old


def test_parent_update_and_context_setting_refresh_snapshot(knowledge_db, monkeypatch):
    _, factory = knowledge_db
    with factory.begin() as session:
        session.add(Document(document_id="parent", title="guide", content="old parent",
                             source="fixture", content_hash="parent-hash"))
    add_chunk(factory, document_id="parent")
    rag = HybridRAGPipeline()
    assert "old parent" in rag._database_bm25_search("squat", 3)[0].content
    with factory.begin() as session:
        session.execute(text("UPDATE knowledge_documents SET content='new parent' WHERE document_id='parent'"))
    assert "new parent" in rag._database_bm25_search("squat", 3)[0].content
    monkeypatch.setattr(settings, "rag_parent_context_chars", 3)
    assert rag._database_bm25_search("squat", 3)[0].content.endswith("new")


def test_empty_index_is_cached_and_concurrent_cold_reads_build_once(knowledge_db):
    engine, factory = knowledge_db
    calls = count_chunk_loads(engine)
    rag = HybridRAGPipeline()
    for _ in range(2):
        assert rag._database_bm25_search("squat", 3) == []
    assert len(calls) == 1
    add_chunk(factory)
    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(lambda _: rag._database_bm25_search("squat", 3), range(12)))
    assert all(len(hits) == 1 for hits in results)
    assert len(calls) == 2


def test_failed_revision_read_does_not_serve_stale_cache(knowledge_db, monkeypatch):
    _, factory = knowledge_db
    add_chunk(factory)
    rag = HybridRAGPipeline()
    assert rag._database_bm25_search("squat", 3)
    def unavailable():
        raise RuntimeError("database unavailable")
    monkeypatch.setattr(db, "SessionLocal", unavailable)
    assert rag._database_bm25_search("squat", 3) == []


def test_concurrent_commit_during_build_retries_before_publishing(knowledge_db, monkeypatch):
    from app.services.rag import pipeline
    _, factory = knowledge_db
    chunk_id = add_chunk(factory)
    rag = HybridRAGPipeline()
    original = pipeline.BM25Index
    builds = []
    def build(documents):
        builds.append(documents)
        if len(builds) == 1:
            with factory.begin() as session:
                session.execute(text("UPDATE knowledge_chunks SET content='cycling cardio' WHERE id=:id"), {"id": chunk_id})
        return original(documents)
    monkeypatch.setattr(pipeline, "BM25Index", build)
    assert rag._database_bm25_search("cycling", 3)[0].content == "cycling cardio"
    assert len(builds) == 2


def test_install_is_idempotent_without_resetting_revision(knowledge_db):
    engine, factory = knowledge_db
    add_chunk(factory)
    before = revision(factory)
    with engine.begin() as connection:
        install_knowledge_versioning(connection)
    assert revision(factory) == before
    add_chunk(factory, "cycling")
    assert revision(factory) > before


@pytest.mark.skipif(os.getenv("FITZEN_DB_TESTS") != "1", reason="isolated PostgreSQL integration is opt-in")
def test_postgres_triggers_commit_rollback_truncate_and_cache(monkeypatch):
    if not settings.database_url.startswith("postgresql"):
        pytest.skip("PostgreSQL URL required")
    schema = "test_rag_cache_" + uuid4().hex
    admin = create_engine(settings.database_url, connect_args={"connect_timeout": 3})
    # Only this test's random schema is created and dropped, never application tables.
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(settings.database_url, connect_args={
        "options": f"-csearch_path={schema},public", "connect_timeout": 3})
    try:
        create_schema(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        monkeypatch.setattr(db, "SessionLocal", factory)
        monkeypatch.setattr(db, "DB_READY", True)
        chunk_id = add_chunk(factory)
        first, second = HybridRAGPipeline(), HybridRAGPipeline()
        assert first._database_bm25_search("squat", 3)
        assert second._database_bm25_search("squat", 3)
        before = revision(factory)
        with factory() as session:
            session.execute(text("UPDATE knowledge_chunks SET content='rollback'"))
            session.rollback()
        assert revision(factory) == before
        with factory.begin() as session:
            session.execute(text("UPDATE knowledge_chunks SET content='cycling cardio' WHERE id=:id"), {"id": chunk_id})
        assert revision(factory) > before
        for rag in (first, second):
            assert rag._database_bm25_search("squat", 3) == []
            assert rag._database_bm25_search("cycling", 3)
        with engine.begin() as connection:
            install_knowledge_versioning(connection)
            connection.execute(text("TRUNCATE knowledge_chunks"))
        assert first._database_bm25_search("cycling", 3) == []
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()
