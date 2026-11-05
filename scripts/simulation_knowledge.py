"""Public knowledge fixtures for isolated simulations; never use the live DB."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "backend/seed/authority_coverage_snapshot.json"
CATALOG = ROOT / "backend/seed/authority_coverage.json"


class KnowledgeFixtureError(ValueError):
    pass


def checked_factory(factory, *, expected_database):
    """Check the destination before issuing any SQL, then verify PG identity."""
    from sqlalchemy import text
    with factory() as session:
        engine = session.get_bind()
        if engine.dialect.name == "sqlite":
            actual = Path(engine.url.database or "").resolve()
            expected = Path(expected_database).resolve()
            if actual != expected or not actual.is_relative_to((ROOT / "evidence").resolve()):
                raise KnowledgeFixtureError("simulation_database_not_isolated")
        elif engine.dialect.name == "postgresql":
            if (not re.fullmatch(r"fitzen_test_[0-9a-f]{32}", str(expected_database))
                    or engine.url.database != expected_database):
                raise KnowledgeFixtureError("simulation_database_not_isolated")
            if session.scalar(text("SELECT current_database()")) != expected_database:
                raise KnowledgeFixtureError("simulation_database_identity_mismatch")
        else:
            raise KnowledgeFixtureError("simulation_database_backend_unsupported")
    return engine


def prepare_public_knowledge(factory, *, expected_database, snapshot=SNAPSHOT, catalog=CATALOG):
    """Import existing source-checked public articles without any model call.

    Vector readiness is separate from successful import. This function never
    substitutes seed data or invented vectors when a source or provider fails.
    """
    from sqlalchemy import func, select
    from app.db.knowledge_version import install_knowledge_versioning
    from app.db.models import KnowledgeChunkModel as Chunk, KnowledgeDocumentModel as Document
    from app.services.rag.ingestion import KnowledgeIngestionService, plan_vector_backfill
    from app.services.rag.knowledge_catalog import load_catalog, load_snapshot
    engine = checked_factory(factory, expected_database=expected_database)
    entries = {e["id"]: e for e in load_catalog(catalog)}
    documents = load_snapshot(snapshot)
    if not documents or any(d.metadata["coverage_id"] not in entries
            or d.metadata["requested_url"] != entries[d.metadata["coverage_id"]]["url"]
            or d.metadata["topics"] != [entries[d.metadata["coverage_id"]]["topic"]]
            for d in documents):
        raise KnowledgeFixtureError("simulation_public_snapshot_catalog_mismatch")
    with engine.begin() as conn:
        install_knowledge_versioning(conn)
    with factory() as session, session.begin():
        imported = KnowledgeIngestionService().ingest_documents(session, documents, embed=False)
    with factory() as session:
        document_count = session.scalar(select(func.count()).select_from(Document))
        chunk_count = session.scalar(select(func.count()).select_from(Chunk))
        pending = len(plan_vector_backfill(session))
    return {"profile": "public_snapshot", "snapshot_sha256": hashlib.sha256(Path(snapshot).read_bytes()).hexdigest(),
            "catalog_sha256": hashlib.sha256(Path(catalog).read_bytes()).hexdigest(),
            "source_documents": len(documents), "documents": document_count, "chunks": chunk_count,
            "vectors_pending": pending, "vectors_ready": chunk_count > 0 and pending == 0,
            "ingestion": asdict(imported), "provider_calls": 0}


def require_live_knowledge(factory, *, expected_database):
    """Fail before spending chat tokens if the promised retrieval profile is absent."""
    from sqlalchemy import func, select
    from app.core.config import settings
    from app.db.models import KnowledgeChunkModel as Chunk
    from app.services.rag.ingestion import plan_vector_backfill
    engine = checked_factory(factory, expected_database=expected_database)
    with factory() as session:
        count = session.scalar(select(func.count()).select_from(Chunk))
        pending = len(plan_vector_backfill(session))
    if not count:
        raise KnowledgeFixtureError("simulation_knowledge_missing")
    if pending:
        raise KnowledgeFixtureError("simulation_knowledge_vectors_pending")
    # The actual vector adapter uses pgvector SQL; populated SQLite vectors do
    # not make this a working hybrid retrieval environment.
    if engine.dialect.name != "postgresql":
        raise KnowledgeFixtureError("simulation_vector_backend_unsupported")
    embedding_ready = bool(settings.retrieval_embedding_base_url and settings.retrieval_embedding_api_key)
    base, key = settings.rerank_base_url, settings.rerank_api_key
    # Match ModelReranker exactly: a partial explicit configuration must fail,
    # rather than silently borrowing embedding credentials.
    if not base and not key and settings.rerank_provider == "dashscope" and settings.rerank_legacy_embedding_config:
        base, key = settings.retrieval_embedding_base_url, settings.retrieval_embedding_api_key
    rerank_ready = bool(settings.rerank_enabled and base and key)
    if not embedding_ready or not rerank_ready:
        raise KnowledgeFixtureError("simulation_retrieval_provider_missing")
    if settings.retrieval_embedding_dimensions != 1024:
        raise KnowledgeFixtureError("simulation_embedding_dimensions_unsupported")
    for url in (settings.retrieval_embedding_base_url, base):
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in {"http", "https"} and bool(parsed.netloc) and not parsed.username and not parsed.password
        except ValueError:
            valid = False
        if not valid:
            raise KnowledgeFixtureError("simulation_retrieval_provider_url_invalid")
    return {"knowledge_chunks": count, "vectors_pending": 0, "embedding_configured": True, "rerank_configured": True}


def prepare_simulation_retrieval(factory, *, expected_database, profile="public_lexical"):
    """Shared simulation setup, called before any agent/model request.

    Explicit profiles keep a lexical-only run from being reported as a normal
    hybrid retrieval acceptance. Importing public text alone cannot promise an
    answerable plan; the existing quality gate still makes that decision.
    """
    if profile not in {"public_lexical", "public_hybrid"}:
        raise KnowledgeFixtureError("simulation_retrieval_profile_unknown")
    result = prepare_public_knowledge(factory, expected_database=expected_database)
    result.update(retrieval_profile=profile, hybrid_ready=False)
    if profile == "public_hybrid":
        result["preflight"] = require_live_knowledge(factory, expected_database=expected_database)
        result["hybrid_ready"] = True
    return result
