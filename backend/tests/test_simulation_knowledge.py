"""Isolated public fixture ingestion, provenance and explicit readiness failures."""
import json
from pathlib import Path
import sys

import pytest
from sqlalchemy import select, text

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from simulation_knowledge import (KnowledgeFixtureError, checked_factory, prepare_public_knowledge,
    require_live_knowledge, prepare_simulation_retrieval)
from app.db.models import KnowledgeChunkModel
from app.services.rag import ingestion
from evals.v2.safety import deny_network


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    def reject(*a, **k):
        raise AssertionError("public fixture preparation must not call an embedding provider")
    monkeypatch.setattr(ingestion.embedding_client, "embed", reject)
    with deny_network():
        yield


def destination(factory):
    with factory() as s:
        return s.get_bind().url.database


def test_full_snapshot_import_is_source_checked_idempotent_and_not_vector_ready(compact_db):
    path = destination(compact_db)
    first = prepare_public_knowledge(compact_db, expected_database=path)
    assert first["source_documents"] == first["documents"] == 86 and first["chunks"] == 646
    assert first["vectors_pending"] == 646 and not first["vectors_ready"] and first["provider_calls"] == 0
    second = prepare_public_knowledge(compact_db, expected_database=path)
    assert second["ingestion"]["documents_added"] == second["ingestion"]["chunks_added"] == 0
    assert second["documents"] == first["documents"] and second["chunks"] == first["chunks"]
    with compact_db() as session:
        for table in ("workout_records", "user_food_log", "user_facts", "memory_turns"):
            assert session.execute(text("SELECT COUNT(*) FROM " + table)).scalar() == 0
        for c in session.scalars(select(KnowledgeChunkModel)):
            assert c.source.startswith("official-coverage:") and c.semantic_embedding is None
            meta = json.loads(c.metadata_json)
            assert meta["source_url"].startswith("https://") and meta["content_sha256"]


@pytest.mark.parametrize("state", ["empty", "imported"])
def test_preflight_rejects_empty_corpus_and_pending_vectors_before_model_calls(compact_db, state):
    path = destination(compact_db)
    if state == "imported":
        prepare_public_knowledge(compact_db, expected_database=path)
    expected = "vectors_pending" if state == "imported" else "knowledge_missing"
    with pytest.raises(KnowledgeFixtureError, match=expected):
        require_live_knowledge(compact_db, expected_database=path)


def test_destination_must_match_checked_isolated_database(compact_db, tmp_path):
    with pytest.raises(KnowledgeFixtureError, match="not_isolated"):
        checked_factory(compact_db, expected_database=tmp_path / "another.sqlite")


def test_snapshot_tampering_leaves_empty_knowledge(compact_db, tmp_path):
    from simulation_knowledge import SNAPSHOT
    source = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    source["documents"][0]["content"] += "tampered"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot_source_or_hash_invalid"):
        prepare_public_knowledge(compact_db, expected_database=destination(compact_db), snapshot=bad)
    with compact_db() as s:
        assert s.execute(text("SELECT COUNT(*) FROM knowledge_chunks")).scalar() == 0


def test_actual_runner_setup_imports_public_corpus_without_claiming_hybrid(compact_db):
    from run_ai_user_simulation import prepare_simulation_retrieval as runner_setup
    result = runner_setup(compact_db, expected_database=destination(compact_db))
    assert result["documents"] == 86 and result["chunks"] == 646
    assert result["retrieval_profile"] == "public_lexical"
    assert result["hybrid_ready"] is False and result["provider_calls"] == 0


def test_hybrid_preflight_rejects_missing_vectors_and_does_not_spend(compact_db):
    with pytest.raises(KnowledgeFixtureError, match="vectors_pending"):
        prepare_simulation_retrieval(compact_db, expected_database=destination(compact_db), profile="public_hybrid")


def test_sqlite_filled_vectors_do_not_claim_pgvector_readiness(compact_db):
    prepare_public_knowledge(compact_db, expected_database=destination(compact_db))
    with compact_db() as session, session.begin():
        candidates = ingestion.plan_vector_backfill(session)
        ingestion.apply_vector_backfill(session, [(c, [1.] + [0.] * 1023) for c in candidates])
    with pytest.raises(KnowledgeFixtureError, match="vector_backend_unsupported"):
        require_live_knowledge(compact_db, expected_database=destination(compact_db))


def test_unknown_profile_rejected_before_import(compact_db):
    with pytest.raises(KnowledgeFixtureError, match="profile_unknown"):
        prepare_simulation_retrieval(compact_db, expected_database=destination(compact_db), profile="typo")
    with compact_db() as s:
        assert s.execute(text("SELECT COUNT(*) FROM knowledge_chunks")).scalar() == 0


@pytest.mark.parametrize("dimensions,url,error", [
    (768, "https://rerank.example.test", "dimensions_unsupported"),
    (1024, "file:///fixture", "provider_url_invalid"),
    (1024, "https://user:synthetic@rerank.example.test", "provider_url_invalid"),
    (1024, "https:///missing-host", "provider_url_invalid"),
    (1024, "https://rerank.example.test", None),
])
def test_provider_preconditions_match_runtime_before_any_network(compact_db, monkeypatch, dimensions, url, error):
    # Unit test only: PG identity/SQL is covered independently. Keep SQL here
    # real and isolated SQLite while simulating the completed PG boundary.
    from types import SimpleNamespace
    import simulation_knowledge as fixture
    from app.core.config import settings
    monkeypatch.setattr(fixture, "checked_factory", lambda *a, **kw: SimpleNamespace(dialect=SimpleNamespace(name="postgresql")))
    for name, value in {"retrieval_embedding_base_url": "https://embedding.example.test/v1",
            "retrieval_embedding_api_key": "synthetic", "retrieval_embedding_dimensions": dimensions,
            "rerank_enabled": True, "rerank_base_url": url, "rerank_api_key": "synthetic"}.items():
        monkeypatch.setattr(settings, name, value)
    with compact_db() as s, s.begin():
        s.add(KnowledgeChunkModel(title="fixture", content="每周训练", source="fixture",
            semantic_embedding=[1.] + [0.] * 1023, embedding_version=settings.retrieval_embedding_index_version))
    if error:
        with pytest.raises(KnowledgeFixtureError, match=error):
            require_live_knowledge(compact_db, expected_database=destination(compact_db))
    else:
        assert require_live_knowledge(compact_db, expected_database=destination(compact_db))["vectors_pending"] == 0
