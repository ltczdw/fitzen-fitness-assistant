"""Public reference ingestion: bounded requests, provenance and atomic writes."""
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select

from app.db.models import KnowledgeChunkModel, KnowledgeDocumentModel
from app.services.rag.crawler import _PageParser
from app.services.rag.coverage_check import representative_entries
from app.services.rag.ingestion import KnowledgeIngestionService, SourceDocument, backfill_missing_vectors
from app.services.rag.knowledge_catalog import crawl_catalog, load_catalog, load_snapshot, safe_url

ENTRY = {"id": "sample", "topic": "training", "title_zh": "训练基础", "url": "https://www.nhs.uk/article", "questions": ["每周训练几次"]}
BODY = "# Strength\n" + "Progress gradually. " * 30


def test_parser_prefers_article_and_preserves_sections():
    parser = _PageParser()
    parser.feed('<title>Official title</title><p>Cookie preferences</p><main><h1>Strength</h1>'
                '<nav><div>Ignore menu</div></nav><p>Start <b>gradually</b>.</p>'
                '<h2>Safety</h2><p>Stop if painful.</p></main><footer>Footer</footer>')
    body = parser.content()
    assert '# Strength' in body and '## Safety' in body and 'gradually' in body
    assert all(text not in body for text in ('Cookie', 'menu', 'Footer', 'Official title'))


@pytest.mark.parametrize('url', ['http://www.nhs.uk/a', 'https://www.nhs.uk.evil.test/a',
    'https://user:password@www.nhs.uk/a', 'file:///a'])
def test_source_urls_reject_untrusted_or_credentialed_addresses(url):
    assert not safe_url(url)


@pytest.mark.parametrize('target', ['/private', 'https://example.org/article'])
def test_redirect_checked_before_request(target):
    calls = []
    def transport(request):
        calls.append(str(request.url))
        if request.url.path == '/robots.txt':
            return httpx.Response(200, text='User-agent: *\nDisallow: /private')
        return httpx.Response(302, headers={'location': target})
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        result = crawl_catalog([ENTRY], client=client, delay=0)
    assert len(result['failures']) == 1 and result['documents'] == []
    assert calls == ['https://www.nhs.uk/robots.txt', ENTRY['url']]


def test_robots_forbidden_does_not_fetch_article():
    calls = []
    def transport(request):
        calls.append(request.url.path)
        return httpx.Response(403)
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        result = crawl_catalog([ENTRY], client=client, delay=0)
    assert calls == ['/robots.txt'] and result['failures'][0]['http_status'] == 403


def test_crawl_snapshot_provenance_and_tamper_detection(tmp_path):
    def transport(request):
        if request.url.path == '/robots.txt':
            return httpx.Response(404)
        return httpx.Response(200, headers={'content-type': 'text/html'},
            text='<title>Strength</title><main><h1>Strength</h1><p>' + BODY + '</p></main>')
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        snapshot = crawl_catalog([ENTRY], client=client, delay=0)
    path = tmp_path / 'snapshot.json'
    path.write_text(json.dumps(snapshot), encoding='utf-8')
    doc = load_snapshot(path)[0]
    assert doc.metadata['coverage_id'] == 'sample'
    assert doc.metadata['content_sha256'] == hashlib.sha256(doc.content.encode()).hexdigest()
    assert doc.metadata['license'] == 'Open Government Licence v3.0'
    snapshot['documents'][0]['content'] += 'tampered'
    path.write_text(json.dumps(snapshot), encoding='utf-8')
    with pytest.raises(ValueError, match='snapshot_source_or_hash_invalid'):
        load_snapshot(path)


def test_catalog_and_snapshot_cover_project_topics():
    root = Path(__file__).resolve().parents[1] / 'seed'
    entries = load_catalog(root / 'authority_coverage.json')
    documents = load_snapshot(root / 'authority_coverage_snapshot.json')
    topics = {'training', 'nutrition', 'weight', 'recovery', 'injury', 'special-populations', 'supplements', 'safety'}
    assert {topic for doc in documents for topic in doc.metadata['topics']} == topics
    assert {entry['topic'] for entry in representative_entries(entries, 8)} == topics
    first = representative_entries(entries, 24)
    rest = representative_entries(entries, 24, offset=24)
    assert len(first + rest) == len(entries)
    assert len({entry['id'] for entry in first + rest}) == len(entries)
    stretch = next(entry for entry in entries if entry['id'] == 'stretching')
    assert stretch['expected_sources']['1'] == ['stretching', 'injury-treatment']
    assert stretch['question_task_types']['1'] == 'safety_triage'
    by_id = {entry['id']: entry for entry in entries}
    for doc in documents:
        assert doc.metadata['requested_url'] == by_id[doc.metadata['coverage_id']]['url']
        assert len(doc.content) >= (30 if doc.metadata['language'] == 'zh' else 200)


def test_translation_cannot_outlive_source_revision(tmp_path):
    root = Path(__file__).resolve().parents[1] / 'seed'
    snapshot = json.loads((root / 'authority_coverage_snapshot.json').read_text(encoding='utf-8'))
    snapshot['documents'][0]['translation']['source_sha256'] = 'outdated'
    path = tmp_path / 'snapshot.json'
    path.write_text(json.dumps(snapshot), encoding='utf-8')
    with pytest.raises(ValueError, match='translation_source_invalid'):
        load_snapshot(path)


def test_deferred_ingestion_is_idempotent_and_vectors_only_targeted(compact_db, monkeypatch):
    from app.services.rag import ingestion
    def unexpected(*args, **kwargs):
        raise AssertionError('deferred import must not call provider')
    monkeypatch.setattr(ingestion.embedding_client, 'embed', unexpected)
    service = KnowledgeIngestionService()
    doc = SourceDocument('力量训练', BODY, 'official-coverage:sample', {})
    other = SourceDocument('其他来源', BODY + 'Different', 'manual', {})
    with compact_db() as session, session.begin():
        first = service.ingest_documents(session, [doc, other], embed=False)
    assert first.documents_added == 2 and first.vectors_pending > 0
    with compact_db() as session, session.begin():
        repeated = service.ingest_documents(session, [doc], embed=False)
        assert repeated.documents_added == repeated.chunks_added == repeated.vectors_backfilled == 0
        target = session.scalar(select(KnowledgeDocumentModel).where(KnowledgeDocumentModel.source == doc.source)).document_id
    monkeypatch.setattr(ingestion.embedding_client, 'embed', lambda texts: [[1.] + [0.] * 1023 for _ in texts])
    filled = backfill_missing_vectors(factory=compact_db, document_ids=[target])
    assert filled.backfilled > 0 and filled.pending == 0
    with compact_db() as session:
        assert all(chunk.semantic_embedding is None for chunk in session.scalars(
            select(KnowledgeChunkModel).where(KnowledgeChunkModel.document_id != target)))


def test_import_rollback_does_not_leave_partial_documents(compact_db):
    with pytest.raises(RuntimeError, match='rollback'):
        with compact_db() as session, session.begin():
            KnowledgeIngestionService().ingest_documents(session, [SourceDocument('Test', BODY, 'test', {})], embed=False)
            raise RuntimeError('rollback')
    with compact_db() as session:
        assert session.scalar(select(func.count()).select_from(KnowledgeDocumentModel)) == 0
        assert session.scalar(select(func.count()).select_from(KnowledgeChunkModel)) == 0


def test_budget_failure_stops_coverage_batch_instead_of_reporting_corpus_failures(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from app.services.rag import coverage_check as check
    from app.services.rag.retrieval import RetrievedChunk, retrieval_traces
    calls = []
    class Pipeline:
        def lexical(self, query, count):
            return []
        async def asearch(self, query, **kwargs):
            calls.append(query)
            retrieval_traces.get().append({'degraded_reason': 'model_budget_exhausted:daily'})
            return [RetrievedChunk('训练', '每周训练逐步增加', 1, metadata={'coverage_id': 'a'})]
    monkeypatch.setattr(check, 'HybridRAGPipeline', Pipeline)
    monkeypatch.setattr(check, 'model_job', lambda *a: nullcontext())
    monkeypatch.setattr(check, 'remaining_daily_calls', lambda: 100)
    entries = [{**ENTRY, 'id': id} for id in ['a', 'b']]
    documents = [SourceDocument('训练', BODY, 'official', {'coverage_id': id, 'content_sha256': id}) for id in ['a', 'b']]
    result = check.verify_coverage(entries, documents, SimpleNamespace(verify_count=2))
    assert len(calls) == 1 and result['live_total'] == 1 and result['live_planned'] == 2
    assert result['stopped_reason'] == 'model_budget_exhausted:daily'
    assert result['live_passed'] == 0
    calls.clear()
    monkeypatch.setattr(check, 'remaining_daily_calls', lambda: 23)
    result = check.verify_coverage(entries, documents, SimpleNamespace(verify_count=2))
    assert calls == [] and result['live_total'] == 0 and result['live_planned'] == 2
    assert result['stopped_reason'] == 'insufficient_daily_budget_with_chat_reserve'


def test_batched_vectors_reuse_existing_cache_and_restore_context(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from app.core.config import settings
    from app.services.rag import coverage_check as check
    from app.services.rag.retrieval import RetrievedChunk, embedding_cache
    seen = []
    class Pipeline:
        def lexical(self, query, count):return []
        async def asearch(self, query, **kwargs):
            seen.append(embedding_cache.get()[(query, settings.retrieval_embedding_model, settings.retrieval_embedding_index_version)])
            return [RetrievedChunk('训练', '每周训练逐步增加', 1, metadata={
                'coverage_id': 'sample', 'rerank_score': .99, 'rerank_status': 'model'})]
    async def fill(queries):
        return {(query, settings.retrieval_embedding_model, settings.retrieval_embedding_index_version): [1., 0.] for query in queries}
    monkeypatch.setattr(check, 'HybridRAGPipeline', Pipeline)
    monkeypatch.setattr(check, 'model_job', lambda *a: nullcontext())
    monkeypatch.setattr(check, 'remaining_daily_calls', lambda: 100)
    monkeypatch.setattr(check, 'prefill_query_embeddings', fill)
    doc = SourceDocument('训练', BODY, 'official', {'coverage_id': 'sample', 'content_sha256': 'sample'})
    before = embedding_cache.get()
    result = check.verify_coverage([ENTRY], [doc], SimpleNamespace(verify_count=1, batch_query_embeddings=True))
    assert result['live_passed'] == 1 and seen == [[1., 0.]]
    assert embedding_cache.get() is before


def test_alternate_source_requires_explicit_mapping_and_keeps_safety_threshold(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from app.services.rag import coverage_check as check
    from app.services.rag.retrieval import RetrievedChunk
    source = ['injury-treatment']
    calls = []
    class Pipeline:
        def lexical(self, query, count):return []
        async def asearch(self, query, **kwargs):
            calls.append(query)
            return [RetrievedChunk('损伤处理', '引发损伤疼痛应停止活动并就医', .95, metadata={
                'coverage_id': source[0], 'rerank_score': .95, 'rerank_status': 'model'})]
    monkeypatch.setattr(check, 'HybridRAGPipeline', Pipeline)
    monkeypatch.setattr(check, 'model_job', lambda *a: nullcontext())
    monkeypatch.setattr(check, 'remaining_daily_calls', lambda: 100)
    entries = [{**ENTRY, 'expected_sources': {'0': ['sample', 'injury-treatment']},
                'question_task_types': {'0': 'safety_triage'}}, {**ENTRY, 'id': 'other'}]
    docs = [SourceDocument('参考', BODY, 'official', {'coverage_id': id, 'content_sha256': id}) for id in ['sample', 'other']]
    args = SimpleNamespace(verify_count=2, verify_ids=['sample'])
    result = check.verify_coverage(entries, docs, args)
    assert len(calls) == 1 and result['live_passed'] == 1
    assert result['live'][0]['task_type'] == 'safety_triage' and result['live'][0]['assessment']['threshold'] == .58
    source[0] = 'unrelated'
    result = check.verify_coverage(entries, docs, args)
    assert result['live_passed'] == 0 and not result['live'][0]['expected_in_top5']
