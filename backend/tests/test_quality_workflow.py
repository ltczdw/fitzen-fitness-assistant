import asyncio
import os
from uuid import uuid4
import pytest
from sqlalchemy import select, func, text
from quality_isolation import checked_engine as create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI
import httpx

from app.db import session as db
from app.db import quality_repository as repo
from app.db.quality_models import ChatRun, FailureCase, AnswerFeedback, ReviewAudit
from app.db.memory_models import MemoryTurn
from app.db.models import KnowledgeDocumentModel as Document, KnowledgeChunkModel as Chunk, KnowledgeIndexVersionModel as Version
from app.db.knowledge_version import install_knowledge_versioning
from app.api.quality import FeedbackInput, ReviewInput, router
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.telemetry import RequestMetrics
from app.services.rag.ingestion import SourceDocument
from app.services.rag.chunking import split_document

@pytest.fixture
def quality_db(tmp_path, monkeypatch):
 from app.core.config import settings
 postgres = os.getenv('FITZEN_QUALITY_DB') == 'postgres'
 schema = 'test_quality_' + uuid4().hex
 admin = None
 if postgres:
  if not settings.database_url.startswith('postgresql'):
   pytest.skip('PostgreSQL configuration required')
  admin = create_engine(settings.database_url, connect_args={'connect_timeout': 3})
  with admin.begin() as conn:
   conn.execute(text(f'CREATE SCHEMA "{schema}"'))
  engine = create_engine(settings.database_url, connect_args={
   'connect_timeout': 3, 'options': f'-csearch_path={schema},public'})
 else:
  engine = create_engine('sqlite:///'+str(tmp_path/'quality.db'))
 try:
  with engine.begin() as conn:
   for model in (MemoryTurn, ChatRun, FailureCase, AnswerFeedback, ReviewAudit, Document, Chunk, Version):
    model.__table__.create(conn)
   install_knowledge_versioning(conn)
   conn.execute(text('CREATE UNIQUE INDEX uq_test_chunks_hash ON knowledge_chunks(content_hash) WHERE content_hash IS NOT NULL'))
  factory=sessionmaker(bind=engine,expire_on_commit=False)
  monkeypatch.setattr(db,'SessionLocal',factory);monkeypatch.setattr(db,'DB_READY',True)
  yield factory
 finally:
  engine.dispose()
  if admin is not None:
   # The schema name is generated here; application/public tables are never removed.
   assert schema.startswith('test_quality_') and schema[13:].isalnum()
   with admin.begin() as conn:
    conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
   admin.dispose()


def add_run(failure=False):
 req=ChatRequest(message='如何安排训练？',conversation_id='quality-test')
 res=ChatResponse(answer='待核实回答',intent='training_qa',run_id=str(uuid4()),message_id=req.request_id+':assistant')
 metrics=RequestMetrics()
 if failure:
  metrics.failure_reasons.add('insufficient_evidence');metrics.degraded_reasons.add('insufficient_evidence')
 repo.record_run(req,res,metrics.summary())
 return req,res

def down(res):
 return FeedbackInput(run_id=res.run_id,message_id=res.message_id,rating='down',note='没有解决')

def approval(version=1):
 return ReviewInput(decision='approve',expected_version=version,note='已人工核查',question='训练指南',answer='逐步安排负荷。',source_url='https://example.org/guide')

def prepared(payload):
 doc=SourceDocument(payload.question,payload.answer,payload.source_url,{'reviewed':True})
 pieces=split_document(doc.content,title=doc.title)
 return doc,pieces,[[1.0]+[0.0]*1023 for _ in pieces]

def counts(factory):
 with factory() as s:
  return [s.scalar(select(func.count()).select_from(m)) for m in (ChatRun,FailureCase,AnswerFeedback,ReviewAudit,Document,Chunk)]

def test_feedback_identity_idempotence_and_conflict(quality_db):
 _,res=add_run()
 with pytest.raises(repo.MissingRecord): repo.feedback(down(res).model_copy(update={'message_id':'different'}))
 first=repo.feedback(down(res));second=repo.feedback(down(res))
 assert first['feedback_id']==second['feedback_id'] and second['replayed']
 with pytest.raises(repo.ReviewConflict):repo.feedback(down(res).model_copy(update={'rating':'up'}))
 assert counts(quality_db)==[1,1,1,0,0,0]

def test_approve_atomic_audit_knowledge_revision_and_replay(quality_db):
 _,res=add_run(True);case=repo.list_cases()['items'][0];payload=approval()
 with quality_db() as s:before=s.scalar(select(Version.revision))
 result=repo.review(case['case_id'],payload,prepared(payload))
 assert result['status']=='approved'
 assert repo.review(case['case_id'],payload)['replayed']
 assert counts(quality_db)==[1,1,0,1,1,1]
 with quality_db() as s:
  assert s.scalar(select(Version.revision))>before
  chunk=s.scalar(select(Chunk));assert 'section_path' in chunk.metadata_json
 assert repo.get_case(case['case_id'])['audit'][0]['note']=='已人工核查'

@pytest.mark.parametrize('reason',['stale','vectors','duplicate'])
def test_review_failure_rolls_back_entire_transaction(quality_db,reason):
 _,res=add_run(True);case=repo.list_cases()['items'][0];payload=approval()
 data=prepared(payload)
 if reason=='stale':payload=approval(2)
 if reason=='vectors':data=(data[0],data[1],[])
 if reason=='duplicate':
  _,other=add_run(True)
  othercase=next(c for c in repo.list_cases()['items'] if c['run_id']==other.run_id)
  repo.review(othercase['case_id'],payload,prepared(payload))
 before=counts(quality_db)
 with quality_db() as s:revision=s.scalar(select(Version.revision))
 with pytest.raises((repo.ReviewConflict,ValueError)):
  repo.review(case['case_id'],payload,data)
 assert counts(quality_db)==before
 assert repo.get_case(case['case_id'])['status']=='pending'
 with quality_db() as s:assert s.scalar(select(Version.revision))==revision

def test_summary_and_delete_hide_private_content(quality_db):
 req,res=add_run(True);repo.feedback(down(res))
 case=repo.list_cases()['items'][0]
 repo.review(case['case_id'],ReviewInput(decision='resolve',expected_version=1,note='人工处理'))
 add_run()
 assert repo.summary()['degraded_rate']==0.5
 repo.scrub_conversation(req.conversation_id)
 assert repo.list_cases(None)['total']==0 and repo.summary()['requests']==0
 with quality_db() as s:
  assert all(not row.question and not row.answer for row in s.scalars(select(ChatRun)))
  assert s.scalar(select(AnswerFeedback.note))==''
  assert s.scalar(select(ReviewAudit.note))==''
 with pytest.raises(repo.MissingRecord):repo.feedback(down(res))

def test_api_failed_embedding_keeps_pending_and_validates_input(quality_db,monkeypatch):
 from app.api import quality
 from app.services.rag.embedding_client import EmbeddingUnavailable
 def unavailable(*args):raise EmbeddingUnavailable('test')
 monkeypatch.setattr(quality.embedding_client,'embed',unavailable)
 _,res=add_run(True);case=repo.list_cases()['items'][0]
 app=FastAPI();app.include_router(router,prefix='/api')
 async def run():
  async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
   response=await client.post('/api/review/cases/'+case['case_id'],json=approval().model_dump())
   assert response.status_code==502
   response=await client.post('/api/feedback',json={**down(res).model_dump(),'injected':'no'})
   assert response.status_code==422
   assert (await client.get('/api/quality')).status_code==200
   assert (await client.get('/api/observability/summary')).json()['requests']==1
 asyncio.run(run())
 assert repo.get_case(case['case_id'])['status']=='pending'
 assert counts(quality_db)==[1,1,0,0,0,0]

def test_deleted_source_does_not_resurrect_diagnostics(quality_db):
 from app.core import clock
 req=ChatRequest(message='已删除的问题',request_id='deleted-request')
 with quality_db.begin() as s:
  s.add(MemoryTurn(turn_id='deleted-turn',request_id=req.request_id,conversation_id=req.conversation_id,deleted_at=clock.utc_now()))
 res=ChatResponse(answer='迟到回答',intent='training_qa',run_id=str(uuid4()),message_id='deleted-turn:assistant')
 assert repo.record_run(req,res,RequestMetrics().summary()) is False
 assert counts(quality_db)==[0,0,0,0,0,0]

@pytest.mark.parametrize('outcome',['success','invalid_vector','source_changed','duplicate_hash'])
def test_rechunk_atomic_replacement_preserves_source_and_refreshes_index(quality_db,monkeypatch,outcome):
 from app.services.rag.rechunk import prepare_rechunk,replace_chunks
 from app.services.rag.pipeline import HybridRAGPipeline
 from app.services.rag.ingestion import content_hash
 from app.services.rag import rechunk
 with quality_db.begin() as s:
  doc=Document(document_id='old',title='guide',content='# quasarheading\nTrain progressively.',source='manual',content_hash=content_hash('guide','# quasarheading\nTrain progressively.'),metadata_json='{}')
  s.add(doc);s.flush();s.add(Chunk(document_id='old',title='guide',content='obsolete text',source='manual',content_hash='oldhash'))
 rag=HybridRAGPipeline();assert rag._database_bm25_search('obsolete',3)
 monkeypatch.setattr(rechunk.embedding_client,'embed',lambda texts:[[1.0]+[0.0]*1023 for _ in texts])
 prepared=prepare_rechunk(doc)
 if outcome=='invalid_vector':prepared=[(prepared[0][0],prepared[0][1],[1.0])]
 if outcome=='duplicate_hash':prepared=prepared+prepared
 if outcome=='source_changed':
  with quality_db.begin() as s:s.scalar(select(Document).where(Document.document_id=='old')).content='updated source'
 if outcome!='success':
  with pytest.raises(Exception):replace_chunks(quality_db,doc,prepared)
  assert rag._database_bm25_search('obsolete',3)
 else:
  assert replace_chunks(quality_db,doc,prepared)==1
  assert not rag._database_bm25_search('obsolete',3)
  hits=rag._database_bm25_search('quasarheading',3)
  assert hits and hits[0].metadata['section_path']=='guide / quasarheading'
  with quality_db() as s:assert s.scalar(select(Document).where(Document.document_id=='old')).content==doc.content



@pytest.mark.skipif(os.getenv('FITZEN_QUALITY_DB') != 'postgres', reason='PostgreSQL row-lock integration is opt-in')
def test_concurrent_feedback_and_reviews_are_single_commit(quality_db):
 from concurrent.futures import ThreadPoolExecutor
 _,res=add_run(True)
 with ThreadPoolExecutor(max_workers=2) as pool:
  results=list(pool.map(lambda _:repo.feedback(down(res)),range(2)))
 assert sorted(r['replayed'] for r in results)==[False,True]
 case=repo.list_cases()['items'][0]
 payload=ReviewInput(decision='resolve',expected_version=1,note='verified')
 def review_once(_):
  try:return repo.review(case['case_id'],payload)
  except repo.ReviewConflict:return {'conflict':True}
 with ThreadPoolExecutor(max_workers=2) as pool:
  results=list(pool.map(review_once,range(2)))
 assert sum(not r.get('replayed') and not r.get('conflict') for r in results)==1
 assert counts(quality_db)==[1,1,1,1,0,0]
 assert repo.get_case(case['case_id'])['version']==2


def test_cancelled_run_never_enters_failure_case_queue(quality_db):
    from app.db.quality_models import FailureCase
    from sqlalchemy import select
    req=ChatRequest(message='cancelled before answer')
    res=ChatResponse(answer='',intent='error',run_id=str(uuid4()),message_id=req.request_id+':assistant')
    metrics=RequestMetrics();metrics.failure_reasons.add('execution_error');metrics.degraded_reasons.add('execution_error')
    repo.record_run(req,res,metrics.summary(),status='cancelled')
    with quality_db() as session:
        assert session.scalar(select(func.count()).select_from(FailureCase))==0
    assert repo.summary()['requests']==0
