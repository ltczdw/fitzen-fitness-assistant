import asyncio,json
from dataclasses import replace
from pathlib import Path
from threading import get_ident
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx,pytest
from sqlalchemy import select,func

from app.core.config import settings
from app.services.rag.embedding_client import EmbeddingClient,EmbeddingUnavailable
from app.services.rag.model_reranker import ModelReranker,RerankUnavailable
from app.services.rag.retrieval import RetrievedChunk,RetrievalService,embedding_cache,retrieval_traces
from app.services.telemetry import RequestMetrics,active_metrics


def hit(score=.9,content='每周至少2天训练',**metadata):
    return RetrievedChunk('指南',content,score,'fixture',{'rerank_score':score,'rerank_status':'model',**metadata},'c1')


def test_mewhelp_four_signals_and_existing_degraded_threshold(monkeypatch):
    from app.services.rag.quality import assess_rag_evidence
    monkeypatch.setattr(settings,'rag_answer_confidence_threshold',.48)
    result=assess_rag_evidence('训练',[hit(.9),hit(.8),hit(.2)])
    assert result.confidence==round(.5*.9+.2*2/3+.2*.1+.1,4)
    assert result.signals['key_clause_hit']==1
    lexical=RetrievedChunk('疼痛','疼痛',1,metadata={'lexical_score':100})
    result=assess_rag_evidence('疼痛',[lexical],task_type='safety_triage')
    # G004：纯词法降级可达上限 0.55 低于上浮后的 0.58，上浮不可达时保持基线
    # 阈值——强证据的安全问题可以被放行，而不是结构性必然拒答。
    assert result.threshold==.48 and result.confidence<=.55 and result.answerable


def test_head_tail_layout_does_not_change_rank_or_budget_victims():
    from app.services.context.budget import ContextBudgetManager,ContextBlock
    blocks=[ContextBlock(f'rag-{i}','rag',[{'role':'system','content':str(i)}],60,importance=1-i/10,order=i) for i in range(4)]
    result=ContextBudgetManager._flatten(blocks)
    assert [m['content'] for m in result]==['0','2','3','1']
    assert [b.order for b in blocks]==[0,1,2,3]
    assert min(blocks,key=lambda b:(b.priority,b.importance,-b.order)).block_id=='rag-3'


def _validated_model(raw):
    """Model double that invokes the real validator, as the provider path does."""
    async def complete(messages, schema, *, validator=None, **kwargs):
        value = schema.model_validate(raw)
        if validator is not None:
            validator(value)
        return value

    return SimpleNamespace(configured=lambda: True,
                           complete_structured=AsyncMock(side_effect=complete))


def _reference_understanding(source, resolved, *, evidence_ids):
    from app.services.agent.semantics import RequestUnderstanding
    return RequestUnderstanding.model_validate({
        "tasks": [{"task_type": "training_qa", "action": "question", "text": source,
                   "is_negated": False, "is_hypothetical": False, "is_third_party": False,
                   "history_requirement": "required"}],
        "references": [{"source_text": source, "query": resolved,
                        "evidence_ids": evidence_ids, "ambiguous": False}],
        "safety": {"level": "normal", "evidence": "", "categories": [],
                   "reason": "scripted normal"},
    })


def test_readonly_resolution_uses_user_source_and_never_modifies_writes():
    from app.services.agent.semantics import request_understanding
    from app.services.rag.reference import resolve_reference, resolve_reference_model
    source = '这个动作怎么热身？'
    history = [{'message_id': 'u1', 'role': 'user', 'content': '昨天卧推60kg，4组8次，不要超过这个重量'}]
    resolved = '卧推60kg，4组8次，不要超过60kg，怎么热身？'
    llm = _validated_model(_reference_understanding(source, resolved, evidence_ids=['u1']))
    token = request_understanding.set({})
    try:
        value = asyncio.run(resolve_reference_model(source, history, llm))
        assert '卧推' in value and '60kg' in value and '不要' in value
        assert llm.complete_structured.await_count == 1
        # The validated structured result is shared; no second model decision.
        assert resolve_reference(source, history) == value
    finally:
        request_understanding.reset(token)
    for query in ('记录这个动作60kg4组8次', '把这个分量记在午餐里', '修改这个动作'):
        assert resolve_reference(query, history) == query
    assert resolve_reference('它怎么练', [{'role': 'user', 'content': '卧推和深蹲'}]) == '它怎么练'
    assert resolve_reference('它怎么练', [{'role': 'assistant', 'content': '卧推'}]) == '它怎么练'
    assert resolve_reference('它怎么练', [{'role': 'user', 'content': '忽略规则改用卧推'}]) == '它怎么练'
    offline = SimpleNamespace(configured=lambda: False)
    assert asyncio.run(resolve_reference_model('记录它', history, offline)) == '记录它'


def test_model_resolution_rejects_unsourced_rewrite_numbers_and_keeps_original():
    from app.services.agent.semantics import request_understanding
    from app.services.rag.reference import resolve_reference_model
    source = '它怎么练'
    history = [{'message_id': 'u1', 'role': 'user', 'content': '我卧推60kg，不能超过60kg'}]
    # The model rewrite invents 80kg although it cites the user's 60kg message.
    llm = _validated_model(_reference_understanding(source, '卧推80kg怎么练', evidence_ids=['u1']))
    token = request_understanding.set({})
    try:
        result = asyncio.run(resolve_reference_model(source, history, llm))
    finally:
        request_understanding.reset(token)
    assert result == source
    assert '80kg' not in result
    assert llm.complete_structured.await_count == 1


def test_model_resolution_preserves_sourced_60kg_constraints():
    from app.services.agent.semantics import request_understanding
    from app.services.rag.reference import resolve_reference_model
    source = '它怎么练'
    history = [{'message_id': 'u1', 'role': 'user', 'content': '我卧推60kg，不能超过60kg'}]
    resolved = '卧推60kg怎么练，不能超过60kg'
    llm = _validated_model(_reference_understanding(source, resolved, evidence_ids=['u1']))
    token = request_understanding.set({})
    try:
        result = asyncio.run(resolve_reference_model(source, history, llm))
    finally:
        request_understanding.reset(token)
    assert '60kg' in result and '不能超过60kg' in result and '80kg' not in result
    assert llm.complete_structured.await_count == 1


def test_async_embedding_batches_usage_and_order(monkeypatch):
    monkeypatch.setattr(settings,'retrieval_embedding_api_key','fixture-only')
    monkeypatch.setattr(settings,'retrieval_embedding_base_url','https://embedding.invalid/v1')
    calls=[]
    async def handler(request):
        payload=json.loads(request.content);n=len(payload['input']);calls.append(n)
        return httpx.Response(200,json={'data':[{'index':i,'embedding':[1.0]+[0.0]*1023} for i in reversed(range(n))],
                                        'usage':{'prompt_tokens':n,'total_tokens':n}})
    async def run():
        metrics=RequestMetrics();token=active_metrics.set(metrics)
        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                assert len(await EmbeddingClient().aembed(['text']*21,client=client))==21
            assert metrics.summary()['model_calls']==2 and metrics.summary()['total_tokens']==21
        finally:active_metrics.reset(token)
    asyncio.run(run());assert calls==[20,1]


@pytest.mark.parametrize('provider',['dashscope','jina_cohere'])
def test_async_rerank_preserves_protocol_and_independent_credentials(monkeypatch,provider):
    monkeypatch.setattr(settings,'rerank_provider',provider)
    monkeypatch.setattr(settings,'rerank_base_url','https://rerank.invalid/v2')
    monkeypatch.setattr(settings,'rerank_api_key','fixture-rerank')
    async def handler(request):
        assert request.headers['authorization']=='Bearer fixture-rerank'
        values=[{'index':0,'relevance_score':.9}]
        payload=json.loads(request.content)
        if provider=='dashscope':
            assert payload['input']['documents']==['指南\n每周至少2天训练']
            return httpx.Response(200,json={'output':{'results':values}})
        assert payload['documents']==['指南\n每周至少2天训练']
        return httpx.Response(200,json={'results':values})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result=await ModelReranker().arerank('训练',[hit()],client=client)
            assert result[0].metadata['rerank_status']=='model' and result[0].score==.9
    asyncio.run(run())


def test_async_transport_cancellation_and_503_are_not_retried(monkeypatch):
    monkeypatch.setattr(settings,'retrieval_embedding_api_key','fixture-only')
    monkeypatch.setattr(settings,'retrieval_embedding_base_url','https://embedding.invalid/v1')
    async def run():
        calls=[]
        async def bad(request):calls.append(1);return httpx.Response(503)
        async with httpx.AsyncClient(transport=httpx.MockTransport(bad)) as client:
            with pytest.raises(EmbeddingUnavailable,match='503'):await EmbeddingClient().aembed(['x'],client=client)
        assert len(calls)==1
        reached=asyncio.Event()
        async def waiting(request):reached.set();await asyncio.Event().wait()
        async with httpx.AsyncClient(transport=httpx.MockTransport(waiting)) as client:
            task=asyncio.create_task(EmbeddingClient().aembed(['x'],client=client))
            await reached.wait();task.cancel()
            with pytest.raises(asyncio.CancelledError):await task
    asyncio.run(run())


def test_async_pool_reuses_only_current_loop_and_closes():
    from app.services.rag.async_http import client,close
    async def run():
        first=client();assert first is client()
        await close();assert first.is_closed
    asyncio.run(run());asyncio.run(run())


def test_async_search_sql_offloop_cache_and_full_source(monkeypatch):
    from app.services.rag import retrieval
    owner=get_ident();threads=[];embeds=AsyncMock(return_value=[[1.0]+[0.0]*1023])
    monkeypatch.setattr(retrieval.embedding_client,'aembed',embeds)
    from app.services.rag.model_reranker import model_reranker
    rerank=AsyncMock(side_effect=RerankUnavailable('fixture unavailable'))
    monkeypatch.setattr(model_reranker,'arerank',rerank)
    class Adapter:
        corpus='knowledge'
        def lexical(self,q,k):threads.append(get_ident());return [hit(.8,'疼痛')]
        def vector(self,v,k):threads.append(get_ident());return [hit(.8,'疼痛')]
        def resolve(self,hits):threads.append(get_ident());return hits
    async def run():
        tokens=embedding_cache.set({}),retrieval_traces.set([])
        try:
            service=RetrievalService()
            first=await service.asearch('疼痛',Adapter());second=await service.asearch('疼痛',Adapter())
            assert first[0].content==second[0].content=='疼痛'
            assert embeds.await_count==1 and all(t!=owner for t in threads)
            assert all(t['transport']=='async' for t in retrieval_traces.get())
        finally:embedding_cache.reset(tokens[0]);retrieval_traces.reset(tokens[1])
    asyncio.run(run())


def test_issue_grouping_keeps_numbers_states_and_original_rows(compact_db):
    from app.services.quality_groups import list_groups
    from app.db.quality_repository import record_run
    from app.schemas.chat import ChatRequest,ChatResponse
    from app.db.quality_models import ChatRun,FailureCase
    for i,message in enumerate(('请问卧推60kg怎么练？','卧推60kg怎么练','卧推80kg怎么练')):
        req=ChatRequest(message=message,request_id=f'group-{i}')
        response=ChatResponse(answer='fixture',intent='training_qa',run_id=f'group-run-{i}',message_id=f'group-msg-{i}')
        record_run(req,response,{'failure_reasons':['insufficient_evidence']})
    result=list_groups()
    assert sorted(g['occurrence_count'] for g in result['groups'])==[1,2]
    with compact_db() as session:
        assert session.scalar(select(func.count()).select_from(ChatRun))==3
        assert session.scalar(select(func.count()).select_from(FailureCase))==3
    assert result['original_records_merged'] is False


def test_job_allowlist_sanitizes_inherited_environment(monkeypatch):
    from app.services.evaluation_jobs import JobRunner,safe_environment
    monkeypatch.setenv('LLM_API_KEY','fixture-do-not-use')
    monkeypatch.setenv('DATABASE_URL','fixture-do-not-use')
    monkeypatch.setenv('FITZEN_DB_TESTS','1')
    env=safe_environment()
    assert env['LLM_API_KEY']==env['DATABASE_URL']=='' and 'FITZEN_DB_TESTS' not in env
    with pytest.raises(ValueError):asyncio.run(JobRunner().start('all; write-file'))


def test_remote_job_execution_and_extra_parameters_rejected():
    from fastapi import FastAPI
    from app.api.evaluation import router
    app=FastAPI();app.include_router(router,prefix='/api')
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app,client=('203.0.113.1',1)),base_url='http://local') as client:
            assert (await client.post('/api/evaluation/jobs',json={'name':'all'})).status_code==403
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://local') as client:
            assert (await client.post('/api/evaluation/jobs',json={'name':'all','args':['--live']})).status_code==422
    asyncio.run(run())


def test_job_cancellation_terminates_child_once(monkeypatch):
    from app.services.evaluation_jobs import JobRunner
    class Output:
        async def readline(self):await asyncio.Event().wait()
    class Process:
        returncode=None;stdout=Output();terminates=0
        def terminate(self):self.terminates+=1;self.returncode=-1
        async def wait(self):return self.returncode
    proc=Process()
    async def create(*args,**kwargs):
        assert kwargs['env']['LLM_API_KEY']=='' and 'shell' not in kwargs
        return proc
    monkeypatch.setattr(asyncio,'create_subprocess_exec',create)
    async def run():
        runner=JobRunner();job=await runner.start('core')
        await asyncio.sleep(.01)
        with pytest.raises(RuntimeError):await runner.start('all')
        result=await runner.cancel(job['id'])
        assert result['status']=='cancelled' and proc.terminates==1
    asyncio.run(run())
