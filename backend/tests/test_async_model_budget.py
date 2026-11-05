import asyncio,json
from unittest.mock import AsyncMock
import httpx
import pytest
from sqlalchemy import select,func

from app.core.config import settings
from app.services.llm.budget import model_scope,active_scope,ModelBudgetError
from app.services.rag.embedding_client import EmbeddingClient,EmbeddingUnavailable


def configure(monkeypatch):
    monkeypatch.setattr(settings,'retrieval_embedding_base_url','https://async-budget.invalid/v1')
    monkeypatch.setattr(settings,'retrieval_embedding_api_key','mock-not-real')


def vectors(request):
    count=len(json.loads(request.content)['input'])
    return httpx.Response(200,json={'data':[{'index':i,'embedding':[1.0]+[0.0]*1023} for i in range(count)],
                                   'usage':{'prompt_tokens':count,'total_tokens':count}})


def test_sync_async_share_parent_allowance(monkeypatch):
    configure(monkeypatch)
    seen=[];original=httpx.Client
    def handle(request):seen.append(request);return vectors(request)
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:original(transport=httpx.MockTransport(handle),**kwargs))
    async def run():
        with model_scope('sync-and-async',max_calls=1):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                assert len(await EmbeddingClient().aembed(['first'],client=client))==1
            with pytest.raises(EmbeddingUnavailable,match='exhausted'):
                await asyncio.to_thread(EmbeddingClient().embed,['second'])
    asyncio.run(run());assert len(seen)==1


def test_async_no_scope_blocks_before_transport(monkeypatch):
    configure(monkeypatch);seen=[]
    def handle(request):seen.append(request);return vectors(request)
    async def run():
        token=active_scope.set(None)
        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                with pytest.raises(EmbeddingUnavailable,match='scope_required'):
                    await EmbeddingClient().aembed(['private-synthetic'],client=client)
        finally:active_scope.reset(token)
    asyncio.run(run());assert not seen


def test_async_failed_batch_resumes_without_repeating_cached_batch(monkeypatch):
    configure(monkeypatch);seen=[]
    def handle(request):
        count=len(json.loads(request.content)['input']);seen.append(count)
        if seen==[20,1]:return httpx.Response(503)
        return vectors(request)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with pytest.raises(EmbeddingUnavailable,match='503'):await EmbeddingClient().aembed(['sample']*21,client=client)
            assert len(await EmbeddingClient().aembed(['sample']*21,client=client))==21
    asyncio.run(run());assert seen==[20,1,1]


def test_policy_stop_preserves_written_record_and_response_replay(compact_db,monkeypatch):
    from app.services.agent.agent import FitnessAgent
    from app.db import repositories
    from app.db.quality_models import FailureCase
    from app.db.models import WorkoutRecordModel
    from app.schemas.chat import ChatRequest
    agent=FitnessAgent();monkeypatch.setattr(agent,'_run_planner_shadow',AsyncMock())
    writes=[]
    async def partial(request,*args):
        writes.append(repositories.create_workout('demo-user','synthetic-workout',1,1,1,idempotency_key='policy-partial'))
        raise ModelBudgetError('model_budget_exhausted:request')
    monkeypatch.setattr(agent,'_respond_impl',partial)
    request=ChatRequest(message='synthetic partial request',request_id='policy-stop',conversation_id='policy-stop')
    async def run():
        first=await agent.respond(request);second=await agent.respond(request)
        assert first.intent=='execution_limited' and '保留' in first.answer
        assert second.replayed and second.answer==first.answer
        assert len(writes)==1
    asyncio.run(run())
    with compact_db() as session:
        assert session.scalar(select(func.count()).select_from(WorkoutRecordModel))==1
        assert session.scalar(select(func.count()).select_from(FailureCase))==0
