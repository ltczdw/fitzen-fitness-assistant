import asyncio
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import select,func
from app.core import clock
from app.core.config import settings
from app.db.model_budget_models import ModelQuota,ModelCallLedger,ModelResultCache
from app.services.llm.budget import (BudgetStore,model_scope,model_stage,paid_attempt,active_scope,key,ModelBudgetError)
from app.services.llm.client import LLMClient,LLMGatewayError
from app.services.llm.jobs import model_job
from app.services.llm.retry_policy import pause_seconds
from app.services.memory import compact
from app.schemas.chat import ChatRequest


def test_parent_budget_is_atomic_across_parallel_threads():
    store=active_scope.get().store
    def call(index):
        try:
            with model_scope('parallel-parent',store=store,max_calls=3):
                with paid_attempt('chat','mock',{'index':index},0):pass
            return True
        except ModelBudgetError:return False
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(call,range(8)))
    assert sum(results)==3


def test_cli_renaming_cannot_reset_same_input_budget():
    first=SimpleNamespace(max_model_calls=1,job_id='label-one')
    second=SimpleNamespace(max_model_calls=1,job_id='label-two')
    with model_job('bounded-job',first,{'source_hash':'synthetic'}):
        with paid_attempt('chat','mock',{},0):pass
    with model_job('bounded-job',second,{'source_hash':'synthetic'}):
        with pytest.raises(ModelBudgetError):
            with paid_attempt('chat','mock',{},0):pass


def test_retry_after_longer_than_deadline_stops_without_sleep():
    with pytest.raises(ModelBudgetError,match='deadline'):
        pause_seconds(1,10,'3600')


def test_request_cancellation_propagates_into_thread_before_next_post():
    async def run():
        with model_scope('cancel-to-thread') as scope:
            scope.cancelled=True
            def call():
                with paid_attempt('embedding','mock',{},0):raise AssertionError('would post')
            with pytest.raises(ModelBudgetError,match='cancelled'):await asyncio.to_thread(call)
    asyncio.run(run())


def test_stream_emitted_content_cannot_auto_retry(monkeypatch):
    monkeypatch.setattr(settings,'llm_api_key','mock-not-real')
    monkeypatch.setattr(settings,'llm_base_url','https://stream-edge.invalid/v1')
    monkeypatch.setattr(settings,'llm_model','mock')
    attempts=[]
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadError('mock disconnected')
    original=httpx.AsyncClient
    def handle(request):attempts.append(request);return httpx.Response(200,stream=BrokenStream())
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handle),**kwargs))
    async def run():
        client=LLMClient();client.admission.token=AsyncMock()
        with pytest.raises(LLMGatewayError):
            async for value in client.stream([{'role':'user','content':'synthetic'}]):assert value=='partial'
    asyncio.run(run());assert len(attempts)==1


def test_new_day_cannot_replenish_original_task_budget(monkeypatch):
    instant=clock.utc_now();monkeypatch.setattr(clock,'utc_now',lambda:instant)
    with model_scope('same-across-days',max_calls=1,ttl_seconds=172800):
        with paid_attempt('chat','mock',{},0):pass
    monkeypatch.setattr(clock,'utc_now',lambda:instant+timedelta(days=1))
    with model_scope('same-across-days',max_calls=99,ttl_seconds=172800):
        with pytest.raises(ModelBudgetError,match='exhausted'):
            with paid_attempt('chat','mock',{},0):pass


def test_derived_cache_deleted_but_spending_not_refunded(compact_db):
    from app.services.llm.budget import store_override
    store=BudgetStore(compact_db);token=store_override.set(store)
    owner=key('demo-user:cache-private')
    request=ChatRequest(message='synthetic',conversation_id='cache-private',request_id='cache-private-one')
    tid,_=compact.begin_turn(request);compact.finish_turn(tid,'synthetic',{'answer':'synthetic','intent':'training_qa'})
    try:
        with model_scope('request:'+request.request_id,store=store,owner_id=owner,max_calls=1) as scope:
            with paid_attempt('chat','mock',{},0):pass
            store.cache('cache-test-private',scope,'chat',{'derived':'synthetic'})
            scope_id=scope.quota_id
        compact.delete_memory('cache-private')
        with compact_db() as session:
            assert session.get(ModelResultCache,'cache-test-private') is None
            assert session.get(ModelQuota,scope_id).calls==1
    finally:store_override.reset(token)
