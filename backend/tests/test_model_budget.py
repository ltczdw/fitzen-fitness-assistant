import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock

import httpx
import pytest
from sqlalchemy import create_engine,select
from sqlalchemy.orm import sessionmaker

from app.core import clock
from app.core.config import settings
from app.db.model_budget_models import ModelQuota,ModelCallLedger,ModelResultCache
from app.services.llm.budget import (BudgetStore,ModelBudgetError,ModelBudgetUnavailable,model_scope,
                                     model_stage,paid_attempt,active_scope)
from app.services.llm.client import LLMClient,LLMGatewayError,LLMErrorCategory


@pytest.fixture
def budget_store(tmp_path):
    engine=create_engine('sqlite:///'+str(tmp_path/'budgets.db'))
    for model in (ModelQuota,ModelCallLedger,ModelResultCache):model.__table__.create(engine)
    factory=sessionmaker(bind=engine,expire_on_commit=False)
    yield BudgetStore(factory)
    engine.dispose()


def attempt(store,name='same',**options):
    with model_scope(name,store=store,**options):
        with paid_attempt('chat','mock-model',{'input':'synthetic'},8):pass


def test_durable_scope_cannot_reset_by_recreating_scope_or_store(budget_store):
    attempt(budget_store,max_calls=1)
    reloaded=BudgetStore(budget_store.factory)
    with pytest.raises(ModelBudgetError,match='exhausted'):
        attempt(reloaded,max_calls=99)


def test_nested_stages_share_one_physical_allowance(budget_store):
    with model_scope('repair',store=budget_store):
        with model_stage('entity',2):
            for n in range(2):
                with paid_attempt('chat','mock',{'enhancement':n},8):pass
            with pytest.raises(ModelBudgetError):
                with paid_attempt('chat','mock',{'enhancement':3},8):pass


def test_unknown_usage_keeps_reserved_tokens_and_failed_calls(budget_store):
    with model_scope('unknown',store=budget_store,max_calls=2):
        with pytest.raises(TimeoutError):
            with paid_attempt('chat','mock',{'input':'x'},8):raise TimeoutError()
        scope_id=active_scope.get().quota_id
    with budget_store.factory() as session:
        quota=session.get(ModelQuota,scope_id)
        assert quota.calls==1 and quota.tokens>8
        call=session.scalar(select(ModelCallLedger))
        assert call.status=='error' and call.usage is None


def test_storage_failure_blocks_before_transport():
    store=BudgetStore(lambda:(_ for _ in ()).throw(RuntimeError('storage down')))
    with model_scope('fail-closed',store=store):
        with pytest.raises(ModelBudgetUnavailable):
            with paid_attempt('chat','mock',{},8):raise AssertionError('HTTP would start')


def test_model_requests_without_scope_fail_closed():
    token=active_scope.set(None)
    try:
        with pytest.raises(ModelBudgetError,match='scope_required'):
            with paid_attempt('chat','mock',{},8):pass
    finally:active_scope.reset(token)


def test_daily_cap_applies_to_new_request_ids(budget_store,monkeypatch):
    monkeypatch.setattr(settings,'model_daily_max_calls',1)
    attempt(budget_store,'one')
    with pytest.raises(ModelBudgetError,match='daily'):attempt(budget_store,'two')


def test_expired_budget_does_not_replenish(budget_store):
    attempt(budget_store,'expiry')
    with budget_store.factory() as session,session.begin():
        quota=session.scalar(select(ModelQuota).where(ModelQuota.kind=='request'))
        quota.expires_at=clock.utc_now()-timedelta(seconds=1)
    with pytest.raises(ModelBudgetError,match='expired'):attempt(budget_store,'expiry')


def test_async_and_to_thread_share_one_scope(budget_store):
    async def run():
        with model_scope('threads',store=budget_store,max_calls=1):
            def call():
                with paid_attempt('embedding','mock',{},0):pass
            await asyncio.to_thread(call)
            with pytest.raises(ModelBudgetError):call()
    asyncio.run(run())


def test_gateway_http_attempts_are_two_and_cached_after_success(budget_store,monkeypatch):
    monkeypatch.setattr(settings,'llm_api_key','mock-not-real')
    monkeypatch.setattr(settings,'llm_base_url','https://mock.invalid/v1')
    monkeypatch.setattr(settings,'llm_model','mock')
    monkeypatch.setattr(settings,'llm_retry_delays','0,0,0,0,0')
    seen=[];original=httpx.AsyncClient
    def handler(request):
        seen.append(request)
        return httpx.Response(503) if len(seen)==1 else httpx.Response(200,json={'choices':[{'message':{'content':'ok'}}],
            'usage':{'prompt_tokens':4,'completion_tokens':2}})
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    async def run():
        with model_scope('gateway',store=budget_store):
            client=LLMClient();client.admission.token=AsyncMock()
            assert await client.complete([{'role':'user','content':'hello'}])=='ok'
            assert await client.complete([{'role':'user','content':'hello'}])=='ok'
    asyncio.run(run())
    assert len(seen)==2


@pytest.mark.parametrize('status',[400,401,402,403,404])
def test_gateway_permanent_status_never_retries(status,budget_store,monkeypatch):
    monkeypatch.setattr(settings,'llm_api_key','mock-not-real')
    monkeypatch.setattr(settings,'llm_base_url',f'https://permanent-{status}.invalid/v1')
    monkeypatch.setattr(settings,'llm_model','mock')
    original=httpx.AsyncClient;seen=[]
    def handler(request):seen.append(request);return httpx.Response(status)
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    async def run():
        with model_scope('permanent-'+str(status),store=budget_store):
            client=LLMClient();client.admission.token=AsyncMock()
            with pytest.raises(LLMGatewayError):await client.complete([{'role':'user','content':'synthetic'}])
    asyncio.run(run())
    assert len(seen)==1
