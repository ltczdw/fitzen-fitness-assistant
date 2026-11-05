import asyncio
import time
from uuid import uuid4
import pytest
from app.core.config import settings
from app.services.llm import client as llm_client
from app.services.llm.client import LLMClient, LLMGatewayError, LLMErrorCategory

@pytest.fixture
def clients(monkeypatch):
    monkeypatch.setattr(settings,"llm_base_url","http://test/"+uuid4().hex)
    monkeypatch.setattr(settings,"llm_model","fixture")
    monkeypatch.setattr(settings,"llm_api_key","fixture")
    monkeypatch.setattr(settings,"llm_max_concurrency",2)
    monkeypatch.setattr(settings,"llm_rate_burst",20)
    monkeypatch.setattr(settings,"llm_requests_per_second",100.)
    monkeypatch.setattr(settings,"llm_total_deadline_seconds",.15)
    return LLMClient(), LLMClient()

def test_shared_concurrency_and_breaker(clients,monkeypatch):
    a,b=clients
    assert a.breaker is b.breaker
    active=0;peak=0
    async def call(*args):
        nonlocal active,peak
        active+=1;peak=max(active,peak)
        await asyncio.sleep(.02)
        active-=1
        return "ok"
    monkeypatch.setattr(a,"_complete_once",call);monkeypatch.setattr(b,"_complete_once",call)
    async def check():
        assert await asyncio.gather(*(client.complete([]) for client in [a,b]*3)) == ["ok"]*6
    asyncio.run(check())
    assert peak==2
    assert a.admission.active==0

@pytest.mark.parametrize("mode",["async","sync","stream"])
def test_total_deadline_all_paths(clients,monkeypatch,mode):
    a,_=clients
    async def slow(*args): await asyncio.sleep(5)
    async def stream(*args):
        yield "first"
        await asyncio.sleep(5)
    monkeypatch.setattr(a,"_complete_once",slow)
    monkeypatch.setattr(a,"_stream_once",stream)
    start=time.monotonic()
    with pytest.raises(LLMGatewayError) as result:
        if mode=="sync": a.complete_sync([])
        elif mode=="async": asyncio.run(a.complete([]))
        else:
            async def consume(): return [v async for v in a.stream([])]
            asyncio.run(consume())
    assert result.value.category==LLMErrorCategory.TRANSPORT
    assert not result.value.retryable
    assert time.monotonic()-start<1
    assert a.admission.active==0


def test_open_fails_without_waiting_for_admission(clients):
    a,b=clients
    for _ in range(3): a.breaker.record_failure()
    start=time.monotonic()
    with pytest.raises(LLMGatewayError): asyncio.run(b.complete([]))
    assert time.monotonic()-start<.1


def test_rate_limit_wait_counts_toward_deadline(clients,monkeypatch):
    a,_=clients
    a.admission.rate=.01;a.admission.tokens=0
    async def forbidden(*args): raise AssertionError("No token available")
    monkeypatch.setattr(a,"_complete_once",forbidden)
    with pytest.raises(LLMGatewayError,match="deadline"):
        asyncio.run(a.complete([]))


def test_retry_sleep_counts_toward_deadline(clients,monkeypatch):
    import httpx
    a,_=clients;attempts=0
    async def fail(*args):
        nonlocal attempts
        attempts+=1
        raise httpx.ReadTimeout("timeout")
    monkeypatch.setattr(a,"_complete_once",fail)
    with pytest.raises(LLMGatewayError,match="deadline"): asyncio.run(a.complete([]))
    assert attempts==1


def test_token_bucket_spaces_upstream_attempts(clients,monkeypatch):
    a,_=clients
    a.admission.tokens=1
    a.admission.rate=20.
    a.admission.burst=1
    monkeypatch.setattr(settings,"llm_total_deadline_seconds",1.)
    calls=[]
    async def upstream(*args):
        calls.append(time.monotonic())
        return "ok"
    monkeypatch.setattr(a,"_complete_once",upstream)
    async def run():
        await asyncio.gather(a.complete([]),a.complete([]),a.complete([]))
    asyncio.run(run())
    assert len(calls)==3
    assert calls[-1]-calls[0]>=.08


def test_deadline_preserves_already_executed_tool_result(clients,monkeypatch):
    a,_=clients
    count=0
    async def upstream(*args):
        nonlocal count
        count+=1
        if count==1:
            return {"tool_calls":[{"id":"one","function":{"name":"save","arguments":"{}"}}]}
        # Simulate the deadline cancellation deterministically: under suite load
        # a real 5s sleep races the deadline check and occasionally re-raises.
        llm_client._call_deadline.set(time.monotonic() - 1)
        raise asyncio.CancelledError()
    async def execute(*args):
        return {"saved":True}
    monkeypatch.setattr(a,"_complete_message_once",upstream)
    result=asyncio.run(a.complete_with_tools([],tools=[{"type":"function","function":{"name":"save"}}],tool_executor=execute))
    assert result["tool_results"]==[{"name":"save","result":{"saved":True}}]
    assert result["error"]=="completion_after_tools_failed"
    assert a.admission.active==0
