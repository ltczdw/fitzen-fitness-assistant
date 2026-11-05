import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock

import pytest
from sqlalchemy import select
from app.core import clock
from app.core.config import settings
from app.db.memory_models import MemoryTurn,MemoryOutbox
from app.db.model_budget_models import ModelQuota
from app.schemas.chat import ChatRequest
from app.services.memory import compact
from app.services.memory.retry_policy import classify_memory_failure
from app.services.llm.client import LLMGatewayError,LLMErrorCategory
from app.services.rag.embedding_client import EmbeddingUnavailable


def queued(factory,identifier='retry-fixture'):
    request=ChatRequest(message='synthetic memory',conversation_id=identifier,request_id=identifier)
    tid,_=compact.begin_turn(request)
    compact.finish_turn(tid,'synthetic answer',{'answer':'synthetic answer','intent':'training_qa'})
    with factory() as session,session.begin():
        turn=session.get(MemoryTurn,tid)
        turn.index_status='pending';compact.enqueue(session,turn)
    return tid


def make_due(factory):
    with factory() as session,session.begin():
        row=session.scalar(select(MemoryOutbox).where(MemoryOutbox.status=='retry'))
        row.next_attempt_at=clock.utc_now()-timedelta(seconds=1)


def test_queue_retries_stop_at_three_and_cannot_reset_across_reenqueue(compact_db,monkeypatch):
    monkeypatch.setattr(settings,'memory_outbox_max_attempts',3)
    tid=queued(compact_db)
    for n in range(3):
        event=compact.claim_event()
        assert event['attempt']==n+1
        compact.retry_event(event,'temporary_transport',retryable=True)
        if n<2:make_due(compact_db)
    assert compact.claim_event() is None
    with compact_db() as session,session.begin():
        old=session.scalar(select(MemoryOutbox))
        assert old.status=='dead_letter' and old.attempts==3
        turn=session.get(MemoryTurn,tid);turn.memory_epoch+=1
        compact.enqueue(session,turn)
        session.flush()
        rows=list(session.scalars(select(MemoryOutbox)))
        assert len(rows)==2 and all(r.attempts==3 for r in rows)


@pytest.mark.parametrize('error',[
    LLMGatewayError('mock credentials invalid',LLMErrorCategory.CONFIG),
    LLMGatewayError('mock payment',LLMErrorCategory.UPSTREAM,status_code=402),
    ValueError('mock invalid schema'),
    EmbeddingUnavailable('embedding_not_configured'),
])
def test_permanent_errors_enter_dead_letter_without_polling(compact_db,error):
    queued(compact_db)
    event=compact.claim_event()
    retryable,code=classify_memory_failure(error)
    assert not retryable
    compact.retry_event(event,code,retryable=retryable)
    with compact_db() as session:
        row=session.scalar(select(MemoryOutbox));assert row.status=='dead_letter' and row.attempts==1
        assert 'mock' not in row.last_error_code
    assert compact.claim_event() is None


def test_task_ttl_checked_before_model_or_attempt_increment(compact_db):
    queued(compact_db)
    with compact_db() as session,session.begin():
        budget=session.scalar(select(ModelQuota).where(ModelQuota.kind=='background'))
        budget.expires_at=clock.utc_now()-timedelta(seconds=1)
    assert compact.claim_event() is None
    with compact_db() as session:
        row=session.scalar(select(MemoryOutbox))
        assert row.attempts==0 and row.status=='dead_letter' and row.last_error_code=='job_budget_expired'


def test_successful_summary_not_repeated_when_embedding_fails(compact_db,monkeypatch):
    from app.services.memory import deepseek_summary as summary_module
    monkeypatch.setattr(settings,'memory_background_enabled',True)
    monkeypatch.setattr(settings,'memory_summary_enabled',True)
    queued(compact_db)
    summarizer=AsyncMock(return_value={'items':[{'speaker':'user','text':'synthetic remembered fact'}]})
    monkeypatch.setattr(summary_module.deepseek_summary,'summarize',summarizer)
    embed=Mock(side_effect=[EmbeddingUnavailable('ReadTimeout'),[[1.0]+[0.0]*1023]])
    monkeypatch.setattr(compact.embedding_client,'embed',embed)
    assert asyncio.run(compact.consume_one()) is True
    make_due(compact_db)
    assert asyncio.run(compact.consume_one()) is True
    assert summarizer.await_count==1 and embed.call_count==2
    with compact_db() as session:
        assert session.scalar(select(MemoryOutbox)).status=='done'


def test_historical_over_limit_tasks_never_call_models(compact_db,monkeypatch):
    from app.services.memory import deepseek_summary as summary_module
    monkeypatch.setattr(settings,'memory_background_enabled',True)
    monkeypatch.setattr(settings,'memory_summary_enabled',True)
    queued(compact_db)
    with compact_db() as session,session.begin():
        row=session.scalar(select(MemoryOutbox));row.attempts=21
    summarize=AsyncMock(side_effect=AssertionError('would charge provider'))
    monkeypatch.setattr(summary_module.deepseek_summary,'summarize',summarize)
    assert asyncio.run(compact.consume_one()) is False
    summarize.assert_not_awaited()
