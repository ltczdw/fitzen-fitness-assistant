import json
from unittest.mock import Mock
import httpx
import pytest
from sqlalchemy import select
from app.core.config import settings
from app.db.model_budget_models import ModelCallLedger
from app.services.llm.budget import model_scope,ModelBudgetError,paid_attempt
from app.services.rag.embedding_client import EmbeddingClient,EmbeddingUnavailable
from app.services.llm.jobs import model_job


def test_embedding_successful_batch_cached_after_later_batch_failure(monkeypatch):
    monkeypatch.setattr(settings,'retrieval_embedding_base_url','https://batch.invalid/v1')
    monkeypatch.setattr(settings,'retrieval_embedding_api_key','mock-not-real')
    monkeypatch.setattr(settings,'model_retry_backoff_seconds',0)
    original=httpx.Client;seen=[];later_succeeds=False
    def handle(request):
        values=json.loads(request.content)['input'];seen.append(len(values))
        if len(values)==1 and not later_succeeds:return httpx.Response(503)
        return httpx.Response(200,json={'data':[{'index':i,'embedding':[1.0]+[0.0]*1023} for i in range(len(values))],
                                       'usage':{'prompt_tokens':len(values),'total_tokens':len(values)}})
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:original(transport=httpx.MockTransport(handle),**kwargs))
    # Same job identity, successful first batch is retained; failed second
    # batch has spent its two-attempt allowance and cannot be silently reset.
    with pytest.raises(EmbeddingUnavailable):EmbeddingClient().embed(['synthetic']*21)
    assert seen==[20,1,1]
    later_succeeds=True
    with pytest.raises(EmbeddingUnavailable):EmbeddingClient().embed(['synthetic']*21)
    assert seen==[20,1,1]


def test_failed_batch_single_attempt_can_resume_without_repeating_success(monkeypatch):
    monkeypatch.setattr(settings,'retrieval_embedding_base_url','https://resume.invalid/v1')
    monkeypatch.setattr(settings,'retrieval_embedding_api_key','mock-not-real')
    monkeypatch.setattr(settings,'model_http_max_attempts',1)
    original=httpx.Client;seen=[]
    def handle(request):
        values=json.loads(request.content)['input'];seen.append(len(values))
        if seen==[20,1]:return httpx.Response(503)
        return httpx.Response(200,json={'data':[{'index':i,'embedding':[1.0]+[0.0]*1023} for i in range(len(values))]})
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:original(transport=httpx.MockTransport(handle),**kwargs))
    with pytest.raises(EmbeddingUnavailable):EmbeddingClient().embed(['synthetic']*21)
    assert len(EmbeddingClient().embed(['synthetic']*21))==21
    assert seen==[20,1,1]


def test_cli_custom_limit_cannot_exceed_ceiling():
    from types import SimpleNamespace
    with pytest.raises(ValueError):
        with model_job('cli',SimpleNamespace(max_model_calls=999999,job_id='bad'),{}):pass
