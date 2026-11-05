import httpx
import pytest

from app.core.config import settings
from app.services.rag.embedding_client import EmbeddingClient, EmbeddingUnavailable
from app.services.memory.retry_policy import classify_memory_failure


@pytest.mark.parametrize('status,expected_calls,can_retry', [(401, 1, False), (503, 2, True)])
def test_sync_embedding_preserves_safe_http_code(monkeypatch, status, expected_calls, can_retry):
    monkeypatch.setattr(settings, 'retrieval_embedding_base_url', 'https://adapter-status.invalid/v1')
    monkeypatch.setattr(settings, 'retrieval_embedding_api_key', 'mock-not-real')
    monkeypatch.setattr(settings, 'model_retry_backoff_seconds', 0)
    seen = []
    original = httpx.Client

    def handle(request):
        seen.append(request)
        return httpx.Response(status)

    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    with pytest.raises(EmbeddingUnavailable) as stopped:
        EmbeddingClient().embed(['synthetic'])
    assert str(stopped.value) == 'embedding_http_' + str(status)
    assert classify_memory_failure(stopped.value) == (can_retry, 'http_' + str(status))
    assert len(seen) == expected_calls
