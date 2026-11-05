"""Unknown/permanent memory failures stop; transport recovery stays finite."""
import asyncio
import re
import httpx
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError
from app.services.llm.budget import ModelBudgetError
from app.services.llm.retry_policy import retryable as transport_retryable


def classify_memory_failure(error):
    # Return only a safe enum/status, never an exception message containing
    # provider response bodies, prompts, identifiers or credential values.
    if isinstance(error,ModelBudgetError):return False,'policy_stop'
    category=getattr(getattr(error,'category',None),'value',None)
    status=getattr(error,'status_code',None)
    if category is not None:
        suffix=':http_'+str(status) if isinstance(status,int) else ''
        return transport_retryable(error),'gateway_'+category+suffix
    if isinstance(error,(ValidationError,ValueError,TypeError,KeyError)):
        return False,'protocol_invalid'
    if isinstance(error,(asyncio.TimeoutError,httpx.TransportError,ConnectionError,OperationalError)):
        return True,'temporary_'+type(error).__name__
    if isinstance(error,httpx.HTTPStatusError):
        return transport_retryable(error),'http_'+str(error.response.status_code)
    if type(error).__name__ in ('EmbeddingUnavailable','RerankUnavailable'):
        # These adapters historically expose canonical codes only. A missing
        # config, invalid vector or exhausted budget cannot improve by polling.
        code=str(error)
        match=re.fullmatch(r'(?:embedding|rerank)_http_(\d{3})',code)
        if match:
            number=int(match[1]);return number in (408,429,500,502,503,504),'http_'+str(number)
        if code in ('ConnectTimeout','ReadTimeout','ConnectError','NetworkError','RemoteProtocolError'):
            return True,'temporary_transport'
        return False,'adapter_permanent'
    return False,'unclassified_stop'
