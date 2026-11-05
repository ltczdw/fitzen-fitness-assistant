"""Finite transport retries only; no SDK or business-loop retry multiplication.

MEWHELP references: app/tools/engine.py transient-only bounded retries;
app/core/rerank.py status allowlist and exponential backoff. Adapted here
with durable physical-call quotas and Retry-After/deadline checks.
"""
import asyncio
import math
import random
import time
import json
from email.utils import parsedate_to_datetime
import httpx
from app.core.config import settings
from app.core import clock
from app.services.llm.budget import require_scope,ModelBudgetError

TRANSIENT_STATUS={408,429,500,502,503,504}


def retryable(error):
    if isinstance(error,ModelBudgetError):return False
    status=getattr(error,'status_code',None)
    if isinstance(error,httpx.HTTPStatusError):status=error.response.status_code
    if status is not None:return status in TRANSIENT_STATUS
    category=getattr(getattr(error,'category',None),'value',None)
    if category in ('config','protocol','policy'):return False
    if hasattr(error,'retryable'):return bool(error.retryable)
    return isinstance(error,(httpx.TransportError,TimeoutError,ConnectionError))


def pause_seconds(delay,remaining,retry_after=None):
    delay=float(delay)
    if not math.isfinite(delay) or delay<0:raise ModelBudgetError('invalid_retry_delay')
    if retry_after is not None:
        try:after=float(retry_after)
        except (ValueError,TypeError):
            try:after=(parsedate_to_datetime(retry_after)-clock.utc_now()).total_seconds()
            except (ValueError,TypeError):after=0
        if math.isfinite(after):delay=max(delay,after)
    delay=delay+random.uniform(0,min(.2,delay*.1))
    if delay>=remaining:raise ModelBudgetError('retry_deadline_exhausted')
    return delay


def bounded_post(client,url,*,payload,headers,capability,model,cache_id,validate=None):
    from app.services.llm.budget import paid_attempt
    from app.services.telemetry import model_attempt
    scope=require_scope()
    cached=scope.store.cached(cache_id)
    if cached is not None:return cached
    last=None
    for number in range(settings.model_http_max_attempts):
        scope.check()
        try:
            with paid_attempt(capability,model,payload) as attempt:
                with model_attempt(model,capability=capability) as metrics:
                    with client.stream('POST',url,json=payload,headers=headers,timeout=min(scope.remaining(),
                        settings.retrieval_embedding_timeout_seconds if capability=='embedding' else settings.rerank_timeout_seconds)) as response:
                        response.raise_for_status()
                        body=bytearray()
                        for chunk in response.iter_bytes():
                            scope.check()
                            body.extend(chunk)
                            if len(body)>2_000_000:raise ModelBudgetError('model_response_too_large')
                        result=json.loads(body);attempt.observe(result);metrics.observe(result)
                    if validate is not None:validate(result)
                    scope.store.cache(cache_id,scope,capability,result)
            return result
        except Exception as exc:
            last=exc
            if not retryable(exc) or number+1>=settings.model_http_max_attempts:raise
            after=exc.response.headers.get('Retry-After') if isinstance(exc,httpx.HTTPStatusError) else None
            time.sleep(pause_seconds(settings.model_retry_backoff_seconds,scope.remaining(),after))
    raise last


async def abounded_post(client,url,*,payload,headers,capability,model,cache_id,validate=None,max_attempts=1):
    from app.services.llm.budget import async_paid_attempt,cached_async,cache_async
    from app.services.telemetry import model_attempt
    scope=require_scope()
    cached=await cached_async(cache_id)
    if cached is not None:return cached
    ceiling=min(settings.model_http_max_attempts,max_attempts)
    for number in range(ceiling):
        scope.check()
        try:
            async with async_paid_attempt(capability,model,payload) as paid:
                with model_attempt(model,capability=capability) as metrics:
                    seconds=min(scope.remaining(),settings.retrieval_embedding_timeout_seconds if capability=='embedding' else settings.rerank_timeout_seconds)
                    async with asyncio.timeout(seconds):
                        response=await client.post(url,json=payload,headers=headers,timeout=seconds)
                        response.raise_for_status()
                        result=response.json();paid.observe(result);metrics.observe(result)
                        if validate is not None:validate(result)
                        await cache_async(cache_id,capability,result)
            return result
        except Exception as exc:
            if not retryable(exc) or number+1>=ceiling:raise
            after=exc.response.headers.get('Retry-After') if isinstance(exc,httpx.HTTPStatusError) else None
            await asyncio.sleep(pause_seconds(settings.model_retry_backoff_seconds,scope.remaining(),after))
