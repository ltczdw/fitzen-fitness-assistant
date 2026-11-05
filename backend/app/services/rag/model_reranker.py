"""Real Qwen relevance model, shared by knowledge and Wiki retrieval."""
import math
from dataclasses import replace
from urllib.parse import urlsplit
import httpx
from app.services.telemetry import model_attempt
from app.core.config import settings
from app.services.llm.budget import cache_key, key, model_stage, ModelBudgetError
from app.services.llm.retry_policy import bounded_post, abounded_post
from app.services.rag.rerank_protocols import PROTOCOLS

class RerankUnavailable(RuntimeError):pass
class ModelReranker:
    def __init__(self):
        self.last_status='not_called'
    def rerank(self,query,chunks,top_n=3):
        if not chunks:return []
        base_url, api_key = settings.rerank_base_url, settings.rerank_api_key
        if not base_url and not api_key and settings.rerank_legacy_embedding_config and settings.rerank_provider == "dashscope":
            base_url, api_key = settings.retrieval_embedding_base_url, settings.retrieval_embedding_api_key
        if not settings.rerank_enabled or not base_url or not api_key:
            raise RerankUnavailable('reranker_not_configured')
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise RerankUnavailable('invalid_reranker_url')
        protocol = PROTOCOLS[settings.rerank_provider]
        endpoint = protocol.endpoint(base_url)
        passages=[]
        for index,hit in enumerate(chunks):
            text=hit.metadata.get('section_path',hit.title)+'\n'+hit.content
            # All source text participates; preserve the complete original in the returned hit.
            passages.extend((index,text[start:start+6000]) for start in range(0,len(text),6000))
        scores={}
        try:
            with httpx.Client(timeout=settings.rerank_timeout_seconds,follow_redirects=False) as client:
                for offset in range(0,len(passages),10):
                    batch=passages[offset:offset+10]
                    payload=protocol.payload(settings.rerank_model,query,[v for _,v in batch])
                    headers={'Authorization':'Bearer '+api_key}
                    identifier=cache_key('rerank',settings.rerank_model,payload,key(endpoint+headers['Authorization']))
                    def validate(data):
                        values=protocol.results(data);seen=set()
                        for item in values:
                            i=item['index'];score=float(item['relevance_score'])
                            if not isinstance(i,int) or isinstance(i,bool) or i in seen or not 0<=i<len(batch) or not math.isfinite(score) or not 0<=score<=1:
                                raise ValueError('invalid_rerank_response')
                            seen.add(i)
                        if len(seen)!=len(batch):raise ValueError('missing_rerank_results')
                    with model_stage('rerank_batch:'+identifier,2):
                        data=bounded_post(client,endpoint,payload=payload,headers=headers,capability='rerank',
                                          model=settings.rerank_model,cache_id=identifier,validate=validate)
                    for item in protocol.results(data):
                        original=batch[item['index']][0]
                        scores[original]=max(scores.get(original,0),float(item['relevance_score']))
        except ModelBudgetError as exc:
            raise RerankUnavailable(exc.code) from None
        except httpx.HTTPStatusError as exc:
            self.last_status = 'rerank_http_' + str(exc.response.status_code)
            raise RerankUnavailable(self.last_status) from None
        except (httpx.HTTPError,KeyError,ValueError,TypeError,RerankUnavailable) as exc:
            self.last_status=str(exc) if isinstance(exc,RerankUnavailable) else type(exc).__name__
            raise RerankUnavailable(self.last_status) from None
        self.last_status='ready'
        ranked=sorted(scores,key=lambda i:scores[i],reverse=True)
        return [replace(chunks[i],score=scores[i],metadata={**chunks[i].metadata,
            'rerank_score':scores[i],'rerank_model':settings.rerank_model,'rerank_status':'model'})
            for i in ranked if scores[i]>=settings.rerank_min_score][:top_n]
    async def arerank(self,query,chunks,top_n=3,*,client=None):
        """Async sibling of the CLI-compatible synchronous API; no automatic retry."""
        if not chunks:return []
        base_url,api_key=settings.rerank_base_url,settings.rerank_api_key
        if not base_url and not api_key and settings.rerank_legacy_embedding_config and settings.rerank_provider=='dashscope':
            base_url,api_key=settings.retrieval_embedding_base_url,settings.retrieval_embedding_api_key
        if not settings.rerank_enabled or not base_url or not api_key:raise RerankUnavailable('reranker_not_configured')
        parsed=urlsplit(base_url)
        if parsed.scheme not in {'http','https'} or not parsed.netloc or parsed.username or parsed.password:
            raise RerankUnavailable('invalid_reranker_url')
        protocol=PROTOCOLS[settings.rerank_provider]
        passages=[]
        for index,hit in enumerate(chunks):
            body=hit.metadata.get('section_path',hit.title)+'\n'+hit.content
            passages.extend((index,body[start:start+6000]) for start in range(0,len(body),6000))
        if len(passages)>100:raise RerankUnavailable('rerank_passage_limit')
        if client is None:
            from app.services.rag.async_http import client as pool_client
            client=pool_client()
        scores={}
        try:
            for offset in range(0,len(passages),10):
                batch=passages[offset:offset+10]
                payload=protocol.payload(settings.rerank_model,query,[v for _,v in batch])
                headers={'Authorization':'Bearer '+api_key}
                identifier=cache_key('rerank',settings.rerank_model,payload,key(protocol.endpoint(base_url)+headers['Authorization']))
                def validate(data):
                    seen=set()
                    for item in protocol.results(data):
                        i=item['index'];score=float(item['relevance_score'])
                        if not isinstance(i,int) or isinstance(i,bool) or i in seen or not 0<=i<len(batch) or not math.isfinite(score) or not 0<=score<=1:
                            raise ValueError('invalid_rerank_response')
                        seen.add(i)
                    if len(seen)!=len(batch):raise ValueError('missing_rerank_results')
                with model_stage('rerank_batch:'+identifier,2):
                    data=await abounded_post(client,protocol.endpoint(base_url),payload=payload,headers=headers,
                        capability='rerank',model=settings.rerank_model,cache_id=identifier,validate=validate,max_attempts=1)
                for item in protocol.results(data):
                    original=batch[item['index']][0];scores[original]=max(scores.get(original,0),float(item['relevance_score']))
        except ModelBudgetError as exc:raise RerankUnavailable(exc.code) from None
        except httpx.HTTPStatusError as exc:raise RerankUnavailable('rerank_http_'+str(exc.response.status_code)) from None
        except (httpx.HTTPError,KeyError,ValueError,TypeError,RerankUnavailable) as exc:
            raise RerankUnavailable(str(exc) if isinstance(exc,RerankUnavailable) else type(exc).__name__) from None
        return [replace(chunks[i],score=scores[i],metadata={**chunks[i].metadata,
            'rerank_score':scores[i],'rerank_model':settings.rerank_model,'rerank_status':'model'})
            for i in sorted(scores,key=lambda i:scores[i],reverse=True) if scores[i]>=settings.rerank_min_score][:top_n]

model_reranker=ModelReranker()
