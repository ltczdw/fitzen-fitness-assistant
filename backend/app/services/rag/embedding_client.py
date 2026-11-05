"""Shared Qwen embedding transport. Failure is explicit; never substitute hash vectors."""
import math
import httpx
from app.services.telemetry import model_attempt
from app.core.config import settings
from app.services.llm.budget import cache_key, key, model_stage, ModelBudgetError
from app.services.llm.retry_policy import bounded_post, abounded_post

class EmbeddingUnavailable(RuntimeError):
    pass

class EmbeddingClient:
    def configured(self):
        return bool(settings.retrieval_embedding_base_url and settings.retrieval_embedding_api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.configured():
            raise EmbeddingUnavailable("embedding_not_configured")
        if settings.retrieval_embedding_dimensions != 1024:
            raise EmbeddingUnavailable("schema_requires_1024_dimensions")
        vectors = []
        try:
            with httpx.Client(timeout=settings.retrieval_embedding_timeout_seconds, follow_redirects=False) as client:
                for start in range(0, len(texts), 20):
                    batch = texts[start:start+20]
                    payload={"model":settings.retrieval_embedding_model,"input":batch,
                             "dimensions":1024,"encoding_format":"float"}
                    url=settings.retrieval_embedding_base_url.rstrip('/')+'/embeddings'
                    headers={"Authorization":"Bearer "+settings.retrieval_embedding_api_key}
                    identifier=cache_key('embedding',settings.retrieval_embedding_model,payload,key(url+headers['Authorization']))
                    def validate(data):
                        values=sorted(data['data'],key=lambda value:value['index'])
                        if [value['index'] for value in values]!=list(range(len(batch))):raise ValueError('invalid_indexes')
                        for value in values:
                            vector=value['embedding']
                            if len(vector)!=1024 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in vector) or not any(vector):
                                raise ValueError('invalid_vector')
                    with model_stage('embedding_batch:'+identifier,2):
                        data=bounded_post(client,url,payload=payload,headers=headers,capability='embedding',
                                          model=settings.retrieval_embedding_model,cache_id=identifier,validate=validate)
                    vectors.extend(value['embedding'] for value in sorted(data['data'],key=lambda value:value['index']))
        except ModelBudgetError as exc:
            raise EmbeddingUnavailable(exc.code) from None
        except httpx.HTTPStatusError as exc:
            raise EmbeddingUnavailable('embedding_http_' + str(exc.response.status_code)) from None
        except EmbeddingUnavailable:
            raise
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise EmbeddingUnavailable(type(exc).__name__) from None
        return vectors

    async def aembed(self, texts: list[str], *, client=None) -> list[list[float]]:
        """Real async transport; one request per batch, no implicit retries."""
        if not texts:return []
        if len(texts)>1280:raise EmbeddingUnavailable('embedding_batch_limit')
        if not self.configured():raise EmbeddingUnavailable('embedding_not_configured')
        if settings.retrieval_embedding_dimensions!=1024:
            raise EmbeddingUnavailable('schema_requires_1024_dimensions')
        if client is None:
            from app.services.rag.async_http import client as pool_client
            client=pool_client()
        vectors=[]
        try:
            for start in range(0,len(texts),20):
                batch=texts[start:start+20]
                url=settings.retrieval_embedding_base_url.rstrip('/')+'/embeddings'
                headers={'Authorization':'Bearer '+settings.retrieval_embedding_api_key}
                payload={'model':settings.retrieval_embedding_model,'input':batch,'dimensions':1024,'encoding_format':'float'}
                identifier=cache_key('embedding',settings.retrieval_embedding_model,payload,key(url+headers['Authorization']))
                def validate(data):
                    values=sorted(data['data'],key=lambda value:value['index'])
                    if [value['index'] for value in values]!=list(range(len(batch))):raise ValueError('invalid_indexes')
                    for value in values:
                        vector=value['embedding']
                        if len(vector)!=1024 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in vector) or not any(vector):
                            raise ValueError('invalid_vector')
                with model_stage('embedding_batch:'+identifier,2):
                    data=await abounded_post(client,url,payload=payload,headers=headers,capability='embedding',
                        model=settings.retrieval_embedding_model,cache_id=identifier,validate=validate,max_attempts=1)
                vectors.extend(value['embedding'] for value in sorted(data['data'],key=lambda value:value['index']))
        except ModelBudgetError as exc:raise EmbeddingUnavailable(exc.code) from None
        except httpx.HTTPStatusError as exc:raise EmbeddingUnavailable('embedding_http_'+str(exc.response.status_code)) from None
        except EmbeddingUnavailable:raise
        except (httpx.HTTPError,ValueError,KeyError,TypeError) as exc:
            raise EmbeddingUnavailable(type(exc).__name__) from None
        return vectors

embedding_client = EmbeddingClient()
