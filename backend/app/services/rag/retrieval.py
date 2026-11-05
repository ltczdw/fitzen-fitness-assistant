"""One BM25/pgvector/RRF pipeline; corpus adapters own retrieval and source resolution."""
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any
import re
from app.core.config import settings
from app.services.rag.embedding_client import embedding_client, EmbeddingUnavailable
from app.services.rag.bm25 import query_terms
from app.services.rag.fusion import reciprocal_rank_fusion
from app.services.rag.reranker import LocalReranker

@dataclass(frozen=True)
class RetrievedChunk:
    title: str
    content: str
    score: float
    source: str = "seed"
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str | None = None

embedding_cache: ContextVar[dict | None] = ContextVar('request_embedding_cache', default=None)
retrieval_traces: ContextVar[list | None] = ContextVar('retrieval_traces', default=None)

class RetrievalService:
    @staticmethod
    def _fuse(query,adapter,lexical,vector):
        scores={}
        for label,values in [('lexical_score',lexical),('vector_score',vector)]:
            for item in values:scores.setdefault(item.chunk_id,{})[label]=item.score
        fused=reciprocal_rank_fusion([lexical,vector])
        fused=[replace(x,metadata={**x.metadata,**scores.get(x.chunk_id,{}),'rrf_score':x.score}) for x in fused]
        if adapter.corpus=='knowledge':fused=LocalReranker().rerank(query,fused,len(fused))
        # Keep synchronous and asynchronous paths identical in this adaptation.
        # G003: 与 BM25 索引相同的重叠二元词规则，按字符偏移切词会丢掉
        # “有疼痛”中的“疼痛”等命中词项。
        terms=query_terms(query)
        allowed=set()
        for hit in fused:
            body=(hit.title+' '+hit.metadata.get('section_path','')+' '+hit.content).lower()
            if hit.metadata.get('vector_score',-1)>=settings.retrieval_vector_threshold or (hit.metadata.get('lexical_score',0)>0 and any(t in body for t in terms)):
                allowed.add(hit.chunk_id)
        return fused,allowed

    async def asearch(self,query,adapter,final_k=3,candidate_k=10,*,strategy='hybrid_rerank'):
        """SQL adapters in bounded workers; network calls await async transports."""
        from app.services.rag.blocking import run_retrieval
        from app.services.rag.model_reranker import model_reranker,RerankUnavailable
        if strategy not in {'bm25','vector','hybrid','hybrid_rerank'}:raise ValueError('unknown retrieval strategy')
        lexical=await run_retrieval(adapter.lexical,query,candidate_k) if strategy!='vector' else []
        vector=[];reason=None
        cache=embedding_cache.get()
        key=(query,settings.retrieval_embedding_model,settings.retrieval_embedding_index_version)
        if strategy!='bm25':
            try:
                if cache is not None and key in cache:qvec=cache[key]
                else:
                    qvec=(await embedding_client.aembed([query]))[0]
                    if cache is not None:cache[key]=qvec
                vector=await run_retrieval(adapter.vector,qvec,candidate_k)
            except EmbeddingUnavailable as exc:reason=str(exc)
        fused,allowed=self._fuse(query,adapter,lexical,vector)
        resolved=await run_retrieval(adapter.resolve,fused[:candidate_k])
        if strategy=='hybrid_rerank':
            try:result=await model_reranker.arerank(query,resolved,final_k)
            except RerankUnavailable as exc:
                result=[replace(h,metadata={**h.metadata,'rerank_status':'degraded_rules','rerank_error':str(exc)}) for h in resolved if h.chunk_id in allowed][:final_k]
                reason=';'.join(filter(None,[reason,str(exc)]))
        else:
            result=[replace(h,metadata={**h.metadata,'rerank_status':'not_requested','retrieval_strategy':strategy}) for h in resolved if h.chunk_id in allowed][:final_k]
        if hasattr(adapter,'validate'):result=await run_retrieval(adapter.validate,result)
        if reason:
            from app.services.telemetry import mark_degraded
            mark_degraded('retrieval_fallback')
        traces=retrieval_traces.get()
        if traces is not None:traces.append({'corpus':adapter.corpus,'lexical_count':len(lexical),
            'vector_count':len(vector),'selected_ids':[x.chunk_id for x in result],
            'degraded_reason':reason,'transport':'async','strategy':strategy})
        return result

    def search(self, query, adapter, final_k=3, candidate_k=10):
        lexical = adapter.lexical(query, candidate_k)
        vector, reason = [], None
        cache = embedding_cache.get()
        key = (query, settings.retrieval_embedding_model, settings.retrieval_embedding_index_version)
        try:
            if cache is not None and key in cache:
                qvec = cache[key]
            else:
                qvec = embedding_client.embed([query])[0]
                if cache is not None:
                    cache[key] = qvec
            vector = adapter.vector(qvec, candidate_k)
        except EmbeddingUnavailable as exc:
            reason = str(exc)
        # Preserve the score source. RRF returns only a rank fusion score.
        scores = {}
        for label, values in [('lexical_score', lexical), ('vector_score', vector)]:
            for item in values:
                scores.setdefault(item.chunk_id, {})[label] = item.score
        fused = reciprocal_rank_fusion([lexical, vector])
        fused = [replace(x, metadata={**x.metadata, **scores.get(x.chunk_id, {}), 'rrf_score': x.score}) for x in fused]
        if adapter.corpus == 'knowledge':
            fused = LocalReranker().rerank(query, fused, len(fused))
        terms = query_terms(query)
        filtered = []
        for hit in fused:
            text = (hit.title+' '+hit.metadata.get('section_path','')+' '+hit.content).lower()
            lexical_match = any(t in text for t in terms)
            if hit.metadata.get('vector_score', -1) >= settings.retrieval_vector_threshold or (hit.metadata.get('lexical_score', 0) > 0 and lexical_match):
                filtered.append(hit)
        # Resolve complete sources before semantic reranking; retrieval summaries never feed answers.
        from app.services.rag.model_reranker import model_reranker, RerankUnavailable
        resolved = adapter.resolve(fused[:candidate_k])
        try:
            result = model_reranker.rerank(query, resolved, final_k)
        except RerankUnavailable as exc:
            allowed = {h.chunk_id for h in filtered}
            result = [replace(h, metadata={**h.metadata, 'rerank_status':'degraded_rules',
                       'rerank_error':str(exc)}) for h in resolved if h.chunk_id in allowed][:final_k]
            reason = ';'.join(filter(None,[reason,str(exc)]))
        if hasattr(adapter, 'validate'):
            result = adapter.validate(result)
        if reason:
            from app.services.telemetry import mark_degraded
            mark_degraded("retrieval_fallback")
        traces = retrieval_traces.get()
        if traces is not None:
            traces.append({'corpus': adapter.corpus, 'lexical_count': len(lexical), 'vector_count': len(vector),
                           'selected_ids': [x.chunk_id for x in result], 'degraded_reason': reason})
        return result

retrieval_service = RetrievalService()
