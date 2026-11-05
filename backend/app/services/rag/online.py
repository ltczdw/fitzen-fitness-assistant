"""Compatibility dispatch for real async RAG and explicit injected/test adapters."""
from app.services.rag.blocking import run_retrieval
from app.services.rag.pipeline import HybridRAGPipeline,SYNC_SEARCH
from app.services.rag.retrieval import retrieval_service

_original_search=SYNC_SEARCH


async def search(pipeline,query,**kwargs):
    # Respect caller-provided synchronous adapters and test boundary mocks.
    if isinstance(pipeline,HybridRAGPipeline) and getattr(pipeline.search,'__func__',None) is _original_search and 'search' not in retrieval_service.__dict__:
        return await pipeline.asearch(query,**kwargs)
    kwargs.pop('skip_rewrite',None)
    return await run_retrieval(pipeline.search,query,**kwargs)
