import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
import httpx
from app.schemas.chat import ChatResponse
from app.schemas.citation import Citation, citations_from_hits, normalize_citations
from app.services.rag.retrieval import RetrievedChunk
from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph


def test_provenance_preserved_and_no_fabricated_url():
    hits = [RetrievedChunk("Title", "content", 1, "document-a", {"page": 7, "url": "https://example.org/doc", "private": "hidden"}, "chunk-3")]
    refs = citations_from_hits(hits)
    assert refs == [{"title": "Title", "source": "document-a", "page": 7, "url": "https://example.org/doc", "chunk_id": "chunk-3"}]
    assert Citation(title="local", source="local", url="javascript:alert(1)").url is None
    assert normalize_citations(["legacy"])[0]["url"] is None
    result = FitnessMultiAgentGraph._finalize_node({"worker_results": [{"answer": "A", "citations": refs}, {"answer": "B", "citations": refs}]})
    assert result["citations"] == refs


def test_sync_and_sse_share_citation_shape(monkeypatch):
    from app.api import chat as module
    from app.main import app
    refs = [{"title":"guide", "source":"book", "chunk_id":"c1", "page":2, "url":None}]
    @asynccontextmanager
    async def scope(request):
        yield None
    async def stream(request):
        yield {"event":"meta", "data":{"intent":"training_qa", "citations":[]}}
        yield {"event":"token", "data":{"content":"answer"}}
        yield {"event":"done", "data":{"model_used":False,"citations":refs}}
    monkeypatch.setattr(module,"require_ready",lambda:None)
    monkeypatch.setattr(module.agent,"_request_scope",scope)
    monkeypatch.setattr(module.agent,"_respond_impl",AsyncMock(return_value=ChatResponse(answer="answer",intent="training_qa",citations=refs)))
    monkeypatch.setattr(module.agent,"_respond_stream_impl",stream)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app),base_url="http://test") as client:
            sync = await client.post("/api/chat",json={"message":"深蹲技巧"})
            sse = await client.post("/api/chat/stream",json={"message":"深蹲技巧"})
        assert sync.status_code == sse.status_code == 200
        done = next(block for block in sse.text.split("\n\n") if block.startswith("event: done"))
        data = json.loads(done.split("data: ",1)[1])
        assert sync.json()["citations"] == data["citations"] == refs
    asyncio.run(run())
