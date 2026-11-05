from typing import Literal
from urllib.parse import urlsplit
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.db import quality_repository as repo
from app.services.rag.blocking import run_retrieval
from app.services.rag.embedding_client import embedding_client, EmbeddingUnavailable
from app.services.rag.ingestion import SourceDocument, clean_text
from app.services.rag.chunking import split_document
from app.core.config import settings

router = APIRouter(tags=["quality"])

@router.get("/quality", include_in_schema=False)
def quality_page():
    from pathlib import Path
    from fastapi.responses import FileResponse
    return FileResponse(Path(__file__).resolve().parents[1] / "static" / "quality.html")


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    run_id: str = Field(min_length=1, max_length=36)
    message_id: str = Field(min_length=1, max_length=160)
    rating: Literal["up", "down"]
    note: str = Field(default="", max_length=2000)


class ReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    decision: Literal["approve", "reject", "resolve"]
    expected_version: int = Field(ge=1)
    note: str = Field(min_length=1, max_length=2000)
    question: str | None = Field(default=None, max_length=300)
    answer: str | None = Field(default=None, max_length=12000)
    source_url: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def approved_source(self):
        if self.decision == "approve":
            url = urlsplit(self.source_url or "")
            if not self.question or not self.answer or url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password:
                raise ValueError("通过入库必须提供核准的问题、答案及 HTTP(S) 来源网址")
        return self


async def invoke(function, *args):
    try:
        return await run_retrieval(function, *args)
    except repo.MissingRecord as exc:
        raise HTTPException(404, str(exc)) from None
    except (repo.ReviewConflict, IntegrityError):
        raise HTTPException(409, "状态已变化或重复提交，请刷新后核对") from None
    except SQLAlchemyError:
        raise HTTPException(503, "质量记录暂不可用，请稍后重试") from None


@router.post("/feedback")
async def feedback(payload: FeedbackInput):
    return await invoke(repo.feedback, payload)


@router.get("/review/cases")
async def cases(status: Literal["pending", "approved", "rejected", "resolved"] | None = "pending",
                offset: int = Query(default=0, ge=0), limit: int = Query(default=20, ge=1, le=100)):
    return await invoke(repo.list_cases, status, offset, limit)


@router.get("/review/cases/{case_id}")
async def case(case_id: str):
    return await invoke(repo.get_case, case_id)


def prepare(payload):
    document = SourceDocument(clean_text(payload.question), clean_text(payload.answer),
                              payload.source_url, {"url": payload.source_url, "reviewed": True})
    chunks = split_document(document.content, title=document.title, chunk_size=settings.rag_chunk_size,
                            overlap=settings.rag_chunk_overlap)
    if not chunks:
        raise ValueError("核准答案没有可入库正文")
    vectors = embedding_client.embed([c.section_path + "\n" + c.content for c in chunks])
    if len(vectors) != len(chunks):
        raise EmbeddingUnavailable("missing_vectors")
    return document, chunks, vectors


@router.post("/review/cases/{case_id}")
async def review(case_id: str, payload: ReviewInput):
    current = await invoke(repo.get_case, case_id)
    if current["status"] != "pending":
        return await invoke(repo.review, case_id, payload)  # Exact replay only; no model call.
    if current["version"] != payload.expected_version:
        raise HTTPException(409, "审核版本已变化，请刷新")
    prepared = None
    if payload.decision == "approve":
        try:
            prepared = await run_retrieval(prepare, payload)
        except EmbeddingUnavailable:
            raise HTTPException(502, "向量化不可用；案例保持待审，知识未写入") from None
        except ValueError:
            raise HTTPException(422, "核准答案没有可入库正文") from None
    return await invoke(repo.review, case_id, payload, prepared)


@router.get("/observability/summary")
async def summary(days: int = Query(default=7, ge=1, le=90)):
    return await invoke(repo.summary, days)


@router.get('/observability/model-budget')
async def model_budget():
    from app.services.llm.observability import model_budget_summary
    return await invoke(model_budget_summary)
