from datetime import timedelta
import hashlib
import json
import math
from uuid import uuid4

from sqlalchemy import select, update, func, delete
from sqlalchemy.exc import IntegrityError

from app.core import clock
from app.db import session as db
from app.db.storage import require_ready
from app.db.quality_models import ChatRun, FailureCase, AnswerFeedback, ReviewAudit
from app.db.models import KnowledgeDocumentModel, KnowledgeChunkModel


class MissingRecord(ValueError):
    pass


class ReviewConflict(ValueError):
    pass


def record_run(request, response, metrics, status="completed"):
    require_ready()
    with db.SessionLocal() as session, session.begin():
        from app.db.memory_models import MemoryTurn
        source = session.scalar(select(MemoryTurn).where(MemoryTurn.request_id == request.request_id).with_for_update())
        if source is not None and source.deleted_at is not None:
            return False
        run = ChatRun(run_id=response.run_id, request_id=request.request_id,
                      message_id=response.message_id, conversation_id=request.conversation_id,
                      question=request.message, answer=response.answer, intent=response.intent,
                      status=status, metrics=metrics, citations=response.model_dump()["citations"])
        session.add(run)
        session.flush()
        reasons = metrics.get("failure_reasons", [])
        if reasons and status != "cancelled":
            session.add(FailureCase(case_id=str(uuid4()), run_id=run.run_id,
                category="knowledge_gap" if "insufficient_evidence" in reasons else "execution",
                reasons=reasons))
    return True


def load_run_for_replay(request_id):
    """Rebuild the replayable response of a completed request from diagnostics.

    Deliberately refuses failed/cancelled/deleted attempts so a retry cannot
    bind an old失败 answer to the request identity.
    """
    require_ready()
    with db.SessionLocal() as session:
        run = session.scalar(select(ChatRun).where(
            ChatRun.request_id == request_id, ChatRun.status == "completed"
        ).order_by(ChatRun.created_at.desc(), ChatRun.run_id.desc()))
        if run is None:
            return None
        return {"answer": run.answer, "intent": run.intent,
                "citations": run.citations or [], "run_id": run.run_id,
                "message_id": run.message_id, "request_id": run.request_id,
                "metrics": run.metrics or {}, "safety_level": "normal",
                "model_used": False, "memory_status": "degraded",
                "diagnostics_status": "ready"}


def feedback(payload):
    require_ready()
    with db.SessionLocal() as session, session.begin():
        run = session.scalar(select(ChatRun).where(ChatRun.run_id == payload.run_id).with_for_update())
        if run is None or run.status == "deleted" or run.message_id != payload.message_id:
            raise MissingRecord("回答不存在或已删除")
        existing = session.scalar(select(AnswerFeedback).where(AnswerFeedback.run_id == run.run_id))
        if existing:
            if existing.rating == payload.rating and existing.note == payload.note:
                return {"feedback_id": existing.feedback_id, "replayed": True}
            raise ReviewConflict("这条回答已经提交反馈")
        row = AnswerFeedback(feedback_id=str(uuid4()), run_id=run.run_id, rating=payload.rating, note=payload.note)
        session.add(row)
        case = session.scalar(select(FailureCase).where(FailureCase.run_id == run.run_id))
        if payload.rating == "down" and case is None:
            case = FailureCase(case_id=str(uuid4()), run_id=run.run_id, category="user_feedback", reasons=["user_feedback"])
            session.add(case)
        elif payload.rating == "down" and case.status == "pending":
            case.reasons = sorted(set(case.reasons + ["user_feedback"]))
        return {"feedback_id": row.feedback_id, "case_id": case.case_id if case else None, "replayed": False}


def _case_out(case, run):
    return {"case_id": case.case_id, "run_id": run.run_id, "message_id": run.message_id,
            "category": case.category, "reasons": case.reasons, "status": case.status,
            "version": case.version, "question": run.question, "answer": run.answer,
            "citations": run.citations, "metrics": run.metrics,
            "approved_document_id": case.approved_document_id}


def list_cases(status="pending", offset=0, limit=20):
    require_ready()
    with db.SessionLocal() as session:
        query = select(FailureCase, ChatRun).join(ChatRun).where(ChatRun.status != "deleted")
        if status:
            query = query.where(FailureCase.status == status)
        total = session.scalar(select(func.count()).select_from(query.subquery()))
        rows = session.execute(query.order_by(FailureCase.created_at, FailureCase.case_id).offset(offset).limit(limit))
        return {"items": [_case_out(c, r) for c, r in rows], "total": total}


def get_case(case_id):
    require_ready()
    with db.SessionLocal() as session:
        row = session.execute(select(FailureCase, ChatRun).join(ChatRun).where(
            FailureCase.case_id == case_id, ChatRun.status != "deleted")).first()
        if not row:
            raise MissingRecord("案例不存在或已删除")
        result = _case_out(*row)
        result["audit"] = [{"decision": r.decision, "note": r.note, "source_url": r.source_url,
                            "reviewer": r.reviewer, "created_at": r.created_at.isoformat()}
                           for r in session.scalars(select(ReviewAudit).where(ReviewAudit.case_id == case_id)
                                                     .order_by(ReviewAudit.created_at))]
        return result


def review_digest(payload):
    return hashlib.sha256(json.dumps(payload.model_dump(mode="json"), sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def review(case_id, payload, prepared=None):
    """CAS + knowledge + audit in ONE transaction. No model/network calls here."""
    require_ready()
    digest = review_digest(payload)
    with db.SessionLocal() as session, session.begin():
        case = session.get(FailureCase, case_id)
        if case is None:
            raise MissingRecord("案例不存在")
        run = session.scalar(select(ChatRun).where(ChatRun.run_id == case.run_id).with_for_update())
        if run.status == "deleted":
            raise MissingRecord("回答已删除")
        if case.status != "pending":
            if case.review_digest == digest:
                return {"status": case.status, "document_id": case.approved_document_id, "replayed": True}
            raise ReviewConflict("案例已被审核")
        status = {"approve": "approved", "reject": "rejected", "resolve": "resolved"}[payload.decision]
        changed = session.execute(update(FailureCase).where(
            FailureCase.case_id == case_id, FailureCase.status == "pending",
            FailureCase.version == payload.expected_version).values(
                status=status, version=FailureCase.version + 1, review_digest=digest))
        if changed.rowcount != 1:
            raise ReviewConflict("审核版本已变化，请刷新")
        document_id = None
        if payload.decision == "approve":
            if prepared is None:
                raise ValueError("缺少已向量化知识")
            document, chunks, vectors = prepared
            from app.services.rag.ingestion import content_hash
            from app.services.rag.chunking import CHUNKER_VERSION
            digest_doc = content_hash(document.title, document.content)
            existing = session.scalar(select(KnowledgeDocumentModel).where(KnowledgeDocumentModel.content_hash == digest_doc))
            if existing:
                raise ReviewConflict("相同知识已存在，请核对后将案例标记为已解决")
            document_id = "doc-" + digest_doc[:24]
            metadata = {**document.metadata, "chunker_version": CHUNKER_VERSION, "review_case_id": case_id}
            session.add(KnowledgeDocumentModel(document_id=document_id, title=document.title,
                content=document.content, source=document.source, content_hash=digest_doc,
                metadata_json=json.dumps(metadata, ensure_ascii=False)))
            session.flush()
            from app.core.config import settings
            for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                session.add(KnowledgeChunkModel(document_id=document_id, parent_id=document_id,
                    title=document.title, content=chunk.content, source=document.source,
                    chunk_index=i, chunk_count=len(chunks),
                    content_hash=content_hash(document.title + "\n" + chunk.section_path, chunk.content),
                    metadata_json=json.dumps({**metadata, "section_path": chunk.section_path,
                                               "content_type": chunk.content_type}, ensure_ascii=False),
                    semantic_embedding=vector, embedding_version=settings.retrieval_embedding_index_version))
            session.execute(update(FailureCase).where(FailureCase.case_id == case_id)
                            .values(approved_document_id=document_id))
        session.add(ReviewAudit(audit_id=str(uuid4()), case_id=case_id, decision=payload.decision,
                               note=payload.note, source_url=payload.source_url))
        return {"status": status, "document_id": document_id, "replayed": False}


def summary(days=7):
    require_ready()
    with db.SessionLocal() as session:
        start = clock.utc_now() - timedelta(days=days)
        query = select(ChatRun.metrics).where(ChatRun.created_at >= start, ChatRun.status.notin_(("deleted", "cancelled")))
        total = session.scalar(select(func.count()).select_from(query.subquery()))
        rows = list(session.scalars(query.order_by(ChatRun.created_at.desc()).limit(10000)))
        latencies = sorted(r["latency_ms"] for r in rows)
        return {"days": days, "requests": len(rows), "total_requests": total,
                "truncated": total > len(rows), "model_calls": sum(r["model_calls"] for r in rows),
                "input_tokens": sum(r["input_tokens"] for r in rows),
                "output_tokens": sum(r["output_tokens"] for r in rows),
                "cached_tokens": sum(r["cached_tokens"] for r in rows),
                "total_tokens": sum(r.get("total_tokens", 0) for r in rows),
                "usage_missing_calls": sum(r["usage_missing_calls"] for r in rows),
                "degraded_rate": sum(bool(r["degraded"]) for r in rows) / len(rows) if rows else None,
                "latency_p50_ms": latencies[math.ceil(len(rows) * .5) - 1] if rows else None,
                "latency_p95_ms": latencies[math.ceil(len(rows) * .95) - 1] if rows else None}


def scrub_conversation(conversation_id):
    """Called inside the same transaction that deletes source memory."""
    # Accept a session at the call site via scrub_in_session to keep deletion atomic.
    require_ready()
    with db.SessionLocal() as session, session.begin():
        scrub_in_session(session, conversation_id)


def scrub_in_session(session, conversation_id, *, all_conversations=False):
    if conversation_id is None and not all_conversations:
        raise ValueError("conversation_id_required; use scrub_all_in_session explicitly")
    condition = True if all_conversations else ChatRun.conversation_id == conversation_id
    run_ids = select(ChatRun.run_id).where(condition)
    case_ids = select(FailureCase.case_id).where(FailureCase.run_id.in_(run_ids))
    session.execute(update(ReviewAudit).where(ReviewAudit.case_id.in_(case_ids)).values(note=""))
    session.execute(update(AnswerFeedback).where(AnswerFeedback.run_id.in_(run_ids)).values(note=""))
    session.execute(update(ChatRun).where(condition)
                    .values(question="", answer="", citations=[], status="deleted"))
