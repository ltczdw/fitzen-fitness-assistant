"""Resumable vector rebuild into semantic_embedding; preserves old 256D embedding.

Reuses the shared backfill service so the repair path and the normal import
path share one implementation of the source/version checks.
"""
import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select, func

from app.db.session import init_db, SessionLocal
from app.db.models import KnowledgeChunkModel as K
from app.core.config import settings
from app.services.llm.jobs import add_job_arguments, model_job, vector_identity
from app.services.rag.ingestion import backfill_missing_vectors, plan_vector_backfill


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',help='Explicitly rebuild vectors; default is read-only preview.')
    add_job_arguments(parser)
    args=parser.parse_args()
    if not args.apply:
        with SessionLocal() as session:
            print({'pending':len(plan_vector_backfill(session)),'apply':False})
        return
    assert init_db(), 'database unavailable'
    version = settings.retrieval_embedding_index_version
    path = Path(__file__).resolve().parents[2] / 'backups' / 'knowledge-before-qwen-rebuild.json'
    with SessionLocal() as s:
        rows = list(s.scalars(select(K).order_by(K.id)))
        if not path.exists():
            path.write_text(json.dumps([{
                'id': x.id, 'title': x.title, 'content': x.content, 'source': x.source,
                'embedding': list(x.embedding) if x.embedding is not None else None,
                'semantic_embedding': list(x.semantic_embedding) if x.semantic_embedding is not None else None,
                'embedding_version': x.embedding_version,
            } for x in rows], ensure_ascii=False, default=float), encoding='utf-8')
        pending = len(plan_vector_backfill(s))
        identity = vector_identity(s)
    with model_job('vector-rebuild', args, identity):
        result = backfill_missing_vectors(factory=SessionLocal)
    with SessionLocal() as s:
        total = s.scalar(select(func.count()).select_from(K))
        ready = s.scalar(select(func.count()).select_from(K).where(
            K.semantic_embedding.is_not(None), K.embedding_version == version))
    report = {
        'total': total,
        'ready': ready,
        'pending_before': pending,
        **result.as_dict(),
        'dimensions': settings.retrieval_embedding_dimensions,
        'index_version': version,
        'legacy_vectors_preserved': True,
    }
    out = Path(__file__).resolve().parents[2] / 'evidence' / 'qwen-rebuild.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report)
    if result.pending:
        raise SystemExit(1)
    assert total == ready


if __name__ == '__main__':
    main()
