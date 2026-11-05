"""Crawl explicit official pages, preview/import, then batch their vectors.

Preview is the default. No schema initialization, user-data writes, chat or
background calls. Import and vector work require explicit switches.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.rag.knowledge_catalog import load_catalog, crawl_catalog, load_snapshot, document_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "seed/authority_coverage.json")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "seed/authority_coverage_snapshot.json")
    parser.add_argument("--crawl", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--vectors", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--verify-count", type=int, default=16)
    parser.add_argument("--verify-offset", type=int, default=0, help="Skip already verified entries in the topic-balanced order")
    parser.add_argument("--verify-question-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--verify-ids", nargs='+', help="Recheck only these changed catalog entries")
    parser.add_argument("--reserve-model-calls", type=int, default=20, help="Keep this many daily calls for normal chat (minimum 20)")
    parser.add_argument("--batch-query-embeddings", action=argparse.BooleanOptionalAction, default=True,
                        help="Batch query vectors into the existing per-request embedding cache")
    from app.services.llm.jobs import add_job_arguments
    add_job_arguments(parser)
    args = parser.parse_args()
    if args.reserve_model_calls < 20:
        parser.error('--reserve-model-calls must be at least 20')
    catalog = load_catalog(args.catalog)
    if args.crawl:
        snapshot = crawl_catalog(catalog, progress=lambda value: print(json.dumps(value), flush=True))
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"fetched": len(snapshot["documents"]), "failed": len(snapshot["failures"])}), flush=True)
    documents = load_snapshot(args.snapshot)
    by_id = {entry['id']: entry for entry in catalog}
    if not documents or any(document.metadata['coverage_id'] not in by_id
            or document.metadata['requested_url'] != by_id[document.metadata['coverage_id']]['url']
            or document.metadata['topics'] != [by_id[document.metadata['coverage_id']]['topic']]
            for document in documents):
        raise ValueError('snapshot_must_match_nonempty_catalog')
    from app.services.rag.chunking import split_document
    from app.core.config import settings
    from app.services.rag.ingestion import KnowledgeIngestionService, backfill_missing_vectors
    ids = document_ids(documents)
    report = {"documents": len(documents), "chunks_preview": sum(len(split_document(d.content, title=d.title,
        chunk_size=settings.rag_chunk_size, overlap=settings.rag_chunk_overlap)) for d in documents),
        "topics": sorted({topic for d in documents for topic in d.metadata["topics"]}), "apply": args.apply,
        "source_failures": json.loads(args.snapshot.read_text(encoding='utf-8')).get('failures', [])}
    print(json.dumps(report), flush=True)
    if args.apply or args.vectors or args.verify:
        from sqlalchemy import text
        from app.db import session as db
        if db.SessionLocal is None: raise RuntimeError("database_unconfigured")
        with db.SessionLocal() as session:
            session.execute(text("SELECT id FROM knowledge_chunks LIMIT 1"))
        db.DB_READY = True  # process-local, after a real read; never runs DDL
    if args.apply:
        with db.SessionLocal() as session, session.begin():
            result = KnowledgeIngestionService().ingest_documents(session, documents, embed=False)
        report["ingestion"] = asdict(result)
        print(json.dumps({"ingestion": report["ingestion"]}), flush=True)
    from app.services.llm.jobs import model_job, vector_identity
    if args.vectors:
        with db.SessionLocal() as session:
            identity = vector_identity(session, ids)
        with model_job("official-coverage-vectors", args, identity):
            result = backfill_missing_vectors(document_ids=ids)
        report["vectors"] = result.as_dict()
        print(json.dumps({"vectors": report["vectors"]}), flush=True)
    if args.verify:
        from app.services.rag.coverage_check import verify_coverage
        report["coverage"] = verify_coverage(catalog, documents, args)
    if args.apply or args.vectors or args.verify:
        from app.services.rag.ingestion import plan_vector_backfill
        with db.SessionLocal() as session:
            report['index_pending_chunks'] = len(plan_vector_backfill(session, ids))
        report['index_ready'] = report['index_pending_chunks'] == 0
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"operations_finished": True, "index_ready": report.get("index_ready"), "report": str(args.report) if args.report else None}), flush=True)
    coverage = report.get("coverage", {})
    return 1 if (report.get("vectors", {}).get("pending")
        or coverage.get('stopped_reason')
        or coverage.get("live_passed", 0) < coverage.get("live_total", 0)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
