"""Preview (default) or fill missing/outdated knowledge vectors.

Only vectors are touched: parents and chunks are never re-inserted, healthy
vectors are never recomputed, and the embedding network call happens outside
the write transaction. Failures leave the lexical index usable.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import session as db
from app.services.llm.jobs import add_job_arguments, model_job, vector_identity
from app.services.rag.ingestion import backfill_missing_vectors, plan_vector_backfill


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Fill the pending vectors instead of only previewing.")
    parser.add_argument("--document-id", action="append", default=None,
                        help="Restrict the run to one or more document ids.")
    add_job_arguments(parser)
    args = parser.parse_args()
    if db.SessionLocal is None:
        raise SystemExit("Database is not configured.")
    with db.SessionLocal() as session:
        candidates = plan_vector_backfill(session, args.document_id)
        identity = vector_identity(session,args.document_id)
    preview = {
        "chunks_pending": len(candidates),
        "documents": sorted({candidate["document_id"] for candidate in candidates if candidate["document_id"]}),
        "apply": args.apply,
    }
    print(json.dumps(preview, ensure_ascii=False), flush=True)
    if not args.apply or not candidates:
        return 0
    with model_job('vector-backfill', args, identity) as budget:
        result = backfill_missing_vectors(document_ids=args.document_id)
    print(json.dumps(result.as_dict(), ensure_ascii=False), flush=True)
    return 1 if result.pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
