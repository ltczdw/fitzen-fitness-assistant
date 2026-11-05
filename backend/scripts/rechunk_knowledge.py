"""Preview by default. --apply backs up each old document before atomic replacement."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import select
from app.db import session as db
from app.db.models import KnowledgeDocumentModel as Document, KnowledgeChunkModel as Chunk
from app.services.rag.chunking import split_document, CHUNKER_VERSION
from app.services.rag.rechunk import prepare_rechunk, replace_chunks
from app.core.config import settings
from app.services.llm.jobs import add_job_arguments, model_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--document-id')
    parser.add_argument('--backup-dir', type=Path, default=None,
                        help='Backup folder (defaults to backups/sections-v2-TIMESTAMP).')
    add_job_arguments(parser)
    args = parser.parse_args()
    if db.SessionLocal is None:
        raise SystemExit('Database is not configured.')
    # Preview is read-only: do not run schema migration implicitly.
    with db.SessionLocal() as session:
        query = select(Document).order_by(Document.document_id)
        if args.document_id:
            query = query.where(Document.document_id == args.document_id)
        documents = list(session.scalars(query))
    if args.document_id and not documents:
        raise SystemExit('Document not found.')
    def needs_rechunk(document):
        try:
            metadata = json.loads(document.metadata_json or '{}')
        except (TypeError, ValueError):
            return True
        return not isinstance(metadata, dict) or metadata.get('chunker_version') != CHUNKER_VERSION
    pending = [d for d in documents if needs_rechunk(d)]
    backup = args.backup_dir or (Path(__file__).resolve().parents[2] / 'backups' / ('sections-v2-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    failures = []
    with model_job('rechunk', args, {'sources':[(d.document_id,d.content_hash) for d in documents],'version':CHUNKER_VERSION,'model':settings.retrieval_embedding_model,'index_version':settings.retrieval_embedding_index_version}):
        for document in pending:
            try:
                size = len(split_document(document.content, title=document.title,
                    chunk_size=settings.rag_chunk_size, overlap=settings.rag_chunk_overlap))
                print(json.dumps({'document_id': document.document_id, 'new_chunks': size, 'apply': args.apply}), flush=True)
                if not args.apply:
                    continue
                prepared = prepare_rechunk(document)  # Failure leaves old chunks intact.
                with db.SessionLocal() as session:
                    chunks = list(session.scalars(select(Chunk).where(Chunk.document_id == document.document_id)))
                    def row_data(row):
                        return {c.name: (getattr(row, c.name).tolist() if hasattr(getattr(row, c.name), 'tolist') else getattr(row, c.name)) for c in row.__table__.columns}
                    data = {'document': row_data(document), 'chunks': [row_data(c) for c in chunks]}
                backup.mkdir(parents=True, exist_ok=True)
                import hashlib
                filename = hashlib.sha256(document.document_id.encode()).hexdigest() + '.json'
                with (backup / filename).open('x', encoding='utf-8') as output:
                    json.dump(data, output, ensure_ascii=False, default=str)
                replace_chunks(db.SessionLocal, document, prepared)
            except Exception as exc:
                failures.append({'document_id': document.document_id, 'error_type': type(exc).__name__})
                print(json.dumps({'document_id': document.document_id, 'status': 'failed', 'error_type': type(exc).__name__}), flush=True)
                continue
    print(json.dumps({'pending_documents': len(pending), 'applied': args.apply,
                      'backup': str(backup) if args.apply and pending else None,
                      'failures': failures}))
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
