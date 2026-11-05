"""Explicit, per-document atomic upgrade; preparation never deletes old chunks."""
import json
import math
from sqlalchemy import select, delete
from app.core.config import settings
from app.db.models import KnowledgeDocumentModel as Document, KnowledgeChunkModel as Chunk
from app.services.rag.chunking import split_document, CHUNKER_VERSION
from app.services.rag.ingestion import content_hash
from app.services.rag.embedding_client import embedding_client


def prepare_rechunk(document):
    chunks = split_document(document.content, title=document.title,
                            chunk_size=settings.rag_chunk_size, overlap=settings.rag_chunk_overlap)
    # Collapse exact repeated content within the same heading to honor the index's unique hash.
    pieces = {content_hash(document.title + '\n' + c.section_path, c.content): c for c in chunks}
    if not pieces:
        raise ValueError('document_has_no_body')
    vectors = embedding_client.embed([c.section_path + '\n' + c.content for c in pieces.values()])
    if len(vectors) != len(pieces):
        raise ValueError('missing_vectors')
    return [(digest, piece, vector) for (digest, piece), vector in zip(pieces.items(), vectors, strict=True)]


def replace_chunks(factory, snapshot, prepared):
    """Caller saves a backup first. A source change aborts the entire transaction."""
    for _, _, vector in prepared:
        if len(vector) != 1024 or not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in vector) or not any(vector):
            raise ValueError('invalid_vector')
    with factory() as session, session.begin():
        current = session.scalar(select(Document).where(
            Document.document_id == snapshot.document_id).with_for_update())
        if current is None or any(getattr(current, field) != getattr(snapshot, field)
                                  for field in ('content', 'title', 'content_hash', 'metadata_json', 'source')):
            raise ValueError('document_changed_during_rechunk')
        if not prepared:
            raise ValueError('empty_replacement')
        metadata = json.loads(current.metadata_json or '{}')
        metadata = metadata if isinstance(metadata, dict) else {}
        metadata['chunker_version'] = CHUNKER_VERSION
        session.execute(delete(Chunk).where(Chunk.document_id == current.document_id))
        for i, (digest, chunk, vector) in enumerate(prepared):
            session.add(Chunk(document_id=current.document_id, parent_id=current.document_id,
                title=current.title, source=current.source, content=chunk.content,
                chunk_index=i, chunk_count=len(prepared), content_hash=digest,
                metadata_json=json.dumps({**metadata, 'section_path': chunk.section_path,
                                          'content_type': chunk.content_type}, ensure_ascii=False),
                semantic_embedding=vector, embedding_version=settings.retrieval_embedding_index_version))
        current.metadata_json = json.dumps(metadata, ensure_ascii=False)
    return len(prepared)
