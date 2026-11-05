"""Explicit bounded CLI jobs; previews create no ledger rows or model calls."""
from contextlib import contextmanager
import json
from app.services.llm.budget import model_scope,key
from app.core.config import settings


def add_job_arguments(parser):
    parser.add_argument('--job-id',help='Run label; renaming the same inputs never replenishes their budget.')
    parser.add_argument('--max-model-calls',type=int,default=settings.model_cli_max_calls)


def vector_identity(session,document_ids=None):
    from sqlalchemy import select
    from app.db.models import KnowledgeChunkModel
    from app.services.rag.ingestion import _chunk_embedding_text
    query=select(KnowledgeChunkModel).order_by(KnowledgeChunkModel.id)
    if document_ids:query=query.where(KnowledgeChunkModel.document_id.in_(document_ids))
    # Include all selected source inputs, not just unfinished vectors. Saving
    # progress does not change job identity; editing the source really does.
    sources=[(row.id,key(_chunk_embedding_text(row))) for row in session.scalars(query)]
    return {'sources':sources,'model':settings.retrieval_embedding_model,'index_version':settings.retrieval_embedding_index_version}


@contextmanager
def model_job(kind,arguments,identity):
    maximum=arguments.max_model_calls
    if not 1<=maximum<=settings.model_cli_max_calls:raise ValueError('max-model-calls exceeds configured ceiling')
    name=key(json.dumps(identity,sort_keys=True,default=str))
    with model_scope(kind+':'+name,kind='cli',max_calls=maximum) as scope:yield scope
