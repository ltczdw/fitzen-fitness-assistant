from fastapi import APIRouter,HTTPException,Query
from pydantic import BaseModel,Field
from app.services.memory import wiki
router=APIRouter(prefix='/wiki',tags=['wiki'])
class EditPage(BaseModel):
    expected_revision:int=Field(ge=1)
    title:str|None=Field(default=None,min_length=1,max_length=200)
    path:str|None=Field(default=None,min_length=1,max_length=240,pattern=r'^[a-zA-Z0-9_/-]+$')
    summary:str|None=Field(default=None,min_length=1,max_length=3000)
class CorrectFact(BaseModel):
    value:str=Field(min_length=1,max_length=2000)
    source_turn_id:str
    source_message_id:str

def call(fn,*args,**kwargs):
    try:return fn(*args,**kwargs)
    except KeyError as e:raise HTTPException(404,str(e)) from None
    except ValueError as e:raise HTTPException(409,str(e)) from None
@router.get('/pages')
def read(address:str='index',follow_links:bool=False,section_id:str|None=None,revision:int|None=Query(None,ge=1)):
    return call(wiki.read_page,address,follow_links,section_id,revision)
@router.patch('/pages/{page_id}')
def edit(page_id:str,body:EditPage):
    return call(wiki.edit_page,page_id,**body.model_dump())
@router.post('/facts/{fact_id}/corrections')
def correct(fact_id:str,body:CorrectFact):
    return call(wiki.correct_fact,fact_id,**body.model_dump())

@router.get('/status')
def status():
    from sqlalchemy import select,func
    from app.db.models import KnowledgeChunkModel as K
    from app.db.wiki_models import WikiPage as P
    from app.db.memory_models import MemoryOutbox
    from app.core.config import settings
    from app.services.rag.model_reranker import model_reranker
    with wiki.session() as s:
        count=lambda model,*conditions:s.scalar(select(func.count()).select_from(model).where(*conditions))
        return {'knowledge_total':count(K),'knowledge_qwen_ready':count(K,K.semantic_embedding.is_not(None),K.embedding_version==settings.retrieval_embedding_index_version),
                'wiki_pages':count(P,P.deleted_at.is_(None)),'wiki_indexed_pages':count(P,P.deleted_at.is_(None),P.summary_embedding.is_not(None)),
                'outbox_pending':count(MemoryOutbox,MemoryOutbox.status.in_(['pending','running','retry'])),
                'outbox_dead_letter':count(MemoryOutbox,MemoryOutbox.status=='dead_letter'),
                'reranker_model':settings.rerank_model,'reranker_last_status':model_reranker.last_status}
