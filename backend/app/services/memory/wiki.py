"""Stable Wiki addressing: hierarchy, sections, evidence, revisions, links and aliases."""
import json
from uuid import uuid4,uuid5,NAMESPACE_URL
from sqlalchemy import select,or_,delete
from app.db.wiki_models import WikiPage,WikiSection,WikiEvidence,WikiFact,WikiRevision,WikiLink,WikiAlias
from app.db.memory_models import MemoryTurn,UserFact
from app.core.config import settings
from app.services.rag.retrieval import RetrievedChunk
from app.services.rag.bm25 import BM25Index
from app.services.memory.compact import session,lock_profile,now,digest

DIRECTORIES=[('index','个人记忆',None),('profile','个人档案','index'),('goals','目标','index'),
 ('constraints','限制与注意事项','index'),('training','训练','index'),
 ('training/preferences','训练偏好','training'),('training/current-plan','当前训练计划','training'),
 ('nutrition','饮食','index'),('nutrition/preferences','饮食偏好','nutrition'),
 ('episodes','历史事件','index'),('conversations','会话','index')]

def uid(key):return str(uuid5(NAMESPACE_URL,'fitzen:wiki:'+key))
def snapshot(s,page):
    sections=list(s.scalars(select(WikiSection).where(WikiSection.page_id==page.page_id).order_by(WikiSection.position)))
    s.add(WikiRevision(id=str(uuid4()),page_id=page.page_id,revision=page.revision,
        snapshot={'title':page.title,'path':page.path,'content':page.content,'summary':page.summary,
                  'sections':[{'section_id':x.section_id,'title':x.title,'content':x.content,'source_message_id':x.source_message_id} for x in sections]}))

def link(s,source,target,relation):
    id=uid(source+':'+target+':'+relation)
    if not s.get(WikiLink,id):s.add(WikiLink(id=id,from_page_id=source,to_page_id=target,relation=relation))

def ensure_directory(s,path,title,parent=None):
    page=s.scalar(select(WikiPage).where(WikiPage.path==path))
    if page:return page
    parent_page=s.scalar(select(WikiPage).where(WikiPage.path==parent)) if parent else None
    page=WikiPage(page_id=uid('dir:'+path),path=path,title=title,kind='directory',
                  parent_id=parent_page.page_id if parent_page else None,content='',summary='',revision=1,memory_epoch=0)
    s.add(page);s.flush();snapshot(s,page)
    if parent_page:link(s,parent_page.page_id,page.page_id,'child')
    return page

def ensure_tree(s):
    for path,title,parent in DIRECTORIES:ensure_directory(s,path,title,parent)

def initialize():
    with session() as s,s.begin():lock_profile(s);ensure_tree(s)

def fact_directory(field,value):
    text=field+' '+value
    if any(t in text for t in ('不能','过敏','限制','受伤','禁忌')):return 'constraints'
    if any(t in text for t in ('目标','减脂','增肌')):return 'goals'
    if any(t in text for t in ('吃','喝','饮食','忌口')):return 'nutrition/preferences'
    return 'training/preferences'

def materialize(s,turn):
    if turn.deleted_at or turn.hidden_at:return None
    ensure_tree(s)
    month=(turn.completed_at or turn.started_at).strftime('%Y-%m')
    ensure_directory(s,'episodes/'+month,month,'episodes')
    cp='conversations/'+turn.conversation_id
    ensure_directory(s,cp,'会话 '+turn.conversation_id,'conversations')
    parent=ensure_directory(s,cp+'/turns','对话原文',cp)
    page=s.scalar(select(WikiPage).where(WikiPage.source_turn_id==turn.turn_id))
    content=json.dumps(turn.messages,ensure_ascii=False)
    summary=turn.retrieval_summary or ''
    if page and page.content==content and page.summary==summary and page.deleted_at is None:
        sync_facts(s,turn,page)
        return page
    if page is None:
        page=WikiPage(page_id=uid('turn:'+turn.turn_id),path=cp+'/turns/'+turn.turn_id,
            title=turn.messages[0]['content'][:80],parent_id=parent.page_id,kind='memory',
            source_turn_id=turn.turn_id,revision=1,memory_epoch=turn.memory_epoch)
        s.add(page);s.flush()
    else:page.revision+=1
    page.content=content;page.summary=summary;page.summary_embedding=turn.summary_embedding
    page.embedding_version=turn.embedding_version;page.memory_epoch=turn.memory_epoch;page.deleted_at=None
    for i,m in enumerate(turn.messages):
        section_id=uid(page.page_id+':'+m['message_id']);section=s.get(WikiSection,section_id)
        if section is None:
            section=WikiSection(section_id=section_id,page_id=page.page_id,title=m['role'],content=m['content'],position=i,source_message_id=m['message_id'])
            s.add(section);s.flush()
        else:section.content=m['content'];section.position=i
        evidence_id=uid('evidence:'+section_id)
        evidence=s.get(WikiEvidence,evidence_id)
        if evidence is None:s.add(WikiEvidence(evidence_id=evidence_id,page_id=page.page_id,section_id=section_id,
            turn_id=turn.turn_id,message_id=m['message_id'],source_hash=digest(m)))
        else:evidence.source_hash=digest(m)
    s.flush();snapshot(s,page)
    link(s,parent.page_id,page.page_id,'child')
    month_page=s.scalar(select(WikiPage).where(WikiPage.path=='episodes/'+month))
    link(s,month_page.page_id,page.page_id,'episode')
    sync_facts(s,turn,page)
    return page

def sync_facts(s,turn,page):
    # A replacement may invalidate facts from another turn. Keep the Wiki
    # projection in the same transaction as the authoritative UserFact rows.
    first = turn.messages[0] if turn.messages else {}
    update = (first.get('app_meta') or {}).get('fact_update') or {}
    for fact_id in update.get('supersedes', []):
        fact = s.get(UserFact, fact_id)
        projection = s.get(WikiFact, fact_id)
        if fact is not None and projection is not None:
            projection.status = fact.status
            projection.valid_to = fact.valid_to
    for f in s.scalars(select(UserFact).where(UserFact.source_turn_id==turn.turn_id)):
        wf=s.get(WikiFact,f.fact_id)
        if wf is None:
            s.add(WikiFact(fact_id=f.fact_id,page_id=page.page_id,field=f.field,value=f.value,
                source_message_id=f.source_message_id,status=f.status,valid_from=f.valid_from,valid_to=f.valid_to))
        else:wf.status=f.status;wf.valid_to=f.valid_to
        topic=s.scalar(select(WikiPage).where(WikiPage.path==fact_directory(f.field,f.value)))
        link(s,topic.page_id,page.page_id,'evidence')

def backfill():
    with session() as s,s.begin():
        lock_profile(s);ensure_tree(s);count=0
        for turn in s.scalars(select(MemoryTurn).where(MemoryTurn.status=='completed',MemoryTurn.deleted_at.is_(None),MemoryTurn.hidden_at.is_(None))):
            materialize(s,turn);count+=1
        return count

def purge_sources(s,turn_ids):
    from app.db.memory_models import MemoryState
    s.execute(delete(MemoryState).where(MemoryState.source_turn_id.in_(turn_ids)))
    ids=list(s.scalars(select(WikiPage.page_id).where(WikiPage.source_turn_id.in_(turn_ids))))
    for id in ids:
        page=s.get(WikiPage,id);page.content='';page.summary='';page.title='已删除';page.summary_embedding=None
        page.deleted_at=now();page.revision+=1
    for model in (WikiFact,WikiEvidence,WikiSection,WikiRevision,WikiAlias):
        s.execute(delete(model).where(model.page_id.in_(ids)))
    s.execute(delete(WikiLink).where(or_(WikiLink.from_page_id.in_(ids),WikiLink.to_page_id.in_(ids))))

def locate(s,address):
    page=s.get(WikiPage,address)
    if page is None:page=s.scalar(select(WikiPage).where(WikiPage.path==address))
    if page is None:
        alias=s.get(WikiAlias,address);page=s.get(WikiPage,alias.page_id) if alias else None
    if page is None or page.deleted_at:raise KeyError('page_not_found')
    return page

def page_data(s,p):
    return {'page_id':p.page_id,'path':p.path,'title':p.title,'kind':p.kind,'revision':p.revision,
            'parent_id':p.parent_id,'content':p.content,'source_turn_id':p.source_turn_id,
            'sections':[{'section_id':x.section_id,'title':x.title,'content':x.content,'source_message_id':x.source_message_id}
                for x in s.scalars(select(WikiSection).where(WikiSection.page_id==p.page_id).order_by(WikiSection.position))],
            'evidence':[{'turn_id':x.turn_id,'message_id':x.message_id,'section_id':x.section_id}
                for x in s.scalars(select(WikiEvidence).where(WikiEvidence.page_id==p.page_id))]}

def read_page(address,follow_links=False,section_id=None,revision=None):
    with session() as s:
        p=locate(s,address);result=page_data(s,p)
        if revision is not None:
            rev=s.scalar(select(WikiRevision).where(WikiRevision.page_id==p.page_id,WikiRevision.revision==revision))
            if rev is None:raise KeyError('revision_not_found')
            result={**result,**rev.snapshot,'revision':revision}
        if section_id:
            found=[x for x in result['sections'] if x['section_id']==section_id]
            if not found:raise KeyError('section_not_found')
            result['sections']=found;result['content']=found[0]['content']
        links=list(s.scalars(select(WikiLink).where(WikiLink.from_page_id==p.page_id).limit(20)))
        result['links']=[{'page_id':x.to_page_id,'relation':x.relation} for x in links]
        if follow_links:
            result['linked_pages']=[page_data(s,q) for x in links if (q:=s.get(WikiPage,x.to_page_id)) and not q.deleted_at]
        if p.kind=='directory':
            target=[x.to_page_id for x in links]
            result['facts']=[{'fact_id':f.fact_id,'field':f.field,'value':f.value,'source_message_id':f.source_message_id,
                             'source_page_id':f.page_id} for f in s.scalars(select(WikiFact).where(WikiFact.page_id.in_(target),WikiFact.status=='active',WikiFact.valid_from<=now(),or_(WikiFact.valid_to.is_(None),WikiFact.valid_to>now())))]
        if p.path in ('profile','training/current-plan'):
            from app.db.models import UserProfileModel,TrainingPlanModel
            from app.db.profile_repository import _serialize as serialize_profile
            from app.db.plan_repository import _serialize as serialize_plan
            if p.path=='profile':
                row=s.get(UserProfileModel,'demo-user')
                result['business_view']=serialize_profile(row) if row else None
            else:
                row=s.scalar(select(TrainingPlanModel).where(TrainingPlanModel.user_id=='demo-user'))
                result['business_view']=serialize_plan(row) if row else None
            result['business_view_source']='postgresql'
        return result

def edit_page(address,expected_revision,title=None,path=None,summary=None):
    # Network outside SQL transaction; re-check revision and deletion after embedding.
    before=read_page(address)
    vector=None
    if summary is not None:
        from app.services.rag.embedding_client import embedding_client
        vector=embedding_client.embed([summary])[0]
    with session() as s,s.begin():
        lock_profile(s);p=locate(s,address)
        if p.revision!=expected_revision:raise ValueError('revision_conflict')
        if path and path!=p.path:
            if p.kind=='directory':raise ValueError('directory_rename_not_supported')
            if s.scalar(select(WikiPage).where(WikiPage.path==path)) or s.get(WikiAlias,path):raise ValueError('path_already_used')
            parent_path=path.rsplit('/',1)[0] if '/' in path else 'index'
            parent=locate(s,parent_path)
            if parent.kind!='directory':raise ValueError('parent_must_be_directory')
            s.add(WikiAlias(alias=p.path,page_id=p.page_id));p.path=path
            s.execute(delete(WikiLink).where(WikiLink.to_page_id==p.page_id,WikiLink.relation=='child'))
            p.parent_id=parent.page_id;link(s,parent.page_id,p.page_id,'child')
        if title is not None:p.title=title
        if summary is not None:
            if p.kind=='directory':raise ValueError('directory_not_indexable')
            p.summary=summary;p.summary_embedding=vector;p.embedding_version=settings.retrieval_embedding_index_version
            source=s.get(MemoryTurn,p.source_turn_id)
            if source:
                source.retrieval_summary=summary;source.summary_embedding=vector;source.embedding_version=p.embedding_version
        p.revision+=1;s.flush();snapshot(s,p);return page_data(s,p)

def correct_fact(fact_id,value,source_turn_id,source_message_id):
    with session() as s,s.begin():
        profile=lock_profile(s);old=s.get(UserFact,fact_id);turn=s.get(MemoryTurn,source_turn_id)
        if old is None or old.status!='active' or turn is None or turn.deleted_at or turn.hidden_at:raise KeyError('fact_or_source_not_found')
        msg=next((m for m in turn.messages if m['message_id']==source_message_id and m['role']=='user'),None)
        if not msg or value not in msg['content']:raise ValueError('value_requires_user_source')
        old.status='superseded';old.valid_to=now()
        oldwiki=s.get(WikiFact,fact_id)
        if oldwiki:oldwiki.status=old.status;oldwiki.valid_to=old.valid_to
        id=str(uuid4());new=UserFact(fact_id=id,field=old.field,value=value,source_turn_id=source_turn_id,
            source_message_id=source_message_id,source_quote=value,memory_epoch=profile.memory_epoch)
        s.add(new);s.flush()
        from app.services.memory.lifecycle import transfer_constraint
        transfer_constraint(s,old,new,owner=profile.user_id,epoch=profile.memory_epoch)
        p=materialize(s,turn)
        wf=s.get(WikiFact,id)
        if wf is None:s.add(WikiFact(fact_id=id,page_id=p.page_id,field=old.field,value=value,source_message_id=source_message_id,
            valid_from=now(),supersedes=fact_id))
        else:wf.supersedes=fact_id
        return {'fact_id':id,'supersedes':fact_id,'page_id':p.page_id}

class WikiAdapter:
    corpus='memory'
    def __init__(self,exclude=()):
        self.exclude=set(exclude)
        with session() as s,s.begin():self.epoch=lock_profile(s).memory_epoch
    def filters(self):
        return [WikiPage.kind=='memory',WikiPage.deleted_at.is_(None),WikiPage.summary!='',
                WikiPage.source_turn_id.not_in(self.exclude)]
    @staticmethod
    def candidate(p,score):
        return RetrievedChunk(p.title,p.summary,score,'wiki',{'page_id':p.page_id,'revision':p.revision,
            'turn_id':p.source_turn_id,'path':p.path},p.page_id)
    def lexical(self,query,k):
        with session() as s:
            pages=list(s.scalars(select(WikiPage).where(*self.filters())))
            if not pages:return []
            idx=BM25Index([p.title+' '+p.summary for p in pages])
            return [self.candidate(pages[i],score) for i,score in idx.rank(query,k) if score>0]
    def vector(self,vector,k):
        with session() as s:
            d=WikiPage.summary_embedding.cosine_distance(vector)
            rows=s.execute(select(WikiPage,d).where(*self.filters(),WikiPage.summary_embedding.is_not(None),
                WikiPage.embedding_version==settings.retrieval_embedding_index_version).order_by(d).limit(k))
            return [self.candidate(p,1-float(distance)) for p,distance in rows]
    def resolve(self,candidates):
        with session() as s,s.begin():
            if lock_profile(s).memory_epoch!=self.epoch:return []
            result=[]
            for hit in candidates:
                p=s.get(WikiPage,hit.chunk_id)
                if p and not p.deleted_at and p.revision==hit.metadata['revision']:
                    source=s.get(MemoryTurn,p.source_turn_id)
                    if source is None or source.deleted_at or source.hidden_at:continue
                    result.append(RetrievedChunk(p.title,p.content,hit.score,'wiki',
                        {**hit.metadata,'source_ids':[m['message_id'] for m in source.messages],'memory_epoch':self.epoch},p.page_id))
            return result
    def validate(self,hits):
        valid={h.chunk_id for h in self.resolve(hits)}
        return [h for h in hits if h.chunk_id in valid]
