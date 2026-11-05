import os,json,unittest
from uuid import uuid4
from unittest.mock import patch,Mock
from contextlib import nullcontext
from sqlalchemy import text,select,func
from quality_isolation import checked_engine as create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.db import session as db
from app.db.models import Base
from app.db.memory_models import MemoryTurn,MemoryOutbox
from app.db.wiki_models import WikiPage,WikiRevision,WikiSection,WikiEvidence,WikiLink
from app.services.memory import compact,wiki
from app.services.rag.retrieval import RetrievedChunk,retrieval_service
from app.services.rag.model_reranker import ModelReranker,RerankUnavailable
from app.services.agent.task_planner import plan_tasks
from app.schemas.chat import ChatRequest

class PlannerContractTests(unittest.TestCase):
    """The live Planner contract: model labels execute, but never invent or negate a write."""
    def spec(self,task_type,text,action='record',**extra):
        item={'task_type':task_type,'action':action,'text':text,'foods':[],'workouts':[]}
        item.update(extra)
        return item
    def test_planner_labels_drive_plan_without_keyword_reclassification(self):
        message='记录两个鸡蛋；评估一下是否适合今天的目标'
        specs=[self.spec('nutrition_record','记录两个鸡蛋'),
               self.spec('nutrition_advice','评估一下是否适合今天的目标',action='question')]
        # The rule classifier would reclassify “记录两个鸡蛋” as a workout; a valid
        # planner label must win, so this path may not call it at all.
        with patch('app.services.agent.task_planner._task_for_clause',side_effect=AssertionError('must not reclassify')):
            result=plan_tasks(message,parsed_input={'tasks':specs})
        self.assertEqual([x['task_type'] for x in result],['nutrition_record','nutrition_advice'])
        self.assertEqual(result[1]['depends_on'],['task-1'])
    def test_generated_task_text_cannot_write(self):
        message='记录两个鸡蛋'
        generated={'tasks':[self.spec('nutrition_record','记录十个鸡蛋')]}
        planned=plan_tasks(message,parsed_input=generated)
        reference=plan_tasks(message)
        fields=('task_type','text','side_effect')
        self.assertEqual([tuple(x.get(f) for f in fields) for x in planned],
                         [tuple(x.get(f) for f in fields) for x in reference])
        self.assertTrue(all(x['text']!='记录十个鸡蛋' for x in planned))
    def test_negated_record_cannot_write(self):
        message='不要记录鸡蛋'
        planned=plan_tasks(message,parsed_input={'tasks':[
            self.spec('nutrition_record',message,foods=[{'name':'鸡蛋'}],is_negated=True)]})
        self.assertTrue(planned)
        self.assertTrue(all(x['side_effect'] is False for x in planned))
        self.assertTrue(all(x['payload'].get('write_authorization')!='authorized' for x in planned))

class ModelRerankerTests(unittest.TestCase):
    def setUp(self):
        # Supply the actual reranker configuration; embedding credentials are unrelated.
        self.enterContext(patch.object(settings, 'rerank_base_url', 'https://rerank.fixture.invalid'))
        self.enterContext(patch.object(settings, 'rerank_api_key', 'fixture-key'))
        self.enterContext(patch.object(settings, 'rerank_model', 'qwen3.7-text-rerank'))
        self.enterContext(patch.object(settings, 'rerank_provider', 'dashscope'))
    def test_rerank_returns_original_sources_and_filters_noise(self):
        original='完整训练原文：卧推10kg，共4组。'
        chunks=[RetrievedChunk('训练',original,.01,chunk_id='a'),RetrievedChunk('天气','带伞',.05,chunk_id='b')]
        response=Mock(status_code=200);response.json.return_value={'output':{'results':[{'index':0,'relevance_score':.9},{'index':1,'relevance_score':.01}]}}
        client=Mock();client.__enter__=Mock(return_value=client);client.__exit__=Mock(return_value=False);client.post.return_value=response
        response.iter_bytes.return_value=[json.dumps(response.json.return_value).encode()]
        client.stream.return_value=nullcontext(response)
        with patch.object(settings,'rerank_enabled',True),patch.object(settings,'retrieval_embedding_api_key','test'),patch('app.services.rag.model_reranker.httpx.Client',return_value=client):
            result=ModelReranker().rerank('卧推怎么练',chunks)
        self.assertEqual([x.chunk_id for x in result],['a'])
        self.assertEqual(result[0].content,original)
        self.assertIn(original,client.stream.call_args.kwargs['json']['input']['documents'][0])
    def test_invalid_indexes_fail_explicitly(self):
        response=Mock(status_code=200);response.json.return_value={'output':{'results':[{'index':3,'relevance_score':.9}]}}
        client=Mock();client.__enter__=Mock(return_value=client);client.__exit__=Mock(return_value=False);client.post.return_value=response
        response.iter_bytes.return_value=[json.dumps(response.json.return_value).encode()]
        client.stream.return_value=nullcontext(response)
        with patch.object(settings,'rerank_enabled',True),patch.object(settings,'retrieval_embedding_api_key','test'),patch('app.services.rag.model_reranker.httpx.Client',return_value=client):
            with self.assertRaises(RerankUnavailable):
                ModelReranker().rerank('q',[RetrievedChunk('a','a',0)])
        client.stream.assert_called_once()  # Config rejection would be a false positive.

@unittest.skipUnless(os.getenv('FITZEN_DB_TESTS')=='1','explicit isolated PostgreSQL integration tests')
class WikiDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema='fitzen_test_'+uuid4().hex
        # Raw SQL consumers need the same scratch schema as ORM tables.
        cls.raw=create_engine(settings.database_url, connect_args={
            'options':f'-csearch_path={cls.schema},public'})
        with cls.raw.begin() as c:c.execute(text('CREATE SCHEMA '+cls.schema))
        cls.engine=cls.raw.execution_options(schema_translate_map={None:cls.schema})
        Base.metadata.create_all(cls.engine)
        from app.db.food_repository import migrate_food_schema
        with cls.engine.begin() as c:migrate_food_schema(c)
    @classmethod
    def tearDownClass(cls):
        # Only the random schema created by this fixture is removed.
        assert cls.schema.startswith('fitzen_test_') and len(cls.schema)==44
        with cls.raw.begin() as c:c.execute(text('DROP SCHEMA '+cls.schema+' CASCADE'))
        cls.raw.dispose()
    def setUp(self):
        self.connection=self.engine.connect();self.transaction=self.connection.begin()
        self.factory=sessionmaker(bind=self.connection,expire_on_commit=False,join_transaction_mode='create_savepoint')
        self.patches=[patch.object(db,'DB_READY',True),patch.object(db,'SessionLocal',self.factory)]
        for p in self.patches:p.start()
        wiki.initialize()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.transaction.rollback();self.connection.close()
    def turn(self,n=0,message=None,semantic=None):
        message=message or f'第{n}次卧推10kg共4组'
        req=ChatRequest(message=message,conversation_id='fixture',request_id=f'fixture-{n}')
        tid,_=compact.begin_turn(req)
        compact.finish_turn(tid,'助手原始完整回答：逐步增加负荷。',
                            semantic_values=[(message,semantic)] if semantic else ())
        return tid
    def test_window_and_atomic_outbox_boundary(self):
        ids=[self.turn(n) for n in range(11)]
        self.assertEqual([r.turn_id for r in compact.recent_rows('fixture')],ids[1:])
        with self.factory() as s:
            jobs=list(s.scalars(select(MemoryOutbox)))
            self.assertEqual(len(jobs),1);self.assertEqual(jobs[0].turn_id,ids[0])
            self.assertEqual(s.scalar(select(func.count()).select_from(WikiPage).where(WikiPage.kind=='memory')),11)
    def test_page_section_revision_alias_and_conflict(self):
        tid=self.turn();page=wiki.read_page('conversations/fixture/turns/'+tid)
        original=page['content'];sid=page['sections'][0]['section_id']
        edited=wiki.edit_page(page['page_id'],1,title='新的标题',path='training/renamed')
        self.assertEqual(edited['revision'],2)
        self.assertEqual(wiki.read_page(page['path'])['page_id'],page['page_id'])
        self.assertEqual(wiki.read_page(page['page_id'],revision=1)['content'],original)
        self.assertEqual(wiki.read_page(page['page_id'],section_id=sid)['content'],'第0次卧推10kg共4组')
        with self.assertRaises(ValueError):wiki.edit_page(page['page_id'],1,title='stale')
    def test_full_original_resolved_before_model(self):
        tid=self.turn()
        with self.factory() as s,s.begin():
            row=s.get(MemoryTurn,tid);row.retrieval_summary='卧推训练摘要';wiki.materialize(s,row)
        adapter=wiki.WikiAdapter();candidates=adapter.lexical('卧推',10)
        hits=adapter.resolve(candidates)
        self.assertIn('助手原始完整回答',hits[0].content)
        self.assertNotIn('卧推训练摘要',hits[0].content)
        with patch('app.services.rag.retrieval.embedding_client.embed',return_value=[[1.0]+[0.0]*1023]),patch.object(adapter,'vector',return_value=[]),patch('app.services.rag.model_reranker.model_reranker.rerank',side_effect=lambda q,h,k:h[:k]) as model:
            result=retrieval_service.search('卧推',adapter)
        self.assertIn('助手原始完整回答',model.call_args.args[1][0].content)
        self.assertEqual(result[0].metadata['page_id'],hits[0].metadata['page_id'])
    def test_delete_scrubs_revisions_evidence_and_stale_jobs(self):
        ids=[self.turn(n) for n in range(11)]
        # This case verifies invalidation of an already claimed job. Make the
        # synthetic job due explicitly, independent of DB/app clock skew.
        from datetime import timedelta
        with self.factory() as s,s.begin():
            jobs=list(s.scalars(select(MemoryOutbox)))
            self.assertEqual(len(jobs),1)
            jobs[0].next_attempt_at=compact.now()-timedelta(seconds=1)
        event=compact.claim_event()
        self.assertIsNotNone(event,'The stale-job fixture must claim a real test outbox event')
        compact.delete_memory('fixture')
        self.assertFalse(compact.commit_index(event,'旧摘要',[1.0]+[0.0]*1023))
        with self.factory() as s:
            pages=list(s.scalars(select(WikiPage).where(WikiPage.kind=='memory')))
            self.assertTrue(all(p.deleted_at and not p.content and not p.summary for p in pages))
            self.assertEqual(s.scalar(select(func.count()).select_from(WikiEvidence)),0)
            self.assertEqual(s.scalar(select(func.count()).select_from(WikiSection)),0)
            self.assertEqual(s.scalar(select(func.count()).select_from(WikiRevision).where(WikiRevision.page_id.in_([p.page_id for p in pages]))),0)
        with self.assertRaises(KeyError):wiki.read_page(pages[0].page_id)
    def test_fact_correction_requires_user_evidence(self):
        from app.db.memory_models import UserFact
        from app.db.wiki_models import WikiFact
        # The live commit path creates a fact only from a validated semantic
        # decision. This test-owned decision is source-bound to this turn; the
        # ordinary turn() helper intentionally stays semantics-free.
        semantic={
            'memory_requests':[{'action':'remember','quote':'我每周五练',
                                'key':'schedule:weekly_training','target_fields':[],
                                'confidence':1}],
            'allowed_memory_fields':[],
            'memory_target_sources':{},
            'needs_clarification':False,
            'current_is_critical':False,
        }
        first=self.turn(1,'请记住：我每周五练',semantic=semantic)
        second=self.turn(2,'现在改为每周三练')
        with self.factory() as s:
            old=s.scalar(select(UserFact));fid=old.fact_id
            self.assertEqual(old.status,'active')
            self.assertEqual(old.source_quote,'我每周五练')
            self.assertEqual(old.source_message_id,first+':user')
            self.assertTrue(old.field.startswith('semantic:'))
            self.assertEqual(s.get(WikiFact,fid).status,'active')
        self.assertEqual(compact.current_facts()[old.field]['text'],'我每周五练')
        # Assistant messages are not user evidence.
        with self.assertRaises(ValueError):
            wiki.correct_fact(fid,'每周三练',second,second+':assistant')
        # A value absent from the real second user message cannot change facts.
        with self.assertRaises(ValueError):
            wiki.correct_fact(fid,'每周四练',second,second+':user')
        with self.factory() as s:
            unchanged=s.get(UserFact,fid)
            self.assertEqual(unchanged.status,'active')
            self.assertEqual(unchanged.valid_to,None)
            self.assertEqual(s.get(WikiFact,fid).status,'active')
        self.assertEqual(next(iter(compact.current_facts().values()))['text'],'我每周五练')
        # The real second user message now supplies the correction value.
        result=wiki.correct_fact(fid,'每周三练',second,second+':user')
        self.assertEqual(result['supersedes'],fid)
        values=compact.current_facts()
        self.assertEqual(next(iter(values.values()))['text'],'每周三练')
        with self.factory() as s:
            self.assertEqual(s.get(UserFact,fid).status,'superseded')
            self.assertEqual(s.get(WikiFact,fid).status,'superseded')
    def test_changed_epoch_drops_reranked_sources(self):
        tid=self.turn()
        with self.factory() as s,s.begin():
            row=s.get(MemoryTurn,tid);row.retrieval_summary='卧推';wiki.materialize(s,row)
        adapter=wiki.WikiAdapter();hits=adapter.resolve(adapter.lexical('卧推',3))
        self.assertTrue(hits)
        compact.delete_memory('fixture')
        self.assertEqual(adapter.validate(hits),[])

    def test_planned_record_executes_and_is_visible_in_frontend_api(self):
        import asyncio
        from unittest.mock import AsyncMock
        import httpx
        from app.main import app
        from app.services.agent import multi_agent_graph
        from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
        from app.services.agent.task_information import PlanInformation, check_plan_information
        message='记录哑铃卧推10kg每组10次共4组，然后给出通用胸部训练计划示例'
        specs=[{'id':'t1','task_type':'workout_record','worker':'workout_record','text':'记录哑铃卧推10kg每组10次共4组，','action':'record','side_effect':True,'depends_on':[],
                'payload':{'write_authorization':'authorized'}},
            {'id':'t2','task_type':'training_plan','worker':'plan_adjust','text':'给出通用胸部训练计划示例','action':'question','side_effect':False,'depends_on':['t1']}]
        graph=FitnessMultiAgentGraph()
        record={'exercise':'哑铃卧推','weight_kg':10.0,'sets':4,'reps':10,'note':''}
        plan_calls=[]
        async def plan_information(messages,schema,*,name=None,**kwargs):
            # Real check_plan_information validates this schema result against
            # the t2 task text and the full current request.
            plan_calls.append(name)
            self.assertEqual(schema,PlanInformation)
            self.assertEqual(name,'return_plan_information')
            task=json.loads(messages[-1]['content'])['task']
            self.assertIn('通用胸部训练计划示例',task)
            self.assertIn('通用胸部训练计划示例',message)
            return PlanInformation(scope='generic',scope_quote='通用胸部训练计划示例',
                                   facts=[],phase_limits=[])
        class _Answerable:
            answerable=True
            def as_dict(self):return {'answerable':True}
        async def run():
            # The scratch schema has no knowledge corpus, so the evidence gate is
            # stubbed here; this test covers execution, dependency order and the
            # frontend read model, not retrieval quality.
            with patch.object(graph.mcp,'call_tool',AsyncMock(return_value=None)),patch.object(graph,'_run_authorized_model_tool',AsyncMock(return_value=None)),patch.object(graph.workout_extraction,'run_many',AsyncMock(return_value=[{'record':record}])),patch.object(graph.rag,'search',return_value=[]),patch.object(multi_agent_graph,'assess_rag_evidence',lambda *a,**k:_Answerable()),patch.object(graph,'_complete_or_fallback_with_mcp',AsyncMock(return_value=('已根据记录制定建议',True))),patch.object(graph,'_load_context_node',AsyncMock(return_value={})),patch.object(graph.llm,'complete_structured',plan_information),patch('pathlib.Path.write_text',side_effect=AssertionError('SQL required')):
                result=await graph.run('demo-user','fixture',message,parsed_input={},planned_tasks=specs,request_id='unique-record-request')
                self.assertEqual(result['completed_tasks'],['t1','t2'])
                self.assertIn('已记录',result['final_answer'])
                self.assertEqual(plan_calls,['return_plan_information'])
                await graph.run('demo-user','fixture',message,parsed_input={},planned_tasks=specs,request_id='unique-record-request')
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                response=await client.get('/api/workout-sessions')
            self.assertEqual(response.status_code,200)
            sessions=response.json()
            self.assertEqual(len(sessions),1)
            exercise=sessions[0]['exercises'][0]
            self.assertEqual(exercise['exercise_name'],'哑铃卧推')
            self.assertEqual(len(exercise['sets']),4)
            self.assertTrue(all(x['weight_kg']==10 and x['reps']==10 for x in exercise['sets']))
            # Minimal negative with the same real validator: a generic scope
            # quote cannot be borrowed by a non-generic source.
            async def reject_unbound_generic():
                fresh=FitnessMultiAgentGraph()
                async def generic(messages,schema,*,name=None,**kwargs):
                    return PlanInformation(scope='generic',scope_quote='通用胸部训练计划示例',
                                           facts=[],phase_limits=[])
                with patch.object(fresh.llm,'complete_structured',generic):
                    information,clarification=await check_plan_information(
                        fresh.llm,{'task_text':'制定胸部训练计划','raw_message':'制定胸部训练计划'})
                self.assertEqual(information,{})
                self.assertIsNotNone(clarification)
                self.assertEqual(clarification.category,'service_unavailable')
            await reject_unbound_generic()
        asyncio.run(run())

    def test_workout_transaction_rolls_back_both_tables(self):
        from app.services.mcp.tools import FitnessMCPTools
        from app.db.models import WorkoutRecordModel
        from app.db.workout_session_models import WorkoutSessionModel
        record={'exercise':'卧推','weight_kg':10,'sets':4,'reps':10}
        with patch('app.db.workout_session_repository._replace_exercises',side_effect=RuntimeError('fixture failure')):
            from app.db.storage import StorageUnavailableError
            with self.assertRaises(StorageUnavailableError):FitnessMCPTools().save_workout_record('demo-user',record)
        with self.factory() as s:
            self.assertEqual(s.scalar(select(func.count()).select_from(WorkoutRecordModel)),0)
            self.assertEqual(s.scalar(select(func.count()).select_from(WorkoutSessionModel)),0)

class CompoundFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_save_blocks_dependent_analysis(self):
        from unittest.mock import AsyncMock
        from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
        graph=FitnessMultiAgentGraph()
        graph.workers['workout_record']=Mock(ainvoke=AsyncMock(return_value={'result':{'status':'error','answer':'保存失败'}}))
        state={'user_id':'demo-user','raw_message':'记录卧推', 'current_task':{'id':'t1','task_type':'workout_record'}}
        result=await graph._run_worker('workout_record',state)
        self.assertEqual(result['completed_tasks'],[])
        dispatch=graph._dispatch_node({'tasks':[{'id':'t2','depends_on':['t1']}],**result})
        self.assertEqual(dispatch['status'],'task_deadlock')

    async def test_false_saved_result_never_claims_success(self):
        from unittest.mock import AsyncMock
        from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
        graph=FitnessMultiAgentGraph()
        with patch.object(graph.workout_extraction,'run_many',AsyncMock(return_value=[{'record':{'exercise':'卧推','weight_kg':10,'sets':4,'reps':10}}])),patch.object(graph,'_run_authorized_model_tool',AsyncMock(return_value=None)),patch.object(graph,'_call_mcp',AsyncMock(return_value={'saved':False})):
            result=await graph.workers['workout_record'].ainvoke({'user_id':'demo-user','raw_message':'记录卧推10kg4组10次'})
        self.assertEqual(result['result']['status'],'error')
        self.assertNotIn('已记录',result['result']['answer'])
