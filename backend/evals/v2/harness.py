"""Execute real FitZen components against labelled cases with isolated boundaries.

No whole-answer mock is used to score semantic generation. Full-graph writing
uses deterministic extraction at the model boundary; SQL records are real but
belong to one temporary SQLite file per scenario. SQLite results do not imply
PostgreSQL concurrent locking or provider accuracy.
"""
import asyncio
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker


class MonkeyPatch:
    """R006：评测直接运行，不依赖 pytest；只覆盖本 harness 用到的 setattr。"""

    def __init__(self) -> None:
        self._undo: list[tuple[Any, str, Any, bool]] = []

    def setattr(self, target: Any, name: str, value: Any) -> None:
        had = hasattr(target, name)
        self._undo.append((target, name, getattr(target, name, None), had))
        setattr(target, name, value)

    def undo(self) -> None:
        for target, name, value, had in reversed(self._undo):
            if had:
                setattr(target, name, value)
            else:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass
        self._undo.clear()

    @classmethod
    @contextmanager
    def context(cls):
        instance = cls()
        try:
            yield instance
        finally:
            instance.undo()


class CheckFailure(AssertionError):
    def __init__(self, failures, actual):
        self.failures, self.actual = failures, actual
        super().__init__('; '.join(failures))


def verify(actual, expected, comparisons):
    failures=[]
    for key in comparisons:
        if actual.get(key) != expected.get(key):
            failures.append(f'{key}: expected={expected.get(key)!r}, actual={actual.get(key)!r}')
    if failures: raise CheckFailure(failures, actual)
    return actual


@contextmanager
def isolated_database(mp):
    from app.db import session as db
    from app.db.models import Base
    from app.db import memory_models, quality_models, workout_session_models  # noqa: F401
    with TemporaryDirectory(prefix='fitzen-eval-v2-') as temporary:
        engine=create_engine('sqlite:///'+str(Path(temporary)/'scenario.db'))
        Base.metadata.create_all(engine)
        with engine.begin() as connection:
            connection.execute(text('''CREATE TABLE user_food_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id VARCHAR(100) NOT NULL,
                meal_group_id VARCHAR(100) NOT NULL, record_time TIMESTAMP NOT NULL,
                meal_type VARCHAR(20), food_id INTEGER, raw_food_name VARCHAR(200) NOT NULL,
                food_name VARCHAR(200) NOT NULL, weight_g NUMERIC NOT NULL,
                total_kcal NUMERIC NOT NULL, total_protein NUMERIC NOT NULL,
                total_carb NUMERIC NOT NULL, total_fat NUMERIC NOT NULL,
                total_fiber NUMERIC NOT NULL DEFAULT 0, is_estimated BOOLEAN DEFAULT 0,
                weight_source VARCHAR(50) NOT NULL DEFAULT 'user_input',
                match_method VARCHAR(50) NOT NULL DEFAULT 'exact', match_confidence FLOAT DEFAULT 1.0,
                idempotency_key VARCHAR(150) NOT NULL, calculation_version VARCHAR(100) DEFAULT 'food-calc-v1',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, UNIQUE(user_id,idempotency_key))'''))
        factory=sessionmaker(bind=engine,expire_on_commit=False)
        mp.setattr(db,'engine',engine)
        mp.setattr(db,'SessionLocal',factory)
        mp.setattr(db,'DB_READY',True)
        try: yield factory
        finally: engine.dispose()


def freeze(mp, case):
    from app.core import clock
    from app.core.config import settings
    moment=datetime.fromisoformat(case['input']['now']).astimezone(timezone.utc)
    mp.setattr(clock,'utc_now',lambda:moment)
    mp.setattr(settings,'app_timezone',case['input']['timezone'])
    mp.setattr(settings,'planner_shadow_enabled',False)
    mp.setattr(settings,'memory_read_backend','compact_v1')


def write_planner(case):
    """Semantic write decision required by the 2026-10-18 contract.

    Evaluation messages deliberately contain a record command; the model
    decision (not keyword routing) supplies the task. Gold expectations and
    every downstream source/authorization gate stay unchanged.
    """
    message=str(case.get('input',{}).get('message') or '')
    if not message:
        return None
    task_type='nutrition_record' if case.get('conditions',{}).get('food') else 'workout_record'
    return {'tasks':[{'task_type':task_type,'action':'record','text':message,
                      'is_negated':False,'is_hypothetical':False,'is_third_party':False}]}


def case_safety(case):
    """Build the case-declared safety judgment at the model boundary.

    The evaluation fixture supplies semantics, not a local symptom classifier.
    Non-normal evidence is bound to the exact current message; ``normal`` keeps
    empty evidence. Expected gold/profile/scenario identities are untouched.
    """
    message=str(case.get('input',{}).get('message') or '')
    level=str(case.get('expected',{}).get('safety_level') or 'normal')
    if level not in {'normal','medical','urgent'}:
        level='normal'
    categories = ({
        'urgent':['current_symptom'],
        'medical':['medical_history'],
    }.get(level, []))
    return {'level':level,'evidence':message if level != 'normal' else '',
            'categories':categories,'reason':'eval fixture'}


def planner_with_entities(case, planner):
    """Keep scenario entities with an explicit write planner.

    The evaluation planner carries the model's task decision; scenario
    conditions carry the deterministic entity fixture. Supplying both avoids
    turning a partial fixture into an unrelated draft-clarification path while
    leaving every real source/authorization check active.
    """
    if not isinstance(planner, dict):
        return planner
    conditions = case.get('conditions') or {}
    tasks=[]
    for original in planner.get('tasks') or []:
        if not isinstance(original, dict):
            continue
        task=dict(original)
        if task.get('task_type') == 'nutrition_record' and not task.get('foods') and conditions.get('food'):
            grams=conditions.get('grams')
            if grams is not None:
                task['foods']=[{'name':'米饭','quantity':1,'unit':'g','weight_g':grams,
                                'meal_type':'午餐','record_date':None}]
        if task.get('task_type') == 'workout_record' and not task.get('workouts'):
            record=conditions.get('mock_workout') or {
                'exercise':'卧推','weight_kg':conditions.get('weight',60),'sets':4,'reps':8}
            task['workouts']=[dict(record)]
        tasks.append(task)
    return {**planner,'tasks':tasks}


def semantic_tasks(case, planner=None):
    """Project the scenario's explicit planner into RequestUnderstanding tasks.

    Reliability/transport and explicit write planners must keep their write,
    negation and third-party semantics. Only a scenario without any planner
    falls back to the historical read-only evaluation task.
    """
    source=str(case.get('input',{}).get('message') or '')
    raw_tasks = planner.get('tasks') if isinstance(planner, dict) else None
    tasks=[]
    for item in raw_tasks or []:
        if not isinstance(item, dict):
            continue
        text=item.get('text')
        task_type=item.get('task_type')
        if not isinstance(text,str) or not text or not isinstance(task_type,str):
            continue
        tasks.append({
            'task_type':task_type,
            'action':str(item.get('action') or 'question'),
            'text':text,
            'is_negated':bool(item.get('is_negated',False)),
            'is_hypothetical':bool(item.get('is_hypothetical',False)),
            'is_third_party':bool(item.get('is_third_party',False)),
        })
    if not tasks:
        tasks=[{'task_type':'training_qa','action':'question','text':source,
                'is_negated':False,'is_hypothetical':False,'is_third_party':False}]
    return tasks


class GraphScenario:
    def __init__(self, mp, case, *, planner=None):
        from app.services.agent.agent import FitnessAgent
        from app.services.agent import execution
        from app.services.agent.semantics import RequestUnderstanding
        self.agent=FitnessAgent()
        self.calls=[]
        self.saves=[]
        self.case=case
        self.before_save=None
        if planner is None and case.get('kind') in {'reliability','transport'}:
            planner=write_planner(case)
        planner=planner_with_entities(case, planner)
        mp.setattr(self.agent.graph,'parse_input',AsyncMock(return_value=planner))
        mp.setattr(self.agent,'_run_planner_shadow',AsyncMock())
        async def semantic(messages, schema, *, name=None, **kwargs):
            if name != "return_request_understanding":
                raise AssertionError(name)
            value = RequestUnderstanding.model_validate({
                'tasks': semantic_tasks(case, planner),
                'safety': case_safety(case),
            })
            return value
        mp.setattr(self.agent.graph.llm, 'configured', lambda: True)
        mp.setattr(self.agent.graph.llm, 'complete_structured', semantic)
        mp.setattr(execution.memory_service,'load',AsyncMock(return_value={'memory_backend':'eval_fixture'}))
        mp.setattr(self.agent.graph.rag,'search',lambda *args,**kwargs:[])
        mp.setattr(self.agent.graph,'_run_authorized_model_tool',AsyncMock(return_value=None))
        mp.setattr(self.agent.graph,'_call_mcp',self.call)
        record=case.get('conditions',{}).get('mock_workout') or {
            'exercise':'卧推','weight_kg':case.get('conditions',{}).get('weight',60),'sets':4,'reps':8}
        if case.get('expected',{}).get('scenario')=='retry_across_midnight':
            from app.core import clock
            record['date']=clock.today().isoformat()
        async def extract(message):
            return [{'record':dict(record)}]
        mp.setattr(self.agent.graph.workout_extraction,'run_many',extract)
        if case.get('conditions',{}).get('food'):
            from app.services.agent import multi_agent_graph as graph_module
            from app.services.agent.food_parser import FoodInput
            grams=case['conditions']['grams']
            async def food_extract(*args):
                return (FoodInput(raw_text=f'米饭{grams}克',requested_name='米饭',quantity=1,unit='g',
                                  weight_g=grams,portion_name=None,meal_type='午餐'),)
            mp.setattr(graph_module,'extract_foods_with_llm',food_extract)
            synthetic_food_catalog(mp)

    async def call(self, name, arguments, **kwargs):
        self.calls.append(name)
        if name=='get_user_profile': return {}
        if name=='get_training_history': return []
        if name=='get_nutrition_summary': return {}
        if name=='calculate_nutrition': return self.agent.graph.tools.calculate_nutrition(**arguments)
        if name=='calculate_nutrition_batch': return self.agent.graph.tools.calculate_nutrition_batch(**arguments)
        if name=='save_workout_record':
            if self.before_save: await self.before_save(name,arguments)
            result=self.agent.graph.tools.save_workout_record(**arguments)
            self.saves.append(result['record'])
            return result
        raise AssertionError('unmocked tool '+name)

    def request(self, suffix='', message=None):
        from app.schemas.chat import ChatRequest
        return ChatRequest(message=message or self.case['input']['message'],
                           conversation_id='eval-'+self.case['id'],request_id=self.case['id']+suffix)

    async def respond(self, request=None):
        return await self.agent.respond(request or self.request())


async def parse_mock_output(raw):
    """Exercise the same Pydantic JSON boundary used by the real planner."""
    import json
    from app.services.agent.deepseek_input_parser import PlannerLLM,ParsedRequest
    wire=raw if isinstance(raw,str) else json.dumps(raw,ensure_ascii=False)
    llm=SimpleNamespace(configured=lambda:True,complete=AsyncMock(return_value=wire))
    planner=PlannerLLM(llm=llm)
    return await planner._call_json('synthetic evaluation planner',ParsedRequest,'synthetic input')


def row_count(factory):
    with factory() as session:
        return session.execute(text('SELECT COUNT(*) FROM workout_records')).scalar()


def food_row_count(factory):
    with factory() as session:
        return session.execute(text('SELECT COUNT(*) FROM user_food_log')).scalar()


def synthetic_food_catalog(mp):
    from app.services.agent import food_calculator
    # Synthetic composition, not a nutritional reference or public-health gold.
    def resolve(name):
        return SimpleNamespace(food_id=None,food_name=name,energy_kcal=100,protein_g=5,
                               carb_g=15,fat_g=2,fiber_g=1,match_method='synthetic_fixture',confidence=1)
    mp.setattr(food_calculator,'resolve_food',resolve)
    mp.setattr(food_calculator,'portion_weight',lambda food_id,unit:100)


def plan(case, mp):
    from app.services.agent.routing import PlannerRoutingMixin
    from app.services.agent import routing
    from app.services.agent.execution_state import _request_route
    from app.core.config import settings
    from app.schemas.chat import ChatRequest
    mode=case['conditions'].get('planner_mode')
    mp.setattr(settings,'planner_shadow_enabled',mode=='shadow')
    mp.setattr(routing,'get_pending_food',lambda *args:None)
    class Agent(PlannerRoutingMixin): pass
    agent=Agent()
    invalid=asyncio.run(parse_mock_output('{"tasks":[')) if mode=='invalid_json' else None
    agent.graph=SimpleNamespace(parse_input=AsyncMock(return_value=invalid))
    token=_request_route.set(None)
    try:
        decision=asyncio.run(agent._route(ChatRequest(message=case['input']['message'],conversation_id=case['id']),{}))
    finally: _request_route.reset(token)
    tasks=list(decision.task_specs)
    # Rejected tasks can remain in the plan for a visible refusal; they are not executable intents.
    live=[t for t in tasks if not t.get('payload',{}).get('write_authorization','').startswith('rejected')]
    actual={'task_types':[t['task_type'] for t in live],
            'tasks':[{'type':t['task_type'],'status':t.get('status'),'depends_on':t.get('depends_on',[])} for t in tasks]}
    failures=[]
    if actual['task_types']!=case['expected']['task_types']: failures.append('task sequence/coverage mismatch')
    # E007 (v2.1.0)：依赖合同从“所有后续建议/计划都依赖第一条记录”的布尔断言，
    # 改为显式必要依赖边 required_dependencies（[dependent_index, on_index]，
    # 1 起始、按 expected task_types 顺序）。同业务建议依赖其数据来源记录；
    # 跨业务仅在原文明确要求使用该结果时才标注必要边（如“根据今天的蛋白质
    # 摄入制定训练计划”）。任务顺序仍由 task_types 断言，二者分开。
    for edge in case['expected'].get('required_dependencies') or []:
        dependent_index, on_index = edge
        dependent = tasks[dependent_index - 1] if 0 < dependent_index <= len(tasks) else None
        source = tasks[on_index - 1] if 0 < on_index <= len(tasks) else None
        if dependent is None or source is None:
            failures.append(f'required dependency edge {edge} out of range')
        elif source['id'] not in dependent.get('depends_on', []):
            failures.append(f'missing required dependency {source["id"]}->task-{dependent_index}')
    if failures: raise CheckFailure(failures,actual)
    return actual


def extract(case, mp):
    if case['kind']=='extract_workout':
        from app.services.agent.workout_extractor import preprocess_workout
        parsed=preprocess_workout(case['input']['message'])
        actual={'status':parsed.status,'record':parsed.record,'missing_fields':list(parsed.missing_fields)}
        expected=case['expected']
        failures=[]
        if sorted(actual['missing_fields'])!=sorted(expected['missing_fields']): failures.append('missing parameter mismatch')
        if not expected['missing_fields']:
            for key,value in expected['record'].items():
                got=(actual['record'] or {}).get(key)
                if str(got)!=str(value) and not (isinstance(got,(int,float)) and isinstance(value,(int,float)) and abs(got-value)<1e-6):
                    failures.append('wrong extracted '+key)
        if failures: raise CheckFailure(failures,actual)
        return actual
    from app.services.agent.food_parser import parse_food_message
    entries=parse_food_message(case['input']['message'])
    actual={'foods':[{'name':e.requested_name,'weight_g':e.weight_g,'quantity':e.quantity,'unit':e.unit,'meal_type':e.meal_type,'record_date':e.record_date} for e in entries]}
    gold=case['expected']['foods']
    failures=[]
    if len(entries)!=len(gold): failures.append('food count mismatch')
    for got,wanted in zip(actual['foods'],gold):
        if any(got.get(key)!=value for key,value in wanted.items()): failures.append('food field mismatch')
    if failures: raise CheckFailure(failures,actual)
    return actual


def write(case, mp, factory):
    planner=None
    if case['conditions'].get('planner_mode')=='omitted_question':
        planner={'tasks':[{'task_type':'workout_record','action':'record',
                          'text':case['input']['message'].rstrip('？?'),
                          'is_negated':False,'is_hypothetical':False,'is_third_party':False}]}
    scenario=GraphScenario(mp,case,planner=planner)
    response=asyncio.run(scenario.respond())
    actual={'workout_rows':row_count(factory),'food_rows':food_row_count(factory),
            'write_calls':sum(scenario.calls.count(name) for name in ('save_workout_record','calculate_nutrition','calculate_nutrition_batch')),
            'answer':response.answer,'metrics':response.metrics}
    failures=[]
    if actual['workout_rows']!=case['expected']['workout_rows']: failures.append('unexpected persisted row count')
    if actual['food_rows']!=case['expected'].get('food_rows',0): failures.append('unexpected food log row count')
    if actual['write_calls']>case['expected']['max_write_calls']: failures.append('unauthorized/extra write dispatch')
    if failures: raise CheckFailure(failures,actual)
    return actual


def urgent(case, mp, factory=None):
    """Urgent cases use the same model-bound SafetyAssessment contract.

    The fixture supplies the case-declared semantic judgment at the fake model
    boundary. This is a model-logic fixture, not a metered provider call: the
    expected physical ``model_calls`` stays zero and no business tool may run.
    """
    scenario=GraphScenario(mp,case)
    mp.setattr(scenario.agent.graph,'parse_input',AsyncMock(side_effect=AssertionError('urgent called planner')))
    response=asyncio.run(scenario.respond())
    actual={'safety_level':response.safety_level,
            'model_calls':response.metrics.get('model_calls',0),
            'write_calls':sum(scenario.calls.count(name) for name in
                              ('save_workout_record','calculate_nutrition','calculate_nutrition_batch')),
            'model_used':response.model_used,'answer':response.answer}
    if response.model_used: raise CheckFailure(['urgent used model'],actual)
    if actual['write_calls']: raise CheckFailure(['urgent dispatched a write'],actual)
    return verify(actual,case['expected'],('safety_level','model_calls','write_calls'))


def tool_denied(case, mp):
    from app.services.mcp.executor import ToolExecutionGateway
    dispatched=[]
    async def operation(args):
        dispatched.append(args)
        return {'ok':True}
    async def run():
        try:
            await ToolExecutionGateway().execute(case['conditions']['tool'],{},schema=None,
                operation=operation,allowed_tools={case['conditions']['tool']},trusted_user_id='demo-user')
        except PermissionError: pass
    asyncio.run(run())
    return verify({'tool_dispatched':bool(dispatched)},case['expected'],('tool_dispatched',))


def rag(case, mp):
    from app.services.rag.pipeline import HybridRAGPipeline
    from app.services.rag.quality import assess_rag_evidence
    from app.services.rag.query import route_query
    from app.core.config import settings
    hits=HybridRAGPipeline().search(case['input']['message'],top_k=4)
    query=case['input']['message']
    route=route_query(query)
    assessment=assess_rag_evidence(query,hits,task_type='safety_triage' if route.task=='safety_triage' else 'training_qa')
    gold=case['expected']
    evidence=gold['evidence']
    def supports(hit,e):
        return hit.source==e['source'] and e['snippet'] in hit.content
    relevant=[any(supports(h,e) for e in evidence) for h in hits]
    covered=[any(supports(h,e) for h in hits) for e in evidence]
    rank=next((i for i,matched in enumerate(relevant,1) if matched),None)
    actual={'answerable':assessment.answerable,'gold_answerable':gold['answerable'],
            'confidence':assessment.confidence,'threshold':assessment.threshold,'mode':assessment.mode,
            'base_threshold':settings.rag_answer_confidence_threshold,
            'sources':[h.source for h in hits],
            'evidence_recall':sum(covered)/len(covered) if covered else None,
            'citation_precision':sum(relevant)/len(hits) if hits and evidence else None,
            'mrr':1/rank if rank else 0.0,
            'generation_evaluated':False,'answer_faithfulness':None}
    failures=[]
    if actual['answerable']!=gold['answerable']: failures.append('default gate disagrees with gold')
    if gold['answerable'] and not all(covered): failures.append('missing required evidence snippets')
    if failures: raise CheckFailure(failures,actual)
    return actual


def memory(case, mp, factory):
    from app.services.memory import compact
    from app.services.memory.compact import CompactMemory
    from app.schemas.chat import ChatRequest
    scenario=case['expected']['scenario']
    marker=case['expected']['marker']
    cid=case['id']
    req=ChatRequest(message=case['input']['message'],conversation_id=cid,request_id=cid)
    tid,_=compact.begin_turn(req)
    response={'answer':f'你的记录为{marker}，4组8次','intent':'training_qa','request_id':cid,'citations':[]}
    if scenario=='tool_observation':
        compact.append_observation(tid,'save_workout_record',{'weight':case['conditions']['weight']},{'saved_weight':marker})
    if scenario=='failed_not_recalled': compact.fail_turn(tid)
    else: compact.finish_turn(tid,response['answer'],response)
    if scenario in ('fact_update','history_order'):
        next_marker=f'{case["conditions"]["weight"]+5}公斤'
        newer=ChatRequest(message=f'记住我现在卧推{next_marker}，之前的重量已经过时，请以这次为准',
                          conversation_id=cid,request_id=cid+'-new')
        newer_tid,_=compact.begin_turn(newer)
        compact.finish_turn(newer_tid,'收到新的重量',{'answer':'收到新的重量','intent':'training_qa'})
    adapter=CompactMemory(SimpleNamespace(delete_conversation=AsyncMock(return_value=True)))
    async def run():
        if scenario=='delete_conversation': await adapter.delete_conversation('demo-user',cid)
        target=cid+'-different' if scenario=='conversation_boundary' else cid
        return await adapter.load_history('demo-user',target)
    result=asyncio.run(run())
    messages=result.get('messages',[])
    import json
    actual={'messages':messages,'marker_present':marker in json.dumps(messages,ensure_ascii=False)}
    if scenario in ('failed_not_recalled','tool_observation','window_numbers'):
        rows=compact.recent_rows(cid)
        canonical=[m for row in rows for m in row.messages]
        actual.update(messages=canonical,marker_present=marker in json.dumps(canonical,ensure_ascii=False))
        messages=canonical
    failures=[]
    if scenario in ('conversation_boundary','delete_conversation','failed_not_recalled'):
        if actual['marker_present']: failures.append('excluded history leaked')
    else:
        if not actual['marker_present']: failures.append('numeric fact lost from history')
        if scenario=='explicit_fact':
            facts=compact.current_facts()
            actual['facts']=facts
            if marker not in json.dumps(facts,ensure_ascii=False): failures.append('explicit fact missing')
        if scenario=='history_order' and [m['role'] for m in messages]!=['user','assistant','user','assistant']:
            failures.append('history role ordering wrong')
        if scenario=='fact_update':
            facts=compact.current_facts();actual['facts']=facts
            # Desired product behavior: an explicit superseding value must not
            # leave contradictory older values active. If unimplemented, report
            # a capability gap, never adjust gold to the implementation.
            fact_text=' '.join(v['text'] for v in facts.values())
            if marker in fact_text or next_marker not in fact_text:failures.append('explicitly superseded fact remains active (capability gap)')
        if scenario=='tool_observation' and not any(m['role']=='tool' and marker in m['content'] for m in messages):
            failures.append('tool observation lost')
        if scenario=='response_replay':
            _,cached=compact.begin_turn(req)
            actual['replay']=cached
            if cached!=response: failures.append('full response replay mismatch')
    if failures: raise CheckFailure(failures,actual)
    return actual


def reliability(case, mp, factory):
    from app.services.memory import compact
    from app.db import repositories
    from app.db.storage import StorageUnavailableError, IdempotencyConflict
    from app.services.agent.execution_state import RequestIdConflict
    from app.core import clock
    scenario=GraphScenario(mp,case)
    action=case['expected']['scenario']
    req=scenario.request()
    errors=[]
    fault={'plan':1,'params':2,'reserve':3}
    if action.startswith('ledger_'):
        count=0
        original=compact.save_request_state
        def broken(rid,state):
            nonlocal count
            count+=1
            if count>=fault[action.removeprefix('ledger_')]: raise StorageUnavailableError('eval injected ledger failure')
            return original(rid,state)
        mp.setattr(compact,'save_request_state',broken)
        response=asyncio.run(scenario.respond(req))
        if '未执行' not in response.answer: errors.append('ledger failure not visible')
        if scenario.calls.count('save_workout_record'): errors.append('write dispatched with unavailable ledger')
    elif action=='cancel_before_write':
        async def cancel_request():
            reached=asyncio.Event()
            async def block(*args):
                reached.set()
                await asyncio.Event().wait()
            scenario.before_save=block
            task=asyncio.create_task(scenario.respond(req))
            await asyncio.wait_for(reached.wait(),5)
            task.cancel()
            try:await task
            except asyncio.CancelledError:pass
            else:errors.append('cancellation swallowed')
        asyncio.run(cancel_request())
        from app.db.quality_models import FailureCase
        with factory() as session:
            if list(session.scalars(select(FailureCase))):errors.append('cancel creates pending failure case')
    elif action=='repository_key_conflict':
        weight=case['conditions']['weight']
        repositories.create_workout('demo-user','卧推',4,8,weight,idempotency_key=req.request_id)
        try: repositories.create_workout('demo-user','卧推',4,8,weight+5,idempotency_key=req.request_id)
        except IdempotencyConflict: pass
        else: errors.append('same-key parameter conflict accepted')
    elif action in ('partial_retry','final_commit_failure','retry_across_midnight'):
        if action=='partial_retry':
            count=0
            async def broken(*args):
                nonlocal count
                count+=1
                if count==2: raise StorageUnavailableError('eval second operation unavailable')
            scenario.before_save=broken
        else:
            from app.services.agent import agent as agent_module
            original_finish=agent_module.finish_turn
            def failed_commit(*args): raise RuntimeError('eval final commit interrupted')
            mp.setattr(agent_module,'finish_turn',failed_commit)
        try:
            response=asyncio.run(scenario.respond(req))
            if action in ('final_commit_failure','retry_across_midnight'):
                from app.db.memory_models import MemoryTurn
                with factory() as session:
                    turn=session.scalar(select(MemoryTurn).where(MemoryTurn.request_id==req.request_id))
                    if response.memory_status!='degraded' or turn.status=='completed' or turn.response is not None:
                        errors.append('final commit failure not reflected in durable state')
            else:errors.append('fault injection not reached')
        except Exception as exc: errors.append('injected:'+type(exc).__name__)
        # Build another real graph for the retry; its mocks deliberately use new instances.
        if action in ('final_commit_failure','retry_across_midnight'):
            mp.setattr(agent_module,'finish_turn',original_finish)
        if action=='retry_across_midnight':
            tomorrow=datetime.fromisoformat(case['input']['now']).astimezone(timezone.utc)+timedelta(days=1)
            mp.setattr(clock,'utc_now',lambda:tomorrow)
        retry=GraphScenario(mp,case)
        asyncio.run(retry.respond(req))
        scenario.calls+=retry.calls
        errors=[e for e in errors if not e.startswith('injected:')]
        if action=='retry_across_midnight':
            with factory() as session:
                dates=list(session.execute(text('SELECT workout_date FROM workout_records')).scalars())
            if dates!=['2026-09-30']:errors.append('retry changed original record date')
    else:
        first=asyncio.run(scenario.respond(req))
        if action=='payload_conflict':
            try: asyncio.run(scenario.respond(scenario.request(message=req.message+'，修改了内容')))
            except RequestIdConflict: pass
            else: errors.append('request payload conflict not rejected')
        elif action=='new_request_same_content': asyncio.run(scenario.respond(scenario.request('-second')))
        elif action in ('replay','retry_across_midnight'):
            if action=='retry_across_midnight':
                tomorrow=datetime.fromisoformat(case['input']['now']).astimezone(timezone.utc)+timedelta(days=1)
                mp.setattr(clock,'utc_now',lambda:tomorrow)
            second=asyncio.run(scenario.respond(req))
            if not second.replayed or second.answer!=first.answer: errors.append('completed request not replayed')
    actual={'workout_rows':row_count(factory),'write_calls':scenario.calls.count('save_workout_record'),
            'errors':errors}
    if actual['workout_rows']!=case['expected']['workout_rows']: errors.append('unexpected final row count')
    if errors: raise CheckFailure(errors,actual)
    return actual


def transport(case, mp, factory):
    from app.services.agent.transport import stream_response
    from app.schemas.chat import ChatResponse
    from app.db import quality_repository as repo
    from app.api.quality import FeedbackInput
    from app.services.memory import compact
    from app.services.memory.compact import CompactMemory
    from app.services.telemetry import RequestMetrics, active_metrics, model_attempt
    scenario=GraphScenario(mp,case)
    req=scenario.request()
    action=case['expected']['scenario']
    failures=[]
    actual={}
    if action=='usage_actual_attempts':
        metrics=RequestMetrics();token=active_metrics.set(metrics)
        try:
            for attempt in range(2):
                try:
                    with model_attempt('synthetic-not-provider') as call:
                        if attempt==0: raise RuntimeError('injected first call failure')
                        call.observe({'usage':{'prompt_tokens':case['conditions']['weight'],'completion_tokens':8}})
                except RuntimeError: pass
            actual=metrics.summary()
            if actual['model_calls']!=2 or actual['input_tokens']!=case['conditions']['weight'] or actual['output_tokens']!=8 or actual['usage_missing_calls']!=1:
                failures.append('attempt/token accounting mismatch')
        finally:active_metrics.reset(token)
    elif action=='cancel_not_failure':
        response=ChatResponse(answer='',intent='error',run_id=req.request_id+'-cancel',request_id=req.request_id,message_id=req.request_id+':assistant')
        repo.record_run(req,response,{'failure_reasons':[],'degraded':False},status='cancelled')
        from app.db.quality_models import FailureCase
        with factory() as session:
            actual={'failure_cases':len(list(session.scalars(select(FailureCase))))}
        if actual['failure_cases']!=0:failures.append('cancel creates pending failure case')
    else:
        first=asyncio.run(scenario.respond(req))
        if action.startswith('feedback') or action=='review_audit':
            payload=FeedbackInput(run_id=first.run_id,message_id=first.message_id,rating='down',note='synthetic evaluation')
            saved=repo.feedback(payload)
            if action=='feedback_duplicate':
                again=repo.feedback(payload)
                actual={'same_feedback':saved['feedback_id']==again['feedback_id'],'replayed':again['replayed']}
                if not actual['same_feedback'] or not actual['replayed']:failures.append('feedback duplicate not idempotent')
            elif action=='review_audit':
                from app.api.quality import ReviewInput
                case_id=saved['case_id']
                reviewed=repo.review(case_id,ReviewInput(decision='resolve',expected_version=1,note='synthetic reviewed case'))
                detail=repo.get_case(case_id)
                actual={'status':reviewed['status'],'audit_count':len(detail['audit'])}
                if actual!={'status':'resolved','audit_count':1}:failures.append('review state/audit mismatch')
            else:
                result=asyncio.run(CompactMemory(SimpleNamespace()).load_history('demo-user',req.conversation_id))
                answers=[m for m in result['messages'] if m['role']=='assistant']
                actual={'feedback':answers[-1].get('feedback'),'run_id':answers[-1].get('run_id')}
                if actual['feedback']!='down' or actual['run_id']!=first.run_id:failures.append('history feedback identity lost')
        else:
            target=req if action=='json_sse_equivalence' else scenario.request(message=req.message+'改动')
            async def collect():return [event async for event in stream_response(scenario.agent,target)]
            events=asyncio.run(collect())
            actual={'events':events,'workout_rows':row_count(factory)}
            if action=='sse_conflict':
                if not any(e['event']=='error' and e['data'].get('code')=='request_id_conflict' for e in events):failures.append('SSE missing structured conflict')
            else:
                answer=''.join(e['data']['content'] for e in events if e['event']=='token')
                done=events[-1]
                if answer!=first.answer or done['event']!='done' or done['data'].get('intent')!=first.intent:
                    failures.append('JSON/SSE result differs')
            if actual['workout_rows']!=1:failures.append('transport duplicated business write')
    if failures:raise CheckFailure(failures,actual)
    return actual


HANDLERS={'plan':plan,'extract_workout':extract,'extract_food':extract,'write':write,
          'urgent':urgent,'tool_denied':tool_denied,'rag':rag,'memory':memory,
          'reliability':reliability,'transport':transport}
DB_KINDS={'write','memory','reliability','transport','urgent'}


def execute(case):
    with MonkeyPatch.context() as mp:
        freeze(mp,case)
        function=HANDLERS[case['kind']]
        if case['kind'] in DB_KINDS:
            with isolated_database(mp) as factory:return function(case,mp,factory)
        return function(case,mp)
