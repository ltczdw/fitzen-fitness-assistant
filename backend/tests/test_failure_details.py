from unittest.mock import AsyncMock
from types import SimpleNamespace
import asyncio

import pytest

from app.services.agent.failure_details import observation_failure, read_failure_details
from app.services.agent.react_graph import ReactDependencies, finalize_node
from app.services.rag.quality import assess_rag_evidence, insufficient_evidence_reply
from evals.v2.safety import deny_network
from tests.react_supervisor_fixtures import tool_call
from tests.react_readonly_fixtures import (
    chunks, environment, semantic_outline, semantic_task, sse_body,
)

SOURCE='如果花生过敏，日常需要注意哪些食物'


def candidate():
    return {'id':'goal','task_type':'nutrition_advice','action':'question','text':SOURCE}


def observation(error, *, status='unavailable', result=None):
    return {'status':status,'error':error,'result':result or {},
            '_source':{'tool_name':'advise_nutrition','permission':'read','task_type':'nutrition_advice'}}


@pytest.mark.parametrize('code,category',[
    ('knowledge_retrieval_failed','service'),('memory_source_unavailable','source'),
    ('semantic_unit_dependency_missing','source'),('context_budget_exceeded','budget'),
    ('context_input_budget_exhausted','budget'),('answer_model_empty','model'),
    ('information_service_unavailable','model'),('plan_worker_unavailable','service'),
    ('plan_worker_failed','service'),('plan_worker_empty','service'),
    ('saved_record_read_failed','service'),('saved_record_query_unavailable','service'),
    ('saved_record_selection_unavailable','source'),('quoted_answer_unavailable','source'),
    ('read_result_unconfirmed_not_retried','source'),('executor_error:SecretException','service')])
def test_known_failure_has_public_remedy_without_raw_payload(code,category):
    detail=observation_failure({**observation(code),'summary':'SECRET','result':{'exception':'SECRET'}})
    assert detail['category']==category and detail['message'] and detail['remedy']
    assert 'SECRET' not in str(detail) and 'SecretException' not in str(detail)
    assert observation_failure(observation(code,status='completed')) is None


def test_unknown_failure_does_not_invent_configuration_or_missing_fields():
    detail=observation_failure(observation('api_key=SECRET'))
    assert detail['category']=='unknown' and detail['missing_fields']==[]
    assert 'SECRET' not in str(detail) and '尚未确认' in detail['message']


@pytest.mark.parametrize('mode,expected',[
    ('empty','matching_knowledge_evidence'),('poor','reliable_knowledge_evidence'),
    ('filtered','applicable_knowledge_evidence'),('unknown','knowledge_evidence')])
def test_evidence_type_comes_from_actual_assessment(mode,expected):
    quality=assess_rag_evidence(SOURCE,[] if mode=='empty' else chunks(SOURCE,sufficient=False))
    result={} if mode=='unknown' else {'retrieval_quality':quality.as_dict(),'evidence_filtered':mode=='filtered'}
    detail=observation_failure(observation('insufficient_evidence',status='rejected',result=result))
    assert detail['missing_fields']==[expected]
    text=insufficient_evidence_reply('nutrition_advice',None if mode=='unknown' else quality,filtered=mode=='filtered')
    assert '缺少：' in text and '个人信息' in text
    assert '知识库里没有' not in text
    if mode=='empty': assert '本次检索未找到' in text
    elif mode=='poor': assert '质量检查未通过' in text
    elif mode=='filtered': assert '适用性检查' in text
    else: assert '尚未确认' in text


def test_current_success_clears_old_failure_and_mixed_writes_suppress_read_override():
    fail=observation('knowledge_retrieval_failed')
    state={'intent_candidates':[candidate()],'observations':[fail]}
    assert read_failure_details(state)[0]['code']=='knowledge_retrieval_failed'
    state.update(covered_goals=['goal'],goal_results={'goal':observation('',status='completed')})
    assert read_failure_details(state)==[]
    state={'intent_candidates':[candidate(),{'id':'write','task_type':'workout_record','action':'record','side_effect':True}],
           'observations':[fail]}
    assert read_failure_details(state)==[]


@pytest.mark.parametrize('boundary',[
    {'safety_level':'urgent'}, {'unbound_write_calls':['write']},
    {'call_facts':[{'permission':'write'}]},
    {'observations':[observation('knowledge_retrieval_failed'),{'status':'commit_unknown','commit_state':'unknown'}]},
])
def test_read_diagnostics_never_override_urgent_or_unconfirmed_writes(boundary):
    state={'intent_candidates':[candidate()],'observations':[observation('knowledge_retrieval_failed')],**boundary}
    assert read_failure_details(state)==[]


@pytest.mark.parametrize('reason,phrase',[
    ('no_progress','未取得进展'),('tool_step_limit','工具执行步数上限'),
    ('supervisor_turn_limit','规划轮数上限')])
def test_stop_names_actual_limit_and_keeps_known_cause(reason,phrase):
    state={'intent_candidates':[candidate()],'observations':[observation('knowledge_retrieval_failed')],
           'termination_reason':reason,'status':'limited'}
    result=finalize_node(state,ReactDependencies(llm=None))
    assert phrase in result['final_answer'] and '知识检索服务' in result['final_answer']
    assert result['status']=='limited'
    state['termination_reason']='urgent_safety_blocked'
    assert '知识检索服务' not in finalize_node(state,ReactDependencies(llm=None))['final_answer']


@pytest.mark.parametrize('transport',['json','sse'])
@pytest.mark.parametrize('failure',['empty','poor','filtered','outage','model_empty','memory','context','gateway_context','dependency'])
def test_actual_response_explains_failure_and_preserves_it_on_replay(tmp_path,monkeypatch,transport,failure):
    from app.services.agent import react_readonly_workers as workers
    from app.services.context.budget import ContextBudgetExceeded, context_budget
    with deny_network() as audit, environment(tmp_path,monkeypatch,'failure') as env:
        before=env.snapshot()
        async def outage(query): raise ConnectionError('SECRET provider body')
        env.mock_rag([] if failure=='empty' else chunks(SOURCE,sufficient=failure!='poor'),
                     callback=outage if failure=='outage' else None)
        if failure=='filtered':
            from app.services.rag import evidence_conflict
            monkeypatch.setattr(evidence_conflict,'filter_hits_for_answer',
                                lambda hits,analysis:SimpleNamespace(hits=[],applied=True))
        if failure=='memory':
            monkeypatch.setattr(workers,'_answer_state',AsyncMock(return_value=None))
        if failure in {'context','dependency'}:
            def reject(**kwargs):
                raise ContextBudgetExceeded(100,50,reason='missing_dependency' if failure=='dependency' else 'input_budget')
            monkeypatch.setattr(context_budget,'assemble',reject)
        def policy(index,messages):
            return {'role':'assistant','content':'','tool_calls':[
                tool_call('attempt-'+str(index),'advise_nutrition',{'query':SOURCE})]}
        llm=env.mock_llm(policy=policy,final='' if failure=='model_empty' else '合成回答',
            semantic=lambda source:semantic_outline(source,[semantic_task('nutrition_advice','question',source)],
                safety={'level':'medical','evidence':source,'categories':[]}))
        if failure=='gateway_context':
            from app.services.llm.client import LLMContextBudgetError
            original=llm.complete_message
            async def gateway(messages,**kwargs):
                if not kwargs.get('tools'):
                    raise LLMContextBudgetError(ContextBudgetExceeded(100,50))
                return await original(messages,**kwargs)
            monkeypatch.setattr(llm,'complete_message',gateway)
        result=env.post('failure-request',message=SOURCE) if transport=='json' else sse_body(env.stream('failure-request',message=SOURCE))
        expected={'empty':'insufficient_evidence','poor':'insufficient_evidence','filtered':'insufficient_evidence','outage':'knowledge_retrieval_failed',
                  'model_empty':'answer_model_empty','memory':'memory_source_unavailable',
                  'context':'context_budget_exceeded','gateway_context':'context_budget_exceeded',
                  'dependency':'semantic_unit_dependency_missing'}[failure]
        details=result['metrics']['failure_details']
        assert len(details)==1 and details[0]['code']==expected,result
        assert 'SECRET' not in str(result) and result['safety_level']=='medical'
        if failure in {'empty','poor','filtered'}:
            assert '缺少：' in result['answer'] and result['metrics']['react_status']=='rejected'
            if failure=='filtered':
                assert details[0]['missing_fields']==['applicable_knowledge_evidence']
        else:
            assert details[0]['message'] in result['answer']
            assert result['metrics']['react_status']=='limited'
        calls=len(llm.calls)
        replay=env.post('failure-request',message=SOURCE)
        assert replay['answer']==result['answer'] and replay['metrics']['failure_details']==details
        assert len(llm.calls)==calls and env.snapshot()==before
    assert audit=={'blocked_connections':0,'blocked_dns':0}


def test_delegated_preview_keeps_filtered_evidence_reason(monkeypatch):
    from app.services.agent.agent import agent
    from app.services.agent.react_readonly_workers import _delegated_plan_preview
    worker=SimpleNamespace(ainvoke=AsyncMock(return_value={'result':{
        'status':'insufficient_evidence','answer':'缺少适用依据。',
        'retrieval_quality':{'reason':'no_hits'},'evidence_filtered':True}}))
    monkeypatch.setitem(agent.graph.workers,'plan_adjust',worker)
    result=asyncio.run(_delegated_plan_preview({'profile':{'goal':'增肌'}},'调整计划')).to_dict()
    assert observation_failure(result)['missing_fields']==['applicable_knowledge_evidence']
