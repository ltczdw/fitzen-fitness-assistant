import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.core import clock
from app.db.memory_models import MemoryTurn
from app.services.agent.task_information import PlanInformation,check_plan_information
from tests.test_plan_semantic_input_budget import preview
from tests.test_plan_followup_contract import fixture,send,c3_env,C3HTTPHarness


def information_without_new_time():
    return PlanInformation(scope='cycle',scope_quote='二分化',cycle_sessions=2,cycle_quote='二分化',
        cycle_quantity_quote='二',facts=[])


def test_preview_revision_inherits_selected_cap_without_inventing_current_quote():
    state,_=preview('unit')
    request={'raw_message':'改成二分化','task_text':'改成二分化',
        'semantic_scope':{'delivery_type':'preview','preview_ref':state['preview_id']},
        'current_plan_preview':{'status':'ok','state':state,'available':[state]}}
    llm=SimpleNamespace(complete_structured=AsyncMock(return_value=information_without_new_time()))
    info,question=asyncio.run(check_plan_information(llm,request))
    assert question is None and info['session_time_seconds']==5400
    assert info['session_time_source']=={'kind':'preview','preview_id':state['preview_id']}
    assert information_without_new_time().facts==[]


def test_missing_or_wrong_preview_cannot_supply_duration():
    state,_=preview('unit')
    for ref,cap in [('wrong',90),(state['preview_id'],None)]:
        current={**state,'session_minutes':cap}
        request={'raw_message':'改成二分化','task_text':'改成二分化',
            'semantic_scope':{'delivery_type':'preview','preview_ref':ref},
            'current_plan_preview':{'status':'ok','state':current,'available':[current]}}
        llm=SimpleNamespace(complete_structured=AsyncMock(return_value=information_without_new_time()))
        info,question=asyncio.run(check_plan_information(llm,request))
        assert info['session_time_seconds'] is None
        if ref == 'wrong':
            assert question and {f.name for f in question.fields} == {'goal','experience'}
        else:
            assert question is None


def seed(engine,harness,*,long=False):
    stored,answer=preview(harness.conversation_id)
    if not long:
        from app.services.agent.plan_preview import _digest
        answer='\n'.join(f'训练日{i+1}：练习{i+1}，上限90分钟。' for i in range(6))
        stored['answer_digest']=_digest(answer)
    with Session(engine) as s,s.begin():
        s.add(MemoryTurn(turn_id=str(uuid4()),request_id=stored['request_id'],
            conversation_id=harness.conversation_id,memory_epoch=0,status='completed',
            completed_at=clock.utc_now(),messages=[{'role':'assistant','content':answer,
                'app_meta':{'plan_preview':stored}}]))


@pytest.mark.parametrize('backend',['sqlite'])
@pytest.mark.parametrize('transport',['json','sse'])
def test_change_split_keeps_ninety_minutes_and_replaces_cycle_count(c3_env,monkeypatch,backend,transport):
    from app.services.agent.agent import agent
    state=fixture(monkeypatch,cycle=True,cycle_count=2,edit=True,plan_source='改成二分化')
    worker=agent.graph.workers['plan_adjust']
    observed=[]
    async def capture(value):
        try: result=await worker.ainvoke(value)
        except Exception as error:
            observed.append({'exception':type(error).__name__,'detail':str(error)[:200]})
            raise
        observed.append({'status':result.get('result',{}).get('status')})
        return result
    monkeypatch.setitem(agent.graph.workers,'plan_adjust',SimpleNamespace(ainvoke=capture))
    harness=C3HTTPHarness(monkeypatch,c3_env)
    seed(c3_env,harness)
    rid=str(uuid4())
    status,response=send(harness,'改成二分化',rid,transport)
    if observed and 'exception' in observed[0]:
        pytest.fail(str(observed[0]))
    assert status==200 and not response['clarifications'],response
    assert response['answer'].count('合计：90分钟')==2,(response,observed,state['calls'])
    artifact=response['metrics']['plan_preview']
    assert artifact['expected_sessions']==2 and len(artifact['days'])==2
    calls=len(state['calls'])
    _,replay=send(harness,'改成二分化',rid)
    assert replay['answer']==response['answer'] and len(state['calls'])==calls


@pytest.mark.parametrize('backend',['sqlite'])
def test_evidence_refusal_is_delivered_without_supervisor_loop(c3_env,monkeypatch,backend):
    from app.services.agent.agent import agent
    text='现有资料适用条件不同，本次未生成修改后的计划。'
    state=fixture(monkeypatch,cycle=True,cycle_count=2,edit=True,plan_source='改成二分化')
    monkeypatch.setitem(agent.graph.workers,'plan_adjust',SimpleNamespace(ainvoke=AsyncMock(return_value={
        'result':{'status':'insufficient_evidence','answer':text,'citations':[]}})))
    harness=C3HTTPHarness(monkeypatch,c3_env)
    seed(c3_env,harness)
    _,response=send(harness,'改成二分化',str(uuid4()))
    assert response['answer']==text,response
    assert response['metrics']['react_status']=='rejected'
    assert response['metrics']['react_termination_reason']=='goals_complete'
    assert sum('plan_training' in names for names in state['calls'])==1


@pytest.mark.parametrize('backend',['sqlite'])
def test_required_large_selected_preview_reports_budget_instead_of_loop(c3_env,monkeypatch,backend):
    from app.services.context.budget import ContextBudgetManager
    # Keep this boundary fixed when the deployment's default window grows.
    monkeypatch.setattr('app.services.agent.multi_agent_graph.context_budget',
        ContextBudgetManager(window=4096,output=1024,margin=.15))
    state=fixture(monkeypatch,cycle=True,cycle_count=2,edit=True,plan_source='改成二分化')
    harness=C3HTTPHarness(monkeypatch,c3_env)
    seed(c3_env,harness,long=True)
    _,response=send(harness,'改成二分化',str(uuid4()))
    assert '上下文预算' in response['answer'] and '执行上限' not in response['answer'],response
    assert response['metrics']['react_status']=='rejected' and 'plan_preview' not in response['metrics']
    assert sum('plan_training' in names for names in state['calls'])==1


@pytest.mark.parametrize('clear',[False,True])
def test_selected_older_preview_and_explicit_reset_are_respected(clear):
    current,_=preview('unit')
    older={**current,'preview_id':'older','session_minutes':45}
    message='改成二分化，不沿用原时长' if clear else '改成二分化'
    value=information_without_new_time()
    if clear:
        value=value.model_copy(update={'preview_time_action':'clear','preview_time_quote':'不沿用原时长'})
    state={'raw_message':message,'task_text':message,
        'semantic_scope':{'delivery_type':'preview','preview_ref':'older'},
        'current_plan_preview':{'status':'ok','state':current,'available':[current,older]}}
    llm=SimpleNamespace(complete_structured=AsyncMock(return_value=value))
    info,question=asyncio.run(check_plan_information(llm,state))
    if clear:
        assert question is None and info['session_time_seconds'] is None
        assert info['preview_time_cleared'] is True and info['timing_requested'] is False
    else:
        assert question is None and info['session_time_seconds']==2700
