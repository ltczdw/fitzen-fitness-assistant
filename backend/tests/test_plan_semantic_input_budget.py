import json
import asyncio
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.core import clock
from app.db.memory_models import MemoryTurn
from app.services.agent.dialogue_context import semantic_messages
from app.services.agent.plan_preview import build_preview_state
from app.services.agent.semantics import PROMPT
from app.services.agent.training_timing import TimedTrainingPlan
from app.services.context.budget import ContextBudgetManager, context_budget, ContextBudgetExceeded
from app.services.llm.client import LLMClient
from tests.test_plan_followup_contract import fixture, send, c3_env, C3HTTPHarness
from tests.test_llm_limits import clients

SOURCE='给我制定三分化训练计划'
COMPLETE=SOURCE+'，在健身房，每次90分钟。'
CONDITIONS='在健身房，每次90分钟'


def preview(cid):
    answer='已生成六个训练日。\n'+('完整训练计划正文与动作说明。'*600)
    plan=TimedTrainingPlan.model_validate({'sessions':[
        {'label':f'训练日{i+1}','warmup':{'seconds':600,'activity':'热身'},
         'main':{'seconds':3600,'exercises':[{'name':'练习'+str(i+1),'sets':4,'reps':10}]},
         'breaks':{'seconds':900},'cooldown':{'seconds':300,'activity':'放松'},
         'time_consistency':'consistent'} for i in range(6)]})
    rid=str(uuid4())
    state=build_preview_state(user_id='demo-user',conversation_id=cid,request_id=rid,
        message_id=rid+':assistant',created_at=clock.utc_now().isoformat(),plan=plan,
        answer=answer,source='每周6次，每次90分钟',expected_sessions=6,session_minutes=90)
    return state,answer


def semantic_request_budget():
    from app.services.agent.semantics import RequestUnderstanding
    specification={'type':'function','function':{'name':'return_request_understanding',
        'description':'Return the required structured result. This function does not execute any action.',
        'parameters':RequestUnderstanding.model_json_schema()}}
    first=LLMClient._payload(semantic_messages(PROMPT,COMPLETE,{}),0,False,
        tools=[specification],tool_choice={'type':'function','function':{'name':'return_request_understanding'}})
    # Fit a complete real request plus bounded preview metadata, rather than
    # reserving one output token while the sender actually requests thousands.
    budget=ContextBudgetManager(window=first.tokens.input_tokens+first['max_tokens']+2501,
                                output=first['max_tokens'],margin=0,counter=context_budget.counter)
    return budget


def test_semantic_preview_is_an_address_not_a_second_full_body():
    state,answer=preview('unit')
    loaded={**state,'content':answer}
    messages=semantic_messages(PROMPT,COMPLETE,{'current_plan_preview':{
        'status':'ok','state':loaded,'available':[loaded]}})
    payload=json.loads(messages[-1]['content'])
    locator=payload['current_plan_preview']
    assert locator['preview_id']==state['preview_id']
    assert locator['days']==state['days'] and locator['expected_sessions']==6
    assert 'content' not in locator and 'content_truncated' not in locator
    assert locator['content_status']=='not_loaded'
    assert all(p['preview_id']!=state['preview_id'] for p in locator.get('available_previews',[]))
    budget=semantic_request_budget()
    assert budget.validate(messages)<=budget.input_budget
    # The complete text remains available server-side for the selected worker.
    assert loaded['content']==answer


@pytest.mark.parametrize('backend',['sqlite'])
@pytest.mark.parametrize('transport',['json','sse'])
def test_complete_new_plan_survives_existing_large_preview(c3_env,monkeypatch,backend,transport):
    # This transport fixture uses the actual API, semantic gateway, ReAct loop,
    # plan information and timed renderer. Only provider responses are mocked.
    state=fixture(monkeypatch,cycle=True,reply=CONDITIONS,plan_source=SOURCE,missing_experience=False,
                  equipment_quote='在健身房',time_quote='每次90分钟',quantity_quote='90分钟')
    harness=C3HTTPHarness(monkeypatch,c3_env)
    stored,answer=preview(harness.conversation_id)
    with Session(c3_env) as s,s.begin():
        s.add(MemoryTurn(turn_id=str(uuid4()),request_id=stored['request_id'],
            conversation_id=harness.conversation_id,memory_epoch=0,status='completed',
            completed_at=clock.utc_now(),messages=[{'role':'assistant','content':answer,
                'app_meta':{'plan_preview':stored}}]))
    budget=semantic_request_budget()
    monkeypatch.setattr('app.services.llm.client.context_budget',budget)
    status,response=send(harness,COMPLETE,str(uuid4()),transport)
    assert status==200 and response['metrics'].get('execution_mode')=='react_readonly',response
    assert not response['clarifications'] and '一个训练循环展示3个训练日' in response['answer'],response
    assert state['general']==[COMPLETE]


def test_gateway_keeps_context_budget_error_identity():
    error=LLMClient._classify_exception(ContextBudgetExceeded(101,100))
    assert error.code=='context_input_budget_exhausted'
    assert error.used==101 and error.budget==100 and error.retryable is False


@pytest.mark.parametrize('backend',['sqlite'])
def test_real_input_overflow_reports_budget_without_fake_model_result(c3_env,monkeypatch,backend):
    state=fixture(monkeypatch,cycle=True,reply=CONDITIONS,plan_source=SOURCE,missing_experience=False)
    harness=C3HTTPHarness(monkeypatch,c3_env)
    budget=ContextBudgetManager(window=512,output=1,margin=0,counter=context_budget.counter)
    monkeypatch.setattr('app.services.llm.client.context_budget',budget)
    status,response=send(harness,COMPLETE,str(uuid4()))
    assert status==200 and response['metrics']['mode_reason']=='context_input_budget_exhausted',response
    assert '上下文预算' in response['answer']
    assert '返回的识别结果' not in response['answer'] and '急诊' not in response['answer']
    assert response['safety_level']=='unknown' and not state['calls']


@pytest.mark.parametrize('kind',['message','complete','stream'])
def test_local_input_limit_does_not_open_provider_circuit(clients,monkeypatch,kind):
    from app.services.llm.client import LLMContextBudgetError
    client,_=clients
    def reject(*args,**kwargs):raise ContextBudgetExceeded(101,100)
    monkeypatch.setattr('app.services.llm.client.context_budget.prepare_request',reject)
    before=client.breaker.snapshot()
    async def run():
        messages=[{'role':'user','content':'输入'}]
        if kind=='message':return await client.complete_message(messages,tools=[])
        if kind=='complete':return await client.complete(messages)
        return [item async for item in client.stream(messages)]
    with pytest.raises(LLMContextBudgetError):asyncio.run(run())
    assert client.breaker.snapshot()==before
