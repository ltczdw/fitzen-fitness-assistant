"""Actual ASGI/ReAct plan questions resume via their bounded current reply."""
import json
import asyncio
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select

from app.db.memory_models import MemoryTurn
from app.services.agent import planning_questions
from tests.test_workout_source_bound_writes import (
    C3HTTPHarness, c3_env, mock_provider_transport, native_tools, semantic_response, saved_rows,
)
from tests.test_react_record_continuation import tool_response

SOURCE='给我制定一个三分化的训练计划'
REPLY='训练一年，在健身房练，每次一个半小时'


def fixture(monkeypatch, *, case='update', reply=REPLY, cycle=False, plan_source=SOURCE,
            equipment_quote='在健身房练',time_quote='每次一个半小时',quantity_quote='一个半小时',
            cycle_count=3,edit=False,missing_experience=True):
    from app.db import profile_repository
    # Six confirmed weekly sessions exercise a three-way split repeated twice.
    # Continuation safety remains tested using a genuinely required field.
    # Equipment/time now have defaults and cannot serve as missing-field fixtures.
    profile_repository.save_profile('demo-user', {'goal':'增肌','available_days':3 if cycle else 6,
        **({} if missing_experience else {'experience':'中级'})})
    state={'general':[], 'calls':[], 'reply_count':0, 'plan_sources':[]}
    def respond(request):
        body=json.loads(request.content)
        names=[t['function']['name'] for t in body.get('tools') or []]
        state['calls'].append(names)
        if 'return_request_understanding' in names:
            semantic_payload=json.loads(body['messages'][-1]['content'])
            query=semantic_payload['query']
            state['general'].append(query)
            result=semantic_response([{'task_type':'training_plan','action':'question','text':query}])
            f=result['choices'][0]['message']['tool_calls'][0]['function']
            v=json.loads(f['arguments'])
            v['tasks'][0].update(plan_result=query!=reply, result_scope='independent',
                                 plan_request_quote=plan_source if plan_source in query else '')
            if edit:
                v['tasks'][0].update(delivery_type='preview',preview_ref=semantic_payload['current_plan_preview']['preview_id'])
            if query==reply: v.update(relation='answer')
            f['arguments']=json.dumps(v,ensure_ascii=False)
        elif 'return_planning_reply' in names:
            payload=json.loads(body['messages'][-1]['content'])
            state['reply_count']+=1
            v={'draft_id':payload['draft_id'], 'action':case if case in {'cancel','unclear','unrelated','explain'} else 'update',
               'quote':reply}
            if case=='explain':
                v['explanation']='三分化是把训练内容拆成三类训练日。每周次数用于安排日历；如果只需要三类训练日的循环模板，可以先按循环生成。'
            if case!='missing_safety':
                v['safety']={'level':'urgent' if case in {'urgent','foreign_safety'} else 'medical' if case=='medical' else 'normal',
                             'evidence':'昨天胸痛' if case=='foreign_safety' else reply if case in {'urgent','medical'} else '',
                             'categories':[]}
            result=tool_response('return_planning_reply',v)
        elif 'return_plan_information' in names:
            data=json.loads(body['messages'][-1]['content'])
            source=data['sources']['current']
            state['plan_sources'].append(source)
            facts=[{'field':name,'source_id':'profile:'+name,'quote':data['sources']['profile:'+name]}
                   for name in (('goal',) if missing_experience else ('goal','experience'))]
            if not cycle:
                facts.append({'field':'available_days','source_id':'profile:available_days','quote':'6','count':6})
            if reply in source:
                facts += ([{'field':'experience','source_id':'current','quote':'训练一年'}] if missing_experience else [])
                facts += [{'field':'equipment','source_id':'current','quote':equipment_quote},
                          {'field':'session_time','source_id':'current','quote':time_quote,
                           'quantity_quote':quantity_quote,'seconds':5400}]
            result=tool_response('return_plan_information',{
                'scope':'cycle' if cycle else 'weekly','scope_quote':'二分化' if cycle_count==2 else '三分化','facts':facts,
                **({'cycle_sessions':cycle_count,'cycle_quote':'二分化' if cycle_count==2 else '三分化',
                    'cycle_quantity_quote':'二' if cycle_count==2 else '三'} if cycle else {})})
        elif 'return_timed_training_plan' in names:
            result=tool_response('return_timed_training_plan',{'sessions':[
                {'label':label,'warmup':{'seconds':600,'activity':'动态热身'},
                 'main':{'seconds':3600,'exercises':[{'name':exercise,'sets':4,'reps':10}]},
                 'breaks':{'seconds':900}, 'cooldown':{'seconds':300,'activity':'放松'},
                 'time_consistency':'consistent'}
                for label,exercise in ([('上肢','卧推'),('下肢','深蹲')] if cycle and cycle_count==2 else [('推','卧推'),('拉','划船'),('腿','深蹲')] if cycle else
                                       [('推1','卧推'),('拉1','划船'),('腿1','深蹲'),
                                        ('推2','卧推'),('拉2','划船'),('腿2','深蹲')])]})
        elif 'plan_training' in names:
            # The actual supervisor/tool matcher retains source and task IDs.
            result=native_tools([('plan-'+str(len(state['calls'])),'plan_training',{'query':plan_source})])
        else:
            raise AssertionError('Unexpected protocol: '+str(names))
        return httpx.Response(200,json=result)
    mock_provider_transport(monkeypatch,[respond])
    from app.services.rag.retrieval import RetrievedChunk
    from app.services.agent.agent import agent
    from unittest.mock import AsyncMock
    hit=RetrievedChunk('训练安排原则','逐步增加训练量并保留恢复时间。',.9,'official',
                       {'rerank_score':.9,'rerank_status':'model'},'plan-fixture')
    monkeypatch.setattr(agent.graph,'_readonly_retrieval',AsyncMock(return_value=[hit]))
    return state


def send(harness,message,rid,transport='json'):
    if transport=='json': return harness.post_json(message,mode='react_readonly',request_id=rid)
    status,events,_=harness.stream(message,mode='react_readonly',request_id=rid)
    return status,{**events[-1],'answer':''.join(str(e.get('content') or '') for e in events)}


@pytest.mark.parametrize('backend',['sqlite'])
@pytest.mark.parametrize('transport',['json','sse'])
def test_gym_duration_reply_resumes_real_plan_and_replays(c3_env,monkeypatch,backend,transport):
    state=fixture(monkeypatch)
    harness=C3HTTPHarness(monkeypatch,c3_env)
    status,first=send(harness,SOURCE,str(uuid4()),transport)
    assert status==200 and first['metrics']['execution_mode']=='react_readonly',first
    assert [f['name'] for f in first['clarifications'][0]['fields']]==['experience'],first
    with c3_env.connect() as c:
        ops=c.execute(select(MemoryTurn.operations).where(MemoryTurn.conversation_id==harness.conversation_id)).scalar_one()
    assert ops['tasks'][planning_questions.KEY]['state']=='waiting'
    rid=str(uuid4())
    status,second=send(harness,REPLY,rid,transport)
    assert status==200 and second['metrics']['execution_mode']=='react_readonly',second
    assert not second['clarifications'] and '卧推' in second['answer'] and '深蹲' in second['answer'],second
    assert REPLY not in state['general'] and state['reply_count']==1,state
    assert any(SOURCE in q and REPLY in q for q in state['plan_sources'])
    assert not saved_rows(c3_env)
    calls=len(state['calls'])
    status,replayed=send(harness,REPLY,rid)
    assert status==200 and replayed['answer']==second['answer']
    assert len(state['calls'])==calls and not saved_rows(c3_env)


@pytest.mark.parametrize('backend',['sqlite'])
@pytest.mark.parametrize('case',['urgent','missing_safety','foreign_safety','cancel','unclear','medical'])
def test_plan_reply_current_safety_and_terminal_actions(c3_env,monkeypatch,backend,case):
    reply=REPLY+('，但现在胸痛' if case=='urgent' else '，肩部不适' if case=='medical' else '')
    state=fixture(monkeypatch,case=case,reply=reply)
    harness=C3HTTPHarness(monkeypatch,c3_env)
    status,first=send(harness,SOURCE,str(uuid4()))
    assert status==200 and first['clarifications'],first
    rid=str(uuid4())
    status,result=send(harness,reply,rid)
    assert status==200 and not saved_rows(c3_env),result
    if case=='urgent': assert result['safety_level']=='urgent' and result['intent']=='symptom_triage'
    elif case in {'missing_safety','foreign_safety'}:
        assert result['metrics']['safety_status']=='unavailable' and result['clarifications'][0]['category']=='service_unavailable'
    elif case=='cancel': assert '已取消' in result['answer']
    elif case=='unclear': assert result['clarifications']==first['clarifications']
    else:
        assert result['safety_level']=='medical' and '卧推' in result['answer'],result
        calls=len(state['calls'])
        _,replayed=send(harness,reply,rid)
        assert replayed['safety_level']=='medical' and len(state['calls'])==calls
    if case!='medical': assert len(state['plan_sources'])==1,state


@pytest.mark.parametrize('case',['unrelated','wrong_target','foreign_quote','budget'])
def test_plan_reply_cannot_claim_unrelated_or_unbound_source(compact_db,case):
    from tests.test_task_information import draft_fixture, begin
    from app.schemas.chat import ChatRequest
    from app.services.memory import compact
    from app.services.llm.budget import ModelBudgetError
    initial,origin=draft_fixture(compact_db)
    request=ChatRequest(message=REPLY,conversation_id=initial.conversation_id,mode='react_readonly')
    tid=begin(request)
    async def decide(messages,schema,**kwargs):
        if case=='budget': raise ModelBudgetError('model_request_budget_exhausted')
        data=json.loads(messages[-1]['content'])
        return schema.model_validate({'draft_id':'foreign' if case=='wrong_target' else data['draft_id'],
             'action':'unrelated' if case=='unrelated' else 'update',
             'quote':'徒手30分钟' if case=='foreign_quote' else REPLY,
             'safety':{'level':'normal','evidence':''}})
    token=compact.current_turn.set(tid)
    try:
        llm=SimpleNamespace(complete_structured=decide)
        if case=='budget':
            with pytest.raises(ModelBudgetError): asyncio.run(planning_questions.continue_request(llm,request))
        elif case=='foreign_quote':
            with pytest.raises(planning_questions.ContinuationError): asyncio.run(planning_questions.continue_request(llm,request))
        else:
            result=asyncio.run(planning_questions.continue_request(llm,request))
            assert result is None if case=='unrelated' else result['clarifications'][0]['category']=='service_unavailable'
        with compact_db() as s:
            assert s.get(MemoryTurn,origin).operations['tasks'][planning_questions.KEY]['state']=='waiting'
            assert s.get(MemoryTurn,tid).operations is None
    finally:
        compact.current_turn.reset(token)


def test_medical_safety_merges_in_both_orders():
    from app.services.agent.agent import agent
    assert agent._merge_safety('normal','medical')=='medical'
    assert agent._merge_safety('medical','normal')=='medical'
    assert agent._merge_safety('medical','urgent')=='urgent'
    assert agent._merge_safety('urgent','medical')=='urgent'
