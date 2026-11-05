import asyncio
from unittest.mock import AsyncMock
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.db.memory_models import MemoryTurn
from app.services.agent import planning_questions
from app.services.agent.task_information import check_plan_information
from tests.test_plan_followup_contract import fixture, send, SOURCE, REPLY, c3_env, C3HTTPHarness


@pytest.mark.parametrize('backend',['sqlite'])
@pytest.mark.parametrize('transport',['json','sse'])
def test_split_cycle_does_not_require_or_invent_weekly_frequency(c3_env,monkeypatch,backend,transport):
    state=fixture(monkeypatch,cycle=True)
    harness=C3HTTPHarness(monkeypatch,c3_env)
    _,first=send(harness,SOURCE,str(uuid4()),transport)
    assert [f['name'] for f in first['clarifications'][0]['fields']]==['experience'],first
    rid=str(uuid4())
    _,result=send(harness,REPLY,rid,transport)
    assert not result['clarifications'] and all(n in result['answer'] for n in ('卧推','划船','深蹲')),result
    assert result['answer'].count('合计：90分钟')==3 and '一个训练循环展示3个训练日' in result['answer'],result
    calls=len(state['calls'])
    _,replayed=send(harness,REPLY,rid)
    assert replayed['answer']==result['answer'] and len(state['calls'])==calls


@pytest.mark.parametrize('backend',['sqlite'])
def test_challenging_a_question_gets_explanation_and_keeps_pending_state(c3_env,monkeypatch,backend):
    challenge='你是不理解三分化什么意思吗，还要我给你补充训练次数？'
    state=fixture(monkeypatch,case='explain',reply=challenge)
    harness=C3HTTPHarness(monkeypatch,c3_env)
    _,first=send(harness,SOURCE,str(uuid4()))
    rid=str(uuid4())
    _,response=send(harness,challenge,rid)
    assert '三类训练日' in response['answer'] and '安排日历' in response['answer'],response
    assert response['answer']!=first['answer']
    assert response['clarifications']==first['clarifications']
    with c3_env.connect() as c:
        operations=c.execute(select(MemoryTurn.operations).where(MemoryTurn.conversation_id==harness.conversation_id)).scalars().all()
    drafts=[o['tasks'][planning_questions.KEY] for o in operations if o and planning_questions.KEY in o.get('tasks',{})]
    assert len(drafts)==1 and drafts[0]['state']=='waiting'
    assert not any(planning_questions.BINDING in (o or {}).get('tasks',{}) for o in operations)
    count=len(state['calls'])
    _,replay=send(harness,challenge,rid)
    assert replay['answer']==response['answer'] and len(state['calls'])==count


def cycle_information(**changes):
    return {'scope':'cycle','scope_quote':'三分化','cycle_sessions':3,
            'cycle_quote':'三分化','cycle_quantity_quote':'三','facts':[
                {'field':'goal','source_id':'current','quote':'增肌'},
                {'field':'experience','source_id':'current','quote':'训练一年'},
                {'field':'equipment','source_id':'current','quote':'健身房'},
                {'field':'session_time','source_id':'current','quote':'每次90分钟',
                 'quantity_quote':'90分钟','seconds':5400}],**changes}


@pytest.mark.parametrize('changes',[{}, {'cycle_sessions':2},{'cycle_quote':'四分化'},
                                 {'cycle_quantity_quote':'三分化'}, {'cycle_sessions':None}])
def test_cycle_count_is_a_source_bound_numeric_field(changes):
    source='目标增肌，训练一年，在健身房，每次90分钟，请给我三分化训练计划。'
    value=cycle_information(**changes)
    from app.services.agent.task_information import PlanInformation
    llm=SimpleNamespace(complete_structured=AsyncMock(side_effect=lambda *a,**k:PlanInformation.model_validate(value)))
    details,question=asyncio.run(check_plan_information(llm,{'raw_message':source,'task_text':source}))
    if not changes:
        assert question is None and details['cycle_sessions']==3
        assert details['available_days_count'] is None
    else:
        assert question.category=='service_unavailable' and question.fields==[]


def test_weekly_scope_still_requires_real_frequency():
    from app.services.agent.task_information import PlanInformation
    source='目标增肌，训练一年，在健身房，每次90分钟，请给我三分化的每周训练计划。'
    value=cycle_information(scope='weekly',scope_quote='每周训练计划',cycle_sessions=None,
                            cycle_quote='',cycle_quantity_quote='')
    llm=SimpleNamespace(complete_structured=AsyncMock(return_value=PlanInformation.model_validate(value)))
    _,question=asyncio.run(check_plan_information(llm,{'raw_message':source,'task_text':source}))
    assert [f.name for f in question.fields]==['available_days']


def test_weekly_missing_fields_are_asked_together():
    from app.services.agent.task_information import PlanInformation
    source='目标增肌，训练一年，请给我每周训练计划。'
    value={'scope':'weekly','scope_quote':'每周训练计划','facts':[
        {'field':'goal','source_id':'current','quote':'增肌'},
        {'field':'experience','source_id':'current','quote':'训练一年'}]}
    llm=SimpleNamespace(complete_structured=AsyncMock(return_value=PlanInformation.model_validate(value)))
    _,question=asyncio.run(check_plan_information(llm,{'raw_message':source,'task_text':source}))
    assert [f.name for f in question.fields]==['available_days']


def test_cycle_count_is_independent_from_weekly_profile_frequency():
    from app.services.agent.training_timing import expected_session_count, count_quantity
    assert expected_session_count('cycle',6,3)==3
    assert expected_session_count('weekly',6,3)==6
    for quote in ('3','三','３'):
        assert count_quantity(quote)==3
    for quote in ('3天','三分化','三或六','3.5','0','22'):
        assert count_quantity(quote) is None


@pytest.mark.parametrize('case',['missing_explanation','foreign_quote','urgent_missing_explanation'])
def test_explanation_cannot_mutate_or_bypass_safety(compact_db,case):
    from tests.test_task_information import draft_fixture, begin
    from app.schemas.chat import ChatRequest
    from app.services.memory import compact
    original,origin=draft_fixture(compact_db)
    reply='我现在胸痛，请解释' if case=='urgent_missing_explanation' else '为什么需要每周次数？'
    request=ChatRequest(message=reply,conversation_id=original.conversation_id,mode='react_readonly')
    tid=begin(request)
    async def decide(messages,schema,**kwargs):
        import json
        payload=json.loads(messages[-1]['content'])
        return schema.model_validate({'draft_id':payload['draft_id'],'action':'explain',
            'quote':'外来问题' if case=='foreign_quote' else reply,
            'explanation':'说明当前问题' if case=='foreign_quote' else '',
            'safety':{'level':'urgent' if case=='urgent_missing_explanation' else 'normal',
                      'evidence':reply if case=='urgent_missing_explanation' else ''}})
    token=compact.current_turn.set(tid)
    try:
        result=asyncio.run(planning_questions.continue_request(SimpleNamespace(complete_structured=decide),request))
        if case=='urgent_missing_explanation':
            assert result['safety_level']=='urgent' and result['intent']=='symptom_triage'
        else:
            assert result['safety_status']=='unavailable'
        with compact_db() as s:
            assert s.get(MemoryTurn,origin).operations['tasks'][planning_questions.KEY]['state']=='waiting'
            assert s.get(MemoryTurn,tid).operations is None
    finally:
        compact.current_turn.reset(token)
