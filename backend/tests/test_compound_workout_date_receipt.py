"""Per-record dates and committed receipts survive mixed write/read delivery."""
import asyncio
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services.agent.react_models import DomainObservation
from app.services.agent.react_write_adapters import WriteAdapters
from app.services.agent.react_graph import completed_delivery_goals, _extract_final_answer
from tests.react_phase_c2_3_support import (
    c3_env, C3HTTPHarness, all_business_rows, workout_rows, mock_provider_transport,
)
from tests.test_workout_save_repair_20261102 import (
    semantic_response, native, native_tools, ref_of, record,
)


@pytest.mark.parametrize('day,expected', [('昨天','2026-11-03'),('今天','2026-11-04'),
                                         ('2026-10-31','2026-10-31'),('','2026-11-04')])
def test_adapter_preserves_date_and_frozen_basis(monkeypatch, day, expected):
    from app.core import clock
    # Wall clock has moved; the request must still resolve against its frozen day.
    monkeypatch.setattr(clock, 'today', lambda: date(2026,11,8))
    source = f'保存{day}哑铃卧推15*8*4'
    captured = []
    async def extract(message, *, reference_date, task_source):
        assert message == source and reference_date == date(2026,11,4)
        assert task_source==source
        return [record('哑铃卧推',15,4,8)]
    async def dispatch(span, canonical, context):
        captured.append(canonical)
        return DomainObservation(status='completed',commit_state='committed',
            receipt_refs=['workout_record:1'], result={'proof_type':'workout_db_v1'})
    runtime=SimpleNamespace(db_date=lambda:'2026-11-04', dispatch_workout=dispatch,
        agent=SimpleNamespace(graph=SimpleNamespace(input_parser=SimpleNamespace(
            workout_extractor=SimpleNamespace(extract=extract)))))
    obs=asyncio.run(WriteAdapters(runtime).record_workout({'source_ref':ref_of(source,source)},
        {'source_text':source,'candidate_span':{'start':0,'end':len(source),'goal_id':'save'}}))
    assert captured[0]['date']==expected
    assert expected in obs.summary and '15 kg × 8 次 × 4 组' in obs.summary


def delivery_state(*, committed=True, status='completed', receipts=True):
    return {'intent_candidates':[
        {'id':'save','task_type':'workout_record','action':'record','side_effect':True},
        {'id':'diet','task_type':'nutrition_advice','action':'question'}],
        'covered_goals':['save','diet'], 'committed_tasks':['save'] if committed else [],
        'goal_results':{
            'save':{'status':status,'commit_state':'committed',
                    'receipt_refs':['workout_record:1'] if receipts else [],
                    'summary':'已保存训练记录：2026-11-03，哑铃卧推，15 kg × 8 次 × 4 组。'},
            'diet':{'status':'completed','summary':'早餐燕麦、午餐米饭鸡肉、晚餐豆腐。'}},
        'messages':[{'role':'assistant','content':'仅饮食总结。'}]}


def test_final_delivery_preserves_both_in_order():
    state=delivery_state()
    answer=_extract_final_answer(state)
    assert answer==state['goal_results']['save']['summary']+'\n\n'+state['goal_results']['diet']['summary']


@pytest.mark.parametrize('mutation',[{'committed':False},{'status':'unavailable'},{'receipts':False}])
def test_missing_commit_evidence_cannot_claim_save(mutation):
    goals=completed_delivery_goals(delivery_state(**mutation))
    assert [g['id'] for g in goals]==['diet']


@pytest.mark.parametrize('backend',['sqlite'])
@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('narrow',[False,True])
def test_real_mixed_request_date_receipt_and_idempotent_replay(c3_env,monkeypatch,backend,stream,narrow):
    from app.services.agent import react_readonly_workers
    from app.core import clock
    save='帮我记录昨天哑铃卧推15*8*4'
    diet='再给我安排一下今天的饮食'
    source=save+'，'+diet
    tasks=[{'task_type':'workout_record','action':'record','text':save,'fact_text':save},
           {'task_type':'nutrition_advice','action':'question','text':diet}]
    from datetime import timedelta
    extracted=record('哑铃卧推',15,4,8)
    extracted['record_date']=(clock.today()-timedelta(days=1)).isoformat()
    responses=[semantic_response(tasks), native_tools([
        ('save','record_workout',{'source_ref':ref_of(source,'哑铃卧推15*8*4' if narrow else save)}),
        ('diet','advise_nutrition',{'source_ref':ref_of(source,diet),'query':diet})]),
        native({'records':[extracted]})]
    seen=mock_provider_transport(monkeypatch,responses * 2)
    async def advice(args,context):
        return DomainObservation(status='completed',summary='今天饮食：早餐燕麦，午餐米饭鸡肉，晚餐豆腐。')
    monkeypatch.setitem(react_readonly_workers.ANSWER_WORKERS,'advise_nutrition',advice)
    harness=C3HTTPHarness(monkeypatch,c3_env,write_enabled=True)
    rid=str(uuid4())
    if stream:
        status,events,body=harness.stream(source,request_id=rid)
        assert status==200
        assert events and 'metrics' in events[-1],(events,body)
        payload={**events[-1],'answer':''.join(str(e.get('content') or '') for e in events)}
    else:
        status,payload=harness.post_json(source,request_id=rid)
        assert status==200,payload
    rows=workout_rows(c3_env)
    from sqlalchemy import text
    with c3_env.connect() as conn:
        frozen=conn.execute(text('SELECT time_base FROM agent_execution_requests WHERE request_id=:r'),{'r':rid}).scalar_one()
    from datetime import datetime,timedelta,timezone
    expected=(datetime.fromisoformat(frozen).astimezone(timezone(timedelta(hours=8))).date()-timedelta(days=1)).isoformat()
    assert len(rows)==1 and rows[0]['workout_date']==expected,(rows,payload)
    assert payload['answer'].startswith('已保存训练记录：'+expected),payload
    assert '早餐燕麦' in payload['answer'] and payload['metrics']['receipt_refs']
    before=all_business_rows(c3_env)
    calls=len(seen)
    status,replay=harness.post_json(source,request_id=rid)
    assert status==200 and replay['answer']==payload['answer']
    assert replay['metrics']['receipt_refs']==payload['metrics']['receipt_refs']
    assert all_business_rows(c3_env)==before and len(seen)==calls
    status,repeated=harness.post_json(source,request_id=str(uuid4()))
    assert status==200 and repeated['answer']==payload['answer']
    assert repeated['metrics']['receipt_refs'] and len(workout_rows(c3_env))==2
    assert all(r['workout_date']==expected for r in workout_rows(c3_env))


def test_multi_action_dates_do_not_share_today_default(monkeypatch):
    from app.core import clock
    monkeypatch.setattr(clock,'today',lambda:date(2026,11,9))
    first='昨天哑铃卧推15*8*4'
    second='今天哑铃划船10*12*3'
    source='保存'+first+'；'+second
    captured=[]
    async def extract(message,*,reference_date,task_source):
        return [record('哑铃卧推',15,4,8),record('哑铃划船',10,3,12)]
    async def dispatch(span,canonical,context):
        captured.append(canonical)
        return DomainObservation(status='completed',commit_state='committed',
            receipt_refs=['workout_record:'+str(len(captured))],result={'proof_type':'workout_db_v1'})
    runtime=SimpleNamespace(db_date=lambda:'2026-11-04',dispatch_workout=dispatch,
        agent=SimpleNamespace(graph=SimpleNamespace(input_parser=SimpleNamespace(
            workout_extractor=SimpleNamespace(extract=extract)))))
    obs=asyncio.run(WriteAdapters(runtime).record_workout({'source_ref':ref_of(source,source)},
        {'source_text':source,'candidate_span':{'start':0,'end':len(source),'goal_id':'save'},
         'workout_sources':[{'exercise':'哑铃卧推','quote':first},
                            {'exercise':'哑铃划船','quote':second}]}))
    assert [r['date'] for r in captured]==['2026-11-03','2026-11-04']
    assert obs.status.value=='completed' and len(obs.receipt_refs)==2
    assert '2026-11-03' in obs.summary and '2026-11-04' in obs.summary


def test_date_prompt_uses_same_frozen_basis():
    from app.services.agent.deepseek_workout import DeepSeekWorkoutExtractor
    prompt=DeepSeekWorkoutExtractor._prompt_for_current_date(date(2026,1,1))
    assert '今天=2026-01-01' in prompt and '昨天=2025-12-31' in prompt


def test_child_receipt_keeps_parent_request_order():
    state=delivery_state()
    state['intent_candidates'].append({**state['intent_candidates'][0],
        'id':'unit','parent_goal_id':'save','source_start':4})
    state['goal_results']['unit']=state['goal_results'].pop('save')
    state['covered_goals']=['unit','diet']
    state['committed_tasks']=['unit']
    assert [g['id'] for g in completed_delivery_goals(state)]==['unit','diet']


def test_extractor_receives_only_authorized_task_context(monkeypatch):
    from app.core.config import settings
    from app.services.agent.deepseek_workout import DeepSeekWorkoutExtractor
    import json
    captured=[]
    class LLM:
        def configured(self):return True
        async def complete(self,messages,**kwargs):
            captured.extend(messages)
            return json.dumps({'records':[record('哑铃卧推',15,4,8)]})
    monkeypatch.setattr(settings,'workout_parser_enabled',True)
    result=asyncio.run(DeepSeekWorkoutExtractor(LLM()).extract('哑铃卧推15*8*4',
        reference_date=date(2026,11,4),task_source='保存昨天哑铃卧推15*8*4'))
    assert result and len(result)==1
    assert json.loads(captured[1]['content'])=={
        'user_message':'哑铃卧推15*8*4','authorized_task':'保存昨天哑铃卧推15*8*4'}
    assert '昨天=2026-11-03' in captured[0]['content']
