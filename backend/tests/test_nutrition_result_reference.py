import asyncio
import json
from uuid import uuid4

import pytest

from app.services.agent.result_reference import result_sources, validate_result_reference
from app.services.agent.semantics import SemanticTask
from evals.v2.safety import deny_network
from tests.react_supervisor_fixtures import tool_call
from tests.react_readonly_fixtures import environment, chunks, semantic_outline, semantic_task, sse_body

SOURCE='给我制定一个减脂饮食计划'
REPLY='重新生成'


def memory():
    return {'short_memory':[
        {'role':'user','message_id':'training-user','content':'制定训练计划'},
        {'role':'assistant','message_id':'training-answer','content':'训练安排',
         'app_meta':{'plan_preview':{'preview_id':'training-preview'}}},
        {'role':'user','message_id':'nutrition-user','content':SOURCE},
        {'role':'assistant','message_id':'nutrition-answer','content':'饮食安排',
         'app_meta':{'result_targets':[{'task_type':'nutrition_advice','delivery_type':'plan'}]}}],
        'current_plan_preview':{'status':'ok','state':{'preview_id':'training-preview','message_id':'training-answer'}}}


def task(kind='nutrition_advice',ref=None,preview=''):
    return SemanticTask.model_validate({**semantic_task(kind,'question',REPLY),
        'delivery_type':'preview' if preview else 'plan','preview_ref':preview,'result_reference':ref})


@pytest.mark.parametrize('change',['missing','latest_old','wrong_domain','foreign','foreign_quote'])
def test_regeneration_rejects_unrelated_training_or_unbound_reference(change):
    state=memory()
    ref={'message_id':'nutrition-answer','selection':'latest','quote':REPLY}
    current=task(ref=ref)
    if change=='missing': current=task('training_plan',preview='training-preview')
    elif change=='latest_old': current=task('training_plan',{'message_id':'training-answer','selection':'latest','quote':REPLY},'training-preview')
    elif change=='wrong_domain': current=task('training_plan',ref)
    elif change=='foreign': current=task(ref={**ref,'message_id':'other-conversation-answer'})
    else: current=task(ref={**ref,'quote':'不存在的依据'})
    assert validate_result_reference(current,'follow_up',state)


def test_latest_nutrition_binds_original_goal_and_explicit_older_training_is_allowed():
    state=memory()
    current=task(ref={'message_id':'nutrition-answer','selection':'latest','quote':REPLY})
    assert validate_result_reference(current,'follow_up',state) is None
    assert current.context_ids==['nutrition-user','nutrition-answer']
    old=task('training_plan',{'message_id':'training-answer','selection':'explicit','quote':'之前的训练计划'},'training-preview')
    old.text='重新生成之前的训练计划'
    assert validate_result_reference(old,'follow_up',state) is None
    new=task('training_plan')
    assert validate_result_reference(new,'new',state) is None
    assert result_sources(state)[0]['task_types']==['nutrition_advice']


def test_legacy_answer_without_target_metadata_still_blocks_stale_preview():
    state=memory()
    state['short_memory'][-1].pop('app_meta')
    assert validate_result_reference(task('training_plan',preview='training-preview'),'follow_up',state)
    state['short_memory'][-1]['app_meta']={'wiki_view':'history'}
    assert all(s['message_id']!='nutrition-answer' for s in result_sources(state))


@pytest.mark.parametrize('transport',['json','sse'])
@pytest.mark.parametrize('unrelated_count',[0,100])
def test_diet_then_regenerate_does_not_use_existing_training_preview(tmp_path,monkeypatch,transport,unrelated_count):
    from app.db.memory_models import MemoryTurn
    from app.services.agent.plan_preview import _digest
    from tests.test_plan_semantic_input_budget import preview
    with deny_network() as audit, environment(tmp_path,monkeypatch,'nutrition-regeneration') as env:
        stored,answer=preview('demo-conversation')
        answer='旧的训练计划，仅用于验证不能抢占饮食目标。'
        stored['answer_digest']=_digest(answer)
        with env.factory.begin() as s:
            s.add(MemoryTurn(turn_id=str(uuid4()),request_id=stored['request_id'],
                conversation_id='demo-conversation',memory_epoch=0,status='completed',
                messages=[{'role':'assistant','message_id':stored['message_id'],'content':answer,
                           'app_meta':{'plan_preview':stored}}]))
            for i in range(unrelated_count):
                s.add(MemoryTurn(turn_id=str(uuid4()),request_id='unrelated-'+str(i),
                    conversation_id='other-conversation',memory_epoch=0,status='completed',
                    messages=[{'role':'user','content':'其他会话，不能作为来源'}]))
        before=env.snapshot()
        from app.services.memory import compact
        async def load(user_id,cid,query=''):
            return {'memory_backend':'compact_v1','recent_result_dialogue':compact.latest_result_dialogue(cid,0)}
        env.load.side_effect=load
        env.mock_rag(chunks(SOURCE))
        payloads=[]
        selected_sources=[]
        def semantic(source):
            t=semantic_task('nutrition_advice','question',source)
            t['delivery_type']='plan'
            if source==REPLY:
                # Same source inventory supplied to the actual semantic pass.
                latest=selected_sources[0]
                t['result_reference']={'message_id':latest['message_id'],'selection':'latest','quote':REPLY}
            return semantic_outline(source,[t]) | {'relation':'follow_up' if source==REPLY else 'new'}
        def policy(index,messages):
            return {'role':'assistant','content':'','tool_calls':[
                tool_call('diet-'+str(index),'advise_nutrition',{'query':SOURCE})]}
        llm=env.mock_llm(policy=policy,semantic=semantic,final='减脂饮食方案：三餐与份量示例。')
        structured=llm.complete_structured
        async def capture_structured(messages,schema,**kwargs):
            if kwargs.get('name')=='return_request_understanding':
                payload=json.loads(messages[-1]['content'])
                selected_sources[:]=payload.get('result_sources') or []
            return await structured(messages,schema,**kwargs)
        monkeypatch.setattr(llm,'complete_structured',capture_structured)
        original=llm.complete_message
        async def capture(messages,**kwargs):
            if not kwargs.get('tools'):payloads.append(messages)
            return await original(messages,**kwargs)
        monkeypatch.setattr(llm,'complete_message',capture)
        first=env.post('diet-first',message=SOURCE)
        with env.factory() as s:
            row=s.query(MemoryTurn).filter_by(request_id='diet-first').one()
            assert row.messages[-1]['app_meta']['result_targets'][0]['task_type']=='nutrition_advice'
        result=env.post('diet-regenerate',message=REPLY) if transport=='json' else sse_body(env.stream('diet-regenerate',message=REPLY))
        assert result['answer']=='减脂饮食方案：三餐与份量示例。',result
        assert result['metrics']['react_status']=='complete' and not result['metrics'].get('plan_preview')
        assert all(h['name']!='plan_training' for h in env.handlers)
        last=json.dumps(payloads[-1],ensure_ascii=False)
        assert SOURCE in last and 'referenced_result' in last
        assert '旧的训练计划，仅用于验证' not in last
        count=len(llm.calls)
        replay=env.post('diet-regenerate',message=REPLY)
        assert replay['answer']==result['answer'] and len(llm.calls)==count
        again=env.post('diet-regenerate-again',message=REPLY)
        assert again['answer']==result['answer'] and not again['metrics'].get('plan_preview')
        assert SOURCE in json.dumps(payloads[-1],ensure_ascii=False)
        # Deleting/hiding the original source cannot be bypassed by the
        # assistant's retained source pointer.
        with env.factory.begin() as s:
            row=s.query(MemoryTurn).filter_by(request_id='diet-first').one()
            from app.core import clock
            row.hidden_at=clock.utc_now()
        env.sql.clear()
        visible=compact.latest_result_dialogue('demo-conversation',0)
        assert all(m.get('content')!=SOURCE for m in visible)
        assert sum(statement.lstrip().upper().startswith('SELECT') for statement in env.sql)==2
        assert all(m.get('content')!='其他会话，不能作为来源' for m in visible)
        assert env.snapshot()==before
    assert audit=={'blocked_connections':0,'blocked_dns':0}
