"""Actual three-turn contract: rename -> preview -> confirm, over ASGI/C1."""
import json
from uuid import uuid4

import httpx
import pytest

from tests.test_workout_save_repair_20261102 import (
    C3HTTPHarness,c3_env,mock_provider_transport,native,native_tools,record,ref_of,
    saved_rows,semantic_response,
)
from tests.test_react_record_continuation import SOURCE,tool_response

NAME_REPLY='是哑铃弯举'


def protocol(monkeypatch, *, confirmation='是', urgent=False, missing_safety=False,
             foreign_safety=False, wrong_target=False, medical=False):
    state={'calls':[],'general_queries':[],'canonical':None,'reply_count':0}
    def response(request):
        body=json.loads(request.content)
        tools=body.get('tools') or []
        names=[tool['function']['name'] for tool in tools]
        state['calls'].append(names)
        if 'return_request_understanding' in names:
            payload=json.loads(body['messages'][-1]['content'])
            source=payload['query']
            state['general_queries'].append(source)
            action='question' if source==SOURCE else 'answer' if source==NAME_REPLY else 'record'
            if source==confirmation: action='confirm'
            result=semantic_response([{'task_type':'workout_record','action':action,
                                       'text':source,'history_requirement':'none'}])
            function=result['choices'][0]['message']['tool_calls'][0]['function']
            value=json.loads(function['arguments'])
            if source==SOURCE:
                value.update(needs_clarification=True,clarification='请确认动作名称。')
            elif source==confirmation:
                # Recorded production failure: a context reference was placed
                # in a field restricted to the one-word current source.
                value['tasks'][0]['workout_sources']=[
                    {'exercise':'哑铃弯举','quote':'哑铃弯举'}]
            elif source!=NAME_REPLY:
                assert '哑铃弯举' in source and '10kg' in source
                state['canonical']=source
                value['tasks'][0]['workout_sources']=[{'exercise':'哑铃弯举','quote':source}]
            function['arguments']=json.dumps(value,ensure_ascii=False)
        elif 'return_record_proposals' in names:
            result=tool_response('return_record_proposals',{'records':[
                {'exercise':'哑铃玩具','source_quote':'哑铃玩具10kg*10*4',
                 'weight_kg':10,'reps':10,'sets':4}]})
        elif 'return_record_reply' in names:
            payload=json.loads(body['messages'][-1]['content'])
            update=payload['reply']==NAME_REPLY
            state['reply_count']+=1
            value={'draft_id':payload['draft_id'],'action':'update' if update else 'accept',
                   'accepts_all':not update,'is_negated':False,'is_hypothetical':False,
                   'is_third_party':False,'patches':[
                       {'index':0,'exercise':'哑铃弯举','source_quote':NAME_REPLY}] if update else []}
            if wrong_target and not update:
                value['draft_id']='foreign-draft'
            fields=tools[names.index('return_record_reply')]['function']['parameters'].get('properties',{})
            if 'safety' in fields and (update or not missing_safety):
                value['safety']={'level':'urgent' if urgent and not update else 'normal',
                                 'evidence':confirmation if urgent and not update else '',
                                 'categories':['current_symptom'] if urgent and not update else [],
                                 'reason':'scripted current-reply assessment'}
                if foreign_safety and not update:
                    value['safety'].update(level='urgent',evidence='昨天胸痛',categories=['current_symptom'])
                if medical and not update:
                    value['safety'].update(level='medical',evidence=confirmation,categories=['current_symptom'])
            result=tool_response('return_record_reply',value)
        elif 'record_workout' in names:
            assert state['canonical']
            result=native_tools([('save-confirmed','record_workout',
                                 {'source_ref':ref_of(state['canonical'],state['canonical'])})])
        else:
            assert not names,names
            result=native({'records':[record('哑铃弯举',10,4,10)]})
        return httpx.Response(200,json=result)
    mock_provider_transport(monkeypatch,[response])
    return state


def send(harness,message,rid,transport='json'):
    if transport=='json': return harness.post_json(message,mode='react_readonly',request_id=rid)
    status,events,_=harness.stream(message,mode='react_readonly',request_id=rid)
    return status,{**events[-1],'answer':''.join(str(e.get('content') or '') for e in events)}


@pytest.mark.parametrize('backend',['sqlite'])
@pytest.mark.parametrize('transport',['json','sse'])
def test_three_turn_confirmation_does_not_reextract_current_workout_facts(
        c3_env,monkeypatch,backend,transport):
    state=protocol(monkeypatch)
    harness=C3HTTPHarness(monkeypatch,c3_env,write_enabled=True)
    status,_=send(harness,SOURCE,str(uuid4()),transport)
    assert status==200 and not saved_rows(c3_env)
    status,preview=send(harness,NAME_REPLY,str(uuid4()),transport)
    assert status==200 and '哑铃弯举' in preview['answer']
    assert not saved_rows(c3_env)
    rid=str(uuid4())
    status,saved=send(harness,'是',rid,transport)
    assert status==200
    assert saved['metrics'].get('react_write')=='completed',saved
    assert saved_rows(c3_env)==[('哑铃弯举',10.0,4,10)]
    assert '是' not in state['general_queries'],state
    assert NAME_REPLY not in state['general_queries'],state
    assert state['reply_count']==2
    calls=len(state['calls'])
    status,replay=send(harness,'是',rid)
    assert status==200 and len(state['calls'])==calls
    assert replay['metrics'].get('receipt_refs')==saved['metrics'].get('receipt_refs')
    assert saved_rows(c3_env)==[('哑铃弯举',10.0,4,10)]


@pytest.mark.parametrize('backend',['sqlite'])
@pytest.mark.parametrize('case',['urgent','missing_safety','foreign_safety','wrong_target','urgent_wrong_target'])
def test_confirmation_requires_its_own_current_safety(c3_env,monkeypatch,backend,case):
    is_urgent=case in {'urgent','urgent_wrong_target'}
    reply='是，但现在胸痛' if is_urgent else '是'
    state=protocol(monkeypatch,confirmation=reply,urgent=is_urgent,
                   missing_safety=case=='missing_safety',foreign_safety=case=='foreign_safety',
                   wrong_target=case in {'wrong_target','urgent_wrong_target'})
    harness=C3HTTPHarness(monkeypatch,c3_env,write_enabled=True)
    send(harness,SOURCE,str(uuid4()))
    send(harness,NAME_REPLY,str(uuid4()))
    status,response=send(harness,reply,str(uuid4()))
    assert status==200 and not saved_rows(c3_env),response
    assert state['canonical'] is None
    if is_urgent:
        assert response['safety_level']=='urgent',response
    else:
        assert response['metrics'].get('react_write')!='completed'
        if case=='wrong_target':
            assert '急诊' not in response['answer'] and '安全语义判断' not in response['answer']


@pytest.mark.parametrize('backend',['sqlite'])
def test_current_reply_medical_status_survives_canonical_record_and_replay(c3_env,monkeypatch,backend):
    reply='是，今天肩部有点不适'
    state=protocol(monkeypatch,confirmation=reply,medical=True)
    harness=C3HTTPHarness(monkeypatch,c3_env,write_enabled=True)
    send(harness,SOURCE,str(uuid4()))
    send(harness,NAME_REPLY,str(uuid4()))
    rid=str(uuid4())
    status,response=send(harness,reply,rid)
    assert status==200 and response['metrics'].get('react_write')=='completed',response
    assert response['safety_level']=='medical',response
    assert saved_rows(c3_env)==[('哑铃弯举',10.0,4,10)]
    calls=len(state['calls'])
    status,replayed=send(harness,reply,rid)
    assert status==200 and replayed['safety_level']=='medical',replayed
    assert len(state['calls'])==calls
