"""Real ASGI/C1 saves after source-bound clarification; providers are mocked."""
import json
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select

from app.db.memory_models import MemoryTurn
from tests.react_phase_c2_3_support import C3SQLFault, set_write_flag

from tests.test_workout_source_bound_writes import (
    C3HTTPHarness, c3_env, mock_provider_transport, native, native_tools,
    record, ref_of, saved_rows, semantic_response,
)

SOURCE = '帮我记录哑铃玩具10kg*10*4'


def tool_response(name, value):
    result = semantic_response([])
    function = result['choices'][0]['message']['tool_calls'][0]['function']
    function.update(name=name, arguments=json.dumps(value, ensure_ascii=False))
    return result


def provider(monkeypatch, *, reply='是弯举', decision='update', accepts=True,
             urgent=False, flags=None):
    state = {'canonical': None, 'calls': []}

    def respond(request):
        body = json.loads(request.content)
        tools = body.get('tools') or []
        names = [tool['function']['name'] for tool in tools]
        state['calls'].append(names)
        if 'return_request_understanding' in names:
            payload = json.loads(body['messages'][-1]['content'])
            query = payload['query']
            action = 'question' if query in {SOURCE, reply} else 'record'
            value = semantic_response([{'task_type': 'workout_record',
                'action': action, 'text': query, 'history_requirement': 'none'}])
            function = value['choices'][0]['message']['tool_calls'][0]['function']
            raw = json.loads(function['arguments'])
            if query == SOURCE:
                raw.update(needs_clarification=True,
                           clarification='请确认“哑铃玩具”的动作名称。')
            elif query == reply:
                assert payload.get('pending_record_draft'), 'reply lost its bound draft context'
                if urgent:
                    raw['safety'] = {'level': 'urgent', 'evidence': reply,
                                     'categories': ['current_symptom'], 'reason': 'scripted urgent'}
            else:
                assert '弯举' in query and '10kg' in query
                assert 'pending_record_draft' not in payload, 'adopted draft remained current context'
                state['canonical'] = query
                raw['tasks'][0]['workout_sources'] = [
                    {'exercise': '弯举', 'quote': query}]
            function['arguments'] = json.dumps(raw, ensure_ascii=False)
        elif 'return_record_proposals' in names:
            value = tool_response('return_record_proposals', {'records': [
                {'kind': 'workout', 'source_quote': '哑铃玩具10kg*10*4',
                 'exercise': '哑铃玩具', 'weight_kg': 10, 'sets': 4, 'reps': 10}]})
        elif 'return_record_reply' in names:
            payload = json.loads(body['messages'][-1]['content'])
            assert payload['reply'] == reply
            value = tool_response('return_record_reply', {
                'draft_id': payload['draft_id'], 'action': decision,
                'accepts_all': accepts,
                'is_negated': False, 'is_hypothetical': False, 'is_third_party': False,
                **(flags or {}),
                'safety': {'level': 'urgent' if urgent else 'normal',
                           'evidence': reply if urgent else '',
                           'categories': ['current_symptom'] if urgent else [],
                           'reason': 'scripted current-reply safety'},
                'patches': ([{'index': 0, 'source_quote': reply, 'exercise': '弯举'}]
                            if decision == 'update' else [])})
        elif 'record_workout' in names:
            assert state['canonical']
            value = native_tools([('save-corrected', 'record_workout',
                                   {'source_ref': ref_of(state['canonical'], state['canonical'])})])
        elif not names:
            value = native({'records': [record('弯举', 10, 4, 10)]})
        else:
            raise AssertionError(f'unexpected provider protocol: {names}')
        return httpx.Response(200, json=value)

    mock_provider_transport(monkeypatch, [respond])
    return state


@pytest.mark.parametrize('backend', ['sqlite'])
@pytest.mark.parametrize('mode', ['auto', 'react_readonly', 'react_write'])
def test_corrected_action_is_saved_by_react_once(c3_env, monkeypatch, backend, mode):
    state = provider(monkeypatch)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    origin_id = str(uuid4())
    status, first = harness.post_json(SOURCE, mode=mode, request_id=origin_id)
    assert status == 200
    assert first['metrics'].get('execution_mode') == 'react_write', first
    assert '哑铃玩具' in first['answer'] and '10' in first['answer'], first
    assert not saved_rows(c3_env)
    first_calls = len(state['calls'])
    status, first_replay = harness.post_json(SOURCE, mode=mode, request_id=origin_id)
    assert status == 200 and first_replay['answer'] == first['answer']
    assert len(state['calls']) == first_calls and not saved_rows(c3_env)
    rid = str(uuid4())
    status, second = harness.post_json('是弯举', mode=mode, request_id=rid)
    assert status == 200
    assert second['metrics'].get('execution_mode') == 'react_write', second
    assert second['metrics'].get('react_write') == 'completed', second
    assert saved_rows(c3_env) == [('弯举', 10.0, 4, 10)], second
    assert second['metrics'].get('receipt_refs')
    from sqlalchemy.orm import sessionmaker
    from app.services.agent.execution_ledger_store import ExecutionLedgerStore
    from tests.react_phase_c2_3_support import API_OWNER
    snapshot = ExecutionLedgerStore(sessionmaker(bind=c3_env)).load_request(API_OWNER, rid)
    assert snapshot['status'] == 'completed', 'typed success left C1 unfinished'
    with c3_env.connect() as connection:
        messages = connection.execute(select(MemoryTurn.messages).where(
            MemoryTurn.request_id == rid)).scalar_one()
    assert messages[0]['content'] == '是弯举', 'canonical command overwrote user history'
    before = len(state['calls'])
    status, replay = harness.post_json('是弯举', mode=mode, request_id=rid)
    assert status == 200
    assert saved_rows(c3_env) == [('弯举', 10.0, 4, 10)]
    assert len(state['calls']) == before
    assert replay['metrics'].get('receipt_refs') == second['metrics'].get('receipt_refs')


@pytest.mark.parametrize('backend', ['sqlite'])
@pytest.mark.parametrize('case', ['cancel', 'urgent', 'third_party'])
def test_pending_reply_still_obeys_safety_and_authorization(c3_env, monkeypatch, backend, case):
    message = {'cancel': '取消保存', 'urgent': '是弯举，但我现在胸痛',
               'third_party': '我朋友说是弯举'}[case]
    state = provider(monkeypatch, reply=message,
                     decision='cancel' if case == 'cancel' else 'update',
                     accepts=case != 'cancel', urgent=case == 'urgent',
                     flags={'is_third_party': True} if case == 'third_party' else None)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    status, first = harness.post_json(SOURCE, mode='auto', request_id=str(uuid4()))
    assert status == 200 and first['metrics'].get('execution_mode') == 'react_write'
    status, response = harness.post_json(message, mode='auto', request_id=str(uuid4()))
    assert status == 200
    assert not saved_rows(c3_env), response
    if case == 'cancel':
        assert '取消' in response['answer']
    if case == 'urgent':
        assert response['safety_level'] == 'urgent'
        assert state['canonical'] is None
        assert not any('record_workout' in call for call in state['calls'])


@pytest.mark.parametrize('backend', ['sqlite'])
def test_disabled_react_does_not_claim_or_fall_back(c3_env, monkeypatch, backend):
    state = provider(monkeypatch)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    status, _ = harness.post_json(SOURCE, mode='auto', request_id=str(uuid4()))
    assert status == 200
    before = len(state['calls'])
    set_write_flag(monkeypatch, False)
    status, response = harness.post_json('是弯举', mode='auto', request_id=str(uuid4()))
    assert status == 200 and response['intent'] == 'execution_limited', response
    assert len(state['calls']) == before and not saved_rows(c3_env)
    set_write_flag(monkeypatch, True)
    status, response = harness.post_json('是弯举', mode='auto', request_id=str(uuid4()))
    # An intervening non-preview reply invalidates implicit full-preview
    # acceptance. Preserve that existing visibility/confirmation contract.
    assert status == 200 and response['intent'] == 'clarification', response
    assert '弯举' in response['answer'] and '10 kg' in response['answer']
    assert not saved_rows(c3_env)
    status, response = harness.post_json('是弯举', mode='auto', request_id=str(uuid4()))
    assert status == 200 and response['metrics'].get('react_write') == 'completed', response
    assert saved_rows(c3_env) == [('弯举', 10.0, 4, 10)]


@pytest.mark.parametrize('backend', ['sqlite'])
def test_bound_correction_resumes_after_tool_result_failure(c3_env, monkeypatch, backend):
    state = provider(monkeypatch)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    status, _ = harness.post_json(SOURCE, mode='auto', request_id=str(uuid4()))
    assert status == 200
    rid = str(uuid4())
    fault = C3SQLFault(c3_env, table='agent_execution_events', action='insert',
                       occurrence=1, match=lambda statement, parameters:
                       'tool_result' in str(parameters) and rid in str(parameters))
    fault.install()
    try:
        status, _ = harness.post_json('是弯举', mode='auto', request_id=rid)
    finally:
        fault.remove()
    assert status == 200 and fault.raised
    assert saved_rows(c3_env) == [('弯举', 10.0, 4, 10)]
    before = len(state['calls'])
    status, resumed = harness.post_json('是弯举', mode='auto', request_id=rid)
    from sqlalchemy.orm import sessionmaker
    from app.services.agent.execution_ledger_store import ExecutionLedgerStore
    from tests.react_phase_c2_3_support import API_OWNER
    snapshot = ExecutionLedgerStore(sessionmaker(bind=c3_env)).load_request(API_OWNER, rid)
    diagnostic = {'response': resumed, 'calls': state['calls'],
                  'checkpoint': snapshot.get('checkpoint'),
                  'events': [{'phase': event.get('phase'),
                              'status': (event.get('observation') or {}).get('status'),
                              'error': (event.get('observation') or {}).get('error')}
                             for event in snapshot.get('events') or []]}
    assert status == 200 and resumed['metrics'].get('react_write') == 'completed', json.dumps(diagnostic, ensure_ascii=False, indent=2)
    assert saved_rows(c3_env) == [('弯举', 10.0, 4, 10)]
    assert len(state['calls']) == before, 'resume repeated draft interpretation or model calls'


@pytest.mark.parametrize('backend', ['sqlite'])
def test_frontend_stream_continuation_and_json_replay(c3_env, monkeypatch, backend):
    state = provider(monkeypatch)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    monkeypatch.setattr(harness.settings, 'chat_default_mode', 'auto')
    status, first, _ = harness.stream(SOURCE, mode=None, request_id=str(uuid4()))
    assert status == 200 and first[-1]['metrics']['execution_mode'] == 'react_write'
    assert first[-1]['metrics']['react_write'] == 'needs_clarification'
    assert not saved_rows(c3_env)
    rid = str(uuid4())
    status, second, _ = harness.stream('是弯举', mode=None, request_id=rid)
    assert status == 200 and second[-1]['metrics']['react_write'] == 'completed', second[-1]
    assert saved_rows(c3_env) == [('弯举', 10.0, 4, 10)]
    before = len(state['calls'])
    status, replay = harness.post_json('是弯举', mode=None, request_id=rid)
    assert status == 200
    assert replay['metrics']['receipt_refs'] == second[-1]['metrics']['receipt_refs']
    assert len(state['calls']) == before and saved_rows(c3_env) == [('弯举', 10.0, 4, 10)]
