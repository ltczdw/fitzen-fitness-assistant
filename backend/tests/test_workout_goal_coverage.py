"""Tests for workout goal coverage."""
import copy
import json
from uuid import uuid4

import pytest

from tests.test_workout_source_bound_writes import (
    C3HTTPHarness, FACT_TWO, SOURCE_TWO, c3_env, mock_provider_transport,
    native, native_tools, record, ref_of, saved_rows, semantic_response,
)


@pytest.mark.parametrize('backend', ['sqlite'])
def test_full_application_goal_reference_saves_both_actions(c3_env, monkeypatch, backend):
    responses = [
        semantic_response([{'task_type': 'workout_record', 'action': 'record',
                            'text': SOURCE_TWO, 'fact_text': FACT_TWO,
                            'workout_sources': [
                                {'exercise': '哑铃侧平举',
                                 'quote': '哑铃侧平举：单只4公斤，2组12次'},
                                {'exercise': '哑铃划船',
                                 'quote': '哑铃划船：单只9公斤，2组8次'},
                            ]}]),
        native_tools([('save-full-goal', 'record_workout',
                       {'source_ref': ref_of(SOURCE_TWO, SOURCE_TWO)})]),
        native({'records': [record('哑铃侧平举', 4, 2, 12),
                            record('哑铃划船', 9, 2, 8)]}),
    ]
    mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    status, payload = harness.post_json(SOURCE_TWO, request_id=str(uuid4()))
    assert status == 200
    assert saved_rows(c3_env) == [('哑铃侧平举', 4.0, 2, 12),
                                  ('哑铃划船', 9.0, 2, 8)], payload
    assert payload['metrics'].get('react_write') == 'completed', payload


@pytest.mark.parametrize('backend', ['sqlite'])
def test_action_without_numeric_facts_is_not_silently_dropped(c3_env, monkeypatch, backend):
    source = '记录哑铃卧推10kg3组12次；俯卧撑也保存。'
    fact = '哑铃卧推10kg3组12次；俯卧撑'
    first = '记录哑铃卧推10kg3组12次；'
    responses = [
        semantic_response([{'task_type': 'workout_record', 'action': 'record',
                            'text': source, 'fact_text': fact,
                            'workout_sources': [
                                {'exercise': '哑铃卧推',
                                 'quote': '哑铃卧推10kg3组12次'},
                                {'exercise': '俯卧撑', 'quote': '俯卧撑'},
                            ]}]),
        native_tools([('save-first-only', 'record_workout',
                       {'source_ref': ref_of(source, first)})]),
        native({'records': [record('哑铃卧推', 10, 3, 12)]}),
    ]
    mock_provider_transport(monkeypatch, responses)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    status, payload = harness.post_json(source, request_id=str(uuid4()))
    assert status == 200
    assert saved_rows(c3_env) == [('哑铃卧推', 10.0, 3, 12)], payload
    assert payload['metrics'].get('react_write') != 'completed', payload
    assert '俯卧撑' in payload.get('answer', ''), payload


@pytest.mark.parametrize('mutation', ['foreign_quote', 'overlap', 'foreign_exercise'])
def test_action_source_contract_rejects_invalid_evidence(mutation):
    from app.services.agent import semantics

    units = [
        {'exercise': '哑铃侧平举', 'quote': '哑铃侧平举：单只4公斤，2组12次'},
        {'exercise': '哑铃划船', 'quote': '哑铃划船：单只9公斤，2组8次'},
    ]
    task = {'task_type': 'workout_record', 'action': 'record',
            'text': SOURCE_TWO, 'fact_text': FACT_TWO, 'workout_sources': units}
    message = semantic_response([task])['choices'][0]['message']
    raw = json.loads(message['tool_calls'][0]['function']['arguments'])
    bad = copy.deepcopy(raw)
    if mutation == 'foreign_quote':
        bad['tasks'][0]['workout_sources'][1]['quote'] = '哑铃划船：单只99公斤，2组8次'
    elif mutation == 'overlap':
        bad['tasks'][0]['workout_sources'][1] = copy.deepcopy(units[0])
    else:
        bad['tasks'][0]['workout_sources'][1]['exercise'] = '杠铃深蹲'
    token = semantics.request_understanding.set({})
    try:
        good = semantics.validate_understanding(SOURCE_TWO, raw)
        assert good['validated_tasks'][0]['workout_sources'] == units
        with pytest.raises(semantics.SemanticValidationError) as error:
            semantics.validate_understanding(SOURCE_TWO, bad)
        assert error.value.code == 'semantic_workout_source_invalid'
        assert semantics.get_understanding(SOURCE_TWO) is None
    finally:
        semantics.request_understanding.reset(token)
