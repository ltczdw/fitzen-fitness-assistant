"""A normal workout must use the same ReAct writer as a clarified workout."""
from uuid import uuid4

import pytest

from tests.test_react_record_continuation import (
    C3HTTPHarness, c3_env, provider, saved_rows,
)
from tests.react_phase_c2_3_support import set_write_flag
from tests.test_workout_save_repair_20261102 import (
    mock_provider_transport, semantic_response, native_tools, native, ref_of,
)


@pytest.mark.parametrize('backend', ['sqlite'])
@pytest.mark.parametrize('mode', ['react_readonly', 'auto', 'react_write'])
def test_normal_workout_routes_to_react_and_replays(c3_env, monkeypatch, backend, mode):
    state = provider(monkeypatch)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    source = '记录弯举10kg*10*4'
    rid = str(uuid4())
    status, response = harness.post_json(source, mode=mode, request_id=rid)
    assert status == 200
    assert response['metrics'].get('execution_mode') == 'react_write', response
    assert response['metrics'].get('react_write') == 'completed', response
    assert saved_rows(c3_env) == [('弯举', 10.0, 4, 10)]
    calls = len(state['calls'])
    status, replay = harness.post_json(source, mode=mode, request_id=rid)
    assert status == 200
    assert replay['metrics'].get('receipt_refs') == response['metrics'].get('receipt_refs')
    assert len(state['calls']) == calls
    assert saved_rows(c3_env) == [('弯举', 10.0, 4, 10)]


@pytest.mark.parametrize('backend', ['sqlite'])
def test_disabled_write_does_not_fall_back_to_legacy(c3_env, monkeypatch, backend):
    provider(monkeypatch)
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    set_write_flag(monkeypatch, False)
    status, response = harness.post_json('记录弯举10kg*10*4', mode='react_readonly',
                                          request_id=str(uuid4()))
    assert status == 200
    assert response['intent'] == 'execution_limited', response
    assert response['metrics'].get('mode_reason') == 'react_write_disabled', response
    assert not saved_rows(c3_env)


@pytest.mark.parametrize('backend', ['sqlite'])
@pytest.mark.parametrize('flag,prefix', [
    ('is_negated', '不要保存'), ('is_hypothetical', '假如记录'),
    ('is_third_party', '我朋友记录'),
])
def test_readonly_upgrade_retains_negative_and_third_party_guards(
        c3_env, monkeypatch, backend, flag, prefix):
    source = prefix+'弯举10kg*10*4'
    mock_provider_transport(monkeypatch, [
        semantic_response([{'task_type':'workout_record', 'action':'record',
                            'text':source, flag:True}]),
        native_tools([('attempt-save','record_workout',{'source_ref':ref_of(source,source)})]),
        native({'answer':'没有保存。'}),
    ])
    harness = C3HTTPHarness(monkeypatch, c3_env, write_enabled=True)
    status, response = harness.post_json(source, mode='react_readonly', request_id=str(uuid4()))
    assert status == 200
    assert response['metrics'].get('execution_mode') != 'legacy', response
    assert not response['metrics'].get('receipt_refs')
    assert not saved_rows(c3_env)
