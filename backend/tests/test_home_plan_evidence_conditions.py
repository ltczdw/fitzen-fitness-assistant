"""Source-bound conditions and actual home-preview continuation; providers mocked."""
import json
from uuid import uuid4
from unittest.mock import AsyncMock

import httpx
import pytest

from app.services.rag.evidence_conflict import (
    EvidenceConflictModel, _select_inputs, _validated_analysis,
    controlled_evidence_answer, answer_system_rules,
)
from tests.test_rag_evidence_conflict import _chunk
from tests.react_phase_c2_3_support import (
    C3HTTPHarness, c3_env, mock_provider_transport, all_business_rows,
)
from tests.test_workout_source_bound_writes import semantic_response, native_tools
from tests.test_react_record_continuation import tool_response

GENERAL = '每个动作尝试8至12次为1组,至少2组,3组可获得更多益处'
EXAMPLE = 'Attempt 3 sets of 5 curls with each arm.'
HOME = '居家训练可以选择靠墙俯卧撑和椅子深蹲，逐步增加训练量。'


def hits():
    return [_chunk('通用力量训练建议', GENERAL, '996'),
            _chunk('居家力量动作', EXAMPLE + HOME, '604')]


def judgement(kind=None, *, supported=True, duplicate=True):
    conditions = [{'text': '通用力量训练建议', 'chunk_id': '996',
                   'quote': GENERAL if duplicate else '8至12次'},
                  {'text': '居家弯举示例', 'chunk_id': '604',
                   'quote': EXAMPLE if duplicate else '5 curls'}]
    if kind is not None:
        for condition in conditions:
            condition['kind'] = kind
    claims = [{'claim': '两种组次建议不同', 'relation': 'condition_difference',
               'conditions': conditions, 'evidence': [
                   {'chunk_id': '996', 'quote': GENERAL},
                   {'chunk_id': '604', 'quote': EXAMPLE}]}]
    if supported:
        claims.append({'claim': '居家训练动作与渐进原则', 'relation': 'consistent',
                       'evidence': [{'chunk_id': '604', 'quote': HOME}]})
    return EvidenceConflictModel.model_validate({'claims': claims})


@pytest.mark.parametrize('kind', [None, 'unknown', 'prescription', 'applicability'])
def test_recycled_conclusions_cannot_be_used_as_applicability_conditions(kind):
    # Even a mistaken applicability label cannot recycle the exact conclusions.
    analysis = _validated_analysis(judgement(kind), _select_inputs(hits())[0])
    assert analysis.status == 'partial' and analysis.invalid_relations == 1
    assert analysis.invalid_references == 0 and analysis.has_insufficient
    assert not analysis.has_condition_difference and not analysis.has_unresolved_conflict
    assert controlled_evidence_answer(analysis) is None
    assert '已限制为insufficient' in answer_system_rules(analysis)
    assert 'prescription' in EvidenceConflictModel.model_json_schema()['$defs']['EvidenceConditionModel']['properties']['kind']['enum']


@pytest.mark.parametrize('kind', [None, 'unknown', 'prescription', 'time_or_version'])
def test_distinct_prescription_spans_still_need_a_source_bound_applicability_role(kind):
    analysis = _validated_analysis(judgement(kind, duplicate=False), _select_inputs(hits())[0])
    assert analysis.invalid_relations == 1 and not analysis.has_condition_difference
    assert controlled_evidence_answer(analysis) is None


@pytest.mark.parametrize('high_risk,supported', [(False, False), (True, True), (True, False)])
def test_missing_comparison_never_upgrades_to_proven_agreement(high_risk, supported):
    analysis = _validated_analysis(judgement(supported=supported), _select_inputs(hits())[0])
    answer = controlled_evidence_answer(analysis, high_risk=high_risk)
    assert '暂未生成具体建议' in answer
    assert analysis.claims[0].relation == 'insufficient'


@pytest.mark.parametrize('backend', ['sqlite', 'postgres'])
@pytest.mark.parametrize('transport', ['json', 'sse'])
def test_home_followup_keeps_selected_preview_cap_and_reaches_generation(c3_env, monkeypatch, backend, transport):
    from app.db import profile_repository
    from app.services.agent.agent import agent
    from app.services.agent.plan_preview import latest_plan_preview

    profile_repository.save_profile('demo-user', {'goal': '增肌', 'experience': '中级'})
    before = all_business_rows(c3_env)
    source, follow = '给我制定一次训练计划，每次60分钟', '是居家训练'
    calls, queries, generations = [], [], []

    def respond(request):
        body = json.loads(request.content)
        names = [t['function']['name'] for t in body.get('tools') or []]
        calls.append(names)
        if 'return_request_understanding' in names:
            data = json.loads(body['messages'][-1]['content'])
            query = data['query']; queries.append(query)
            result = semantic_response([{'task_type': 'training_plan', 'action': 'question', 'text': query}])
            function = result['choices'][0]['message']['tool_calls'][0]['function']
            value = json.loads(function['arguments'])
            task = value['tasks'][0]
            task.update(plan_result=True, result_scope='independent', plan_request_quote=query, delivery_type='plan')
            if query == follow:
                task.update(delivery_type='preview', preview_ref=data['current_plan_preview']['preview_id'],
                            history_requirement='required')
                value['relation'] = 'answer'
            function['arguments'] = json.dumps(value, ensure_ascii=False)
            return httpx.Response(200, json=result)
        if 'return_plan_information' in names:
            data = json.loads(body['messages'][-1]['content'])
            facts = [{'field': field, 'source_id': 'profile:' + field,
                      'quote': data['sources']['profile:' + field]} for field in ('goal', 'experience')]
            if data['task'] == source:
                facts.append({'field': 'session_time', 'source_id': 'current', 'quote': '每次60分钟',
                              'quantity_quote': '60分钟', 'seconds': 3600})
            else:
                assert data['task'] == follow
                facts.append({'field': 'equipment', 'source_id': 'current', 'quote': '居家训练'})
            return httpx.Response(200, json=tool_response('return_plan_information', {'scope': 'session', 'facts': facts}))
        if 'return_evidence_conflict_judgement' in names:
            # Replay the captured false pair, with real verified quotes; do not
            # mock the decoder, conflict guard, or final-answer handling.
            return httpx.Response(200, json=tool_response('return_evidence_conflict_judgement', judgement().model_dump()))
        if 'return_timed_training_plan' in names:
            prompt = json.dumps(body['messages'], ensure_ascii=False)
            home = queries[-1] == follow
            if home:
                assert '居家训练' in prompt and 'insufficient' in prompt
                assert '3600' in prompt
            generations.append(home)
            value = {'sessions': [{'label': '居家训练日' if home else '训练日',
                'warmup': {'seconds': 300, 'activity': '动态活动'},
                'main': {'seconds': 2400, 'exercises': [
                    {'name': '靠墙俯卧撑' if home else '卧推', 'sets': 3, 'reps': 10}]},
                'breaks': {'seconds': 600}, 'cooldown': {'seconds': 300, 'activity': '轻柔拉伸'},
                'time_consistency': 'consistent'}]}
            return httpx.Response(200, json=tool_response('return_timed_training_plan', value))
        if 'plan_training' in names:
            return httpx.Response(200, json=native_tools([('plan', 'plan_training', {'query': queries[-1]})]))
        raise AssertionError(names)

    mock_provider_transport(monkeypatch, [respond])
    monkeypatch.setattr(agent.graph, '_readonly_retrieval', AsyncMock(return_value=hits()))
    harness = C3HTTPHarness(monkeypatch, c3_env)

    def send(message, rid):
        if transport == 'json':
            return harness.post_json(message, request_id=rid, mode='react_readonly')
        status, events, _ = harness.stream(message, request_id=rid, mode='react_readonly')
        return status, {**events[-1], 'answer': ''.join(event.get('content', '') for event in events)}

    status, first = send(source, str(uuid4()))
    assert status == 200 and not first['clarifications'], first
    old_result = latest_plan_preview('demo-user', harness.conversation_id)
    assert old_result['status'] == 'ok'
    old = old_result['state']
    rid = str(uuid4()); status, result = send(follow, rid)
    assert status == 200 and not result['clarifications'], result
    assert '靠墙俯卧撑' in result['answer'] and '60分钟' in result['answer']
    assert '不能合并为同一条件' not in result['answer']
    assert result['metrics']['react_status'] == 'complete'
    current_result = latest_plan_preview('demo-user', harness.conversation_id)
    assert current_result['status'] == 'ok'
    current = current_result['state']
    assert current['base_preview_id'] == old['preview_id'] and current['session_minutes'] == 60
    assert current['days'][0]['exercises'] == ['靠墙俯卧撑']
    assert generations == [False, True]
    count = len(calls); status, replay = send(follow, rid)
    assert status == 200 and replay['answer'] == result['answer'] and len(calls) == count
    assert all_business_rows(c3_env) == before
