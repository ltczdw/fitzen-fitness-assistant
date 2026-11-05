"""Source-bound memory lifecycle; actual SQL and mocked model boundaries."""
import asyncio
import copy
import json
import hashlib
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select, update

from app.core.config import settings
from app.db.memory_models import MemoryState, MemoryTurn, UserFact
from app.schemas.chat import ChatRequest
from app.services.agent import dialogue_context as dc, semantics
from app.services.agent.execution_state import _request_memory
from app.services.context.budget import ContextBudgetManager
from app.services.context.token_counter import TokenCounter
from app.services.memory import compact, lifecycle as lc, wiki_context as wc
from tests.test_wiki_addressed_context import memory_db, load, add_turn, prepare, selector


def structured(source, memory=None, **extra):
    return semantics.validate_understanding(source, {
        'tasks': [{'task_type': 'training_qa', 'action': 'question', 'text': source,
                   'is_negated': False, 'is_hypothetical': False, 'is_third_party': False}],
        'safety': {'level': 'normal', 'evidence': '', 'categories': [], 'reason': 'scripted'},
        **extra}, context=memory or {})


def condition(quote, key='location', action='set', scope='conversation', domain='training'):
    return dict(key=key, quote=quote, action=action, scope=scope, domain=domain)


def finish(tid, source, **extra):
    value = structured(source, **extra)
    compact.finish_turn(tid, '回答', {'answer': '回答'}, semantic_values=[(source, value)])


def temporary(quote, **kwargs):
    request = ChatRequest(message=quote, conversation_id=kwargs.pop('cid', 'wiki-chat'))
    tid, _ = compact.begin_turn(request)
    finish(tid, quote, context_states=[condition(quote, **kwargs)])
    return tid


def remember(quote, *, key='restriction', kind='constraint', domain='training', memory=None, targets=None):
    source = '请记住' + quote
    tid, _ = compact.begin_turn(ChatRequest(message=source, conversation_id='wiki-chat'))
    finish(tid, source, memory=memory, memory_requests=[dict(action='replace' if targets else 'remember',
        key=key, quote=quote, confidence=1., kind=kind, domain=domain, target_fields=targets or [])])
    return tid


def test_arrival_version_wins_when_older_request_finishes_later(memory_db):
    a, _ = compact.begin_turn(ChatRequest(message='我在健身房', conversation_id='wiki-chat'))
    b, _ = compact.begin_turn(ChatRequest(message='我改在家里', conversation_id='wiki-chat'))
    finish(b, '我改在家里', context_states=[condition('我改在家里')])
    finish(a, '我在健身房', context_states=[condition('我在健身房')])
    view = load()
    assert view['current_states']['context:training:location']['text'] == '我改在家里'
    assert a not in {e['turn_id'] for e in view['wiki_catalog']['entries']}
    with memory_db.factory() as s:
        rows = list(s.scalars(select(MemoryState).where(MemoryState.kind == 'transient').order_by(MemoryState.version)))
        assert [r.status for r in rows] == ['superseded', 'active']
        assert not list(s.scalars(select(UserFact)))


@pytest.mark.parametrize('terminal', ['expiry', 'clear'])
def test_no_resurrection_no_read_renewal_and_explicit_archive(memory_db, terminal):
    a = temporary('我在健身房')
    b = temporary('我改在家里')
    with memory_db.factory() as s:
        before = s.scalar(select(MemoryState.expires_at).where(MemoryState.source_turn_id == b, MemoryState.kind == 'transient'))
    load(); load()
    with memory_db.factory.begin() as s:
        assert s.scalar(select(MemoryState.expires_at).where(MemoryState.source_turn_id == b, MemoryState.kind == 'transient')) == before
        if terminal == 'expiry':
            s.execute(update(MemoryState).where(MemoryState.source_turn_id == b, MemoryState.kind == 'transient')
                .values(expires_at=compact.now() - timedelta(seconds=5)))
    if terminal == 'clear':
        temporary('不再限定地点', action='clear')
    view = load()
    assert not view['current_states']
    assert not {a, b} & {e['turn_id'] for e in view['wiki_catalog']['entries']}
    archive = wc.catalog('wiki-chat', view['memory_epoch'], '健身房', 16000, include_history=True)
    assert a in {e['turn_id'] for e in archive['entries']}
    prepare(view, [{'include_history': True, 'query': '健身房', 'done': False},
                   {'include_history': True, 'page_ids': [next(e['page_id'] for e in archive['entries'] if e['turn_id'] == a)]}])
    assert all(m['app_meta']['wiki_view'] == 'history' for m in view['short_memory'])
    assert view['short_memory'][0]['content'] == '我在健身房'


def test_ttl_scope_replay_and_atomic_completion(memory_db):
    tid = temporary('这次在家', scope='request')
    assert not load()['current_states']
    assert tid not in {e['turn_id'] for e in load()['wiki_catalog']['entries']}
    other = temporary('我在健身房', cid='other-chat')
    assert not load()['current_states'] and load(cid='other-chat')['current_states']
    with memory_db.factory() as s:
        rows_before = [(r.state_id, r.expires_at) for r in s.scalars(select(MemoryState))]
        row = s.get(MemoryTurn, other)
        request = ChatRequest(message='我在健身房', request_id=row.request_id, conversation_id='other-chat')
    repeated, _ = compact.begin_turn(request)
    assert repeated == other
    compact.finish_turn(other, '重试', {'answer': '重试'})
    with memory_db.factory() as s:
        assert [(r.state_id, r.expires_at) for r in s.scalars(select(MemoryState))] == rows_before


def test_fact_latest_tie_conflict_and_expiry_do_not_pick_arbitrary_value(memory_db):
    tids = [add_turn('合成版本甲'), add_turn('合成版本乙')]
    older = compact.now() - timedelta(days=2)
    newer = compact.now() - timedelta(days=1)
    with memory_db.factory.begin() as s:
        for i, (tid, value, instant) in enumerate([(tids[0], '旧值', older), (tids[0], '新值甲', newer), (tids[1], '新值乙', newer)]):
            s.add(UserFact(fact_id=f'f{i}', field='versioned', value=value, source_quote=value,
                source_turn_id=tid, source_message_id=tid + ':user', memory_epoch=0, status='active', valid_from=instant))
    view = lc.snapshot('wiki-chat')
    assert 'versioned' not in view['long_memory'] and len(view['memory_conflicts']['versioned']) == 2
    with memory_db.factory.begin() as s:
        s.execute(update(UserFact).where(UserFact.fact_id.in_(['f1', 'f2'])).values(valid_to=compact.now()-timedelta(seconds=1)))
    assert 'versioned' not in compact.current_facts()


def test_authorized_constraint_is_durable_independent_and_domain_scoped(memory_db):
    remember('训练时避免跳跃')
    temporary('本周在家里')
    with memory_db.factory.begin() as s:
        s.execute(update(MemoryState).where(MemoryState.kind == 'transient').values(expires_at=compact.now()-timedelta(seconds=1)))
        policy = s.scalar(select(MemoryState).where(MemoryState.kind == 'constraint'))
        assert policy.value == '' and policy.expires_at is None and policy.source_fact_id
    view = load()
    assert view['persistent_constraints'] and not view['current_states']
    worker = dc.scoped_worker_memory(view, {'fact_fields': [], 'critical_fact_fields': []}, 'training_qa')
    assert not worker['long_memory'] and worker['persistent_constraints']
    assert not lc.channels(view, 'nutrition_advice', {})['persistent_constraints']
    assert not lc.channels(view, 'training_qa', {'is_third_party': True})['persistent_constraints']
    assert not lc.channels(view, 'training_qa', {'quoted': True})['persistent_constraints']
    assert not lc.channels(view, 'training_qa', {'reply_mode': 'conversation'})['persistent_constraints']
    manager = ContextBudgetManager(window=1600, output=128, margin=0, counter=TokenCounter(tokenizer_path=''))
    worker['short_memory'] = [{'role':'user','content':'无关历史'*1000}, {'role':'assistant','content':'旧回答'*1000}]
    packed = manager.assemble(system='规则', query='如何安排训练', state=worker, hits=[])
    assert '训练时避免跳跃' in json.dumps(packed, ensure_ascii=False)
    assert '无关历史' not in json.dumps(packed, ensure_ascii=False)


def test_legacy_constraint_label_is_source_bound_idempotent_and_does_not_create_fact(memory_db):
    remember('训练时避免跳跃', kind='fact')
    memory = load()
    field = next(iter(memory['long_memory']))
    for _ in range(2):
        text = '如何安排训练'
        tid, _ = compact.begin_turn(ChatRequest(message=text, conversation_id='wiki-chat'))
        finish(tid, text, memory=memory, constraint_fields=[{'field':field,'domain':'training'}])
    with memory_db.factory() as s:
        assert s.scalar(select(func.count()).select_from(UserFact)) == 1
        assert s.scalar(select(func.count()).select_from(MemoryState).where(MemoryState.kind == 'constraint')) == 1
    assert field in load()['persistent_constraints']


@pytest.mark.parametrize('action,scope', [('set','conversation'), ('clear','conversation'), ('set','request')])
def test_current_input_overrides_previous_condition_before_generation(memory_db, action, scope):
    temporary('我在健身房')
    source = '今天改在家里' if action == 'set' else '不再限定地点'
    request = ChatRequest(message=source, conversation_id='wiki-chat')
    tid, _ = compact.begin_turn(request)
    token = compact.current_turn.set(tid)
    try:
        memory = load()
        field = 'context:training:location'
        understanding = structured(source, memory, context_states=[condition(source, action=action, scope=scope)])
        understanding['tasks'][0]['fact_fields'] = [field]
        lc.apply_labels(memory, understanding)
        checked = asyncio.run(dc.restore_worker_scope(request, memory, understanding['tasks'][0], 'training_qa'))
        states = checked['current_states']
        assert (states['context:training:location']['text'] == source) if action == 'set' else not states
        scoped = dc.scoped_worker_memory(checked, understanding['tasks'][0], 'training_qa')
        if action == 'clear': assert field not in scoped['long_memory']
        compact.finish_turn(tid, '回答', {'answer':'回答'}, semantic_values=[(source, understanding)])
    finally:
        compact.current_turn.reset(token)
    expected = source if scope == 'conversation' and action == 'set' else ('我在健身房' if scope == 'request' else None)
    values = [value['text'] for value in load()['current_states'].values()]
    assert values == ([expected] if expected else [])


def test_cold_resume_rejects_expired_or_changed_sources(memory_db):
    temporary('我在健身房')
    memory = load()
    result = structured('如何安排训练', memory)
    lc.apply_labels(memory, result)
    result['__memory_view__'] = json.loads(json.dumps(lc.binding(memory, result)))
    lc.validate_binding(load(), result)
    with memory_db.factory.begin() as s:
        s.execute(update(MemoryState).where(MemoryState.kind == 'transient').values(expires_at=compact.now()-timedelta(seconds=1)))
    with pytest.raises(wc.WikiReadError, match='memory_state_changed'):
        lc.validate_binding(load(), result)


def test_constraint_replacement_inherits_protected_channel(memory_db):
    remember('训练时避免跳跃')
    memory = load()
    old = next(iter(memory['long_memory']))
    source = '请记住改成只做无跳跃动作'
    result = structured(source, memory, memory_requests=[dict(action='replace', key='restriction',
        quote='改成只做无跳跃动作', confidence=1., target_fields=[old])])
    lc.apply_labels(memory, result)
    assert old not in memory['persistent_constraints']
    assert [v['text'] for v in memory['persistent_constraints'].values()] == ['改成只做无跳跃动作']
    remember('改成只做无跳跃动作', memory=load(), targets=[old], kind='fact')
    view = load()
    assert [v['text'] for v in view['persistent_constraints'].values()] == ['改成只做无跳跃动作']
    assert all(v['domain'] == 'training' for v in view['persistent_constraints'].values())


def test_actual_readonly_worker_receives_unselected_constraints_and_rechecks_visibility(memory_db):
    from app.services.agent.react_readonly_workers import _answer_state
    original = remember('训练时避免跳跃')
    request = ChatRequest(message='如何安排训练', conversation_id='wiki-chat')
    tid, _ = compact.begin_turn(request)
    turn_token = compact.current_turn.set(tid)
    memory_token = _request_memory.set(load())
    context = {'conversation_id':'wiki-chat','request_id':request.request_id,'source_text':request.message,
               'task_text':request.message, 'semantic_scope': {'fact_fields':[], 'critical_fact_fields':[]}}
    try:
        state = asyncio.run(_answer_state(context, 'training_qa'))
        assert state['persistent_constraints']
        with memory_db.factory.begin() as s:
            s.get(MemoryTurn, original).hidden_at = compact.now()
        assert asyncio.run(_answer_state(context, 'training_qa')) is None
    finally:
        _request_memory.reset(memory_token)
        compact.current_turn.reset(turn_token)


@pytest.mark.parametrize('count', [1, 24])
def test_batch_state_writes_and_snapshot_reads_have_constant_query_count(memory_db, count):
    source = '条件' + '；'.join(str(i) for i in range(count))
    tid, _ = compact.begin_turn(ChatRequest(message=source, conversation_id='wiki-chat'))
    # The normal semantic contract caps a batch at 16; low-level SQL can batch more.
    with memory_db.factory.begin() as s:
        turn = s.get(MemoryTurn, tid)
        version = turn.messages[0]['app_meta']['_memory_version']
        items = [dict(kind='transient', conversation_id='wiki-chat', scope='conversation',
            logical_key=f'training:key{i}', domain='training', value=f'{i}', status='active',
            source_turn_id=tid, source_message_id=tid+':user') for i in range(count)]
        memory_db.queries.clear()
        lc.write_versions(s, items, owner='demo-user', epoch=0, version=version, instant=compact.now())
        s.flush()
        assert len(memory_db.queries) == 3  # canonical MAX summary, canonical UPDATE, batched INSERT
        turn.status = 'completed'; turn.completed_at = compact.now()
    memory_db.queries.clear()
    assert len(lc.snapshot('wiki-chat')['current_states']) == count
    assert len(memory_db.queries) == 4  # profile, facts, policy, current states


@pytest.mark.parametrize('defect', ['quote', 'duplicate', 'target', 'unknown_policy'])
def test_semantic_boundary_rejects_unbound_or_ambiguous_lifecycle_data(memory_db, defect):
    source = '这次在家'
    data = {'context_states':[condition(source)]}
    if defect == 'quote': data['context_states'][0]['quote'] = '不是用户原文'
    if defect == 'duplicate': data['context_states'] *= 2
    if defect == 'target': data = {'memory_requests':[dict(action='replace',key='x',quote=source,confidence=1.,kind='transient',target_fields=['invented'])]}
    if defect == 'unknown_policy': data = {'constraint_fields':[dict(field='invented',domain='general')]}
    with pytest.raises(semantics.SemanticValidationError):
        structured(source, **data)


def test_critical_input_never_gets_ttl_and_urgent_gate_survives_conflict(memory_db):
    source = '膝盖受伤需要避免跳跃'
    tid, _ = compact.begin_turn(ChatRequest(message=source, conversation_id='wiki-chat'))
    finish(tid, source, current_is_critical=True, context_states=[condition(source)])
    assert not load()['current_states']
    memory = load()
    memory['constraint_conflicts'] = {'x':{'domain':'training','candidates':[]}}
    with pytest.raises(wc.WikiReadError, match='persistent_constraint_conflict'):
        lc.channels(memory, 'training_qa', {})
    lc.apply_labels(memory, {'safety':{'level':'urgent'}, 'tasks':[{'task_type':'safety_triage'}]})
    assert not memory['persistent_constraints']


def test_wiki_correction_preserves_constraint_then_privacy_delete_purges_it(memory_db):
    from app.services.memory import wiki
    original = remember('训练时避免跳跃')
    prior = next(iter(load()['persistent_constraints'].values()))
    source = add_turn('改成不跑步也不跳跃')
    wiki.correct_fact(prior['fact_id'], '不跑步也不跳跃', source, source+':user')
    assert [v['text'] for v in load()['persistent_constraints'].values()] == ['不跑步也不跳跃']
    temporary('在家训练')  # next source version must follow the API correction too
    compact.delete_memory('wiki-chat')
    with memory_db.factory() as s:
        assert s.scalar(select(func.count()).select_from(MemoryState)) == 0
        assert s.scalar(select(func.count()).select_from(UserFact)) == 0
    assert not load()['persistent_constraints'] and not load()['current_states']


def test_transient_explicit_request_has_honest_receipt_and_no_permanent_fact(memory_db):
    remember('今天在家训练', kind='transient', key='location')
    memory = load()
    assert memory['current_states']['context:training:location']['text'] == '今天在家训练'
    with memory_db.factory() as s:
        assert s.scalar(select(func.count()).select_from(UserFact)) == 0
        row = s.scalar(select(MemoryTurn).where(MemoryTurn.status == 'completed'))
        assert '临时记住' in row.response['answer']


def test_same_version_constraint_conflict_never_silently_drops_the_channel(memory_db):
    original = remember('训练时避免跳跃')
    with memory_db.factory.begin() as s:
        fact = s.scalar(select(UserFact))
        s.add(UserFact(fact_id='conflicting', field=fact.field, value='可以跳跃', source_quote='可以跳跃',
            source_turn_id=original, source_message_id=original+':user', memory_epoch=0,
            status='active', valid_from=fact.valid_from))
    memory = load()
    assert not memory['persistent_constraints'] and memory['constraint_conflicts']
    with pytest.raises(wc.WikiReadError, match='persistent_constraint_conflict'):
        dc.scoped_worker_memory(memory, {}, 'training_qa')


def test_current_constraint_replacement_revalidates_old_source(memory_db):
    original = remember('训练时避免跳跃')
    source = '请记住改成避免跑步'
    request = ChatRequest(message=source, conversation_id='wiki-chat')
    tid, _ = compact.begin_turn(request)
    token = compact.current_turn.set(tid)
    try:
        memory = load()
        old = next(iter(memory['persistent_constraints']))
        result = structured(source, memory, memory_requests=[dict(action='replace', key='restriction',
            quote='改成避免跑步', confidence=1., target_fields=[old])])
        lc.apply_labels(memory, result)
        assert asyncio.run(dc.restore_worker_scope(request, memory, result['tasks'][0], 'training_qa'))['persistent_constraints']
        with memory_db.factory.begin() as s:
            s.get(MemoryTurn, original).hidden_at = compact.now()
        with pytest.raises(dc.DialogueContextUnavailable):
            asyncio.run(dc.restore_worker_scope(request, memory, result['tasks'][0], 'training_qa'))
    finally:
        compact.current_turn.reset(token)


def test_actual_legacy_graph_passes_constraints_to_worker_without_model_selection(memory_db):
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    remember('训练时避免跳跃')
    request = ChatRequest(message='如何安排训练', conversation_id='wiki-chat')
    tid, _ = compact.begin_turn(request)
    token = compact.current_turn.set(tid)
    memory = load()
    mt = _request_memory.set(memory)
    worker = SimpleNamespace(ainvoke=AsyncMock(return_value={'result': {'status':'complete','answer':'ok'}}))
    graph = object.__new__(FitnessMultiAgentGraph)
    graph.workers = {'training_qa': worker}
    try:
        result = asyncio.run(graph._run_worker('training_qa', {**memory,
            'conversation_id':'wiki-chat','request_id':request.request_id,'user_id':'demo-user',
            'raw_message':request.message,'current_task':{'id':'q1','task_type':'training_qa',
            'action':'question','text':request.message,'worker':'training_qa'}}))
        assert result['completed_tasks'] == ['q1']
        assert worker.ainvoke.await_args.args[0]['persistent_constraints']
        assert not worker.ainvoke.await_args.args[0]['long_memory']
    finally:
        _request_memory.reset(mt)
        compact.current_turn.reset(token)


@pytest.mark.parametrize('reverse', [False, True])
def test_request_scope_overrides_conversation_scope_only_for_this_answer(memory_db, reverse):
    source = '平时在家，这次在健身房'
    states = [condition('平时在家'), condition('这次在健身房', scope='request')]
    if reverse: states.reverse()
    result = structured(source, context_states=states)
    memory = load()
    lc.apply_labels(memory, result)
    assert memory['current_states']['context:training:location']['text'] == '这次在健身房'
    tid, _ = compact.begin_turn(ChatRequest(message=source, conversation_id='wiki-chat'))
    compact.finish_turn(tid, '回答', {'answer':'回答'}, semantic_values=[(source,result)])
    assert load()['current_states']['context:training:location']['text'] == '平时在家'


@pytest.mark.parametrize('change', ['expiry', 'newer', 'hidden_classified_fact'])
def test_controller_snapshot_restore_rechecks_lifecycle_without_model_call(memory_db, change):
    from app.services.agent.react_write_runtime import WriteRuntimeSession
    source = '如何安排训练'
    if change == 'hidden_classified_fact':
        original = remember('训练时避免跳跃', kind='fact')
        memory = load()
        result = structured(source, memory, constraint_fields=[dict(field=next(iter(memory['long_memory'])), domain='training')])
    else:
        temporary('我在健身房')
        memory = load()
        result = structured(source, memory)
    lc.apply_labels(memory, result)
    result['__memory_view__'] = lc.binding(memory, result)
    frozen = json.loads(json.dumps({'tasks':[{'task_type':'react_controller', 'parameters': {
        'input_fingerprint':hashlib.sha256(source.encode()).hexdigest(), 'semantic_understanding':result}}]}))
    if change == 'hidden_classified_fact':
        with memory_db.factory.begin() as s: s.get(MemoryTurn, original).hidden_at = compact.now()
    elif change == 'expiry':
        with memory_db.factory.begin() as s:
            s.execute(update(MemoryState).where(MemoryState.kind == 'transient').values(expires_at=compact.now()-timedelta(seconds=1)))
    else: temporary('我改在家里')
    token = semantics.request_understanding.set({})
    no_model = SimpleNamespace(complete_structured=AsyncMock(side_effect=AssertionError('resume must reuse its decision')))
    try:
        WriteRuntimeSession._restore_source_understanding(frozen, source)
        with pytest.raises(wc.WikiReadError, match='memory_state_changed'):
            asyncio.run(semantics.understand(no_model, source, context=load()))
        no_model.complete_structured.assert_not_awaited()
    finally: semantics.request_understanding.reset(token)
