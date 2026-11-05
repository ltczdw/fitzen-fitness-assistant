"""Addressed chat memory: real isolated SQL, scripted model boundaries only."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event, select

from app.core.config import settings
from app.db import session as db
from app.db.memory_models import MemoryTurn
from app.db.models import UserProfileModel
from app.db.wiki_models import WikiPage
from app.schemas.chat import ChatRequest
from app.services.agent import dialogue_context, semantics
from app.services.agent.execution_state import _request_memory
from app.services.memory import compact, wiki_context as wc
from tests.react_phase_c1_support import isolated


@pytest.fixture(params=['sqlite', 'postgres'])
def memory_db(request, tmp_path, monkeypatch):
    with isolated(monkeypatch, tmp_path / 'wiki-context', request.param) as engine:
        monkeypatch.setattr(settings, 'wiki_enabled', True)
        # These cases isolate addressed-only retrieval. The production hybrid
        # default is covered separately with its real ten-turn SQL window.
        monkeypatch.setattr(settings, 'memory_short_window_turns', 0)
        monkeypatch.setattr(settings, 'wiki_semantic_units_enabled', False)
        monkeypatch.setattr(settings, 'wiki_catalog_max_tokens', 16000)
        monkeypatch.setattr(settings, 'llm_context_window_tokens', 65536)
        queries = []
        event.listen(engine, 'before_cursor_execute', lambda c, u, sql, p, x, many: queries.append(sql))
        token = semantics.request_understanding.set({})
        mt = _request_memory.set(None)
        try:
            yield SimpleNamespace(factory=db.SessionLocal, queries=queries)
        finally:
            _request_memory.reset(mt)
            semantics.request_understanding.reset(token)


def add_turn(message, answer='选项一周一，选项二周三。', cid='wiki-chat'):
    tid, _ = compact.begin_turn(ChatRequest(message=message, conversation_id=cid))
    compact.finish_turn(tid, answer, {'answer': answer})
    return tid


def load(source='继续', cid='wiki-chat'):
    legacy = SimpleNamespace(initialize=AsyncMock(side_effect=AssertionError('no Redis context read')))
    return asyncio.run(compact.CompactMemory(legacy).load('demo-user', cid, source))


def selector(plans, semantic=None):
    async def call(messages, schema, *, name, **kwargs):
        if name == 'return_wiki_read_plan':
            value = next(plans)
        elif name == 'return_request_understanding':
            value = semantic
        elif name == 'return_dialogue_answer':
            value = {'answer': '第二项是周三。', 'requires_grounding': False}
        else:
            raise AssertionError(name)
        result = schema.model_validate(value)
        if kwargs.get('validator'):
            kwargs['validator'](result)
        return result
    return SimpleNamespace(configured=lambda: True, complete_structured=AsyncMock(side_effect=call))


def prepare(memory, plans):
    llm = selector(iter(plans))
    asyncio.run(wc.prepare(llm, '继续', memory, semantics.PROMPT))
    return llm


def pid(memory, tid):
    return next(e['page_id'] for e in memory['wiki_catalog']['entries'] if e['turn_id'] == tid)


def test_load_has_no_recent_window_or_original_body(memory_db, monkeypatch):
    for index in range(15):
        add_turn(f'合成问题{index}', 'BODY-ONLY-SENTINEL')
    monkeypatch.setattr(compact, 'recent_rows', lambda *a: pytest.fail('fixed window invoked'))
    memory = load()
    assert memory['short_memory'] == [] and memory['retrieved_history'] == []
    assert memory['context_strategy'] == 'wiki_addressed'
    assert 'BODY-ONLY-SENTINEL' not in json.dumps(memory)


def test_more_than_six_sections_and_old_sources_are_complete(memory_db):
    tids = [add_turn('旧方案' if i == 0 else f'问题{i}', f'完整回答{i}') for i in range(15)]
    memory = load('旧方案和前面几个安排')
    chosen = [pid(memory, tid) for tid in tids[:7]]
    prepare(memory, [{'page_ids': chosen}])
    assert len(memory['short_memory']) == 14
    assert [m['content'] for m in memory['short_memory'] if m['role'] == 'assistant'] == [f'完整回答{i}' for i in range(7)]
    assert memory['wiki_read_tokens'] <= memory['wiki_read_limit']
    assert tids[0] in {m['turn_id'] for m in memory['short_memory']}


@pytest.mark.parametrize('count', [1, 30])
def test_selected_reads_are_batched_and_repeated_reads_are_free(memory_db, count):
    tids = [add_turn(f'问题{i}', '回答') for i in range(count)]
    memory = load()
    memory.update(wiki_catalog_entries={e['page_id']: e for e in memory['wiki_catalog']['entries']}, wiki_loaded_pages={})
    chosen = [pid(memory, tid) for tid in tids]
    memory_db.queries.clear()
    pages, cost = wc.read_pages(memory, chosen, 1000000)
    assert len(pages) == count and sum(len(p['messages']) for p in pages) == count * 2
    selects = [sql for sql in memory_db.queries if sql.lstrip().upper().startswith('SELECT')]
    assert len(selects) == 3
    memory['wiki_loaded_pages'].update({p['page_id']: p for p in pages})
    memory_db.queries.clear()
    assert wc.read_pages(memory, chosen, 1000000) == ([], 0)
    assert not memory_db.queries and cost > 0


def test_over_budget_rejects_before_body_read_without_partial_context(memory_db):
    tid = add_turn('长方案', '很长的原文' * 2000)
    memory = load()
    memory.update(wiki_catalog_entries={e['page_id']: e for e in memory['wiki_catalog']['entries']}, wiki_loaded_pages={})
    memory_db.queries.clear()
    with pytest.raises(wc.WikiReadError, match='wiki_read_budget_exceeded'):
        wc.read_pages(memory, [pid(memory, tid)], 500)
    assert not any(sql.startswith('SELECT wiki_sections.page_id, wiki_sections.source_message_id') for sql in memory_db.queries)
    assert not memory['short_memory'] and not memory['wiki_loaded_pages']


@pytest.mark.parametrize('change', ['hidden', 'deleted', 'epoch', 'revision', 'foreign_conversation'])
def test_addresses_are_bound_to_visible_unchanged_sources(memory_db, change):
    tid = add_turn('原方案')
    memory = load()
    memory.update(wiki_catalog_entries={e['page_id']: e for e in memory['wiki_catalog']['entries']}, wiki_loaded_pages={})
    with memory_db.factory.begin() as s:
        turn = s.get(MemoryTurn, tid)
        if change == 'hidden': turn.hidden_at = compact.now()
        if change == 'deleted': turn.deleted_at = compact.now()
        if change == 'epoch': s.get(UserProfileModel, 'demo-user').memory_epoch += 1
        if change == 'revision': s.get(WikiPage, pid(memory, tid)).revision += 1
        if change == 'foreign_conversation': turn.conversation_id = 'another-conversation'
    with pytest.raises(wc.WikiReadError):
        wc.read_pages(memory, [pid(memory, tid)], 1000000)


def test_unknown_address_is_rejected_and_user_namespace_is_not_shared(memory_db):
    add_turn('私有方案')
    memory = load()
    memory.update(wiki_catalog_entries={}, wiki_loaded_pages={})
    with pytest.raises(wc.WikiReadError, match='not_offered'):
        wc.read_pages(memory, ['invented-page'], 100000)
    legacy = SimpleNamespace()
    other = asyncio.run(compact.CompactMemory(legacy).load('another-user', 'wiki-chat', '方案'))
    assert other['degraded_reason'] == 'memory_owner_unavailable' and not other['short_memory']


def test_cumulative_budget_and_addressing_loop_are_bounded(memory_db, monkeypatch):
    tids = [add_turn(f'问题{i}') for i in range(2)]
    memory = load()
    memory.update(wiki_catalog_entries={e['page_id']: e for e in memory['wiki_catalog']['entries']}, wiki_loaded_pages={})
    pages, cost = wc.read_pages(memory, [pid(memory, tids[0])], 100000)
    memory['wiki_loaded_pages'].update({p['page_id']: p for p in pages})
    with pytest.raises(wc.WikiReadError, match='budget_exceeded'):
        wc.read_pages(memory, [pid(memory, tids[1])], cost - 1)
    monkeypatch.setattr(settings, 'wiki_memory_max_rounds', 2)
    llm = selector(iter([{'query': '', 'done': False}, {'query': '', 'done': False}]))
    with pytest.raises(wc.WikiReadError, match='round_limit'):
        asyncio.run(wc.prepare(llm, '继续', load(), semantics.PROMPT))
    assert llm.complete_structured.await_count == 2


def test_self_contained_query_does_not_read_any_originals(memory_db):
    add_turn('一个旧问题')
    memory = load('什么是肌肥大')
    llm = prepare(memory, [{'page_ids': []}])
    assert memory['wiki_read_tokens'] == 0 and memory['short_memory'] == []
    prepare(memory, [])
    assert llm.complete_structured.await_count == 1


def test_followup_understanding_reads_original_then_reuses_it_for_restore(memory_db):
    tid = add_turn('给我两个选项')
    memory = load('第二项是什么')
    raw = {'tasks': [{'task_type': 'training_qa', 'action': 'question', 'text': '第二项是什么',
        'is_negated': False, 'is_hypothetical': False, 'is_third_party': False,
        'history_requirement': 'required'}],
        'dialogue_ids': [tid + ':user', tid + ':assistant'], 'reply_mode': 'conversation',
        'relation': 'follow_up', 'safety': {'level': 'normal', 'evidence': '', 'categories': [], 'reason': 'scripted'}}
    llm = selector(iter([{'page_ids': [pid(memory, tid)]}]), raw)
    result = asyncio.run(semantics.understand(llm, '第二项是什么', context=memory))
    assert result['dialogue_ids'] == raw['dialogue_ids']
    assert [call.kwargs['name'] for call in llm.complete_structured.await_args_list] == [
        'return_request_understanding', 'return_wiki_read_plan']
    assert asyncio.run(semantics.understand(llm, '第二项是什么', context=memory)) == result
    assert llm.complete_structured.await_count == 2
    request = ChatRequest(message='第二项是什么', conversation_id='wiki-chat')
    live, _ = compact.begin_turn(request)
    token = compact.current_turn.set(live)
    memory_db.queries.clear()
    try:
        restored = asyncio.run(dialogue_context.restore(request, memory, result))
        assert restored['short_memory'] == memory['short_memory']
        assert not any(sql.startswith('SELECT memory_turns.messages') for sql in memory_db.queries)
        with memory_db.factory.begin() as s:
            s.get(MemoryTurn, tid).hidden_at = compact.now()
        with pytest.raises(dialogue_context.DialogueContextUnavailable):
            asyncio.run(dialogue_context.restore(request, memory, result))
    finally:
        compact.current_turn.reset(token)


def test_answer_persists_only_reference_addresses_for_next_turn(memory_db, monkeypatch):
    old = add_turn('旧计划')
    for i in range(12): add_turn(f'无关话题{i}')
    memory = load('旧计划')
    chosen = pid(memory, old)
    prepare(memory, [{'page_ids': [chosen]}])
    tid, _ = compact.begin_turn(ChatRequest(message='继续改旧计划', conversation_id='wiki-chat'))
    token = _request_memory.set(memory)
    try:
        compact.finish_turn(tid, '已调整计划', {'answer': '已调整计划'})
    finally:
        _request_memory.reset(token)
    monkeypatch.setattr(settings, 'wiki_catalog_batch_size', 2)
    next_memory = load('继续')
    assert chosen in {e['page_id'] for e in next_memory['wiki_catalog']['entries']}
    anchor = next(e for e in next_memory['wiki_catalog']['entries'] if e['current_anchor'])
    assert anchor['references'] == [chosen]


def test_other_conversation_is_explicit_historical_evidence_not_local_dialogue(memory_db):
    tid = add_turn('独特旧训练计划', cid='other-chat')
    memory = load('独特旧训练计划')
    prepare(memory, [{'page_ids': [pid(memory, tid)]}])
    assert memory['short_memory'] == []
    assert memory['retrieved_history'][0]['turn_id'] == tid
    payload = json.loads(dialogue_context.semantic_messages('规则', '回忆', memory)[-1]['content'])
    assert payload['addressed_history'][0]['turn_id'] == tid and payload['history'] == []


def test_resumed_bound_semantics_reads_sources_without_another_model_call(memory_db):
    tid = add_turn('原任务')
    raw = {'dialogue_ids': [tid + ':assistant'], 'tasks': []}
    semantics.remember_understanding('继续', raw)
    memory = load()
    llm = SimpleNamespace(complete_structured=AsyncMock(side_effect=AssertionError('resume re-planned')))
    assert asyncio.run(semantics.understand(llm, '继续', context=memory)) == raw
    assert len(memory['short_memory']) == 2
    llm.complete_structured.assert_not_awaited()
    with memory_db.factory.begin() as s:
        s.get(MemoryTurn, tid).hidden_at = compact.now()
    with pytest.raises(wc.WikiReadError, match='cached_source_missing'):
        asyncio.run(semantics.understand(llm, '继续', context=load()))


def test_small_catalog_budget_pages_without_skipping_unseen_entries(memory_db, monkeypatch):
    tids = {add_turn(f'分页问题{i}') for i in range(40)}
    token_budget = 2200
    value = wc.catalog('wiki-chat', 0, '', token_budget, initial=True)
    seen = {e['turn_id'] for e in value['entries']}
    assert 0 < len(seen) < 32 and value['has_more']
    while value['has_more']:
        previous = value['next_offset']
        value = wc.catalog('wiki-chat', 0, '', token_budget, offset=previous)
        seen.update(e['turn_id'] for e in value['entries'])
        assert not value['has_more'] or value['next_offset'] > previous
    assert seen == tids


def test_cross_conversation_source_is_rechecked_even_without_local_reference_ids(memory_db):
    tid = add_turn('旧的私人计划', cid='old-chat')
    memory = load('旧的私人计划')
    prepare(memory, [{'page_ids': [pid(memory, tid)]}])
    request = ChatRequest(message='回忆计划', conversation_id='wiki-chat')
    live, _ = compact.begin_turn(request)
    token = compact.current_turn.set(live)
    try:
        with memory_db.factory.begin() as s:
            s.get(MemoryTurn, tid).hidden_at = compact.now()
        with pytest.raises(dialogue_context.DialogueContextUnavailable):
            asyncio.run(dialogue_context.restore(request, memory, {}))
    finally:
        compact.current_turn.reset(token)


def test_real_agent_followup_uses_addressed_pair_and_commits_reply(memory_db, monkeypatch):
    from test_retry_operations import RetryScenario
    from app.services.agent import execution
    tid = add_turn('给两个选项')
    memory = load('第二项是什么')
    scenario = RetryScenario(monkeypatch, memory_db.factory)
    monkeypatch.setattr(execution.memory_service, 'load', AsyncMock(return_value=memory))
    raw = {'tasks': [{'task_type': 'training_qa', 'action': 'question', 'text': '第二项是什么',
        'is_negated': False, 'is_hypothetical': False, 'is_third_party': False,
        'history_requirement': 'required'}],
        'dialogue_ids': [tid + ':assistant'], 'reply_mode': 'conversation', 'relation': 'follow_up',
        'safety': {'level': 'normal', 'evidence': '', 'categories': [], 'reason': 'scripted'}}
    llm = selector(iter([{'page_ids': [pid(memory, tid)]}]), raw)
    monkeypatch.setattr(scenario.agent.graph.llm, 'complete_structured', llm.complete_structured)
    response = asyncio.run(scenario.respond(ChatRequest(message='第二项是什么', conversation_id='wiki-chat')))
    assert response.answer == '第二项是周三。'
    assert not scenario.saves and not scenario.tool_calls
    assert [call.kwargs['name'] for call in llm.complete_structured.await_args_list] == [
        'return_request_understanding', 'return_wiki_read_plan', 'return_dialogue_answer']
    with memory_db.factory() as s:
        row = s.scalar(select(MemoryTurn).where(MemoryTurn.request_id == response.request_id))
        assert row.status == 'completed' and row.messages[-1]['content'] == response.answer


@pytest.mark.parametrize('urgent', [False, True])
def test_memory_budget_failure_cannot_suppress_current_urgent_gate(memory_db, urgent):
    tid = add_turn('旧计划', '完整原文' * 20000)
    source = '我现在胸痛' if urgent else '继续旧计划'
    memory = load(source)
    raw = {'tasks': [{'task_type': 'safety_triage' if urgent else 'training_qa',
        'action': 'question', 'text': source, 'is_negated': False, 'is_hypothetical': False,
        'is_third_party': False, 'history_requirement': 'none' if urgent else 'required'}],
        'safety': {'level': 'urgent' if urgent else 'normal', 'evidence': source if urgent else '',
                   'categories': [], 'reason': 'scripted'}}
    required = {'page_ids': [pid(memory, tid)], 'required_page_ids': [pid(memory, tid)]}
    llm = selector(iter([required] * settings.wiki_memory_max_rounds), raw)
    if urgent:
        result = asyncio.run(semantics.understand(llm, source, context=memory))
        assert result['safety']['level'] == 'urgent'
    else:
        with pytest.raises(wc.WikiReadError, match='budget_exceeded'):
            asyncio.run(semantics.understand(llm, source, context=memory))
    assert llm.complete_structured.await_count == (
        1 if urgent else settings.wiki_memory_max_rounds + 1)
    assert memory['short_memory'] == []


def test_unindexed_content_beyond_title_is_addressable_without_loading_the_body(memory_db):
    tid = add_turn('普通背景' * 40 + '之后补充：右手腕旧伤', '保留完整原文，不自动建立事实。')
    for i in range(35): add_turn(f'无关话题{i}')
    memory = load('右手腕旧伤')
    chosen = pid(memory, tid)
    entry = next(e for e in memory['wiki_catalog']['entries'] if e['page_id'] == chosen)
    assert '右手腕旧伤' not in entry['title'] and entry['preview'] == ''
    assert entry['match_score'] > 0 and memory['short_memory'] == []
    prepare(memory, [{'page_ids': [chosen]}])
    assert '右手腕旧伤' in memory['short_memory'][0]['content']


@pytest.mark.parametrize('backend,wiki_enabled,blocked', [
    ('compact_v1', True, True), ('compact_v1', False, False), ('legacy', True, False)])
def test_database_outage_cannot_silently_switch_wiki_to_a_window(monkeypatch, backend, wiki_enabled, blocked):
    monkeypatch.setattr(db, 'DB_READY', False)
    monkeypatch.setattr(settings, 'memory_read_backend', backend)
    monkeypatch.setattr(settings, 'wiki_enabled', wiki_enabled)
    legacy = SimpleNamespace(load=AsyncMock(return_value={'short_memory': [{'role': 'user', 'content': '旧窗口'}]}))
    memory = compact.CompactMemory(legacy)
    result = asyncio.run(memory.load('demo-user', 'wiki-chat', '继续'))
    if blocked:
        assert result['wiki_error'] == 'wiki_database_unavailable' and result['short_memory'] == []
        assert memory.status()['context_strategy'] == 'wiki_addressed'
        legacy.load.assert_not_awaited()
    else:
        assert result['short_memory'][0]['content'] == '旧窗口'
        legacy.load.assert_awaited_once()


def test_catalog_budget_exhaustion_cannot_masquerade_as_no_history(compact_db, monkeypatch):
    from app.services.context.token_counter import TokenCounter
    # Fixed byte-sized catalog control, independent of the active provider's
    # tokenizer. Keep the empty-but-has-more safety assertion unchanged.
    monkeypatch.setattr(wc, 'TokenCounter', lambda: TokenCounter(tokenizer_path=""))
    monkeypatch.setattr(settings, 'wiki_enabled', True)
    monkeypatch.setattr(settings, 'wiki_catalog_max_tokens', 256)
    add_turn('原来的任务', '需要保留的回答')
    memory = load()
    assert memory['wiki_catalog']['entries'] == [] and memory['wiki_catalog']['has_more']
    with pytest.raises(wc.WikiReadError, match='wiki_catalog_budget_exceeded'):
        prepare(memory, [])
