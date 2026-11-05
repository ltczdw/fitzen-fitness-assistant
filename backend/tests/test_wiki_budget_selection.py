"""Budget fitting and directed source dependencies, with real isolated SQL."""
import asyncio
import hashlib
import json

import pytest
from app.services.context.token_counter import TokenCounter

from app.core.config import settings
from app.services.agent import dialogue_context, semantics
from app.services.context.budget import ContextBlock, ContextBudgetExceeded, ContextBudgetManager
from app.services.memory import wiki_context as wc
from app.services.memory.selection import fit_pages
from app.services.llm.structured import StructuredOutputError
from test_wiki_addressed_context import memory_db, add_turn, load, pid, prepare, selector


@pytest.fixture(autouse=True)
def clean_semantic_state():
    token = semantics.request_understanding.set({})
    yield
    semantics.request_understanding.reset(token)


def indexed(memory):
    memory.update(wiki_catalog_entries={e['page_id']: e for e in memory['wiki_catalog']['entries']}, wiki_loaded_pages={})
    return memory


def read_limit(monkeypatch, memory, limit):
    # ``prepare`` now uses the independent source-read budget; the final answer
    # context budget remains a separate ContextBudgetManager input budget.
    monkeypatch.setattr(settings, 'wiki_source_read_max_tokens', max(1000, limit))


def test_catalog_exposes_conservative_page_size_without_body(memory_db):
    answer = 'BODY-SENTINEL-' + '中文原文' * 30
    tid = add_turn('目录测试', answer)
    memory = load()
    entry = next(e for e in memory['wiki_catalog']['entries'] if e['turn_id'] == tid)
    assert entry['estimated_tokens'] == len('目录测试'.encode()) + len(answer.encode()) + 400
    assert 'BODY-SENTINEL' not in json.dumps(memory)
    llm = prepare(memory, [{'page_ids': [entry['page_id']]}])
    payload = json.loads(llm.complete_structured.await_args.args[0][-1]['content'])
    offered = next(e for e in payload['catalog']['entries'] if e['page_id'] == entry['page_id'])
    assert offered['read_cost'] == entry['estimated_tokens']


def test_optional_oversized_page_does_not_block_useful_smaller_ones(memory_db):
    tids = [add_turn('large', 'L' * 8000), add_turn('medium', 'M' * 500),
            add_turn('tiny', 'T' * 20), add_turn('required', 'R' * 50)]
    memory = indexed(load())
    large, medium, tiny, required = [pid(memory, tid) for tid in tids]
    limit = sum(memory['wiki_catalog_entries'][p]['estimated_tokens'] for p in (required, medium))
    plan = wc.WikiReadPlan(page_ids=[large, medium, tiny, required], required_page_ids=[required])
    memory_db.queries.clear()
    pages, used = wc.read_pages(memory, plan.page_ids, limit, plan=plan)
    # Incremental value density is a heuristic: the shortest complete unit can
    # win a slot over a longer, higher-ranked optional page. The oversized page
    # never blocks a fitting smaller alternative.
    assert {p['page_id'] for p in pages} == {tiny, required}
    assert used == sum(memory['wiki_catalog_entries'][p]['estimated_tokens'] for p in (required, tiny))
    assert memory['wiki_selection_feedback']['skipped'] == [large, medium]
    assert sum(sql.lstrip().upper().startswith('SELECT') for sql in memory_db.queries) == 3
    assert sum(len(p['messages']) for p in pages) == 4


def test_optional_dependency_bundle_is_atomic_and_required_closure_reserved(memory_db):
    tids = [add_turn('main', 'A' * 200), add_turn('prerequisite', 'B' * 200), add_turn('alternative', 'C')]
    memory = indexed(load())
    main, dependency, alternative = [pid(memory, tid) for tid in tids]
    sizes = {p: memory['wiki_catalog_entries'][p]['estimated_tokens'] for p in (main, dependency, alternative)}
    plan = wc.WikiReadPlan(page_ids=[main, alternative], dependencies={main: [dependency]})
    pages, used = wc.read_pages(memory, plan.page_ids, sizes[alternative], plan=plan)
    assert [p['page_id'] for p in pages] == [alternative]
    assert used == sizes[alternative]
    plan = wc.WikiReadPlan(page_ids=[alternative, main], required_page_ids=[main], dependencies={main: [dependency]})
    pages, used = wc.read_pages(memory, plan.page_ids, sizes[main] + sizes[dependency], plan=plan)
    assert {p['page_id'] for p in pages} == {main, dependency}
    assert set(memory['wiki_selection_feedback']['required']) == {main, dependency}


def test_required_overflow_automatically_reselects_with_size_feedback(memory_db, monkeypatch):
    big_turn = add_turn('过宽的旧背景', '大段原文' * 5000)
    small_turn = add_turn('实际需要的原始安排', '周三做两组徒手训练。')
    memory = load()
    big, small = pid(memory, big_turn), pid(memory, small_turn)
    size = next(e['estimated_tokens'] for e in memory['wiki_catalog']['entries'] if e['page_id'] == small)
    read_limit(monkeypatch, memory, size)
    llm = prepare(memory, [
        {'page_ids': [big], 'required_page_ids': [big]},
        {'page_ids': [small], 'required_page_ids': [small],
         'reselection_reason': '当前任务只需要这份安排，上一份只是可省略背景。'}])
    assert llm.complete_structured.await_count == 2
    assert {m['turn_id'] for m in memory['short_memory']} == {small_turn}
    assert memory['wiki_read_tokens'] == size
    feedback = json.loads(llm.complete_structured.await_args_list[1].args[0][-1]['content'])['selection_feedback']
    assert feedback['required_tokens'] > feedback['remaining_tokens']
    assert memory['wiki_reselection_history'][-1]['resolved'] is True
    assert memory['wiki_reselection_history'][-1]['reason']


def test_reselection_reuses_cached_original_and_cumulative_read_budget(memory_db, monkeypatch):
    first = add_turn('必要来源一', '一')
    big = add_turn('过宽来源', '很长' * 10000)
    last = add_turn('必要来源二', '二')
    memory = load()
    a, b, c = [pid(memory, t) for t in (first, big, last)]
    cost = {e['page_id']: e['estimated_tokens'] for e in memory['wiki_catalog']['entries']}
    read_limit(monkeypatch, memory, cost[a] + cost[c])
    llm = prepare(memory, [
        {'page_ids': [a], 'required_page_ids': [a], 'done': False},
        {'page_ids': [b], 'required_page_ids': [b]},
        {'page_ids': [a, c], 'required_page_ids': [a, c], 'reselection_reason': '已读来源一与来源二足够，过宽来源不是本任务必要内容。'}])
    assert llm.complete_structured.await_count == 3
    assert memory['wiki_read_tokens'] == cost[a] + cost[c]
    assert set(memory['wiki_required_pages']) == {a, c}
    payload = json.loads(llm.complete_structured.await_args_list[2].args[0][-1]['content'])
    assert payload['loaded'][0]['page_id'] == a and payload['loaded'][0]['read_cost'] == 0
    assert len(memory['short_memory']) == 4


def test_required_overflow_can_search_for_a_smaller_source(memory_db, monkeypatch):
    from sqlalchemy import select
    from app.db.wiki_models import WikiPage
    target = add_turn('原始简短安排', '周三训练。')
    for i in range(34): add_turn(f'较新无关{i}')
    oversized = add_turn('当前宽泛背景', '很长的原文' * 8000)
    memory = load()
    with memory_db.factory() as s:
        target_id = s.scalar(select(WikiPage.page_id).where(WikiPage.source_turn_id == target))
    assert target_id not in {e['page_id'] for e in memory['wiki_catalog']['entries']}
    big = pid(memory, oversized)
    read_limit(monkeypatch, memory, 1000)
    llm = prepare(memory, [
        {'page_ids': [big], 'required_page_ids': [big], 'query': '原始简短安排'},
        {'page_ids': [target_id], 'required_page_ids': [target_id], 'reselection_reason': '找到直接回答当前任务的原始简短安排。'}])
    assert {m['turn_id'] for m in memory['short_memory']} == {target}
    assert llm.complete_structured.await_count == 2


@pytest.mark.parametrize('bad', [
    {'page_ids': [], 'required_page_ids': ['unoffered']},
    {'page_ids': [], 'dependencies': {'unoffered': ['another']}},
])
def test_model_cannot_invent_required_or_dependency_addresses(memory_db, bad):
    add_turn('真实来源')
    memory = load()
    with pytest.raises(StructuredOutputError):
        prepare(memory, [bad])
    assert memory['wiki_read_tokens'] == 0 and memory['short_memory'] == []


@pytest.mark.parametrize('case', ['local', 'foreign', 'hidden', 'wrong_scope'])
def test_cold_resume_restores_dependency_bindings_without_model_or_unrelated_sources(memory_db, case):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.db.memory_models import MemoryTurn
    from app.services.memory import compact
    from app.services.agent.react_write_runtime import WriteRuntimeSession
    source = '继续，共用依据B'
    a_turn = add_turn('任务A', 'A-ORIGINAL-BODY')
    b_turn = add_turn('共用依据B', 'B-ORIGINAL-BODY', cid='other-chat' if case == 'foreign' else 'wiki-chat')
    c_turn = add_turn('无关任务C')
    memory = load(source)
    a, b, c = [pid(memory, turn) for turn in (a_turn, b_turn, c_turn)]
    raw = {'tasks': [{'task_type': 'training_qa', 'action': 'question', 'text': source,
        'is_negated': False, 'is_hypothetical': False, 'is_third_party': False,
        'history_requirement': 'required'}],
        'dialogue_ids': [a_turn + ':assistant'], 'relation': 'follow_up',
        'safety': {'level': 'normal', 'evidence': '', 'categories': [], 'reason': 'scripted'}}
    llm = selector(iter([{'page_ids': [a, c], 'dependencies': {a: [b], c: [b]}}]), raw)
    result = asyncio.run(semantics.understand(llm, source, context=memory))
    frozen = result['__wiki_context__']
    assert {page['page_id'] for page in frozen['pages']} == {a, b}
    assert 'ORIGINAL-BODY' not in json.dumps(frozen)
    snapshot = {'tasks': [{'task_type': 'react_controller', 'parameters': {
        'input_fingerprint': hashlib.sha256(source.encode()).hexdigest(), 'semantic_understanding': result}}]}
    token = semantics.request_understanding.set({})
    try:
        WriteRuntimeSession._restore_source_understanding(snapshot, source)
        if case == 'hidden':
            with memory_db.factory.begin() as s:
                s.get(MemoryTurn, b_turn).hidden_at = compact.now()
        resumed = load(source, cid='wrong-chat' if case == 'wrong_scope' else 'wiki-chat')
        no_model = SimpleNamespace(complete_structured=AsyncMock(side_effect=AssertionError('resume re-planned')))
        if case in {'hidden', 'wrong_scope'}:
            with pytest.raises(wc.WikiReadError):
                asyncio.run(semantics.understand(no_model, source, context=resumed))
            assert resumed['short_memory'] == []
        else:
            restored = asyncio.run(semantics.understand(no_model, source, context=resumed))
            selected = dialogue_context.answer_memory(resumed, restored)
            assert set(selected['wiki_selected_pages']) == {a, b}
            assert c not in selected['wiki_selected_pages']
            if case == 'foreign':
                assert len(selected['short_memory']) == 2 and selected['retrieved_history'][0]['page_id'] == b
            else:
                assert len(selected['short_memory']) == 4
        no_model.complete_structured.assert_not_awaited()
    finally:
        semantics.request_understanding.reset(token)


def test_task_scope_keeps_only_directed_dependencies_and_complete_pairs(memory_db):
    tids = [add_turn('任务A'), add_turn('共用依据B'), add_turn('另一任务C')]
    memory = load()
    a, b, c = [pid(memory, tid) for tid in tids]
    prepare(memory, [{'page_ids': [a, c], 'dependencies': {a: [b], c: [b]}}])
    selected = dialogue_context.scoped_worker_memory(memory, {'context_ids': [tids[0] + ':assistant']})
    assert {m['turn_id'] for m in selected['short_memory']} == set(tids[:2])
    assert len(selected['short_memory']) == 4
    assert set(selected['wiki_selected_pages']) == {a, b}
    assert all(not m['app_meta']['wiki_root'] for m in selected['short_memory'] if m['turn_id'] == tids[1])
    viewed = dialogue_context.answer_memory(memory, {'dialogue_ids': [tids[0] + ':assistant']})
    assert viewed['short_memory'] == selected['short_memory']


def test_final_prompt_preserves_model_priority_over_recency(memory_db):
    old = add_turn('早期的相关安排', 'HIGH-' + 'A' * 350)
    new = add_turn('较新的可选背景', 'LOW-' + 'B' * 350)
    memory = load()
    prepare(memory, [{'page_ids': [pid(memory, old), pid(memory, new)]}])
    # Include the source-time provenance carried by the actual final unit wire.
    high_state = {**memory,
                  'short_memory': [m for m in memory['short_memory'] if m['turn_id'] == old],
                  'wiki_units': [u for u in memory['wiki_units'] if u['turn_id'] == old]}
    needed = ContextBudgetManager(window=100000, output=100, margin=0).assemble(
        system='规则', query='继续', state=high_state, hits=[]).report['used_tokens']
    manager = ContextBudgetManager(window=needed + 100, output=100, margin=0)
    result = manager.assemble(system='规则', query='继续', state=memory, hits=[])
    assert any('HIGH-' in m['content'] for m in result)
    assert not any('LOW-' in m['content'] for m in result)
    assert result.report['used_tokens'] <= result.report['input_budget']


def test_shared_dependencies_and_cycles_are_counted_once():
    result = fit_pages(['A', 'C'], ['A', 'C'], {'A': ['B'], 'C': ['B']},
                       {'A': 100, 'B': 100, 'C': 100}, 300)
    assert set(result['accepted']) == {'A', 'B', 'C'} and result['selected_tokens'] == 300
    result = fit_pages(['A', 'C'], [], {'A': ['B'], 'B': ['A']}, {'A': 200, 'B': 200, 'C': 100}, 100)
    assert result['accepted'] == ['C']


def blocks(required=False):
    return [ContextBlock('A', 'recent', [{'role': 'user', 'content': 'A' * 200}], 40,
                         importance=0.9, required=required, group_id='wiki:A', depends_on=('wiki:B',)),
            ContextBlock('B', 'recent', [{'role': 'assistant', 'content': 'B' * 200}], 40,
                         importance=0.1, group_id='wiki:B', root=False),
            ContextBlock('C', 'recent', [{'role': 'user', 'content': 'C' * 100}], 40,
                         importance=0.8, group_id='wiki:C')]


def test_final_budget_skips_oversized_bundle_and_keeps_fitting_alternative():
    manager = ContextBudgetManager(window=350, output=100, margin=0, counter=TokenCounter(tokenizer_path=""))
    result = manager.build(blocks())
    assert [m['content'] for m in result] == ['C' * 100]
    assert set(result.report['dropped_blocks']) == {'A', 'B'}


def test_required_dependency_cannot_be_pruned_to_make_required_parent_fit():
    manager = ContextBudgetManager(window=350, output=100, margin=0, counter=TokenCounter(tokenizer_path=""))
    with pytest.raises(ContextBudgetExceeded):
        manager.build(blocks(required=True))


def test_final_budget_removes_optional_dependency_cycles_atomically():
    items = blocks()
    items[1].depends_on = ('wiki:A',)
    result = ContextBudgetManager(window=350, output=100, margin=0, counter=TokenCounter(tokenizer_path="")).build(items)
    assert [m['content'] for m in result] == ['C' * 100]
