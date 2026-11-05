"""Only an explicit isolated store may waive the daily call/token ceilings."""
from datetime import timedelta
from pathlib import Path
import sys

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.core import clock
from app.core.config import settings
from app.db.model_budget_models import ModelQuota, ModelCallLedger, ModelResultCache
from app.services.llm.budget import (BudgetStore, ModelBudgetError, key, model_scope,
                                     model_stage, paid_attempt)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from run_ai_hybrid_simulation import budget_policy, parse_args, simulation_budget_store


@pytest.fixture
def factory(tmp_path):
    engine = create_engine('sqlite:///' + str(tmp_path / 'daily-policy.db'))
    for model in (ModelQuota, ModelCallLedger, ModelResultCache):
        model.__table__.create(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False)
    engine.dispose()


def spend(store, name='default', *, usage=None, **options):
    with model_scope(name, store=store, **options):
        with paid_attempt('chat', 'mock-model', {'synthetic': name}, 8) as attempt:
            if usage is not None:
                attempt.observe({'usage': usage})


def budget_state(factory):
    with factory() as session:
        daily = session.scalar(select(ModelQuota).where(ModelQuota.kind == 'daily'))
        quotas = {q.quota_id: (q.kind, q.calls, q.tokens, q.max_calls, q.max_tokens)
                  for q in session.scalars(select(ModelQuota))}
        ledgers = [(row.call_id, row.status, row.reserved_tokens, tuple(row.quota_ids), row.usage)
                   for row in session.scalars(select(ModelCallLedger).order_by(ModelCallLedger.started_at))]
        return {'daily': None if daily is None else
                (daily.calls, daily.tokens, daily.max_calls, daily.max_tokens),
                'quotas': quotas, 'ledgers': ledgers}


def test_default_store_still_blocks_daily_call_limit(factory, monkeypatch):
    monkeypatch.setattr(settings, 'model_daily_max_calls', 1)
    monkeypatch.setattr(settings, 'model_daily_max_tokens', 1000000)
    store = BudgetStore(factory)
    spend(store, 'first')
    used = budget_state(factory)
    assert used['daily'][0] == 1 and used['daily'][1] > 0 and len(used['ledgers']) == 1
    with pytest.raises(ModelBudgetError, match='exhausted:daily'):
        spend(store, 'second')
    assert budget_state(factory) == used


def test_default_store_still_blocks_daily_token_limit(factory, monkeypatch):
    monkeypatch.setattr(settings, 'model_daily_max_calls', 10)
    monkeypatch.setattr(settings, 'model_daily_max_tokens', 1)
    store = BudgetStore(factory)
    with pytest.raises(ModelBudgetError, match='token_budget_exhausted:daily'):
        spend(store, 'daily-token')
    assert budget_state(factory) == {'daily': None, 'quotas': {}, 'ledgers': []}


def test_isolated_store_keeps_daily_usage_ledger_and_cache(factory, monkeypatch):
    monkeypatch.setattr(settings, 'model_daily_max_calls', 1)
    monkeypatch.setattr(settings, 'model_daily_max_tokens', 1)
    store = BudgetStore(factory, enforce_daily_limits=False)
    assert store.enforce_daily_limits is False
    for index in range(2):
        spend(store, 'isolated-' + str(index), usage={'total_tokens': 5})
    before = budget_state(factory)
    assert before['daily'][0] == 2 and before['daily'][1] == 10
    assert len(before['ledgers']) == 2

    reloaded = BudgetStore(factory, enforce_daily_limits=False)
    spend(reloaded, 'isolated-2', usage={'total_tokens': 5})
    after = budget_state(factory)
    assert after['daily'][0] == 3 and after['daily'][1] == 15
    assert len(after['ledgers']) == 3
    assert {row[0] for row in before['ledgers']} <= {row[0] for row in after['ledgers']}
    assert all(row[1] == 'success' and row[4] == {'total_tokens': 5} for row in after['ledgers'])
    assert all(any(qid.startswith('daily:') for qid in row[3]) for row in after['ledgers'])
    status = reloaded.daily_status()
    assert status['calls'] == 3 and status['max_calls'] == 1
    assert status['remaining_calls'] == 0 and status['daily_enforced'] is False

    with model_scope('cache-scope', store=reloaded) as scope:
        assert reloaded.cache('daily-policy-cache', scope, 'chat', {'answer': 'synthetic'})
        assert reloaded.cached('daily-policy-cache') == {'answer': 'synthetic'}
    assert BudgetStore(factory, enforce_daily_limits=False).cached('daily-policy-cache') == {
        'answer': 'synthetic'}


def test_request_stage_deadline_and_cancel_limits_remain(factory, monkeypatch):
    monkeypatch.setattr(settings, 'model_daily_max_calls', 1)
    monkeypatch.setattr(settings, 'model_daily_max_tokens', 1)
    store = BudgetStore(factory, enforce_daily_limits=False)
    empty = budget_state(factory)
    with pytest.raises(ModelBudgetError, match='token_budget_exhausted:request'):
        spend(store, 'request-token', max_tokens=1)
    assert budget_state(factory) == empty

    spend(store, 'request-calls', max_calls=1, max_tokens=100000)
    used = budget_state(factory)
    with pytest.raises(ModelBudgetError, match='exhausted:request'):
        spend(store, 'request-calls', max_calls=99)
    assert budget_state(factory) == used

    with model_scope('stage-calls', store=store, max_calls=10, max_tokens=100000):
        with model_stage('bounded', max_calls=1):
            with paid_attempt('chat', 'mock', {'stage': 1}, 8):
                pass
            used = budget_state(factory)
            with pytest.raises(ModelBudgetError, match='exhausted:stage'):
                with paid_attempt('chat', 'mock', {'stage': 2}, 8):
                    pass
            assert budget_state(factory) == used

    with model_scope('stage-token', store=store, max_calls=10, max_tokens=100000) as scope:
        moment = clock.utc_now()
        stage_id = 'stage:' + key(scope.quota_id + 'bounded-token')
        with factory() as session, session.begin():
            session.add(ModelQuota(quota_id=stage_id, kind='stage', calls=0, tokens=0,
                                   max_calls=10, max_tokens=1, created_at=moment,
                                   expires_at=moment + timedelta(seconds=600)))
        used = budget_state(factory)
        with model_stage('bounded-token', max_calls=10):
            with pytest.raises(ModelBudgetError, match='token_budget_exhausted:stage'):
                with paid_attempt('chat', 'mock', {'stage-token': 1}, 8):
                    pass
        assert budget_state(factory) == used

    with model_scope('deadline', store=store) as scope:
        scope.started -= scope.ttl_seconds + 1
        used = budget_state(factory)
        with pytest.raises(ModelBudgetError, match='deadline'):
            with paid_attempt('chat', 'mock', {'deadline': 1}, 8):
                pass
        assert budget_state(factory) == used

    with model_scope('cancel', store=store) as scope:
        scope.cancelled = True
        used = budget_state(factory)
        with pytest.raises(ModelBudgetError, match='cancelled'):
            with paid_attempt('chat', 'mock', {'cancel': 1}, 8):
                pass
        assert budget_state(factory) == used


def test_daily_exception_requires_explicit_factory_and_does_not_pollute_default(factory,
                                                                               monkeypatch):
    with pytest.raises(ValueError, match='explicit_factory'):
        BudgetStore(enforce_daily_limits=False)
    assert BudgetStore().enforce_daily_limits is True
    assert BudgetStore(factory).enforce_daily_limits is True

    monkeypatch.setattr(settings, 'model_daily_max_calls', 1)
    monkeypatch.setattr(settings, 'model_daily_max_tokens', 1)
    isolated = BudgetStore(factory, enforce_daily_limits=False)
    spend(isolated, 'isolated-one', usage={'total_tokens': 5})
    spend(isolated, 'isolated-two', usage={'total_tokens': 5})
    used = budget_state(factory)
    with pytest.raises(ModelBudgetError, match='exhausted:daily'):
        spend(BudgetStore(factory), 'default-blocked')
    assert budget_state(factory) == used
    spend(isolated, 'isolated-three', usage={'total_tokens': 5})
    after = budget_state(factory)
    assert after['daily'][0] == 3 and len(after['ledgers']) == 3


def test_simulation_parser_defaults_to_isolated_daily_exception_and_explicit_enforce(factory,
                                                                                     monkeypatch):
    default = parse_args(['--live', '--allow-unpriced-limited'])
    assert default.enforce_daily_budget is False
    explicit = parse_args(['--live', '--allow-unpriced-limited', '--enforce-daily-budget'])
    assert explicit.enforce_daily_budget is True
    with pytest.raises(SystemExit):
        parse_args([])
    policy = budget_policy(settings, daily_enforced=default.enforce_daily_budget)
    assert policy['daily_enforced'] is False
    assert policy['daily_enforcement'] == 'disabled_isolated_simulation'
    assert policy['request_max_calls'] == settings.model_request_max_calls
    assert policy['request_max_tokens'] == settings.model_request_max_tokens
    assert policy['stage_max_calls'] == settings.model_stage_max_calls
    assert budget_policy(settings, daily_enforced=explicit.enforce_daily_budget)['daily_enforced'] is True

    monkeypatch.setattr(settings, 'model_daily_max_calls', 1)
    monkeypatch.setattr(settings, 'model_daily_max_tokens', 1)
    default_store = simulation_budget_store(factory, default)
    assert default_store.factory is factory and default_store.enforce_daily_limits is False
    spend(default_store, 'simulation-default-one')
    spend(default_store, 'simulation-default-two')
    assert budget_state(factory)['daily'][0] == 2

    enforced_store = simulation_budget_store(factory, explicit)
    assert enforced_store.factory is factory and enforced_store.enforce_daily_limits is True
    used = budget_state(factory)
    with pytest.raises(ModelBudgetError, match='exhausted:daily'):
        spend(enforced_store, 'simulation-enforced')
    assert budget_state(factory) == used
