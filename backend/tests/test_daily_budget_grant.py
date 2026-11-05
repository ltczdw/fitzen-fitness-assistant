"""One authorized day extension must never reset spending or other limits."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker

from app.core import clock
from app.core.config import settings
from app.db.model_budget_models import ModelQuota, ModelCallLedger, ModelResultCache
from app.services.llm.budget import BudgetStore, ModelBudgetError, daily_window, model_scope, paid_attempt


@pytest.fixture
def store(tmp_path, monkeypatch):
    engine=create_engine('sqlite:///'+str(tmp_path/'grant.db'))
    for model in (ModelQuota,ModelCallLedger,ModelResultCache):model.__table__.create(engine)
    monkeypatch.setattr(settings,'model_daily_max_calls',2)
    monkeypatch.setattr(clock,'utc_now',lambda:datetime(2026,10,20,7,0,tzinfo=timezone.utc))
    yield BudgetStore(sessionmaker(bind=engine,expire_on_commit=False))
    engine.dispose()


def spend(store,name,**kwargs):
    with model_scope(name,store=store,**kwargs):
        with paid_attempt('embedding','mock',{'synthetic':name}):pass


def test_preview_and_apply_preserve_usage_and_are_durable_idempotent(store):
    spend(store,'one');spend(store,'two')
    with store.session() as s:
        before=s.scalar(select(ModelQuota).where(ModelQuota.kind=='daily'))
        tokens=before.tokens
    with pytest.raises(ModelBudgetError,match='exhausted:daily'):spend(store,'blocked')
    preview=store.grant_daily_calls(150,day='2026-10-20')
    assert preview['max_calls']==2 and preview['remaining_calls']==0 and not preview['apply']
    first=store.grant_daily_calls(150,day='2026-10-20',apply=True)
    assert first['grant_added'] and first['calls']==2 and first['max_calls']==152
    assert first['expires_at']=='2026-10-20T16:00:00+00:00'
    second=BudgetStore(store.factory).grant_daily_calls(150,day='2026-10-20',apply=True)
    assert not second['grant_added'] and second['max_calls']==152
    with store.session() as s:
        assert s.scalar(select(ModelQuota).where(ModelQuota.kind=='daily')).tokens==tokens
        assert s.scalar(select(func.count()).select_from(ModelCallLedger))==2
    spend(store,'three')
    assert store.daily_status()['calls']==3 and store.daily_status()['remaining_calls']==149


@pytest.mark.parametrize('amount,day',[(0,'2026-10-20'),(-1,'2026-10-20'),(True,'2026-10-20'),
    (150,'2026-10-19'),(150,'2026-10-21')])
def test_invalid_or_other_day_grants_do_not_write(store,amount,day):
    with pytest.raises(ValueError):store.grant_daily_calls(amount,day=day,apply=True)
    with store.session() as s:
        assert s.scalar(select(func.count()).select_from(ModelQuota))==0


def test_grant_cannot_be_changed_or_replayed_for_more_calls(store):
    store.grant_daily_calls(150,day='2026-10-20',apply=True)
    with pytest.raises(ValueError,match='different_daily_grant'):
        store.grant_daily_calls(151,day='2026-10-20',apply=True)
    assert store.daily_status()['max_calls']==152


def test_grant_expires_at_shanghai_midnight_and_default_returns(store,monkeypatch):
    spend(store,'today')
    store.grant_daily_calls(150,day='2026-10-20',apply=True)
    monkeypatch.setattr(clock,'utc_now',lambda:datetime(2026,10,20,16,0,tzinfo=timezone.utc))
    status=store.daily_status()
    assert status['extra_calls']==0 and status['max_calls']==2 and status['calls']==0
    spend(store,'tomorrow-1');spend(store,'tomorrow-2')
    with pytest.raises(ModelBudgetError,match='exhausted:daily'):spend(store,'tomorrow-3')
    with pytest.raises(ValueError):store.grant_daily_calls(150,day='2026-10-20',apply=True)


def test_other_caps_and_tightening_are_not_relaxed_by_day_grant(store,monkeypatch):
    spend(store,'bounded',max_calls=1)
    store.grant_daily_calls(150,day='2026-10-20',apply=True)
    with pytest.raises(ModelBudgetError,match='exhausted:request'):spend(store,'bounded',max_calls=99)
    # Grant only changes calls; a token cap still fails before transport.
    with pytest.raises(ModelBudgetError,match='token_budget'):
        spend(store,'tiny-token-cap',max_tokens=1)
    monkeypatch.setattr(settings,'model_daily_max_calls',1000)
    assert store.daily_status()['max_calls']==152  # config increase cannot refill
    monkeypatch.setattr(settings,'model_daily_max_calls',1)
    spend(store,'tightened')
    assert store.daily_status()['max_calls']==151
    monkeypatch.setattr(settings,'model_daily_max_calls',2)
    store.grant_daily_calls(150,day='2026-10-20',apply=True)
    assert store.daily_status()['max_calls']==151  # replay cannot undo tightening


def test_parallel_grant_is_once_and_usage_stays_atomic(store):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:store.grant_daily_calls(3,day='2026-10-20',apply=True),range(4)))
    assert sum(result['grant_added'] for result in results)==1
    def call(index):
        try:spend(store,'parallel-'+str(index));return True
        except ModelBudgetError:return False
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(call,range(8)))
    assert sum(results)==5 and store.daily_status()['remaining_calls']==0


def test_daily_tokens_are_not_increased_by_call_grant(store,monkeypatch):
    monkeypatch.setattr(settings,'model_daily_max_tokens',1)
    store.grant_daily_calls(150,day='2026-10-20',apply=True)
    with pytest.raises(ModelBudgetError,match='token_budget_exhausted:daily'):
        spend(store,'daily-tokens')
    assert store.daily_status()['calls']==0


def test_summary_reports_authorized_effective_limit(store):
    from app.services.llm.observability import model_budget_summary
    spend(store,'used')
    store.grant_daily_calls(150,day='2026-10-20',apply=True)
    with model_scope('summary',store=store):
        report=model_budget_summary()
    assert report['ready'] and report['daily_max_calls']==152 and report['daily_default_max_calls']==2
    assert report['daily_extra_calls']==150 and report['reserved_attempts']==1 and report['remaining_calls']==151
