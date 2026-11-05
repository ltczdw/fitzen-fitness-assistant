"""Keep pytest away from application fallback files and the live database."""
import builtins
import io
from pathlib import Path

import pytest


_USER_FOOD_LOG_DDL = """
CREATE TABLE IF NOT EXISTS user_food_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(100) NOT NULL,
    meal_group_id VARCHAR(100) NOT NULL,
    record_time TIMESTAMP NOT NULL,
    meal_type VARCHAR(20),
    food_id INTEGER,
    raw_food_name VARCHAR(200) NOT NULL,
    food_name VARCHAR(200) NOT NULL,
    weight_g NUMERIC NOT NULL,
    total_kcal NUMERIC NOT NULL,
    total_protein NUMERIC NOT NULL,
    total_carb NUMERIC NOT NULL,
    total_fat NUMERIC NOT NULL,
    total_fiber NUMERIC NOT NULL DEFAULT 0,
    is_estimated BOOLEAN NOT NULL DEFAULT 0,
    weight_source VARCHAR(50) NOT NULL DEFAULT 'user_input',
    match_method VARCHAR(50) NOT NULL DEFAULT 'exact',
    match_confidence FLOAT NOT NULL DEFAULT 1.0,
    idempotency_key VARCHAR(150) NOT NULL,
    calculation_version VARCHAR(100) NOT NULL DEFAULT 'food-calc-v1',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, idempotency_key)
)
"""


@pytest.fixture
def compact_db(tmp_path, monkeypatch):
    """Temporary SQLite database with compact memory and diagnostics tables."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    from app.core.config import settings
    from app.db import session as db
    from app.db.models import Base

    engine = create_engine("sqlite:///" + str(tmp_path / "compact.db"))
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text(_USER_FOOD_LOG_DDL))
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db, "SessionLocal", factory)
    monkeypatch.setattr(db, "DB_READY", True)
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(settings, "memory_read_backend", "compact_v1")
    monkeypatch.setattr(settings, "planner_shadow_enabled", False)
    yield factory
    engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def pin_legacy_chat_default_for_suite():
    """Keep the existing suite on the legacy baseline it was written against.

    Production now defaults to react_readonly via settings.chat_default_mode.
    Tests that need the ReAct default override settings for their own scope;
    the production default value itself is asserted in test_chat_mode_default.py.
    """
    from app.core.config import settings
    previous = settings.chat_default_mode
    settings.chat_default_mode = "legacy"
    try:
        yield
    finally:
        settings.chat_default_mode = previous


@pytest.fixture(autouse=True)
def isolate_application_storage(tmp_path, monkeypatch):
    from app.db import session as db
    from app.db import (repositories, workout_session_repository,
                        nutrition_repository, profile_repository, plan_repository,
                        food_repository, food_log_listing, food_log_mutations,
                        food_summary)

    # Tests that need SQL must explicitly supply their own session factory.
    monkeypatch.setattr(db, "DB_READY", False)
    monkeypatch.setattr(db, "SessionLocal", None)
    data_root = (Path(__file__).resolve().parents[1] / "data").resolve()
    def guard(original):
        def guarded(file, mode="r", *args, **kwargs):
            if isinstance(file, (str, bytes, Path)) and any(flag in mode for flag in "wax+"):
                target = Path(file).resolve()
                if target == data_root or data_root in target.parents:
                    raise AssertionError("Tests must not write backend/data; use tmp_path")
            return original(file, mode, *args, **kwargs)
        return guarded
    monkeypatch.setattr(builtins, "open", guard(builtins.open))
    monkeypatch.setattr(io, "open", guard(io.open))


@pytest.fixture(autouse=True)
def isolate_model_budget_storage(tmp_path, request):
    """Model tests never read/write the application's real quota database."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db.model_budget_models import ModelQuota, ModelCallLedger, ModelResultCache
    from app.services.llm.budget import BudgetStore, store_override, model_scope
    engine = create_engine('sqlite:///' + str(tmp_path / 'model-budget-fixture.db'))
    for model in (ModelQuota, ModelCallLedger, ModelResultCache):
        model.__table__.create(engine)
    store = BudgetStore(sessionmaker(bind=engine, expire_on_commit=False))
    token = store_override.set(store)
    try:
        with model_scope('pytest:' + tmp_path.name, kind='test', store=store) as budget:
            # unittest captures its coroutine context at collection time,
            # before pytest fixtures. Install the same isolated store there.
            instance_context = getattr(request.instance, '_asyncioTestContext', None)
            context_tokens = None
            if instance_context is not None:
                from app.services.llm.budget import active_scope
                context_tokens = (instance_context.run(store_override.set, store),
                                  instance_context.run(active_scope.set, budget))
            try:
                yield
            finally:
                if context_tokens is not None:
                    instance_context.run(active_scope.reset, context_tokens[1])
                    instance_context.run(store_override.reset, context_tokens[0])
    finally:
        store_override.reset(token)
        engine.dispose()
