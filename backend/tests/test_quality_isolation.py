"""Regression checks for fixture cleanup, database identity and init failures."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import event, text
from quality_isolation import isolated_database, TempPGDatabase, checked_engine


def test_init_failure_restores_database_globals(monkeypatch, tmp_path):
    from app.core.config import settings
    from app.db import session as db
    old = (db.engine, db.SessionLocal, db.DB_READY, settings.database_url)
    disposed = []
    on_dispose = disposed.append
    event.listen(__import__("sqlalchemy").engine.Engine, "engine_disposed", on_dispose)
    monkeypatch.setattr(db, "init_db", lambda: False)
    try:
        with pytest.raises(AssertionError, match="initialization failed"):
            with isolated_database(monkeypatch, tmp_path, "sqlite"):
                pytest.fail("must never reach yield after initialization failure")
        assert (db.engine, db.SessionLocal, db.DB_READY, settings.database_url) == old
        # Production configuration first recycles an unused file pool; the
        # context then disposes the same engine on initialization failure.
        assert len(disposed) == 2 and disposed[0] is disposed[1]
        assert disposed[-1] is not old[0]
    finally:
        event.remove(__import__("sqlalchemy").engine.Engine, "engine_disposed", on_dispose)


def test_assertion_failure_restores_database_globals(monkeypatch, tmp_path):
    from app.core.config import settings
    from app.db import session as db
    old = (db.engine, db.SessionLocal, db.DB_READY, settings.database_url)
    with pytest.raises(RuntimeError, match="test failure"):
        with isolated_database(monkeypatch, tmp_path, "sqlite") as engine:
            assert db.engine is engine and db.DB_READY
            raise RuntimeError("test failure")
    assert (db.engine, db.SessionLocal, db.DB_READY, settings.database_url) == old


def test_business_database_url_rejected_before_connect():
    with pytest.raises(AssertionError, match="non-test database rejected"):
        checked_engine("postgresql+psycopg://invalid@127.0.0.1/business")


def test_never_drop_database_when_create_did_not_succeed():
    temporary = TempPGDatabase.__new__(TempPGDatabase)
    temporary.created = False
    temporary.raw = Mock()
    temporary.drop()
    temporary.raw.connect.assert_not_called()
    temporary.raw.dispose.assert_called_once()


def test_wrong_database_identity_stops_before_application_sql():
    engine = checked_engine("postgresql+psycopg://invalid@127.0.0.1/fitzen_test_" + "a" * 32)
    connection = Mock(autocommit=False)
    connection.cursor.return_value.__enter__ = Mock(return_value=connection.cursor.return_value)
    connection.cursor.return_value.__exit__ = Mock(return_value=False)
    connection.cursor.return_value.fetchone.return_value = ("business",)
    try:
        with pytest.raises(AssertionError, match="identity mismatch"):
            engine.pool.dispatch.checkout(connection, None, None)
        assert connection.autocommit is False
        connection.cursor.return_value.execute.assert_called_once_with("SELECT current_database()")
    finally:
        engine.dispose()


def test_actual_pg_identity_on_reused_connection(monkeypatch, tmp_path):
    with isolated_database(monkeypatch, tmp_path, "postgres") as engine:
        identities = []
        for _ in range(3):
            with engine.connect() as conn:
                identities.append(conn.execute(text("SELECT current_database()")).scalar())
        assert identities == [engine.url.database] * 3
