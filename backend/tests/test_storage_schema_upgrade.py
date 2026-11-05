"""Tests for storage schema upgrade."""
import pytest
from sqlalchemy import event,inspect,text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from test_manual_data_consistency import api,USER,create_mirror
from test_workout_record_bindings import snapshot


@pytest.mark.parametrize("phase",["before_commit","after_commit"])
def test_business_validation_respects_commit_attempt_boundary(api,monkeypatch,record_property,phase):
    from app.db import storage
    rec,_=create_mirror(api);before=snapshot(api)
    if phase=="after_commit":
        actual=Session.commit
        def late(session):
            actual(session)
            raise storage.BusinessValidationError("quality-late-error-after-real-commit")
        monkeypatch.setattr(Session,"commit",late)
    caught=None
    try:
        with storage.storage_session() as s:
            s.execute(text("UPDATE workout_records SET note='late-commit-evidence' WHERE id=:r"),{"r":rec["id"]})
            if phase=="before_commit":raise storage.BusinessValidationError("quality-pre-commit-input")
            s.commit()
    except Exception as exc:caught=exc
    after=snapshot(api)
    record_property("exception_type",type(caught).__name__)
    record_property("real_update_committed",str(after!=before))
    if phase=="before_commit":
        assert type(caught) is storage.BusinessValidationError
        assert after==before
    else:
        assert after["workout_records"][0]["note"]=="late-commit-evidence"
        assert isinstance(caught,storage.CommitUnknownError),type(caught).__name__


def test_new_schema_binding_unique_fk_and_set_null(api):
    rec,s=create_mirror(api)
    inspector=inspect(api.engine)
    assert "source_record_id" in {x["name"] for x in inspector.get_columns("workout_session_exercises")}
    fks=inspector.get_foreign_keys("workout_session_exercises")
    assert any(x["constrained_columns"]==["source_record_id"] and x["referred_table"]=="workout_records" and x["options"].get("ondelete","").upper()=="SET NULL" for x in fks),fks
    if api.engine.dialect.name=="sqlite":
        with api.engine.connect() as c:
            assert c.exec_driver_sql("PRAGMA foreign_keys").scalar()==1
    with pytest.raises(IntegrityError):
        with api.engine.begin() as c:
            c.execute(text("INSERT INTO workout_session_exercises(session_id,source_record_id,exercise_name,order_index,difficulty,note) VALUES (:s,:r,'duplicate-binding',99,'未记录','')"),{"s":s["id"],"r":rec["id"]})
    with pytest.raises(IntegrityError):
        with api.engine.begin() as c:
            c.execute(text("UPDATE workout_session_exercises SET source_record_id=999999 WHERE id=:i"),{"i":s["exercises"][0]["id"]})
    with api.engine.begin() as c:c.execute(text("DELETE FROM workout_records WHERE id=:r"),{"r":rec["id"]})
    with api.engine.connect() as c:
        assert c.execute(text("SELECT source_record_id FROM workout_session_exercises WHERE id=:i"),{"i":s["exercises"][0]["id"]}).scalar() is None
        assert c.execute(text("SELECT source_record_id FROM workout_sessions WHERE id=:i"),{"i":s["id"]}).scalar() is None


def test_existing_sqlite_schema_support_boundary(api,record_property,monkeypatch,tmp_path):
    if api.engine.dialect.name!="sqlite":pytest.skip("Direct historical SQLite schema upgrade only")
    from sqlite_workout_fixtures import TABLES,legacy_database,database_snapshot,assert_upgrade,emit
    missing=("workout_session_exercises",)
    # Build an independent historical database before init_db; never RENAME or
    # DROP a current parent, which can rewrite child FK targets or cascade rows.
    with legacy_database(monkeypatch,tmp_path,missing=missing) as engine:
        before=database_snapshot(engine)
        assert [len(before["rows"][t]) for t in TABLES]==[2,3,4,7]
        upgraded=assert_upgrade(engine,before,missing)
        emit(record_property,"direct_old_schema_before",before)
        emit(record_property,"direct_old_schema_upgraded",upgraded)
        record_property("sqlite_old_schema_upgraded","True; original IDs/fields/rows preserved; new bindings NULL; init twice")


@pytest.mark.parametrize("fault",[False,True],ids=["upgrade_idempotent","upgrade_transaction_rollback"])
def test_old_postgres_binding_upgrade_preserves_rows(api,fault,record_property):
    if api.engine.dialect.name!="postgresql":pytest.skip("Actual PostgreSQL upgrade required")
    from app.db import session as db
    rec,s=create_mirror(api)
    with api.engine.begin() as c:c.execute(text("ALTER TABLE workout_session_exercises DROP COLUMN source_record_id CASCADE"))
    before=snapshot(api);hit=[]
    def fail(conn,cursor,stmt,params,context,many):
        if "CREATE UNIQUE INDEX IF NOT EXISTS uq_workout_session_exercises_source_record" in stmt:
            hit.append(True);raise RuntimeError("quality-upgrade-index-failure")
    if fault:event.listen(api.engine,"before_cursor_execute",fail)
    try:ok=db.init_db()
    finally:
        if fault:event.remove(api.engine,"before_cursor_execute",fail)
    if fault:
        assert hit and not ok
        assert "source_record_id" not in {x["name"] for x in inspect(api.engine).get_columns("workout_session_exercises")}
        assert snapshot(api)==before
    else:
        assert ok and db.init_db()
        after=snapshot(api)
        for t in before:
            stripped=[{k:v for k,v in r.items() if t!="workout_session_exercises" or k!="source_record_id"} for r in after[t]]
            assert stripped==before[t]
        assert all(x["source_record_id"] is None for x in after["workout_session_exercises"])
        assert after["workout_sessions"][0]["source_record_id"]==rec["id"]
        record_property("old_pg_upgrade","idempotent; existing relations preserved; no startup adoption")
