"""SQLite production defaults, direct old-schema compatibility and real DDL atomicity."""
import platform
import sqlite3

import pytest
import sqlalchemy
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError

from sqlite_workout_fixtures import (assert_legacy_rows_preserved, BINDINGS, INDEX, TABLES, USER, assert_binding_schema,
    assert_upgrade, binding_schema, database_snapshot, emit, legacy_database)


@pytest.fixture(autouse=True)
def sqlite_dialect_evidence(record_property):
    record_property('dialect', 'sqlite')


def physical(connection):
    return connection.connection.dbapi_connection


def test_production_factory_first_parallel_reuse_dispose(tmp_path, monkeypatch, record_property):
    from app.db import session as db
    from sqlalchemy.orm import sessionmaker
    engine = db.create_database_engine('sqlite:///' + str(tmp_path / 'factory.db'), pool_size=3, max_overflow=0)
    monkeypatch.setattr(db, 'engine', engine)
    monkeypatch.setattr(db, 'SessionLocal', sessionmaker(bind=engine))
    try:
        with engine.connect() as a, engine.connect() as b, engine.connect() as c:
            raw = [physical(x) for x in (a, b, c)]
            assert len({id(x) for x in raw}) == 3
            flags = [x.exec_driver_sql('PRAGMA foreign_keys').scalar() for x in (a, b, c)]
            assert flags == [1, 1, 1]
            mode = getattr(raw[0], 'autocommit', 'attribute-unavailable')
            emit(record_property, 'runtime', {'python': platform.python_version(), 'sqlite': sqlite3.sqlite_version,
                'sqlalchemy': sqlalchemy.__version__, 'autocommit': mode, 'legacy_constant': sqlite3.LEGACY_TRANSACTION_CONTROL,
                'isolation_level': raw[0].isolation_level, 'factory': 'app.db.session.create_database_engine'})
            emit(record_property, 'physical_connections', {'distinct': 3, 'ids': [id(x) for x in raw], 'foreign_keys': flags})
        with engine.connect() as c:
            assert any(physical(c) is r for r in raw)
            assert c.exec_driver_sql('PRAGMA foreign_keys').scalar() == 1
            emit(record_property, 'pooled_reuse', True)
        engine.dispose()
        with engine.connect() as c:
            assert all(physical(c) is not r for r in raw)
            assert c.exec_driver_sql('PRAGMA foreign_keys').scalar() == 1
            emit(record_property, 'dispose_new_physical_connection', True)
        assert db.init_db() and db.DB_READY and db.init_db() and db.DB_READY
        emit(record_property, 'new_schema', assert_binding_schema(engine))
    finally:
        engine.dispose()


@pytest.mark.parametrize('active', [False, True], ids=['idle-file-recycle', 'active-transaction-preserved'])
def test_configure_existing_file_engine(tmp_path, active, record_property):
    from app.db import session as db
    engine = create_engine('sqlite:///' + str(tmp_path / 'existing.db'))
    try:
        with engine.begin() as c:
            assert c.exec_driver_sql('PRAGMA foreign_keys').scalar() == 0
            c.exec_driver_sql('CREATE TABLE sentinel(id INTEGER PRIMARY KEY)')
        if active:
            with engine.connect() as c:
                transaction = c.begin()
                c.exec_driver_sql('INSERT INTO sentinel VALUES(1)')
                raw = physical(c)
                db.configure_sqlite_engine(engine)
                assert physical(c) is raw and raw.in_transaction and transaction.is_active
                assert c.exec_driver_sql('SELECT COUNT(*) FROM sentinel').scalar() == 1
                # An old, checked-out connection is intentionally not upgraded.
                assert c.exec_driver_sql('PRAGMA foreign_keys').scalar() == 0
                transaction.rollback()
            engine.dispose()  # Caller has now reached a safe recycling point.
        else:
            db.configure_sqlite_engine(engine)
        listener_count = len(list(engine.pool.dispatch.connect))
        for _ in range(3): assert db.configure_sqlite_engine(engine) is engine
        assert len(list(engine.pool.dispatch.connect)) == listener_count
        with engine.connect() as c:
            assert c.exec_driver_sql('PRAGMA foreign_keys').scalar() == 1
            assert c.exec_driver_sql('SELECT COUNT(*) FROM sentinel').scalar() == 0
        emit(record_property, 'configuration', {'active_transaction_preserved': active, 'sentinel_committed_rows': 0,
            'new_physical_foreign_keys': 1, 'listener_count_unchanged': listener_count})
    finally:
        engine.dispose()


def test_new_binding_sql_unique_null_fk_set_null_and_cascade(tmp_path, monkeypatch, record_property):
    from app.db import session as db
    with legacy_database(monkeypatch, tmp_path, fresh=True) as engine:
        assert db.init_db() and db.init_db()
        emit(record_property, 'constraints', assert_binding_schema(engine))
        for table, cloneid in zip(BINDINGS, (22, 31)):
            with pytest.raises(IntegrityError):
                with engine.begin() as c: c.exec_driver_sql(f'UPDATE {table} SET source_record_id=10 WHERE id={cloneid}')
            with pytest.raises(IntegrityError):
                with engine.begin() as c: c.exec_driver_sql(f'UPDATE {table} SET source_record_id=999999 WHERE id={cloneid}')
        with engine.begin() as c:
            c.exec_driver_sql("INSERT INTO workout_sessions(id,user_id,session_date,title,note) VALUES(23,'sqlite-legacy-user','2026-10-03','null-control','')")
            c.exec_driver_sql("INSERT INTO workout_session_exercises(id,session_id,exercise_name,difficulty,order_index,note) VALUES(34,23,'null-control','未记录',0,'')")
            assert c.exec_driver_sql('SELECT COUNT(*) FROM workout_sessions WHERE source_record_id IS NULL').scalar() == 2
            assert c.exec_driver_sql('SELECT COUNT(*) FROM workout_session_exercises WHERE source_record_id IS NULL').scalar() == 3
        before = database_snapshot(engine)['rows']
        with engine.begin() as c: c.exec_driver_sql('DELETE FROM workout_records WHERE id=10')
        after = database_snapshot(engine)['rows']
        assert next(r for r in after[BINDINGS[0]] if r['id'] == 20)['source_record_id'] is None
        assert next(r for r in after[BINDINGS[1]] if r['id'] == 30)['source_record_id'] is None
        assert after[TABLES[3]] == before[TABLES[3]]
        with engine.begin() as c: c.exec_driver_sql('DELETE FROM workout_sessions WHERE id=20')
        final = database_snapshot(engine)['rows']
        assert {r['id'] for r in final[TABLES[2]]} == {32, 33, 34}
        assert {r['id'] for r in final[TABLES[3]]} == {43, 44, 45, 46}
        assert [r for r in final[TABLES[1]] if r['id'] == 21] == [r for r in before[TABLES[1]] if r['id'] == 21]
        assert [r for r in final[TABLES[0]] if r['id'] == 11] == [r for r in before[TABLES[0]] if r['id'] == 11]
        emit(record_property, 'sql_controls', {'unique_rejected': 2, 'invalid_fk_rejected': 2, 'multiple_null': True,
            'both_set_null': True, 'session_exercise_set_cascade': True, 'unrelated_preserved': True})


SHAPES = [('exercise-missing', (BINDINGS[1],), True), ('session-missing', (BINDINGS[0],), True),
          ('both-missing', BINDINGS, True), ('columns-no-unique', (), False), ('already-upgraded', (), True)]


@pytest.mark.parametrize('shape,missing,unique', SHAPES, ids=[x[0] for x in SHAPES])
def test_five_direct_old_shapes_idempotent_asgi(tmp_path, monkeypatch, record_property, shape, missing, unique):
    from test_manual_data_consistency import API
    with legacy_database(monkeypatch, tmp_path, missing=missing, unique=unique) as engine:
        before = database_snapshot(engine)
        assert set(before['rows']) == set(TABLES)
        assert [len(before['rows'][t]) for t in TABLES] == [2, 3, 4, 7]
        emit(record_property, 'direct_old_fixture', before)
        first = assert_upgrade(engine, before, missing)
        emit(record_property, 'upgrade_rows', first['rows'])
        emit(record_property, 'upgrade_schema', binding_schema(engine))
        api = API(engine)
        # Edit an existing independent old action; its heterogeneous facts and
        # original IDs survive, and the other old mirror/user remain unchanged.
        response = api.patch('/api/workout-session-sets/45', params=USER, json={'weight_kg': 110, 'reps': 5})
        assert response.status_code == 200, response.text
        after = database_snapshot(engine)['rows']
        assert {r['id'] for r in after[TABLES[3]]} == {r['id'] for r in first['rows'][TABLES[3]]}
        assert_legacy_rows_preserved([next(r for r in before['rows'][TABLES[3]] if r['id'] == 46)], [next(r for r in after[TABLES[3]] if r['id'] == 46)])
        assert after[TABLES[0]] == first['rows'][TABLES[0]]
        created = api.post('/api/workouts', json={**USER, 'exercise': '新镜像卧推', 'sets': 2, 'reps': 8, 'weight_kg': 60,
            'set_details': [{'weight_kg': 60, 'reps': 8}, {'weight_kg': 62.5, 'reps': 6}]})
        assert created.status_code == 200, created.text
        rec = created.json()
        sessions = api.get('/api/workout-sessions', params=USER).json()
        mirror = next(s for s in sessions if s['id'] == rec['session_id'])
        groups = mirror['exercises'][0]['sets']
        assert [(g['weight_kg'], g['reps']) for g in groups] == [(60, 8), (62.5, 6)]
        assert api.delete(f"/api/workouts/{rec['id']}", params=USER).status_code == 200
        assert api.delete('/api/workout-sessions/22', params=USER).status_code == 200
        final = database_snapshot(engine)['rows']
        for t in TABLES:
            removed = {TABLES[0]: set(), TABLES[1]: {22}, TABLES[2]: {33}, TABLES[3]: {45, 46}}[t]
            assert final[t] == [r for r in first['rows'][t] if r['id'] not in removed]
        emit(record_property, 'asgi_controls', {'patch_old_set': response.status_code, 'created_record': rec['id'],
            'heterogeneous_new_mirror': True, 'deleted_new_mirror_and_old_independent': True, 'old_mirror_and_other_user_unchanged': True})


@pytest.mark.parametrize('missing,fault_table', [((BINDINGS[0],), BINDINGS[0]), ((BINDINGS[1],), BINDINGS[1]), (BINDINGS, BINDINGS[1])],
                         ids=['session-alter-index-fault', 'exercise-alter-index-fault', 'first-table-upgraded-second-fault'])
def test_actual_ddl_rollback_after_successful_alter(tmp_path, monkeypatch, record_property, missing, fault_table, caplog):
    from app.db import session as db
    with legacy_database(monkeypatch, tmp_path, missing=missing) as engine:
        before = database_snapshot(engine); successful = []; faults = []
        def complete(conn, cursor, statement, params, context, many):
            if statement.lstrip().upper().startswith(('ALTER TABLE', 'CREATE UNIQUE INDEX')): successful.append(statement)
        def fail(conn, cursor, statement, params, context, many):
            if 'CREATE UNIQUE INDEX IF NOT EXISTS ' + INDEX[fault_table] in statement:
                faults.append(statement); raise RuntimeError('sqlite-acceptance-after-alter-index-fault')
        event.listen(engine, 'after_cursor_execute', complete)
        event.listen(engine, 'before_cursor_execute', fail)
        try: ok = db.init_db()
        finally:
            event.remove(engine, 'after_cursor_execute', complete); event.remove(engine, 'before_cursor_execute', fail)
        after = database_snapshot(engine)
        emit(record_property, 'rollback', {'successful_statements': successful, 'fault_statements': faults,
             'init_ok': ok, 'db_ready': db.DB_READY, 'complete_schema_and_data_unchanged': before == after,
             'before': before, 'after': after})
        assert faults and any('ALTER TABLE ' + fault_table in s for s in successful)
        if len(missing) == 2:
            assert any(INDEX[BINDINGS[0]] in s for s in successful)
        assert not ok and not db.DB_READY and after == before
        assert 'sqlite-acceptance-after-alter-index-fault' in caplog.text
        assert_upgrade(engine, before, missing)
        emit(record_property, 'retry_after_fault_removed', True)


INVALID = [(t, k) for t in BINDINGS for k in
    ('duplicate', 'wrong_target', 'wrong_action', 'wrong_type', 'not_null', 'missing_fk')]


@pytest.mark.parametrize('invalid', INVALID, ids=[t + '-' + k for t, k in INVALID])
def test_invalid_existing_shape_rejected_with_complete_rollback(tmp_path, monkeypatch, record_property, caplog, invalid):
    from app.db import session as db
    missing = (BINDINGS[0],) if invalid[0] == BINDINGS[1] else ()
    with legacy_database(monkeypatch, tmp_path, missing=missing, unique=False, invalid=invalid) as engine:
        before = database_snapshot(engine)
        ok = db.init_db(); after = database_snapshot(engine)
        reason = ('UNIQUE constraint failed' if invalid[1] == 'duplicate' else
                  'has an unsupported shape' if invalid[1] in ('wrong_type', 'not_null') else 'foreign key is unsupported')
        emit(record_property, 'invalid_shape', {'input': invalid, 'schema': binding_schema(engine), 'init_ok': ok,
            'db_ready': db.DB_READY, 'complete_schema_and_rows_unchanged': before == after, 'before': before, 'after': after,
            'safe_reason': [str(r.exc_info[1]) for r in caplog.records if r.exc_info]})
        assert not ok and not db.DB_READY and before == after
        assert 'database_initialization_failed' in caplog.text
        assert reason in caplog.text


@pytest.mark.parametrize('table', BINDINGS)
@pytest.mark.parametrize('kind', ['nonunique', 'unique-other-column'])
@pytest.mark.parametrize('alternative', [False, True], ids=['no-valid-source-unique', 'genuine-alternative-source-unique'])
def test_same_name_wrong_index_requires_actual_source_uniqueness(tmp_path, monkeypatch, record_property, table, kind, alternative):
    from app.db import session as db
    with legacy_database(monkeypatch, tmp_path, unique=False, wrong_index=(table, kind), alternative=alternative) as engine:
        before = database_snapshot(engine); old_indexes = binding_schema(engine)[table]['indexes']
        ok = db.init_db(); ready = db.DB_READY; after = database_snapshot(engine)
        new_indexes = binding_schema(engine)[table]['indexes']
        duplicate_accepted = False
        with engine.connect() as c:
            transaction = c.begin()
            cols = [r['name'] for r in c.exec_driver_sql(f'PRAGMA table_info({table})').mappings()]
            selections = [('900' if col == 'id' else '10' if col == 'source_record_id' else col) for col in cols]
            original_id = 20 if table == BINDINGS[0] else 30
            duplicate_sql = f'INSERT INTO {table}({",".join(cols)}) SELECT {",".join(selections)} FROM {table} WHERE id={original_id}'
            try:
                c.exec_driver_sql(duplicate_sql)
                duplicate_accepted = True
            except IntegrityError: pass
            finally: transaction.rollback()
        emit(record_property, 'index_shape', {'table': table, 'wrong_index_kind': kind, 'alternative_correct_unique': alternative,
            'before_indexes': old_indexes, 'after_indexes': new_indexes, 'init_ok': ok, 'db_ready': ready,
            'duplicate_source_accepted': duplicate_accepted, 'duplicate_insert_sql': duplicate_sql,
            'complete_database_unchanged': before == after})
        assert new_indexes == old_indexes, 'Original wrong index must not be rewritten/dropped'
        for t in TABLES: assert_legacy_rows_preserved(before['rows'][t], after['rows'][t])
        assert database_snapshot(engine) == after, 'Probe transaction must leave no synthetic mutation'
        if alternative:
            assert ok and ready and not duplicate_accepted
            assert db.init_db() and db.DB_READY
            assert_binding_schema(engine)
        else:
            assert not ok and not ready and before == after, {'init_ok': ok, 'db_ready': ready, 'duplicate_source_accepted': duplicate_accepted}
