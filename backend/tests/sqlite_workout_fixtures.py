"""Direct historical SQLite schemas, synthetic rows and complete rollback snapshots.

No current table is renamed/dropped to manufacture history. Engines are always
created by the production factory, inside pytest's exact temporary directory.
"""
from contextlib import contextmanager
import json
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

TABLES = ('workout_records', 'workout_sessions', 'workout_session_exercises', 'workout_session_sets')
BINDINGS = TABLES[1:3]
INDEX = {'workout_sessions': 'uq_workout_sessions_source_record',
         'workout_session_exercises': 'uq_workout_session_exercises_source_record'}
USER = {'user_id': 'sqlite-legacy-user'}


def emit(record_property, name, value):
    record_property(name, json.dumps(value, ensure_ascii=False, default=str))


@contextmanager
def legacy_database(monkeypatch, tmp_path, *, missing=(), unique=True, invalid=None, wrong_index=None, alternative=False, fresh=False):
    from app.core.config import settings
    from app.db import session as db
    path = (tmp_path / 'direct-legacy-workout.db').resolve()
    assert path.is_relative_to(tmp_path.resolve()) and not path.exists()
    engine = db.create_database_engine('sqlite:///' + str(path))
    with monkeypatch.context() as m:
        m.setattr(db, 'engine', engine)
        m.setattr(db, 'SessionLocal', sessionmaker(bind=engine, expire_on_commit=False))
        m.setattr(db, 'DB_READY', False)
        m.setattr(settings, 'database_url', 'sqlite:///' + str(path))
        m.setattr(settings, 'memory_read_backend', 'compact_v1')
        try:
            if fresh:
                assert db.init_db() and db.DB_READY
            build_history(engine, missing, unique, invalid, wrong_index, alternative, seed_only=fresh)
            yield engine
        finally:
            engine.dispose()


def build_history(engine, missing, unique, invalid, wrong_index, alternative, seed_only=False):
    def source(table):
        if table in missing:
            return ''
        kind = invalid[1] if invalid and invalid[0] == table else ''
        target = 'workout_sessions' if kind == 'wrong_target' else 'workout_records'
        action = 'CASCADE' if kind == 'wrong_action' else 'SET NULL'
        coltype = 'TEXT' if kind == 'wrong_type' else 'INTEGER'
        nullable = ' NOT NULL' if kind == 'not_null' else ''
        fk = '' if kind == 'missing_fk' else f' REFERENCES {target}(id) ON DELETE {action}'
        return f'source_record_id {coltype}{nullable}{fk},'
    ddl = {
        'workout_records': '''id INTEGER PRIMARY KEY, user_id VARCHAR(100) NOT NULL,
            workout_date DATE NOT NULL, exercise VARCHAR(200) NOT NULL, sets INTEGER NOT NULL,
            reps INTEGER NOT NULL, weight_kg FLOAT NOT NULL, difficulty VARCHAR(20) NOT NULL,
            completed BOOLEAN NOT NULL, note TEXT NOT NULL, set_details TEXT,
            idempotency_key VARCHAR(150), created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP''',
        'workout_sessions': f'''id INTEGER PRIMARY KEY, user_id VARCHAR(100) NOT NULL,
            {source('workout_sessions')} session_date DATE NOT NULL, title VARCHAR(200) NOT NULL,
            note TEXT NOT NULL, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP''',
        'workout_session_exercises': f'''id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL REFERENCES workout_sessions(id) ON DELETE CASCADE,
            exercise_name VARCHAR(200) NOT NULL, difficulty VARCHAR(20) NOT NULL,
            order_index INTEGER NOT NULL, note TEXT NOT NULL, {source('workout_session_exercises').rstrip(',')}'''.rstrip(', '),
        'workout_session_sets': '''id INTEGER PRIMARY KEY,
            exercise_id INTEGER NOT NULL REFERENCES workout_session_exercises(id) ON DELETE CASCADE,
            set_index INTEGER NOT NULL, weight_kg FLOAT NOT NULL, reps INTEGER NOT NULL,
            difficulty VARCHAR(20) NOT NULL, completed BOOLEAN NOT NULL, note TEXT NOT NULL''',
    }
    # Seed FK targets before dependants; these are synthesized data, not records
    # copied from the application database.
    with engine.begin() as c:
        if not seed_only:
            for table in TABLES:
                c.exec_driver_sql(f'CREATE TABLE {table} ({ddl[table]})')
        details = json.dumps([{'weight_kg': 80, 'reps': 8}, {'weight_kg': 82.5, 'reps': 6}])
        for rid, user, name in ((10, USER['user_id'], '深蹲'), (11, 'sqlite-unrelated-user', '卧推')):
            c.execute(text('''INSERT INTO workout_records VALUES
                (:id,:user,'2026-10-03',:name,2,8,80,'未记录',1,'legacy-original',:details,NULL,'2026-10-03 01:02:03')'''),
                {'id': rid, 'user': user, 'name': name, 'details': details})
        for sid, user, rid in ((20, USER['user_id'], 10), (21, 'sqlite-unrelated-user', 11), (22, USER['user_id'], None)):
            cols, vals = 'id,user_id,session_date,title,note,created_at,updated_at', ':id,:user,\'2026-10-03\',\'legacy-title\',\'legacy-note\',\'2026-10-03 01:02:03\',\'2026-10-03 01:02:03\''
            params = {'id': sid, 'user': user}
            if 'workout_sessions' not in missing:
                cols += ',source_record_id'; vals += ',:source'
                params['source'] = rid
                if invalid and invalid == ('workout_sessions', 'wrong_target'):
                    params['source'] = sid if rid is not None else None
                if invalid and invalid == ('workout_sessions', 'not_null') and rid is None: params['source'] = 11
            c.execute(text(f'INSERT INTO workout_sessions({cols}) VALUES({vals})'), params)
        for eid, sid, name, order, rid in ((30, 20, '深蹲', 0, 10), (31, 20, '划船', 1, None), (32, 21, '卧推', 0, 11), (33, 22, '硬拉', 0, None)):
            cols, vals = 'id,session_id,exercise_name,difficulty,order_index,note', ':id,:session,:name,\'未记录\',:order,\'original-action\''
            params = {'id': eid, 'session': sid, 'name': name, 'order': order}
            if 'workout_session_exercises' not in missing:
                cols += ',source_record_id'; vals += ',:source'; params['source'] = rid
                if invalid and invalid == ('workout_session_exercises', 'wrong_target'):
                    params['source'] = {10: 20, 11: 21}.get(rid)
                if invalid and invalid == ('workout_session_exercises', 'not_null') and rid is None: params['source'] = 10
            c.execute(text(f'INSERT INTO workout_session_exercises({cols}) VALUES({vals})'), params)
        for gid, eid, order, weight, reps in ((40, 30, 0, 80, 8), (41, 30, 1, 82.5, 6), (42, 31, 0, 30, 10), (43, 32, 0, 80, 8), (44, 32, 1, 82.5, 6), (45, 33, 0, 100, 5), (46, 33, 1, 102.5, 4)):
            c.execute(text("INSERT INTO workout_session_sets(id,exercise_id,set_index,weight_kg,reps,difficulty,completed,note) VALUES (:id,:ex,:slot,:weight,:reps,'未记录',1,'original-set')"),
                      {'id': gid, 'ex': eid, 'slot': order, 'weight': weight, 'reps': reps})
        if invalid and invalid[1] == 'duplicate':
            c.exec_driver_sql(f'UPDATE {invalid[0]} SET source_record_id=10 WHERE id=' + ('22' if invalid[0] == BINDINGS[0] else '31'))
        for table in (() if seed_only else BINDINGS):
            if table in missing: continue
            if unique and not (invalid and invalid[0] == table):
                c.exec_driver_sql(f'CREATE UNIQUE INDEX {INDEX[table]} ON {table}(source_record_id)')
        if wrong_index:
            table, kind = wrong_index
            unique_sql, column = ('', 'source_record_id') if kind == 'nonunique' else ('UNIQUE ', 'id')
            c.exec_driver_sql(f'CREATE {unique_sql}INDEX {INDEX[table]} ON {table}({column})')
            if alternative:
                c.exec_driver_sql(f'CREATE UNIQUE INDEX genuine_source_unique ON {table}(source_record_id)')


def database_snapshot(engine):
    with engine.connect() as c:
        schema = [tuple(r) for r in c.exec_driver_sql("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name")]
        tables = [r[0] for r in c.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        rows = {t: [dict(r) for r in c.exec_driver_sql(f'SELECT * FROM "{t}" ORDER BY rowid').mappings()] for t in tables}
        return {'schema': schema, 'rows': rows}


def binding_schema(engine):
    with engine.connect() as c:
        result = {}
        for t in BINDINGS:
            indexes = []
            for row in c.exec_driver_sql(f'PRAGMA index_list({t})').mappings():
                entry = dict(row)
                entry['columns'] = [r['name'] for r in c.exec_driver_sql(f'PRAGMA index_info("{row["name"]}")').mappings()]
                indexes.append(entry)
            result[t] = {'columns': [dict(r) for r in c.exec_driver_sql(f'PRAGMA table_info({t})').mappings()],
                         'foreign_keys': [dict(r) for r in c.exec_driver_sql(f'PRAGMA foreign_key_list({t})').mappings()], 'indexes': indexes}
        return result


def assert_binding_schema(engine):
    info = binding_schema(engine)
    for t, value in info.items():
        col = next(r for r in value['columns'] if r['name'] == 'source_record_id')
        assert col['type'].upper() == 'INTEGER' and col['notnull'] == 0
        fk = [r for r in value['foreign_keys'] if r['from'] == 'source_record_id']
        assert len(fk) == 1 and (fk[0]['table'], fk[0]['to'], fk[0]['on_delete']) == ('workout_records', 'id', 'SET NULL')
        assert any(i['unique'] and i['columns'] == ['source_record_id'] and not i['partial'] for i in value['indexes']), info
    return info


def assert_upgrade(engine, before, missing):
    from app.db import session as db
    assert db.init_db() and db.DB_READY
    first = database_snapshot(engine)
    assert db.init_db() and db.DB_READY
    assert database_snapshot(engine) == first
    assert_binding_schema(engine)
    for t in TABLES:
        old, new = before['rows'][t], first['rows'][t]
        assert [{k: r[k] for k in old[0]} for r in new] == old
        if t in missing: assert all(r['source_record_id'] is None for r in new)
    return first


def assert_legacy_rows_preserved(before, after):
    """All old fields must match; only the two new NULL metric columns may appear."""
    assert len(before) == len(after)
    for old, new in zip(before, after):
        assert {key: new[key] for key in old} == old
        added = set(new) - set(old)
        assert added <= {'distance_km', 'duration_minutes'}
        assert all(new[key] is None for key in added)
