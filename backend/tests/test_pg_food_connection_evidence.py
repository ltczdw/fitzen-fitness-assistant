"""Limited PG control for the R015 verdict (glm-index-verdict-r015-20261004 items 5-6).

Proves that a food PUT opens genuinely new physical connections inside the
request deny-network scope against the isolated authorized test database,
performs the real catalog food match (not a post-exception fallback), and that
direct localhost getaddrinfo stays prohibited. No API or database operation is
mocked; the pinned numeric hostaddr comes from the isolation helper.
"""
import socket

import pytest
from sqlalchemy import event, text

from quality_isolation import isolated_database

USER = {'user_id': 'consistency-user'}
MEAL = {'user_id': 'consistency-user', 'meal_type': '午餐',
        'foods': [{'food_name': '米饭', 'weight_g': 100}, {'food_name': '鸡蛋', 'weight_g': 50}]}


def test_food_put_real_new_physical_connections_under_deny_network(monkeypatch, tmp_path, record_property):
    from test_manual_data_consistency import API
    with isolated_database(monkeypatch, tmp_path, "postgres") as engine:
        api = API(engine)
        created = api.post('/api/nutrition/food', json=MEAL, headers={'Idempotency-Key': 'r015-evidence-create'})
        assert created.status_code == 200, created.text
        group = created.json()['meal_group_id']
        physical_ids = []
        event.listens_for(engine, 'connect')(lambda dbapi, record: physical_ids.append(id(dbapi)))

        # Phase A: cold pool - dispose only when no connection is checked out, so
        # both concurrent sessions of the PUT must open fresh physical connections.
        assert engine.pool.checkedout() == 0
        before_cold = len(physical_ids)
        engine.dispose()
        updated = api.put(f'/api/nutrition/food/{group}', params=USER,
                          json={'items': [{'food_name': '米饭', 'weight_g': 150}]})
        assert updated.status_code == 200, updated.text
        cold_new = len(physical_ids) - before_cold
        assert cold_new >= 2, {'cold_new_physical_connections': cold_new}

        # Phase B: keep every pooled connection checked out to force the PUT onto
        # overflow connections; the nested second session must still be a new one.
        held = [engine.connect() for _ in range(engine.pool.size())]
        try:
            before_held = len(physical_ids)
            renamed = api.put(f'/api/nutrition/food/{group}', params=USER,
                              json={'items': [{'food_name': '米饭', 'weight_g': 180}]})
            assert renamed.status_code == 200, renamed.text
            overflow_new = len(physical_ids) - before_held
            assert overflow_new >= 2, {'overflow_new_physical_connections': overflow_new}
        finally:
            for connection in held:
                connection.close()

        # Real database outcome: the match came from the seeded catalog via the
        # deterministic alias path, with the actual stored row values.
        with engine.connect() as connection:
            rows = connection.execute(text(
                'SELECT l.raw_food_name, l.food_name, l.weight_g, l.match_method, l.food_id, '
                'f.food_name AS catalog_name FROM user_food_log l '
                'LEFT JOIN food_nutrition f ON f.id = l.food_id '
                'WHERE l.meal_group_id = :g AND l.user_id = :u ORDER BY l.id'),
                {'g': group, 'u': 'consistency-user'}).mappings().all()
        assert len(rows) == 1, rows
        row = rows[0]
        assert row['raw_food_name'] == '米饭'
        assert row['food_name'] == '熟大米' and row['catalog_name'] == '熟大米'
        assert row['weight_g'] == 180.0
        assert row['food_id'] is not None and row['match_method'] in ('exact', 'alias_exact'), row
        record_property('dialect', engine.dialect.name)
        record_property('database', engine.url.database)
        record_property('r015_connection_evidence', str({
            'cold_new_physical_connections': cold_new,
            'overflow_new_physical_connections': overflow_new,
            'total_physical_connections': len(physical_ids),
            'row': dict(row),
        }))


def test_localhost_getaddrinfo_remains_prohibited():
    from evals.v2.safety import deny_network
    with deny_network() as audit:
        with pytest.raises(RuntimeError, match='evaluation_dns_prohibited'):
            socket.getaddrinfo('localhost', 5432, type=socket.SOCK_STREAM)
    assert audit['blocked_dns'] == 1 and audit['blocked_connections'] == 0
