"""Real API consistency contracts, actual SQLite/PostgreSQL parameterization."""
import asyncio
import pytest
from sqlalchemy import event, text
from quality_isolation import isolated_database
from evals.v2.safety import deny_network

class API:
    def __init__(self, engine):
        import app.main as main
        self.engine = engine
        self.app = main.app
    def request(self, method, path, **kwargs):
        async def call():
            from httpx import AsyncClient, ASGITransport
            with deny_network() as audit:
                async with AsyncClient(transport=ASGITransport(app=self.app, raise_app_exceptions=False), base_url="http://api.test") as client:
                    response = await client.request(method, path, **kwargs)
                assert audit["blocked_connections"] == audit["blocked_dns"] == 0
            return response
        return asyncio.run(call())
    def get(self, path, **kwargs): return self.request("GET", path, **kwargs)
    def post(self, path, **kwargs): return self.request("POST", path, **kwargs)
    def put(self, path, **kwargs): return self.request("PUT", path, **kwargs)
    def patch(self, path, **kwargs): return self.request("PATCH", path, **kwargs)
    def delete(self, path, **kwargs): return self.request("DELETE", path, **kwargs)

@pytest.fixture(params=["sqlite", "postgres"])
def api(request, monkeypatch, tmp_path, record_property):
    with isolated_database(monkeypatch, tmp_path, request.param) as engine:
        record_property("dialect", engine.dialect.name)
        record_property("database", engine.url.database)
        yield API(engine)

@pytest.fixture()
def api_pg(monkeypatch, tmp_path, record_property):
    with isolated_database(monkeypatch, tmp_path, "postgres") as engine:
        record_property("dialect", engine.dialect.name)
        record_property("database", engine.url.database)
        yield API(engine)

USER = {'user_id': 'consistency-user'}


# ---------------------------------------------------------------------------
# W：训练记录与镜像（session/exercise/sets）一致性
# ---------------------------------------------------------------------------

def test_w1_create_with_set_details_mirrors_into_sessions(api):
    response = api.post('/api/workouts', json={**USER, 'exercise': '深蹲', 'sets': 2,
                                               'reps': 8, 'weight_kg': 80.0,
                                               'set_details': [
                                                   {'weight_kg': 80.0, 'reps': 8},
                                                   {'weight_kg': 82.5, 'reps': 6}]})
    assert response.status_code == 200, response.text
    record = response.json()
    assert record['session_id']
    sessions = api.get('/api/workout-sessions', params={**USER}).json()
    mirrored = next(s for s in sessions if s['id'] == record['session_id'])
    assert mirrored['source_record_id'] == record['id']
    assert mirrored['exercises'][0]['exercise_name'] == '深蹲'
    sets = mirrored['exercises'][0]['sets']
    assert [(s['weight_kg'], s['reps']) for s in sets] == [(80.0, 8), (82.5, 6)]


def test_w2_update_record_syncs_mirror(api):
    created = api.post('/api/workouts', json={**USER, 'exercise': '卧推', 'sets': 2,
                                              'reps': 8, 'weight_kg': 60.0}).json()
    updated = api.put(f"/api/workouts/{created['id']}", params={**USER}, json={
        'exercise': '卧推', 'sets': 3, 'reps': 10, 'weight_kg': 70.0})
    assert updated.status_code == 200, updated.text
    sessions = api.get('/api/workout-sessions', params={**USER}).json()
    mirrored = next(s for s in sessions if s['id'] == created['session_id'])
    sets = mirrored['exercises'][0]['sets']
    # 产品合同：镜像用于向前端暴露该记录；记录更新后镜像不得保留旧值。
    assert len(sets) == 3, f'镜像组数未随记录更新: {sets}'
    assert all(s['weight_kg'] == 70.0 and s['reps'] == 10 for s in sets), sets


def test_w3_delete_record_removes_mirror_session(api):
    created = api.post('/api/workouts', json={**USER, 'exercise': '硬拉', 'sets': 1,
                                              'reps': 5, 'weight_kg': 100.0}).json()
    deleted = api.delete(f"/api/workouts/{created['id']}", params={**USER})
    assert deleted.status_code == 200
    sessions = api.get('/api/workout-sessions', params={**USER}).json()
    orphans = [s for s in sessions if s['id'] == created['session_id']]
    # 产品合同：记录删除后镜像会话不得残留（否则前端训练视图仍显示已删记录）。
    assert orphans == [], f'镜像会话成为孤儿: {orphans}'


def test_w4_patch_set_updates_legacy_record_details(api):
    created = api.post('/api/workouts', json={**USER, 'exercise': '划船', 'sets': 2,
                                              'reps': 10, 'weight_kg': 50.0,
                                              'set_details': [
                                                  {'weight_kg': 50.0, 'reps': 10},
                                                  {'weight_kg': 50.0, 'reps': 10}]}).json()
    sessions = api.get('/api/workout-sessions', params={**USER}).json()
    mirrored = next(s for s in sessions if s['id'] == created['session_id'])
    set_id = mirrored['exercises'][0]['sets'][0]['id']
    patched = api.patch(f'/api/workout-session-sets/{set_id}', params={**USER},
                        json={'weight_kg': 55.5, 'reps': 9})
    assert patched.status_code == 200, patched.text
    record = api.get('/api/workouts', params={**USER}).json()[0]
    details = record.get('set_details') or []
    # 产品合同：两个视图展示同一次训练；编辑组数据后记录明细不得保留旧值。
    assert details and (details[0]['weight_kg'], details[0]['reps']) == (55.5, 9), \
        f'PATCH 组数据未同步回记录明细: {details}'


def test_w5_workout_input_bounds(api):
    negative = api.post('/api/workouts', json={**USER, 'exercise': '深蹲', 'sets': 1,
                                               'reps': 8, 'weight_kg': -5})
    assert negative.status_code == 422
    zero_reps = api.post('/api/workouts', json={**USER, 'exercise': '深蹲', 'sets': 1,
                                                'reps': 0, 'weight_kg': 50})
    assert zero_reps.status_code == 422
    zero_weight = api.post('/api/workouts', json={**USER, 'exercise': '自重俯卧撑', 'sets': 1,
                                                  'reps': 20, 'weight_kg': 0})
    assert zero_weight.status_code == 200
    mismatch = api.post('/api/workouts', json={**USER, 'exercise': '深蹲', 'sets': 2,
                                               'reps': 8, 'weight_kg': 50,
                                               'set_details': [{'weight_kg': 50, 'reps': 8}]})
    assert mismatch.status_code == 422
    decimal = api.post('/api/workouts', json={**USER, 'exercise': '深蹲', 'sets': 1,
                                              'reps': 5, 'weight_kg': 62.5}).json()
    assert decimal['weight_kg'] == 62.5


def test_w5b_start_end_date_order_enforced(api):
    bad = api.get('/api/workouts', params={'start_date': '2026-02-01', 'end_date': '2026-01-01'})
    assert bad.status_code == 422


# ---------------------------------------------------------------------------
# F1-F6：饮食餐次与统一列表/汇总一致性（含 2026-09-15 BUG-002 回归）
# ---------------------------------------------------------------------------

def _meal(key, foods=None, meal_type='午餐'):
    return json_headers(key), {'user_id': 'consistency-user', 'meal_type': meal_type,
                               'foods': foods or [{'food_name': '米饭', 'weight_g': 100},
                                                  {'food_name': '鸡蛋', 'weight_g': 50}]}


def json_headers(key):
    return {'Idempotency-Key': key}


def test_f1_structured_meal_visible_in_unified_list(api_pg):
    headers, payload = _meal('consistency-f1')
    created = api_pg.post('/api/nutrition/food', json=payload, headers=headers)
    assert created.status_code == 200, created.text
    body = created.json()
    assert body['status'] == 'calculated'
    listing = api_pg.get('/api/nutrition', params={**USER}).json()
    meals = [m for m in listing if m.get('id') == body.get('meal_group_id')]
    # 2026-09-15 BUG-002 回归：结构化餐次必须出现在统一列表且带明细。
    assert meals, f'结构化餐次未出现在统一列表: {listing}'
    items = meals[0].get('items') or []
    # raw_food_name 保留用户原话；food_name 允许食物库规范化（米饭→熟大米）。
    assert len(items) == 2
    assert {i['raw_food_name'] for i in items} == {'米饭', '鸡蛋'}
    kcal_sum = round(sum(i['calories'] for i in items), 1)
    assert abs(meals[0]['calories'] - kcal_sum) < 0.2, (meals[0], items)


def test_f2_meal_edit_updates_listing_and_totals(api_pg):
    headers, payload = _meal('consistency-f2')
    created = api_pg.post('/api/nutrition/food', json=payload, headers=headers).json()
    group = created['meal_group_id']
    listing = api_pg.get('/api/nutrition', params={**USER}).json()
    meal = next(m for m in listing if m['id'] == group)
    item_ids = {i['raw_food_name']: i['id'] for i in meal['items']}
    updated = api_pg.put(f'/api/nutrition/food/{group}', params={'user_id': 'consistency-user'}, json={
        'items': [{'id': item_ids['米饭'], 'food_name': '米饭', 'weight_g': 150}]})
    assert updated.status_code == 200, updated.text
    listing = api_pg.get('/api/nutrition', params={**USER}).json()
    meal = next(m for m in listing if m['id'] == group)
    # 编辑删除鸡蛋、米饭增至150克：列表与合计必须反映编辑结果。
    # 观察项（P3，登记 backlog）：编辑后 raw_food_name 被规范化名（熟大米）
    # 替换，用户原话不再保留。
    assert {i['food_name'] for i in meal['items']} == {'熟大米'}
    rice = meal['items'][0]
    assert rice['weight_g'] == 150.0
    assert abs(meal['calories'] - rice['calories']) < 0.2


def test_f3_meal_delete_removes_rows(api_pg):
    headers, payload = _meal('consistency-f3')
    created = api_pg.post('/api/nutrition/food', json=payload, headers=headers).json()
    group = created['meal_group_id']
    deleted = api_pg.delete(f'/api/nutrition/food/{group}', params={**USER})
    assert deleted.status_code == 200
    listing = api_pg.get('/api/nutrition', params={**USER}).json()
    assert not [m for m in listing if m['id'] == group]
    with api_pg.engine.connect() as conn:
        rows = conn.execute(text('SELECT COUNT(*) FROM user_food_log WHERE meal_group_id = :g'),
                            {'g': group}).scalar()
    assert rows == 0, '物理行未随餐次删除'


def test_f4_summary_matches_listing_totals(api_pg):
    import datetime as dt
    today = dt.datetime.now(dt.timezone.utc).isoformat()
    headers, payload = _meal('consistency-f4a', meal_type='早餐')
    payload['record_time'] = today
    api_pg.post('/api/nutrition/food', json=payload, headers=headers)
    listing = api_pg.get('/api/nutrition', params={**USER}).json()
    assert listing, '统一列表为空'
    dates = [str(m.get('date', '')) for m in listing]
    assert dates == sorted(dates, reverse=True), f'统一列表未按日期降序: {dates}'
    # 汇总是聚合视图：entries 应等于 legacy 条数 + 结构化餐次数，热量合计一致。
    summary = api_pg.get('/api/nutrition/summary', params={'user_id': 'consistency-user', 'days': 7}).json()
    assert summary, summary
    meal_calories = sum(m['calories'] for m in listing if m.get('source') == 'ai_food_log')
    legacy_calories = sum(m['calories'] for m in listing if m.get('source') != 'ai_food_log')
    # entries 语义 = legacy 日志条数 + 结构化餐次的食物行数，两者从数据库
    # 直接计数，不对行数语义做猜测；热量合计必须与列表一致。
    with api_pg.engine.connect() as conn:
        legacy_rows = conn.execute(text(
            "SELECT COUNT(*) FROM nutrition_logs WHERE user_id = 'consistency-user'")).scalar()
        food_rows = conn.execute(text(
            "SELECT COUNT(*) FROM user_food_log WHERE user_id = 'consistency-user'")).scalar()
    assert summary['entries'] == legacy_rows + food_rows, (
        summary['entries'], legacy_rows, food_rows)
    assert abs(summary['calories'] - (meal_calories + legacy_calories)) < 0.5, (
        summary['calories'], meal_calories, legacy_calories)


def test_f5_legacy_log_still_listed_and_merged(api_pg):
    legacy = api_pg.post('/api/nutrition', json={
        **USER, 'meal_type': '晚餐', 'description': '蛋白粉一杯', 'calories': 120.0,
        'protein_g': 24.0, 'carbs_g': 3.0, 'fat_g': 1.0})
    assert legacy.status_code == 200, legacy.text
    headers, payload = _meal('consistency-f5')
    api_pg.post('/api/nutrition/food', json=payload, headers=headers)
    listing = api_pg.get('/api/nutrition', params={**USER}).json()
    sources = {m.get('description'): m for m in listing if m.get('description')}
    assert any('蛋白粉' in key for key in sources), listing
    assert any(m.get('source') == 'ai_food_log' for m in listing), listing


def test_f6_same_idempotency_key_keeps_single_meal(api_pg):
    headers, payload = _meal('consistency-f6')
    first = api_pg.post('/api/nutrition/food', json=payload, headers=headers).json()
    second = api_pg.post('/api/nutrition/food', json=payload, headers=headers).json()
    assert first['meal_group_id'] == second['meal_group_id']
    listing = api_pg.get('/api/nutrition', params={**USER}).json()
    matches = [m for m in listing if m['id'] == first['meal_group_id']]
    assert len(matches) == 1, '同一幂等键重复提交产生多份餐次'
    with api_pg.engine.connect() as conn:
        rows = conn.execute(text('SELECT COUNT(*) FROM user_food_log WHERE meal_group_id = :g'),
                            {'g': first['meal_group_id']}).scalar()
    assert rows == 2, '重复提交不应新增食物行'


def test_f7_unknown_food_is_clarified_not_saved(api_pg):
    headers, payload = _meal('consistency-f7', foods=[{'food_name': '火星尘蛋糕', 'weight_g': 100}])
    response = api_pg.post('/api/nutrition/food', json=payload, headers=headers)
    assert response.status_code == 400
    listing = api_pg.get('/api/nutrition', params={**USER}).json()
    assert not [m for m in listing if '火星' in str(m.get('description', ''))], listing
    with api_pg.engine.connect() as conn:
        rows = conn.execute(text('SELECT COUNT(*) FROM user_food_log')).scalar()
    assert rows == 0, '无法解析的餐次不得写入任何行'


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def create_mirror(api):
    response = api.post("/api/workouts", json={**USER, "exercise": "深蹲", "sets": 2, "reps": 8, "weight_kg": 80,
        "set_details": [{"weight_kg": 80, "reps": 8}, {"weight_kg": 82.5, "reps": 6}]})
    assert response.status_code == 200, response.text
    record = response.json()
    mirror = next(s for s in api.get("/api/workout-sessions", params=USER).json() if s["id"] == record["session_id"])
    return record, mirror

def source_record(api, rid):
    return next(r for r in api.get("/api/workouts", params=USER).json() if r["id"] == rid)

def test_r012_delete_mirror_session_removes_source_record(api):
    record, mirror = create_mirror(api)
    response = api.delete(f"/api/workout-sessions/{mirror['id']}", params=USER)
    assert response.status_code == 200
    assert not [r for r in api.get("/api/workouts", params=USER).json() if r["id"] == record["id"]]

@pytest.mark.parametrize("mutation", ["patch", "add", "delete", "delete_exercise", "update_session"])
def test_r012_reverse_mutation_updates_source(api, mutation):
    record, mirror = create_mirror(api)
    exercise = mirror["exercises"][0]
    set_id = exercise["sets"][0]["id"]
    if mutation == "patch":
        response = api.patch(f"/api/workout-session-sets/{set_id}", params=USER, json={"weight_kg": 90})
        expected = [(90, 8), (82.5, 6)]
    elif mutation == "add":
        response = api.post(f"/api/workout-sessions/{mirror['id']}/exercises/{exercise['id']}/sets", params=USER, json={"weight_kg": 85, "reps": 5})
        expected = [(80, 8), (82.5, 6), (85, 5)]
    elif mutation == "delete":
        response = api.delete(f"/api/workout-session-sets/{set_id}", params=USER)
        expected = [(82.5, 6)]
    elif mutation == "delete_exercise":
        response = api.delete(f"/api/workout-sessions/{mirror['id']}/exercises/{exercise['id']}", params=USER)
        expected = None
    else:
        response = api.put(f"/api/workout-sessions/{mirror['id']}", params=USER, json={**USER, "session_date": "2026-10-03", "title": "深蹲", "exercises": [{"exercise_name": "深蹲", "sets": [{"weight_kg": 90, "reps": 8}, {"weight_kg": 82.5, "reps": 6}]}]})
        expected = [(90, 8), (82.5, 6)]
    assert response.status_code == 200, response.text
    if expected is None:
        assert not [r for r in api.get("/api/workouts", params=USER).json() if r["id"] == record["id"]]
    else:
        after = source_record(api, record["id"])
        assert [(s["weight_kg"], s["reps"]) for s in after["set_details"]] == expected, after

def test_r012_independent_session_and_user_isolation(api):
    record, mirror = create_mirror(api)
    independent = api.post("/api/workout-sessions", json={**USER, "title": "独立训练", "exercises": [{"exercise_name": "划船", "sets": [{"weight_kg": 50, "reps": 10}]}]}).json()
    assert independent["source_record_id"] is None
    assert api.delete(f"/api/workout-sessions/{independent['id']}", params=USER).status_code == 200
    sid = mirror["exercises"][0]["sets"][0]["id"]
    assert api.patch(f"/api/workout-session-sets/{sid}", params={"user_id": "another-user"}, json={"weight_kg": 1}).status_code == 404
    assert source_record(api, record["id"])["weight_kg"] == 80

def test_r012_update_keeps_ids_and_updates_values(api):
    record, mirror = create_mirror(api)
    def ids(m): return (m["id"], m["exercises"][0]["id"], tuple(s["id"] for s in m["exercises"][0]["sets"]))
    response = api.put(f"/api/workouts/{record['id']}", params=USER, json={"exercise": "深蹲", "sets": 2, "reps": 12, "weight_kg": 85,
        "set_details": [{"weight_kg": 85, "reps": 12}, {"weight_kg": 87.5, "reps": 8}]})
    assert response.status_code == 200
    after = next(s for s in api.get("/api/workout-sessions", params=USER).json() if s["id"] == mirror["id"])
    assert ids(after) == ids(mirror)
    assert [(s["weight_kg"], s["reps"]) for s in after["exercises"][0]["sets"]] == [(85, 12), (87.5, 8)]

@pytest.mark.parametrize("direction", ["record", "set"])
def test_r012_transaction_failure_rolls_back_both_sides(api, direction):
    # Inject at actual SQL flush, not at a guessed future repository function.
    record, mirror = create_mirror(api)
    statements = []
    def reject_update(conn, cursor, stmt, params, context, many):
        if stmt.lstrip().upper().startswith("UPDATE") and any(t in stmt.lower() for t in ("workout_records", "workout_session_sets")):
            statements.append(stmt.split()[1])
            raise RuntimeError("quality-forced-write-failure")
    event.listen(api.engine, "before_cursor_execute", reject_update)
    try:
        if direction == "record":
            # Round 2: top-only edits to heterogeneous facts are 409. Use legal
            # explicit details to reach SQL, retaining the real fault/rollback checks.
            response = api.put(f"/api/workouts/{record['id']}", params=USER, json={
                "exercise": "深蹲", "sets": 2, "reps": 20, "weight_kg": 99,
                "set_details": [{"weight_kg": 99, "reps": 20},
                                {"weight_kg": 82.5, "reps": 6}]})
        else:
            sid = mirror["exercises"][0]["sets"][0]["id"]
            response = api.patch(f"/api/workout-session-sets/{sid}", params=USER, json={"weight_kg": 99})
    finally:
        event.remove(api.engine, "before_cursor_execute", reject_update)
    assert statements, "failure injection did not intercept an actual UPDATE"
    assert response.status_code >= 500, response.text
    assert source_record(api, record["id"]) == {k: v for k, v in record.items() if k != "session_id"}
    after = next(s for s in api.get("/api/workout-sessions", params=USER).json() if s["id"] == mirror["id"])
    assert after == mirror

@pytest.mark.parametrize("direction", ["record", "set"])
def test_r012_second_representation_failure_rolls_back_first_write(api, direction):
    record, mirror = create_mirror(api)
    writes, rejected = [], []
    # Reject the other representation only after an actual first-side mutation.
    first = "workout_records" if direction == "record" else "workout_session_sets"
    second = "workout_session_sets" if direction == "record" else "workout_records"
    def fail_second(conn, cursor, stmt, params, context, many):
        upper = stmt.lstrip().upper()
        if not upper.startswith(("UPDATE ", "INSERT ", "DELETE ")):
            return
        if first in stmt.lower():
            writes.append(first)
        if second in stmt.lower():
            rejected.append(second)
            raise RuntimeError("quality-forced-second-representation-failure")
    event.listen(api.engine, "before_cursor_execute", fail_second)
    try:
        if direction == "record":
            response = api.put(f"/api/workouts/{record['id']}", params=USER, json={
                "exercise": "深蹲", "sets": 2, "reps": 8, "weight_kg": 99,
                "set_details": [{"weight_kg": 99, "reps": 8}, {"weight_kg": 82.5, "reps": 6}]})
        else:
            sid = mirror["exercises"][0]["sets"][0]["id"]
            response = api.patch(f"/api/workout-session-sets/{sid}", params=USER, json={"weight_kg": 99})
    finally:
        event.remove(api.engine, "before_cursor_execute", fail_second)
    assert writes, "first-side mutation was not reached"
    assert rejected, "associated second representation was never mutated"
    assert response.status_code >= 500, response.text
    assert source_record(api, record["id"]) == {k: v for k, v in record.items() if k != "session_id"}
    after = next(s for s in api.get("/api/workout-sessions", params=USER).json() if s["id"] == mirror["id"])
    assert after == mirror


@pytest.mark.parametrize("mode", ["weight_only", "same_name", "rename", "add_local", "add_custom"])
def test_r013_raw_name_contract(api_pg, mode):
    headers, payload = _meal("raw-contract", foods=[{"food_name": "米饭", "weight_g": 100}])
    created = api_pg.post("/api/nutrition/food", json=payload, headers=headers)
    assert created.status_code == 200, created.text
    group = created.json()["meal_group_id"]
    meal = next(m for m in api_pg.get("/api/nutrition", params=USER).json() if m["id"] == group)
    item_id = meal["items"][0]["id"]
    items = [{"id": item_id, "weight_g": 150}]
    if mode == "same_name": items[0]["food_name"] = "米饭"
    if mode == "rename": items[0]["food_name"] = "白米饭"
    if mode == "add_local": items.append({"food_name": "鸡蛋", "weight_g": 50})
    if mode == "add_custom": items.append({"food_name": "自制蛋糕", "weight_g": 80, "source": "custom", "nutrition_per_100g": {"kcal": 200, "protein_g": 5, "carbs_g": 30, "fat_g": 6, "fiber_g": 1}})
    response = api_pg.put(f"/api/nutrition/food/{group}", params=USER, json={"items": items})
    assert response.status_code == 200, response.text
    after = next(m for m in api_pg.get("/api/nutrition", params=USER).json() if m["id"] == group)
    rice = next(i for i in after["items"] if i["id"] == item_id)
    assert rice["raw_food_name"] == ("白米饭" if mode == "rename" else "米饭"), rice
    assert rice["food_name"] == "熟大米"
    if mode.startswith("add_"):
        raw = "鸡蛋" if mode == "add_local" else "自制蛋糕"
        assert any(i["raw_food_name"] == raw for i in after["items"]), after
