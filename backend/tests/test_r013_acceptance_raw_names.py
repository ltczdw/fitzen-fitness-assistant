"""Real HTTP and SQL mutation tests. SQLite uses a minimal synthetic exact/alias
catalog; PostgreSQL retains its real isolated startup schema. No vector claim."""
import json
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

import pytest
from sqlalchemy import event, text
from test_manual_data_consistency import api, USER


@pytest.fixture
def food_api(api, record_property):
    if api.engine.dialect.name == "sqlite":
        # Only SQL schema adaptation: real production resolve_food/mutation code
        # runs exact and alias queries, with synthetic rows, without matcher mocks.
        with api.engine.begin() as c:
            c.execute(text("""CREATE TABLE food_nutrition (
                id INTEGER PRIMARY KEY, food_name TEXT, category TEXT,
                preparation_state TEXT, energy_kcal NUMERIC, protein_g NUMERIC,
                carb_g NUMERIC, fat_g NUMERIC, fiber_g NUMERIC,
                source_version TEXT DEFAULT 'synthetic-v1', is_active BOOLEAN DEFAULT TRUE)"""))
            c.execute(text("""CREATE TABLE food_alias_index (
                id INTEGER PRIMARY KEY, food_id INTEGER, normalized_alias TEXT,
                confidence FLOAT DEFAULT 1, is_active BOOLEAN DEFAULT TRUE, embedding TEXT)"""))
            c.execute(text("""CREATE TABLE user_custom_food (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, food_name TEXT,
                energy_kcal NUMERIC, protein_g NUMERIC, carb_g NUMERIC, fat_g NUMERIC,
                fiber_g NUMERIC DEFAULT 0, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, last_used_at TIMESTAMP,
                use_count INTEGER DEFAULT 0, is_active BOOLEAN DEFAULT TRUE,
                UNIQUE(user_id,food_name))"""))
            c.execute(text("""INSERT INTO food_nutrition
                (id,food_name,category,preparation_state,energy_kcal,protein_g,carb_g,fat_g,fiber_g)
                VALUES (1,'熟大米','谷薯','熟',116,2.6,25.9,0.3,0.3),
                       (6,'鸡蛋','蛋类','熟',144,12.6,1.1,9.5,0)"""))
            for i, name in enumerate(("米饭", "白米饭", "白饭"), start=1):
                c.execute(text("INSERT INTO food_alias_index (id,food_id,normalized_alias) VALUES (:i,1,:n)"), {"i":i,"n":name})
        record_property("food_fixture", "SQLite synthetic exact/alias catalog; no pgvector coverage")
    yield api


def create_meal(api, user=USER["user_id"]):
    response = api.post("/api/nutrition/food", headers={"Idempotency-Key":uuid4().hex},
        json={"user_id":user,"meal_type":"午餐","record_time":"2026-10-03T03:00:00Z",
              "foods":[{"food_name":"米饭","weight_g":100},{"food_name":"鸡蛋","weight_g":50}]})
    assert response.status_code == 200, response.text
    return response.json()["meal_group_id"]


def rows(api, group):
    with api.engine.connect() as c:
        return [dict(r) for r in c.execute(text("SELECT * FROM user_food_log WHERE meal_group_id=:g ORDER BY id"), {"g":group}).mappings()]


def update(api, group, **payload):
    return api.put(f"/api/nutrition/food/{group}", params=USER, json=payload)


@pytest.mark.parametrize("mode", ["weight","meal","date","same_name","rename"])
def test_r013_raw_and_canonical_names_real_sql(food_api, mode, record_property):
    api = food_api
    group = create_meal(api)
    before = rows(api, group)
    rice, egg = before
    body = {}
    if mode == "meal": body["meal_type"] = "晚餐"
    elif mode == "date": body["record_time"] = "2026-10-04T03:00:00Z"
    else:
        body["items"] = [{"id":rice["id"],"weight_g":150}, {"id":egg["id"],"weight_g":50}]
        if mode in ("same_name","rename"): body["items"][0]["food_name"] = "米饭" if mode == "same_name" else "白米饭"
    response = update(api, group, **body)
    assert response.status_code == 200, response.text
    after = rows(api, group)
    record_property("actual_names", json.dumps([(r["raw_food_name"],r["food_name"]) for r in after],ensure_ascii=False))
    assert after[0]["raw_food_name"] == ("白米饭" if mode == "rename" else "米饭")
    assert after[0]["food_name"] == "熟大米"
    assert after[1]["raw_food_name"] == "鸡蛋" and after[1]["food_name"] == "鸡蛋"
    assert [r["id"] for r in after] == [r["id"] for r in before]
    assert float(after[0]["total_kcal"]) == (116 if mode in ("meal","date") else 174)
    assert float(after[1]["total_kcal"]) == 72
    if mode == "date": assert all(str(r["record_time"]).startswith("2026-10-04") for r in after)
    if mode == "meal": assert all(r["meal_type"] == "晚餐" for r in after)


@pytest.mark.parametrize("source", ["local","custom","ai"])
def test_r013_additions_and_same_custom_name_weight_edit(food_api, source):
    api = food_api
    group = create_meal(api)
    before = rows(api, group)
    added = {"food_name":"白饭" if source == "local" else "合成自制食物","weight_g":80,"source":source}
    if source != "local":
        added["nutrition_per_100g"] = {"kcal":200,"protein_g":5,"carbs_g":30,"fat_g":6,"fiber_g":1}
    response = update(api, group, items=[*({"id":r["id"],"weight_g":float(r["weight_g"])} for r in before),added])
    assert response.status_code == 200, response.text
    after = rows(api, group)
    assert after[:2] == before
    item = after[2]
    assert item["raw_food_name"] == added["food_name"]
    assert item["food_name"] == ("熟大米" if source == "local" else added["food_name"])
    assert item["match_method"] == ("alias_exact" if source == "local" else "ai_estimate" if source == "ai" else "user_custom")
    assert bool(item["is_estimated"]) == (source == "ai")
    assert float(item["total_kcal"]) == (92.8 if source == "local" else 160)
    if source != "local":
        response = update(api,group,items=[*({"id":r["id"],"weight_g":float(r["weight_g"])} for r in before),
            {"id":item["id"],"food_name":added["food_name"],"weight_g":160}])
        assert response.status_code == 200, response.text
        edited = rows(api,group)[2]
        assert edited["raw_food_name"] == item["raw_food_name"] and float(edited["total_kcal"]) == 320


def test_r013_wrong_user_and_foreign_item_are_rejected(food_api, record_property):
    api = food_api
    group = create_meal(api)
    other_group = create_meal(api, "other-synthetic-user")
    before, other = rows(api,group), rows(api,other_group)
    wrong_user = api.put(f"/api/nutrition/food/{group}",params={"user_id":"other-synthetic-user"},json={"meal_type":"晚餐"})
    assert wrong_user.status_code == 404, wrong_user.text
    bad_item = update(api,group,items=[{"id":other[0]["id"],"weight_g":150}])
    record_property("foreign_item_status",bad_item.status_code)
    record_property("both_users_sql_unchanged",rows(api,group) == before and rows(api,other_group) == other)
    # Existing route catches ValueError as HTTP 400, not schema-level 422.
    assert bad_item.status_code == 400, bad_item.text
    assert rows(api,group) == before and rows(api,other_group) == other


def test_r013_second_item_sql_failure_rolls_back_first_update(food_api, record_property):
    api = food_api
    group = create_meal(api)
    before = rows(api,group)
    successful, rejected = [], []
    def is_update(statement): return statement.lstrip().upper().startswith("UPDATE USER_FOOD_LOG SET MEAL_TYPE=")
    def success(conn,cursor,statement,parameters,context,executemany):
        if is_update(statement): successful.append(statement.split(" WHERE")[0])
    def fail(conn,cursor,statement,parameters,context,executemany):
        if is_update(statement) and successful:
            rejected.append(True)
            raise RuntimeError("quality-second-food-update-failure")
    def fail_batch(cursor,statement,parameters,context):
        if is_update(statement) and len(parameters) > 1:
            # Execute the first real update, then fail before the second DBAPI
            # row. Both per-statement and executemany paths must roll it back.
            cursor.execute(statement,parameters[0])
            assert cursor.rowcount == 1
            successful.append(statement.split(" WHERE")[0])
            rejected.append(True)
            raise RuntimeError("quality-second-food-update-failure")
    event.listen(api.engine,"do_executemany",fail_batch)
    event.listen(api.engine,"after_cursor_execute",success)
    event.listen(api.engine,"before_cursor_execute",fail)
    try:
        response = update(api,group,items=[{"id":r["id"],"weight_g":150} for r in before])
    finally:
        event.remove(api.engine,"do_executemany",fail_batch)
        event.remove(api.engine,"after_cursor_execute",success)
        event.remove(api.engine,"before_cursor_execute",fail)
    record_property("successful_first_updates",len(successful))
    assert successful and rejected and response.status_code >= 500
    assert rows(api,group) == before


@pytest.mark.parametrize("mode", ["weight","meal","date"])
def test_r013_actual_frontend_payload_does_not_passively_overwrite_raw(food_api, mode, record_property):
    api = food_api
    group = create_meal(api)
    before = rows(api,group)
    script = Path(__file__).resolve().parents[2]/"frontend/tests/nutrition-edit-harness.mjs"
    node = shutil.which("node")
    assert node, "Node required for actual component payload capture"
    capture = subprocess.run([node,str(script),mode],capture_output=True,text=True,encoding="utf-8",check=True)
    call = json.loads(capture.stdout)
    body = call["body"]
    # Adapt only synthetic IDs to this database; fields come from actual Vue save.
    body["items"][0]["id"] = before[0]["id"]
    body["items"].append({"id":before[1]["id"],"weight_g":50})
    record_property("actual_component_payload",json.dumps(body,ensure_ascii=False))
    response = update(api,group,**body)
    assert response.status_code == 200, response.text
    after = rows(api,group)
    record_property("sql_raw_before_after",json.dumps([before[0]["raw_food_name"],after[0]["raw_food_name"]],ensure_ascii=False))
    assert after[0]["raw_food_name"] == before[0]["raw_food_name"], after[0]
