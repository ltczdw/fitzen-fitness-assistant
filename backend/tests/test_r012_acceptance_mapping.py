"""Real API/SQL checks of heterogeneous facts, association, IDs and transactions."""
from copy import deepcopy
from datetime import date
import json

import pytest
from sqlalchemy import event, text
from test_manual_data_consistency import api, USER, create_mirror, source_record


def test_r012_reverse_edit_consumers_use_set_facts_despite_stale_summary(api, record_property):
    from app.api.dashboard import build_dashboard
    from app.db import session as db
    rec, session = create_mirror(api)
    group = session["exercises"][0]["sets"][0]
    response = api.patch(f"/api/workout-session-sets/{group['id']}",params=USER,json={"weight_kg":90,"reps":7})
    assert response.status_code == 200, response.text
    source = source_record(api,rec["id"])
    current = mirror(api,session["id"])
    assert facts(source["set_details"]) == facts(current["exercises"][0]["sets"])
    assert (source["weight_kg"], source["reps"], source["sets"], source["summary_mode"]) == (90,7,2,"first_set")
    from app.services.mcp.tools import FitnessMCPTools
    history = FitnessMCPTools().get_training_history(USER["user_id"])
    entry = next(x for x in history if x["id"] == rec["id"])
    assert (entry["weight_kg"],entry["reps"],entry["summary_mode"]) == (90,7,"first_set")
    assert facts(entry["set_details"]) == [(90,7),(82.5,6)]
    record_property("legacy_summary_after_edit",json.dumps({k:source[k] for k in ("weight_kg","reps")}))
    record_property("set_facts_after_edit",json.dumps(facts(source["set_details"])))
    if api.engine.dialect.name == "sqlite":
        # Dashboard always queries a food catalog even for workout-only users.
        # Supply an empty compatible table; do not mock its training read model.
        with api.engine.begin() as c:
            c.execute(text("""CREATE TABLE food_nutrition (
                id INTEGER PRIMARY KEY,food_name TEXT,energy_kcal NUMERIC,
                protein_g NUMERIC,carb_g NUMERIC,fat_g NUMERIC,fiber_g NUMERIC,
                is_active BOOLEAN DEFAULT TRUE)"""))
    # Exercise actual production dashboard read model, without inventing a
    # new top-level average/max rule. Its volume sum is an existing contract.
    with db.SessionLocal() as sql:
        dashboard = build_dashboard(sql,USER["user_id"],7,today=date.fromisoformat(source["date"]))
    record_property("actual_dashboard",json.dumps(dashboard,ensure_ascii=False,default=str))
    daily = next(d for d in dashboard["daily"] if d["date"] == source["date"])
    assert daily["training_records"] == 1 and daily["completed_sets"] == 2
    assert daily["training_volume"] == 90*7+82.5*6


def mirror(api, sid):
    return next(s for s in api.get("/api/workout-sessions", params=USER).json() if s["id"] == sid)


def facts(items):
    return [(s["weight_kg"], s["reps"]) for s in items]


def fields(s):
    return {k:s[k] for k in ("weight_kg", "reps", "difficulty", "completed", "note")}


def put_record(api, original, **changes):
    payload = {k:original[k] for k in ("exercise","sets","reps","weight_kg","difficulty","completed","note")}
    payload.update(changes)
    return api.put(f"/api/workouts/{original['id']}", params=USER, json=payload)


@pytest.mark.parametrize("change", [{"difficulty": "困难"}, {"completed": False}, {"note": "仅更新备注"}],
                         ids=["difficulty", "completed", "note"])
def test_r012_metadata_put_does_not_flatten_existing_heterogeneous_facts(api, record_property, change):
    rec, before = create_mirror(api)
    response = put_record(api, rec, **change)
    assert response.status_code == 200, response.text
    after = mirror(api, before["id"])
    record_property("before_details", json.dumps(rec["set_details"], ensure_ascii=False))
    record_property("after_record", json.dumps(source_record(api, rec["id"]), ensure_ascii=False))
    record_property("after_mirror_details", json.dumps(after["exercises"][0]["sets"], ensure_ascii=False))
    assert facts(after["exercises"][0]["sets"]) == [(80,8),(82.5,6)], after
    assert facts(source_record(api, rec["id"])["set_details"]) == [(80,8),(82.5,6)]


def test_r012_uniform_no_detail_record_and_explicit_detail_edit_are_distinct(api):
    rec = api.post("/api/workouts", json={**USER,"exercise":"卧推","sets":2,"reps":8,"weight_kg":60}).json()
    response = put_record(api, rec, weight_kg=70, sets=3, reps=10)
    assert response.status_code == 200
    assert facts(mirror(api, rec["session_id"])["exercises"][0]["sets"]) == [(70,10)]*3
    response = put_record(api, source_record(api, rec["id"]), sets=2,
        set_details=[{"weight_kg":70,"reps":10},{"weight_kg":75,"reps":6}])
    assert response.status_code == 200, response.text
    assert facts(mirror(api, rec["session_id"])["exercises"][0]["sets"]) == [(70,10),(75,6)]


def test_r012_middle_delete_then_patch_keeps_other_facts_and_ids(api):
    rec = api.post("/api/workouts", json={**USER,"exercise":"深蹲","sets":3,"reps":8,"weight_kg":80,
        "set_details":[{"weight_kg":80,"reps":8,"note":"first"},{"weight_kg":82.5,"reps":6,"note":"middle"},
                       {"weight_kg":85,"reps":5,"note":"last"}]}).json()
    before = mirror(api, rec["session_id"])
    groups = before["exercises"][0]["sets"]
    assert api.delete(f"/api/workout-session-sets/{groups[1]['id']}", params=USER).status_code == 200
    assert api.patch(f"/api/workout-session-sets/{groups[2]['id']}", params=USER, json={"reps":4}).status_code == 200
    after = mirror(api, rec["session_id"])["exercises"][0]["sets"]
    assert [s["id"] for s in after] == [groups[0]["id"],groups[2]["id"]]
    assert after[0] == groups[0]
    assert after[1]["note"] == "last" and after[1]["weight_kg"] == 85
    assert facts(source_record(api, rec["id"])["set_details"]) == [(80,8),(85,4)]


def session_payload(s):
    return {**USER, "session_date":s["session_date"], "title":s["title"], "note":s["note"],
        "exercises":[{"exercise_name":ex["exercise_name"],"difficulty":ex["difficulty"],"note":ex["note"],
                      "sets":[fields(g) for g in ex["sets"]]} for ex in s["exercises"]]}


def multi_mirror(api, *, extra_first=False, same_name=False):
    rec, s = create_mirror(api)
    body = session_payload(s)
    extra = {"exercise_name":"深蹲" if same_name else "划船","note":"unmapped-own-marker",
             "sets":[{"weight_kg":30,"reps":10,"note":"unmapped-set"}]}
    body["exercises"].insert(0 if extra_first else 1, extra)
    response = api.put(f"/api/workout-sessions/{s['id']}", params=USER, json=body)
    assert response.status_code == 200, response.text
    return rec, mirror(api, s["id"])


@pytest.mark.parametrize("mutation", ["patch", "add", "delete_set", "delete_exercise"])
def test_r012_unmapped_action_mutations_do_not_change_source(api, mutation):
    rec, before = multi_mirror(api)
    original = source_record(api, rec["id"])
    extra = before["exercises"][1]
    if mutation == "patch":
        response = api.patch(f"/api/workout-session-sets/{extra['sets'][0]['id']}",params=USER,json={"weight_kg":45})
    elif mutation == "add":
        response = api.post(f"/api/workout-sessions/{before['id']}/exercises/{extra['id']}/sets",params=USER,json={"weight_kg":35,"reps":6})
    elif mutation == "delete_set":
        response = api.delete(f"/api/workout-session-sets/{extra['sets'][0]['id']}",params=USER)
    else:
        response = api.delete(f"/api/workout-sessions/{before['id']}/exercises/{extra['id']}",params=USER)
    assert response.status_code == 200, response.text
    assert source_record(api, rec["id"]) == original
    assert mirror(api,before["id"])["exercises"][0] == before["exercises"][0]


def test_r012_same_name_second_action_patch_keeps_first_mapping(api):
    rec, before = multi_mirror(api, same_name=True)
    original = source_record(api, rec["id"])
    extra = before["exercises"][1]
    assert api.patch(f"/api/workout-session-sets/{extra['sets'][0]['id']}", params=USER,json={"weight_kg":45}).status_code == 200
    assert source_record(api,rec["id"]) == original
    assert mirror(api,before["id"])["exercises"][0] == before["exercises"][0]


def test_r012_record_rename_collision_does_not_overwrite_unrelated_action(api, record_property):
    rec, before = multi_mirror(api)
    original = source_record(api,rec["id"])
    response = put_record(api, original, exercise="划船", set_details=original["set_details"])
    assert response.status_code == 200, response.text
    after = mirror(api,before["id"])
    record_property("before_mirror",json.dumps(before,ensure_ascii=False))
    record_property("after_mirror",json.dumps(after,ensure_ascii=False))
    by_id = {ex["id"]:ex for ex in after["exercises"]}
    assert by_id[before["exercises"][1]["id"]] == before["exercises"][1], after
    assert by_id[before["exercises"][0]["id"]]["exercise_name"] == "划船"
    assert facts(by_id[before["exercises"][0]["id"]]["sets"]) == [(80,8),(82.5,6)]


def test_r012_session_rename_does_not_rebind_record_to_other_action(api, record_property):
    rec,before = multi_mirror(api, extra_first=True)
    body=session_payload(before)
    body["exercises"][1]["exercise_name"]="硬拉"
    response=api.put(f"/api/workout-sessions/{before['id']}",params=USER,json=body)
    assert response.status_code==200,response.text
    actual=source_record(api,rec["id"])
    record_property("before_mirror",json.dumps(before,ensure_ascii=False))
    record_property("request_body",json.dumps(body,ensure_ascii=False))
    record_property("after_record",json.dumps(actual,ensure_ascii=False))
    assert actual["exercise"]=="硬拉",actual
    assert facts(actual["set_details"])==[(80,8),(82.5,6)]


def test_r012_observe_mapped_action_delete_without_inventing_projection_rule(api, record_property):
    rec,before=multi_mirror(api)
    assert api.delete(f"/api/workout-sessions/{before['id']}/exercises/{before['exercises'][0]['id']}",params=USER).status_code==200
    after=mirror(api,before["id"])
    assert not any(r["id"] == rec["id"] for r in api.get("/api/workouts",params=USER).json())
    assert after["source_record_id"] is None
    record_property("round2_delete_contract",json.dumps({"remaining_session":after},ensure_ascii=False))
    assert after["exercises"]==[before["exercises"][1]]
    # Round 2 explicitly deletes the source and detaches the surviving session.


def test_r012_observe_full_session_put_id_replacement(api, record_property):
    rec,before=create_mirror(api)
    # Keep another session's higher row IDs alive so SQLite cannot mask replacements via ID reuse.
    other=api.post("/api/workout-sessions",json={**USER,"title":"independent",
        "exercises":[{"exercise_name":"划船","sets":[{"weight_kg":20,"reps":10}]}]}).json()
    body=session_payload(before);body["note"]="metadata only"
    assert api.put(f"/api/workout-sessions/{before['id']}",params=USER,json=body).status_code==200
    after=mirror(api,before["id"])
    record_property("full_put_id_compatibility",json.dumps({"before":before,"after":after},ensure_ascii=False))
    assert facts(after["exercises"][0]["sets"])==facts(before["exercises"][0]["sets"])
    assert after["exercises"][0]["id"] == before["exercises"][0]["id"]
    assert [x["id"] for x in after["exercises"][0]["sets"]] == [x["id"] for x in before["exercises"][0]["sets"]]
    assert mirror(api,other["id"])==other
    assert source_record(api,rec["id"])["set_details"]==rec["set_details"]
    # No-ID compatibility must also preserve the unchanged action/set identities.


@pytest.mark.parametrize("direction", ["record_update","set_update","record_delete","session_delete"])
def test_r012_second_side_failure_after_successful_sql_rolls_back_all(api, record_property, direction):
    rec,before=create_mirror(api)
    original=source_record(api,rec["id"])
    first,second = (("workout_records","workout_session_sets") if direction in {"record_update","session_delete","record_delete"}
                    else ("workout_session_sets","workout_records"))
    executed,rejected=[],[]
    def after_sql(conn,cursor,stmt,params,context,many):
        if stmt.lstrip().upper().startswith(("UPDATE ","DELETE ","INSERT ")) and first in stmt.lower():
            executed.append(stmt.split()[0])
    def fail_second(conn,cursor,stmt,params,context,many):
        if executed and stmt.lstrip().upper().startswith(("UPDATE ","DELETE ","INSERT ")) and second in stmt.lower():
            rejected.append(stmt.split()[0]);raise RuntimeError("quality-second-side-after-first-success")
    event.listen(api.engine,"after_cursor_execute",after_sql)
    event.listen(api.engine,"before_cursor_execute",fail_second)
    try:
        if direction=="record_update":
            response=put_record(api,original,weight_kg=99,set_details=[{"weight_kg":99,"reps":8},{"weight_kg":82.5,"reps":6}])
        elif direction=="set_update":
            response=api.patch(f"/api/workout-session-sets/{before['exercises'][0]['sets'][0]['id']}",params=USER,json={"weight_kg":99})
        elif direction=="record_delete":
            response=api.delete(f"/api/workouts/{rec['id']}",params=USER)
        else:
            response=api.delete(f"/api/workout-sessions/{before['id']}",params=USER)
    finally:
        event.remove(api.engine,"after_cursor_execute",after_sql)
        event.remove(api.engine,"before_cursor_execute",fail_second)
    record_property("successful_first_side_sql",json.dumps(executed))
    record_property("rejected_second_side_sql",json.dumps(rejected))
    assert executed and rejected,{"first":executed,"second":rejected,"http":response.status_code}
    assert response.status_code>=500,response.text
    assert source_record(api,rec["id"])==original
    assert mirror(api,before["id"])==before
