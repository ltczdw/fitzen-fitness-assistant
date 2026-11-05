"""Tests for workout record bindings."""
from contextlib import contextmanager
from copy import deepcopy
import pytest
from sqlalchemy import event,text
from test_manual_data_consistency import api,USER,create_mirror,source_record
from test_r012_acceptance_mapping import mirror,multi_mirror,session_payload,put_record,facts

TABLES=("workout_records","workout_sessions","workout_session_exercises","workout_session_sets")
def snapshot(api):
    with api.engine.connect() as c:
        return {t:[dict(r) for r in c.execute(text(f"SELECT * FROM {t} ORDER BY id")).mappings()] for t in TABLES}

@contextmanager
def business_dml(api):
    writes=[]
    def audit(conn,cursor,stmt,params,context,many):
        if stmt.lstrip().upper().startswith(("UPDATE ","INSERT ","DELETE ")) and any(t in stmt.lower() for t in TABLES):writes.append(stmt.split()[0:3])
    event.listen(api.engine,"before_cursor_execute",audit)
    try:yield writes
    finally:event.remove(api.engine,"before_cursor_execute",audit)

def id_payload(s):
    p=session_payload(s)
    for x,original in zip(p["exercises"],s["exercises"]):
        x["id"]=original["id"]
        for g,old in zip(x["sets"],original["sets"]):g["id"]=old["id"]
    return p

def binding(api,rid):
    with api.engine.connect() as c:
        return [dict(r) for r in c.execute(text("SELECT id,session_id,source_record_id FROM workout_session_exercises WHERE source_record_id=:r"),{"r":rid}).mappings()]

def test_top_only_heterogeneous_edit_409_zero_sql(api):
    rec,s=create_mirror(api);before=snapshot(api)
    with business_dml(api) as writes:
        response=api.put(f"/api/workouts/{rec['id']}",params=USER,json={"exercise":"深蹲","sets":2,"reps":20,"weight_kg":99})
    assert response.status_code == 409,response.text
    assert writes == [],writes
    assert snapshot(api) == before


@pytest.mark.parametrize("legacy",["new","single","unique_name","ambiguous","no_candidate"])
def test_binding_creation_and_legacy_adoption(api,legacy):
    rec,s=(multi_mirror(api,same_name=legacy=="ambiguous") if legacy in {"unique_name","ambiguous","no_candidate"} else create_mirror(api))
    original=source_record(api,rec["id"])
    assert binding(api,rec["id"]) == [{"id":s["exercises"][0]["id"],"session_id":s["id"],"source_record_id":rec["id"]}]
    if legacy != "new":
        with api.engine.begin() as c:
            c.execute(text("UPDATE workout_session_exercises SET source_record_id=NULL WHERE source_record_id=:r"),{"r":rec["id"]})
            if legacy=="no_candidate":c.execute(text("UPDATE workout_session_exercises SET exercise_name='无候选动作' WHERE session_id=:s"),{"s":s["id"]})
    before=snapshot(api)
    with business_dml(api) as writes:response=put_record(api,original,note="binding-transaction")
    if legacy in {"ambiguous","no_candidate"}:
        assert response.status_code==409,response.text
        assert writes==[] and snapshot(api)==before
    else:
        assert response.status_code==200,response.text
        assert binding(api,rec["id"])[0]["id"]==s["exercises"][0]["id"]
        assert source_record(api,rec["id"])["note"]=="binding-transaction"


@pytest.mark.parametrize("other_user",[False,True],ids=["other_session","other_user"])
def test_corrupt_binding_cannot_mutate_foreign_action(api,other_user,record_property):
    rec,s=create_mirror(api)
    other=api.post("/api/workout-sessions",json={"user_id":"different-user" if other_user else USER["user_id"],
        "title":"foreign","exercises":[{"exercise_name":"划船","sets":[{"weight_kg":30,"reps":10}]}]}).json()
    with api.engine.begin() as c:
        c.execute(text("UPDATE workout_session_exercises SET source_record_id=NULL WHERE source_record_id=:r"),{"r":rec["id"]})
        c.execute(text("UPDATE workout_session_exercises SET source_record_id=:r WHERE id=:i"),{"r":rec["id"],"i":other["exercises"][0]["id"]})
    before=snapshot(api)
    with business_dml(api) as writes:response=put_record(api,source_record(api,rec["id"]),note="must-not-touch-foreign",set_details=[{"weight_kg":99,"reps":8},{"weight_kg":82.5,"reps":6}],weight_kg=99)
    import json
    record_property("foreign_binding_before",json.dumps(before,ensure_ascii=False,default=str))
    record_property("foreign_binding_after",json.dumps(snapshot(api),ensure_ascii=False,default=str))
    assert response.status_code==409,{"http":response.status_code,"body":response.text,"dml":writes}
    assert writes==[] and snapshot(api)==before


@pytest.mark.parametrize("mode",["reorder","rename_collision","edit_group","insert_unrelated","legacy_slots","legacy_insert"])
def test_full_put_preserves_undeleted_ids_and_binding(api,mode):
    rec,s=multi_mirror(api);body=session_payload(s) if mode.startswith("legacy") else id_payload(s)
    if mode=="reorder":body["exercises"].reverse();body["exercises"][1]["sets"].reverse()
    elif mode=="rename_collision":body["exercises"][0]["exercise_name"]="划船"
    elif mode in {"edit_group","legacy_slots"}:body["exercises"][0]["sets"][0]["weight_kg"]=99
    elif mode in {"insert_unrelated","legacy_insert"}:body["exercises"].insert(0,{"exercise_name":"无关新动作","sets":[{"weight_kg":20,"reps":10}]})
    response=api.put(f"/api/workout-sessions/{s['id']}",params=USER,json=body)
    assert response.status_code==200,response.text
    after=mirror(api,s["id"]);by_id={x["id"]:x for x in after["exercises"]}
    for old in s["exercises"]:
        assert old["id"] in by_id,after
        assert {g["id"] for g in by_id[old["id"]]["sets"]}=={g["id"] for g in old["sets"]}
    assert binding(api,rec["id"])[0]["id"]==s["exercises"][0]["id"]
    current=by_id[s["exercises"][0]["id"]]
    assert facts(source_record(api,rec["id"])["set_details"])==facts(current["sets"])


@pytest.mark.parametrize("mode",["foreign_action","duplicate_action","foreign_group","duplicate_group","wrong_parent_group","legacy_ambiguous"])
def test_invalid_full_put_409_zero_mutation(api,mode):
    rec,s=multi_mirror(api);body=id_payload(s)
    other=api.post("/api/workout-sessions",json={"user_id":"different-user","exercises":[{"exercise_name":"划船","sets":[{"weight_kg":20,"reps":10}]}]}).json()
    if mode=="foreign_action":body["exercises"][0]["id"]=other["exercises"][0]["id"]
    elif mode=="duplicate_action":body["exercises"][1]["id"]=body["exercises"][0]["id"]
    elif mode=="foreign_group":body["exercises"][0]["sets"][0]["id"]=other["exercises"][0]["sets"][0]["id"]
    elif mode=="duplicate_group":body["exercises"][0]["sets"][1]["id"]=body["exercises"][0]["sets"][0]["id"]
    elif mode=="wrong_parent_group":body["exercises"][0]["sets"][0]["id"]=body["exercises"][1]["sets"][0]["id"]
    else:
        with api.engine.begin() as c:
            # Make identical old actions in a standalone session; exact matches are ambiguous.
            sid=other["id"];eid=other["exercises"][0]["id"]
            c.execute(text("UPDATE workout_sessions SET user_id=:u WHERE id=:s"),{"u":USER["user_id"],"s":sid})
        body=session_payload(mirror(api,sid));body["exercises"].append(deepcopy(body["exercises"][0]))
        assert api.put(f"/api/workout-sessions/{sid}",params=USER,json=body).status_code==200
        s=mirror(api,sid);body=session_payload(s)
    before=snapshot(api)
    with business_dml(api) as writes:response=api.put(f"/api/workout-sessions/{s['id']}",params=USER,json=body)
    assert response.status_code==409,response.text
    assert snapshot(api)==before
    # Validation can flush earlier legitimate in-transaction changes; all roll back.


@pytest.mark.parametrize("kind",["source_action","source_record","last_source_set","whole_session"])
def test_source_delete_preserves_unrelated_action(api,kind):
    rec,s=multi_mirror(api);source,extra=s["exercises"]
    independent=api.post("/api/workout-sessions",json={**USER,"title":"independent","exercises":[{"exercise_name":"划船","sets":[{"weight_kg":20,"reps":10}]}]}).json()
    if kind=="source_action":path=f"/api/workout-sessions/{s['id']}/exercises/{source['id']}"
    elif kind=="source_record":path=f"/api/workouts/{rec['id']}"
    elif kind=="whole_session":path=f"/api/workout-sessions/{s['id']}"
    else:
        assert api.delete(f"/api/workout-session-sets/{source['sets'][1]['id']}",params=USER).status_code==200
        path=f"/api/workout-session-sets/{source['sets'][0]['id']}"
    response=api.delete(path,params=USER);assert response.status_code==200,response.text
    assert not any(x["id"]==rec["id"] for x in api.get("/api/workouts",params=USER).json())
    assert binding(api,rec["id"])==[]
    if kind=="whole_session":assert not any(x["id"]==s["id"] for x in api.get("/api/workout-sessions",params=USER).json())
    else:
        after=mirror(api,s["id"])
        assert after["source_record_id"] is None and after["exercises"]==[extra]
    assert mirror(api,independent["id"])==independent


@pytest.mark.parametrize("mode",["delete_first","delete_middle","add","edit"])
def test_current_first_set_summary_counts_and_mcp(api,mode):
    from app.services.mcp.tools import FitnessMCPTools
    rec=api.post("/api/workouts",json={**USER,"exercise":"深蹲","sets":3,"reps":8,"weight_kg":80,
        "set_details":[{"weight_kg":80,"reps":8},{"weight_kg":82.5,"reps":6},{"weight_kg":85,"reps":5}]}).json()
    s=mirror(api,rec["session_id"]);ex=s["exercises"][0]
    if mode.startswith("delete"):
        i=0 if mode=="delete_first" else 1;response=api.delete(f"/api/workout-session-sets/{ex['sets'][i]['id']}",params=USER)
    elif mode=="add":response=api.post(f"/api/workout-sessions/{s['id']}/exercises/{ex['id']}/sets",params=USER,json={"weight_kg":90,"reps":4})
    else:response=api.patch(f"/api/workout-session-sets/{ex['sets'][0]['id']}",params=USER,json={"weight_kg":99,"reps":3})
    assert response.status_code==200,response.text
    expected=facts(mirror(api,s["id"])["exercises"][0]["sets"])
    for current in (source_record(api,rec["id"]),next(x for x in FitnessMCPTools().get_training_history(USER["user_id"]) if x["id"]==rec["id"])):
        assert (current["weight_kg"],current["reps"])==expected[0]
        assert current["sets"]==len(expected) and current["summary_mode"]=="first_set"
        assert facts(current["set_details"])==expected


@pytest.mark.parametrize("prompt",["qa","planning"])
def test_real_model_context_keeps_summary_mode_and_heterogeneous_details(api,monkeypatch,prompt,record_property):
    import asyncio,json
    from unittest.mock import AsyncMock
    from evals.v2.safety import deny_network
    from app.services.mcp.tools import FitnessMCPTools
    from app.api.chat import agent
    rec,_=create_mirror(api)
    history=FitnessMCPTools().get_training_history(USER["user_id"])
    state={"raw_message":"分析两组训练负荷","training_history":history}
    graph=agent.graph
    messages=graph._qa_prompt(state,[]) if prompt=="qa" else graph._planning_prompt("训练计划Agent",state,[])
    mocked=AsyncMock(return_value="mock-heterogeneous-answer")
    monkeypatch.setattr(graph.llm,"complete",mocked)
    with deny_network() as audit:asyncio.run(graph._complete_or_fallback(messages,"fixture fallback"))
    assert audit["blocked_connections"]==audit["blocked_dns"]==0
    sent=mocked.call_args.args[0]
    fact=next(m["content"] for m in sent if "SQL/MCP训练事实：" in m["content"])
    value=json.loads(fact.split("：",1)[1])
    record_property("actual_model_training_fact",json.dumps(value,ensure_ascii=False))
    assert value["id"]==rec["id"] and value["summary_mode"]=="first_set"
    assert (value["weight_kg"],value["reps"],value["sets"])==(80,8,2)
    assert facts(value["set_details"])==[(80,8),(82.5,6)]


def test_uniform_mode_and_mismatched_explicit_details(api):
    rec=api.post("/api/workouts",json={**USER,"exercise":"卧推","sets":2,"reps":8,"weight_kg":60}).json()
    assert source_record(api,rec["id"])["summary_mode"]=="uniform"
    before=snapshot(api)
    with business_dml(api) as writes:response=put_record(api,rec,set_details=[{"weight_kg":60,"reps":8}])
    assert response.status_code==422,response.text
    assert writes==[] and snapshot(api)==before
