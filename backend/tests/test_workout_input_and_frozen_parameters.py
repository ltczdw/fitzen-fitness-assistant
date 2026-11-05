"""Tests for workout input and frozen parameters."""
from copy import deepcopy
import pytest
from test_r011_acceptance_semantics import (
    chat_db, AcceptanceHarness, make_harness, record, seed_cache,
)


@pytest.mark.parametrize("field,value", [
    ("weight_kg","60"),("sets","4"),("reps","8"),
    ("sets",4.5),("set_details",[{"weight_kg":"60","reps":8}]*4),
], ids=["numeric_string_weight","numeric_string_sets","numeric_string_reps","fractional_sets","numeric_string_detail"])
def test_raw_types_rejected_before_freeze(chat_db,monkeypatch,record_property,field,value):
    message="记录卧推60kg，4组8次"
    h=make_harness(monkeypatch,chat_db,message,[record(**{field:value})],record_property)
    h.assert_no_write(h.run(message))


@pytest.mark.parametrize("message,details", [
    ("记录卧推2组，第2组62.5kg6次，第1组60kg8次",[(62.5,6),(60,8)]),
    ("记录卧推2组，第1组60kg8次，第1组62.5kg6次",[(60,8),(62.5,6)]),
    ("记录卧推2组，第1组60kg8次，第3组62.5kg6次",[(60,8),(62.5,6)]),
], ids=["out_of_order_mispaired","duplicate_number","missing_number"])
def test_explicit_group_numbers_cannot_lend_facts(chat_db,monkeypatch,record_property,message,details):
    payload=record(sets=2,reps=details[0][1],weight_kg=details[0][0],
        set_details=[{"weight_kg":w,"reps":r} for w,r in details])
    h=make_harness(monkeypatch,chat_db,message,[payload],record_property)
    h.assert_no_write(h.run(message))


@pytest.mark.parametrize("message,details", [
    ("记录卧推2组，第2组62.5kg6次，第1组60kg8次",[(60,8),(62.5,6)]),
    ("记录卧推2组，每组60.5kg8次",[(60.5,8),(60.5,8)]),
    ("记录卧推2组，第1组60kg8次，第2组62.5kg6次",[(60,8),(62.5,6)]),
],ids=["out_of_order_correct_position","uniform_decimal","heterogeneous_control"])
def test_numbered_and_uniform_controls(chat_db,monkeypatch,record_property,message,details):
    h=make_harness(monkeypatch,chat_db,message,[record(sets=2,weight_kg=details[0][0],reps=details[0][1],
        set_details=[{"weight_kg":w,"reps":r} for w,r in details])],record_property)
    h.run(message)
    assert len(h.rows()) == len(h.payloads) == 1
    assert [(x["weight_kg"],x["reps"]) for x in h.payloads[0]["set_details"]] == details


@pytest.mark.parametrize("mode", ["duplicate","negative","bool","text","numeric_text","fractional","null","short","long","not_list","keys_duplicate","keys_short","keys_wrong","keys_not_list","shared_validation"])
def test_present_cache_metadata_corruption_fails_closed(chat_db,monkeypatch,record_property,mode):
    from app.services.agent.operation_store import OperationStore
    message="记录卧推60kg，4组8次，深蹲80kg，3组5次"
    records=[record(date="2026-10-03"),record(exercise="深蹲",weight_kg=80,sets=3,reps=5,date="2026-10-03")]
    if mode == "shared_validation": records[0]["weight_kg"]=999
    h=AcceptanceHarness(monkeypatch,chat_db,property_writer=record_property)
    rid,cid,store=seed_cache(h,message,records,source=message,validated=True)
    original_operations=deepcopy(store.load()["operations"])
    original=OperationStore.task_params
    def corrupted(store,task):
        p=deepcopy(original(store,task));p["positions"]=[0,1]
        p["operation_keys"]=[f"workout-task-{rid}-t1-{i}" for i in range(2)]
        if mode in {"duplicate","shared_validation"}:p["positions"]=[0,0];p["operation_keys"]=[f"workout-task-{rid}-t1-0"]*2
        elif mode=="negative":p["positions"]=[-1,1];p["operation_keys"][0]=f"workout-task-{rid}-t1--1"
        elif mode=="bool":p["positions"]=[False,1]
        elif mode=="text":p["positions"]=["bad",1]
        elif mode=="numeric_text":p["positions"]=["0",1]
        elif mode=="fractional":p["positions"]=[0.5,1]
        elif mode=="null":p["positions"]=None
        elif mode=="short":p["positions"]=[0]
        elif mode=="long":p["positions"]=[0,1,2]
        elif mode=="not_list":p["positions"]="bad"
        elif mode=="keys_duplicate":p["operation_keys"]=[p["operation_keys"][0]]*2
        elif mode=="keys_short":p["operation_keys"]=p["operation_keys"][:1]
        elif mode=="keys_wrong":p["operation_keys"][0]="other-request-t1-0"
        elif mode=="keys_not_list":p["operation_keys"]=None
        return p
    monkeypatch.setattr(OperationStore,"task_params",corrupted)
    record_property("boundary","synthetic accessor corruption; actual frozen plan/HTTP/SQL retained")
    record_property("metadata_fault",mode)
    h.run(message,request_id=rid,conversation_id=cid)
    assert h.payloads==[] and h.rows()==[],{"payloads":h.payloads,"rows":h.rows()}
    assert store.load()["operations"]==original_operations,"Corrupt metadata must not reserve a new operation identity"


def test_repeated_action_instances_do_not_borrow_later_weight(chat_db,monkeypatch,record_property):
    message="记录卧推60kg4组8次，然后卧推70kg3组6次"
    h=make_harness(monkeypatch,chat_db,message,[record(weight_kg=70),record(weight_kg=60,sets=3,reps=6)],record_property)
    response=h.run(message)
    h.assert_structured_path(message,[record(weight_kg=70),record(weight_kg=60,sets=3,reps=6)])
    assert h.trace["reservations"] == [],h.trace
    h.assert_no_write(response)


def test_repeated_action_instances_legal_control(chat_db,monkeypatch,record_property):
    message="记录卧推60kg4组8次，然后卧推70kg3组6次"
    h=make_harness(monkeypatch,chat_db,message,[record(),record(weight_kg=70,sets=3,reps=6)],record_property)
    h.run(message)
    h.assert_structured_path(message,[record(),record(weight_kg=70,sets=3,reps=6)])
    assert len(h.rows())==len(h.payloads)==2
    assert [(p["weight_kg"],p["sets"],p["reps"]) for p in h.payloads]==[(60,4,8),(70,3,6)]


def test_business_edit_does_not_rewrite_frozen_operation_parameters(chat_db,monkeypatch,record_property):
    from uuid import uuid4
    from app.services.agent.operation_store import OperationStore
    from test_manual_data_consistency import API
    message="记录卧推60kg，4组8次"
    h=make_harness(monkeypatch,chat_db,message,[record()],record_property)
    rid,cid=str(uuid4()),str(uuid4())
    h.run(message,request_id=rid,conversation_id=cid)
    original=deepcopy(OperationStore(rid).load())
    from sqlalchemy import text
    assert len(h.rows())==1
    # The chat route selects the effective test user; read that actual owner,
    # instead of assuming request.user_id survives the route's context policy.
    with chat_db.connect() as c:owner=c.execute(text("SELECT user_id FROM workout_records")).scalar_one()
    api=API(chat_db);user={"user_id":owner}
    rec=api.get("/api/workouts",params=user).json()[0]
    response=api.put(f"/api/workouts/{rec['id']}",params=user,json={"exercise":"卧推","sets":4,"reps":8,"weight_kg":65})
    assert response.status_code==200,response.text
    after=OperationStore(rid).load()
    assert after["operations"]==original["operations"]
    assert after["tasks"]==original["tasks"]
