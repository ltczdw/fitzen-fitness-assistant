"""All write gates on damaged representations, plus existing independent controls."""
from copy import deepcopy
import json
import pytest
from sqlalchemy import text
from test_manual_data_consistency import api, USER, create_mirror, source_record
from test_r012_acceptance_mapping import put_record
from test_quality_round2_r012 import snapshot, business_dml, id_payload

ENTRIES = ("record_put", "record_delete", "session_put", "session_delete", "exercise_delete", "set_add", "set_patch", "set_delete")


@pytest.mark.parametrize("fault", ["missing_action_wrong_user", "missing_session_source", "wrong_session_source", "foreign_session_binding", "foreign_user_binding"])
@pytest.mark.parametrize("entry", ENTRIES)
def test_damaged_association_all_writes_409_zero_dml(api, fault, entry, record_property):
    rec, s = create_mirror(api)
    unrelated = api.post("/api/workout-sessions", json={**USER, "title": "unrelated-control",
        "exercises": [{"exercise_name": "划船", "sets": [{"weight_kg": 20, "reps": 10}]}]}).json()
    reverse_user = USER.copy()
    other = None
    if fault == "wrong_session_source":
        other = api.post("/api/workouts", json={**USER, "exercise": "硬拉", "weight_kg": 100, "sets": 1, "reps": 5}).json()
    with api.engine.begin() as c:
        if fault == "missing_action_wrong_user":
            c.execute(text("UPDATE workout_session_exercises SET source_record_id=NULL WHERE session_id=:s"), {"s": s["id"]})
            c.execute(text("UPDATE workout_sessions SET user_id='different-user' WHERE id=:s"), {"s": s["id"]})
            reverse_user = {"user_id": "different-user"}
        elif fault == "missing_session_source":
            c.execute(text("UPDATE workout_sessions SET source_record_id=NULL WHERE id=:s"), {"s": s["id"]})
        elif fault == "wrong_session_source":
            # Keep the unique source_record_id constraint enabled. Free only the
            # synthetic other session's pointer before assigning it to the target;
            # its record and action remain, giving an actual inconsistent second
            # representation instead of failing during fixture preparation.
            c.execute(text("UPDATE workout_sessions SET source_record_id=NULL WHERE id=:s"), {"s": other["session_id"]})
            c.execute(text("UPDATE workout_sessions SET source_record_id=:r WHERE id=:s"), {"r": other["id"], "s": s["id"]})
        else:
            c.execute(text("UPDATE workout_session_exercises SET source_record_id=NULL WHERE session_id=:s"), {"s": s["id"]})
            c.execute(text("UPDATE workout_session_exercises SET source_record_id=:r WHERE id=:e"), {"r": rec["id"], "e": unrelated["exercises"][0]["id"]})
            if fault == "foreign_user_binding":
                c.execute(text("UPDATE workout_sessions SET user_id='different-user' WHERE id=:s"), {"s": unrelated["id"]})
    before = snapshot(api)
    original = source_record(api, rec["id"])
    ex, group = s["exercises"][0], s["exercises"][0]["sets"][0]
    body = id_payload(s); body.update(reverse_user); body["title"] = "must-not-mutate"
    with business_dml(api) as writes:
        if entry == "record_put": response = put_record(api, original, note="must-not-mutate")
        elif entry == "record_delete": response = api.delete(f"/api/workouts/{rec['id']}", params=USER)
        elif entry == "session_put": response = api.put(f"/api/workout-sessions/{s['id']}", params=reverse_user, json=body)
        elif entry == "session_delete": response = api.delete(f"/api/workout-sessions/{s['id']}", params=reverse_user)
        elif entry == "exercise_delete": response = api.delete(f"/api/workout-sessions/{s['id']}/exercises/{ex['id']}", params=reverse_user)
        elif entry == "set_add": response = api.post(f"/api/workout-sessions/{s['id']}/exercises/{ex['id']}/sets", params=reverse_user, json={"weight_kg": 99, "reps": 3})
        elif entry == "set_patch": response = api.patch(f"/api/workout-session-sets/{group['id']}", params=reverse_user, json={"weight_kg": 99, "reps": 3})
        else: response = api.delete(f"/api/workout-session-sets/{group['id']}", params=reverse_user)
    after = snapshot(api)
    for key, value in (("fault", fault), ("write_entry", entry), ("http_status", response.status_code), ("http_body", response.text),
                       ("business_dml", writes), ("all_facts_before", before), ("all_facts_after", after)):
        record_property(key, json.dumps(value, ensure_ascii=False, default=str))
    assert response.status_code == 409 and writes == [] and after == before, {"status": response.status_code, "body": response.text, "dml": writes, "facts_unchanged": after == before}
