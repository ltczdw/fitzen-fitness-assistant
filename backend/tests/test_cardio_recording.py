import copy
from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlalchemy.orm import sessionmaker

from app.api.dashboard import aggregate
from app.api.workout_sessions_api import router
from app.db import session as db
from app.db.cardio_metrics_schema import ensure_cardio_metrics_schema
from app.db.models import WorkoutRecordModel
from app.db.workout_session_models import WorkoutSessionModel, WorkoutExerciseModel, WorkoutSetModel


@pytest.fixture
def client(tmp_path, monkeypatch):
    engine = db.create_database_engine('sqlite:///' + str(tmp_path / 'cardio.db'))
    for model in (WorkoutRecordModel, WorkoutSessionModel, WorkoutExerciseModel, WorkoutSetModel):
        model.__table__.create(engine)
    monkeypatch.setattr(db, 'SessionLocal', sessionmaker(bind=engine))
    monkeypatch.setattr(db, 'DB_READY', True)
    app = FastAPI()
    app.include_router(router, prefix='/api')
    with TestClient(app) as value:
        yield value
    engine.dispose()


def body():
    return dict(user_id='cardio-test',session_date='2026-10-21',title='混合训练',note='',exercises=[
        dict(exercise_name='跑步',sets=[dict(distance_km=5.25,duration_minutes=31.5)]),
        dict(exercise_name='卧推',sets=[dict(weight_kg=40,reps=8)])])


def test_round_trip_edit_append_patch_and_strength_aggregation(client):
    response = client.post('/api/workout-sessions',json=body())
    assert response.status_code == 200, response.text
    record = response.json()
    cardio, strength = record['exercises']
    assert cardio['sets'][0]['distance_km'] == 5.25
    assert cardio['sets'][0]['duration_minutes'] == 31.5
    assert cardio['sets'][0]['reps'] == cardio['sets'][0]['weight_kg'] == 0
    assert 'duration_minutes' not in strength['sets'][0]
    strength_snapshot = copy.deepcopy(strength)
    payload = {key:record[key] for key in ('user_id','session_date','title','note','exercises')}
    for exercise in payload['exercises']:
        for item in exercise['sets']:
            item.pop('set_index')
    payload['exercises'][0]['sets'][0]['duration_minutes'] = 32.25
    response = client.put(f"/api/workout-sessions/{record['id']}?user_id=cardio-test",json=payload)
    assert response.status_code == 200, response.text
    updated=response.json()
    assert updated['exercises'][0]['sets'][0]['id'] == cardio['sets'][0]['id']
    assert updated['exercises'][1] == strength_snapshot
    response=client.post(f"/api/workout-sessions/{record['id']}/exercises/{cardio['id']}/sets?user_id=cardio-test",json=dict(distance_km=1.5,duration_minutes=10))
    assert response.status_code == 200, response.text
    set_id=response.json()['set']['id']
    response=client.patch(f'/api/workout-session-sets/{set_id}?user_id=cardio-test',json=dict(duration_minutes=11))
    assert response.status_code == 200, response.text
    assert response.json()['set']['distance_km'] == 1.5
    response=client.patch(f'/api/workout-session-sets/{set_id}?user_id=cardio-test',json=dict(weight_kg=3))
    assert response.status_code == 422
    response=client.get('/api/workout-sessions?user_id=cardio-test')
    assert response.status_code == 200
    saved=response.json()[0]
    assert saved['exercises'][0]['sets'][1]['weight_kg'] == 0
    assert saved['exercises'][0]['sets'][1]['duration_minutes'] == 11
    days,best=aggregate([], [saved], [], date(2026,10,21),1)
    assert days[0]['cardio_distance_km'] == 6.75
    assert days[0]['cardio_duration_minutes'] == 43.25
    assert days[0]['training_volume'] == 320
    assert days[0]['completed_sets'] == 1
    assert [item['exercise'] for item in best] == ['卧推']
    assert client.get('/api/workout-sessions?user_id=other-user').json() == []


@pytest.mark.parametrize('metrics',[
    dict(distance_km=2),dict(duration_minutes=20),dict(distance_km=-1,duration_minutes=20),
    dict(distance_km=0,duration_minutes=0),dict(distance_km=2,duration_minutes=20,reps=5),
    dict(distance_km=2,duration_minutes=20,weight_kg=5),dict(reps=0),
])
def test_invalid_or_mixed_units_rejected_without_saving(client,metrics):
    payload=body();payload['exercises'][0]['sets']=[metrics]
    assert client.post('/api/workout-sessions',json=payload).status_code == 422
    assert client.get('/api/workout-sessions?user_id=cardio-test').json() == []


def test_one_action_cannot_mix_measurement_kinds(client):
    payload=body();payload['exercises'][0]['sets'].append(dict(reps=8,weight_kg=40))
    assert client.post('/api/workout-sessions',json=payload).status_code == 422


def test_upgrade_is_idempotent_and_does_not_reinterpret_legacy_numbers(tmp_path):
    engine=db.create_database_engine('sqlite:///' + str(tmp_path/'old-cardio.db'))
    with engine.begin() as connection:
        connection.execute(text('CREATE TABLE workout_session_sets(id INTEGER PRIMARY KEY,weight_kg FLOAT,reps INTEGER)'))
        connection.execute(text('INSERT INTO workout_session_sets VALUES(1,40,8)'))
        ensure_cardio_metrics_schema(connection)
        ensure_cardio_metrics_schema(connection)
        row=connection.execute(text('SELECT * FROM workout_session_sets')).mappings().one()
        assert dict(row) == dict(id=1,weight_kg=40,reps=8,distance_km=None,duration_minutes=None)
        assert len(inspect(connection).get_columns('workout_session_sets')) == 5
    engine.dispose()


def test_linked_record_snapshot_keeps_cardio_fields_and_action_identity(client):
    import json
    from app.db.workout_session_repository import mirror_sets_from_record
    with db.SessionLocal() as session:
        source=WorkoutRecordModel(user_id='cardio-test',workout_date=date(2026,10,21),exercise='跑步',sets=1,reps=1,weight_kg=0,note='old snapshot')
        session.add(source);session.flush()
        workout=WorkoutSessionModel(user_id='cardio-test',source_record_id=source.id,session_date=date(2026,10,21),title='有氧训练',note='')
        session.add(workout);session.flush()
        exercise=WorkoutExerciseModel(session_id=workout.id,source_record_id=source.id,exercise_name='跑步',order_index=0,note='')
        session.add(exercise);session.flush()
        row=WorkoutSetModel(exercise_id=exercise.id,set_index=0,weight_kg=0,reps=1,note='')
        session.add(row);session.commit()
        sid,eid,setid,sourceid=workout.id,exercise.id,row.id,source.id
    payload=dict(user_id='cardio-test',session_date='2026-10-21',title='有氧训练',note='',exercises=[dict(id=eid,exercise_name='跑步',sets=[dict(id=setid,distance_km=3.2,duration_minutes=22.5)])])
    response=client.put(f'/api/workout-sessions/{sid}?user_id=cardio-test',json=payload)
    assert response.status_code==200,response.text
    assert response.json()['exercises'][0]['sets'][0]['id']==setid
    with db.SessionLocal() as session:
        source=session.get(WorkoutRecordModel,sourceid)
        details=json.loads(source.set_details)
        assert details[0]['distance_km']==3.2
        assert details[0]['duration_minutes']==22.5
        assert source.reps==0 and source.weight_kg==0
        assert mirror_sets_from_record(source)[0]['distance_km']==3.2
        assert mirror_sets_from_record(source)[0]['duration_minutes']==22.5
