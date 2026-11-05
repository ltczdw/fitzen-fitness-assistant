from unittest.mock import AsyncMock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from app.api import profile as api
from app.db import session as db
from app.db.models import UserProfileModel
from app.db.profile_schema import ensure_profile_schema


@pytest.fixture
def profile_client(tmp_path,monkeypatch):
    engine=db.create_database_engine('sqlite:///'+str(tmp_path/'profile.db'))
    UserProfileModel.__table__.create(engine)
    monkeypatch.setattr(db,'DB_READY',True)
    monkeypatch.setattr(db,'SessionLocal',sessionmaker(bind=engine))
    sync=AsyncMock()
    monkeypatch.setattr(api.memory_service,'sync_profile',sync)
    app=FastAPI();app.include_router(api.router,prefix='/api')
    with TestClient(app) as client:
        yield client,sync
    engine.dispose()


def test_personal_information_roundtrip_partial_update_and_clear(profile_client):
    client,sync=profile_client
    payload=dict(user_id='profile-test',display_name='测试用户',height_cm=175.5,body_weight_kg=70.2,body_fat_pct=18.5,target_weight_kg=68,goal='提高跑步耐力',experience='1年训练',available_days=4)
    response=client.put('/api/profile',json=payload)
    assert response.status_code==200,response.text
    assert response.json()==payload
    assert client.get('/api/profile?user_id=profile-test').json()==payload
    assert sync.await_args.args[1]['body_fat_pct']==18.5
    response=client.put('/api/profile',json=dict(user_id='profile-test',goal='保持健康'))
    assert response.status_code==200
    assert response.json()['body_fat_pct']==18.5
    assert response.json()['height_cm']==175.5
    response=client.put('/api/profile',json=dict(user_id='profile-test',body_fat_pct=None))
    assert response.status_code==200 and response.json()['body_fat_pct'] is None
    other=client.get('/api/profile?user_id=other-test').json()
    assert other['height_cm'] is None and other['body_fat_pct'] is None


@pytest.mark.parametrize('patch',[
    {'body_fat_pct':0},{'body_fat_pct':100},{'body_fat_pct':-1},{'body_fat_pct':'nan'},
    {'height_cm':99},{'body_weight_kg':401},{'available_days':8},{'available_days':2.5},{'goal':' '},
])
def test_invalid_profile_rejected_without_writes(profile_client,patch):
    client,sync=profile_client
    response=client.put('/api/profile',json=dict(user_id='invalid-profile',**patch))
    assert response.status_code==422,response.text
    current=client.get('/api/profile?user_id=invalid-profile').json()
    assert current['height_cm'] is None and current['body_fat_pct'] is None
    sync.assert_not_awaited()


def test_nullable_upgrade_is_idempotent_and_preserves_old_profile(tmp_path):
    engine=db.create_database_engine('sqlite:///'+str(tmp_path/'old-profile.db'))
    with engine.begin() as connection:
        connection.execute(text('CREATE TABLE user_profiles(user_id TEXT PRIMARY KEY,height_cm FLOAT,goal TEXT)'))
        connection.execute(text("INSERT INTO user_profiles VALUES('synthetic',175,'保持健康')"))
        ensure_profile_schema(connection);ensure_profile_schema(connection)
        value=dict(connection.execute(text('SELECT * FROM user_profiles')).mappings().one())
        assert value==dict(user_id='synthetic',height_cm=175,goal='保持健康',body_fat_pct=None)
    engine.dispose()
