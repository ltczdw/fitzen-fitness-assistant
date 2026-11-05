"""Library writes use temporary SQLite or mocked repositories, never live data."""
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.db import session as db
from app.db.food_repository import delete_catalog_food, list_food_catalog
from app.db.custom_food_repository import delete_custom_food
from app.api import nutrition

@pytest.fixture
def library_db(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///" + str(tmp_path / "food-library.db"))
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(db, "DB_READY", True)
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "SessionLocal", factory)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE food_nutrition(id INTEGER PRIMARY KEY, food_name TEXT, category TEXT, preparation_state TEXT, energy_kcal REAL, protein_g REAL, carb_g REAL, fat_g REAL, fiber_g REAL, is_active BOOLEAN)"))
        conn.execute(text("CREATE TABLE user_custom_food(id INTEGER PRIMARY KEY, user_id TEXT, is_active BOOLEAN)"))
        conn.execute(text("CREATE TABLE user_food_log(id INTEGER PRIMARY KEY, food_id INTEGER REFERENCES food_nutrition(id), user_id TEXT, record_time TEXT, total_kcal REAL)"))
        conn.execute(text("INSERT INTO food_nutrition VALUES (1,'合成米饭','主食','熟',116,2.6,25.9,.3,.3,TRUE)"))
        conn.execute(text("INSERT INTO user_food_log VALUES(7,1,'alice','2026-10-18',116)"))
        conn.execute(text("INSERT INTO user_custom_food VALUES(3,'alice',TRUE)"))
    yield engine
    engine.dispose()

def test_catalog_identity_soft_delete_idempotency_and_history(library_db):
    rows=list_food_catalog(user_id="alice")
    assert rows and rows[0]["id"] == 1
    assert delete_catalog_food(1) is True
    assert delete_catalog_food(1) is True
    assert delete_catalog_food(99) is False
    assert list_food_catalog(user_id="alice") == []
    with library_db.connect() as conn:
        assert conn.execute(text("SELECT food_id,total_kcal FROM user_food_log")).one() == (1,116)
        assert conn.execute(text("SELECT is_active FROM food_nutrition WHERE id=1")).scalar() == 0

def test_custom_delete_is_user_scoped(library_db):
    assert delete_custom_food(3,"bob") is False
    with library_db.connect() as conn:
        assert conn.execute(text("SELECT is_active FROM user_custom_food WHERE id=3")).scalar() == 1
    assert delete_custom_food(3,"alice") is True
    assert delete_custom_food(3,"alice") is True

def test_delete_unavailable_fails_without_success():
    with pytest.raises(RuntimeError): delete_catalog_food(1)
    with pytest.raises(RuntimeError): delete_custom_food(1)

@pytest.mark.parametrize("path,repository",[("/nutrition/food/catalog/4","delete_catalog_food"),("/nutrition/custom-foods/4","delete_custom_food")])
def test_delete_api_contract(monkeypatch,path,repository):
    app=FastAPI(); app.include_router(nutrition.router)
    with TestClient(app) as client:
        calls=[]
        monkeypatch.setattr(nutrition,repository,lambda *args: calls.append(args) or True)
        response=client.delete(path+"?user_id=alice")
        assert response.status_code == 200 and response.json()=={"deleted":True,"id":4}
        assert calls == ([(4,"alice")] if repository=="delete_custom_food" else [(4,)])
        monkeypatch.setattr(nutrition,repository,lambda *args: False)
        assert client.delete(path).status_code == 404
        def unavailable(*args): raise RuntimeError("数据库暂时不可用")
        monkeypatch.setattr(nutrition,repository,unavailable)
        assert client.delete(path).status_code == 503
        assert client.delete(path.rsplit("/",1)[0]+"/0").status_code == 422

def test_new_food_validation_and_save_response(monkeypatch):
    app=FastAPI();app.include_router(nutrition.router)
    calls=[]
    monkeypatch.setattr(nutrition,"upsert_custom_food",lambda *args,**kw: calls.append((args,kw)) or {"id":8,"name":"合成食物"})
    payload={"user_id":"alice","food_name":"合成食物","nutrition_per_100g":{"kcal":120,"protein_g":10,"carbs_g":15,"fat_g":2}}
    with TestClient(app) as client:
        assert client.post("/nutrition/custom-foods",json=payload).json()["food"]["id"] == 8
        payload["nutrition_per_100g"]["kcal"]=-1
        assert client.post("/nutrition/custom-foods",json=payload).status_code == 422
        assert len(calls)==1
