"""Library edits use isolated SQLite and API mocks only."""
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.db import session as db
from app.db import food_repository as catalog
from app.db.food_library_mutations import update_library_food
from app.api import nutrition

VALUES={"kcal":180,"protein_g":12,"carbs_g":18,"fat_g":5,"fiber_g":2}

@pytest.fixture
def library(tmp_path,monkeypatch):
    engine=create_engine("sqlite:///"+str(tmp_path/"edit.db"))
    monkeypatch.setattr(db,"DB_READY",True)
    monkeypatch.setattr(db,"engine",engine)
    monkeypatch.setattr(db,"SessionLocal",sessionmaker(bind=engine))
    with engine.begin() as c:
        c.execute(text("CREATE TABLE food_nutrition(id INTEGER PRIMARY KEY,food_name TEXT,category TEXT,preparation_state TEXT,energy_kcal REAL,protein_g REAL,carb_g REAL,fat_g REAL,fiber_g REAL,is_active BOOLEAN DEFAULT TRUE,UNIQUE(food_name,preparation_state))"))
        c.execute(text("CREATE TABLE food_alias_index(food_id INTEGER,alias_text TEXT,normalized_alias TEXT,alias_type TEXT,is_active BOOLEAN DEFAULT TRUE,UNIQUE(food_id,normalized_alias))"))
        c.execute(text("CREATE TABLE user_custom_food(id INTEGER PRIMARY KEY,user_id TEXT,food_name TEXT,energy_kcal REAL,protein_g REAL,carb_g REAL,fat_g REAL,fiber_g REAL,use_count INTEGER,last_used_at TEXT,updated_at TEXT,is_active BOOLEAN DEFAULT TRUE,UNIQUE(user_id,food_name))"))
        c.execute(text("CREATE TABLE user_food_log(food_id INTEGER,food_name TEXT,total_kcal REAL)"))
        c.execute(text("INSERT INTO food_nutrition VALUES(1,'合成米饭','主食','熟',116,2.6,25.9,.3,.3,TRUE),(2,'合成燕麦','主食','熟',120,4,20,2,1,TRUE)"))
        c.execute(text("INSERT INTO user_custom_food VALUES(1,'alice','合成沙拉',130,10,12,3,1,9,'2026-10-18',NULL,TRUE),(2,'alice','合成汤',30,1,4,1,0,1,NULL,NULL,TRUE),(3,'bob','合成沙拉',140,10,14,4,1,3,NULL,NULL,TRUE)"))
        c.execute(text("INSERT INTO user_food_log VALUES(1,'合成米饭',116)"))
    yield engine
    engine.dispose()

def test_builtin_edit_same_id_preserves_history_and_survives_seed(library,monkeypatch):
    food=update_library_food(1,"改名米饭",VALUES,source="local")
    assert food["id"]==1 and food["name"]=="改名米饭" and food["kcal"]==180 and food["category"]=="主食"
    monkeypatch.setattr(catalog,"SEED_FOODS",[("合成米饭","主食","熟",116,2.6,25.9,.3,.3)])
    monkeypatch.setattr(catalog,"SEED_ALIASES",{})
    monkeypatch.setattr(catalog,"SEED_PORTIONS",{})
    with library.begin() as c:
        catalog._seed_catalog(c)
        assert c.execute(text("SELECT COUNT(*) FROM food_nutrition")).scalar()==2
        assert c.execute(text("SELECT food_name,total_kcal FROM user_food_log")).one()==("合成米饭",116)
    assert update_library_food(999,"未知",VALUES,source="local") is None

def test_custom_edit_is_scoped_and_keeps_usage(library):
    assert update_library_food(1,"越权",VALUES,source="custom",user_id="bob") is None
    food=update_library_food(1,"新沙拉",VALUES,source="custom",user_id="alice")
    assert food["id"]==1 and food["name"]=="新沙拉" and food["usage_count"]==9
    with library.begin() as c:
        assert c.execute(text("SELECT food_name FROM user_custom_food WHERE id=3")).scalar()=="合成沙拉"
        assert c.execute(text("SELECT food_name FROM food_nutrition WHERE id=1")).scalar()=="合成米饭"
        c.execute(text("UPDATE user_custom_food SET is_active=FALSE WHERE id=1"))
    assert update_library_food(1,"不能复活",VALUES,source="custom",user_id="alice") is None

@pytest.mark.parametrize("source,name",[("local","合成燕麦"),("custom","合成汤")])
def test_duplicate_rename_rolls_back(library,source,name):
    with pytest.raises(ValueError,match="同名"):
        update_library_food(1,name,VALUES,source=source,user_id="alice")
    with library.connect() as c:
        table="food_nutrition" if source=="local" else "user_custom_food"
        assert c.execute(text(f"SELECT energy_kcal FROM {table} WHERE id=1")).scalar() in (116,130)
        assert c.execute(text("SELECT COUNT(*) FROM food_alias_index")).scalar()==0

def test_unavailable_edit_fails_closed():
    with pytest.raises(RuntimeError):update_library_food(1,"合成",VALUES,source="local")

@pytest.mark.parametrize("source,path",[("local","/nutrition/food/catalog/1"),("custom","/nutrition/custom-foods/1")])
def test_api_updates_validated_target_and_reports_errors(monkeypatch,source,path):
    app=FastAPI();app.include_router(nutrition.router)
    calls=[]
    def save(*args,**kw):calls.append((args,kw));return {"id":1,"name":"合成修改"}
    monkeypatch.setattr(nutrition,"update_library_food",save)
    payload={"user_id":"alice","food_name":"合成修改","nutrition_per_100g":dict(VALUES)}
    with TestClient(app) as client:
        response=client.put(path,json=payload)
        assert response.status_code==200 and response.json()["saved"] is True
        assert calls[0][0]==(1,"合成修改",VALUES)
        assert calls[0][1]=={"source":source,"user_id":"alice"}
        payload["food_name"]=" "
        assert client.put(path,json=payload).status_code==422
        payload["food_name"]="合成修改";payload["nutrition_per_100g"]["kcal"]=-1
        assert client.put(path,json=payload).status_code==422
        assert len(calls)==1
        payload["nutrition_per_100g"]=VALUES
        monkeypatch.setattr(nutrition,"update_library_food",lambda *a,**kw:None)
        assert client.put(path,json=payload).status_code==404
        def conflict(*a,**kw):raise ValueError("已存在同名食物")
        monkeypatch.setattr(nutrition,"update_library_food",conflict)
        assert client.put(path,json=payload).status_code==409
        def unavailable(*a,**kw):raise RuntimeError("数据库暂时不可用")
        monkeypatch.setattr(nutrition,"update_library_food",unavailable)
        assert client.put(path,json=payload).status_code==503


@pytest.mark.skipif(__import__("os").environ.get("FITZEN_DB_TESTS")!="1",reason="explicit isolated PostgreSQL opt-in required")
def test_postgres_edit_and_seed_use_temporary_tables(monkeypatch):
    from app.core.config import settings
    engine=create_engine(settings.database_url)
    assert engine.dialect.name=="postgresql"
    try:
        with engine.connect() as connection:
            transaction=connection.begin()
            try:
                # Names shadow public tables only on this connection. Rollback removes everything.
                connection.execute(text("CREATE TEMP TABLE food_nutrition(id BIGSERIAL PRIMARY KEY,food_name VARCHAR(200),category VARCHAR(100),preparation_state VARCHAR(50),energy_kcal NUMERIC(10,2),protein_g NUMERIC(10,2),carb_g NUMERIC(10,2),fat_g NUMERIC(10,2),fiber_g NUMERIC(10,2),is_active BOOLEAN DEFAULT TRUE,UNIQUE(food_name,preparation_state)) ON COMMIT DROP"))
                connection.execute(text("CREATE TEMP TABLE food_alias_index(food_id BIGINT,alias_text VARCHAR(200),normalized_alias VARCHAR(200),alias_type VARCHAR(50),UNIQUE(food_id,normalized_alias)) ON COMMIT DROP"))
                connection.execute(text("CREATE TEMP TABLE user_custom_food(id BIGSERIAL PRIMARY KEY,user_id VARCHAR(100),food_name VARCHAR(200),energy_kcal NUMERIC(10,2),protein_g NUMERIC(10,2),carb_g NUMERIC(10,2),fat_g NUMERIC(10,2),fiber_g NUMERIC(10,2),use_count INTEGER DEFAULT 0,last_used_at TIMESTAMPTZ,updated_at TIMESTAMPTZ,is_active BOOLEAN DEFAULT TRUE,UNIQUE(user_id,food_name)) ON COMMIT DROP"))
                monkeypatch.setattr(db,"DB_READY",True)
                monkeypatch.setattr(db,"engine",engine)
                monkeypatch.setattr(db,"SessionLocal",sessionmaker(bind=connection))
                monkeypatch.setattr(catalog,"SEED_FOODS",[("合成米饭","主食","熟",116,2.6,25.9,.3,.3)])
                monkeypatch.setattr(catalog,"SEED_ALIASES",{})
                monkeypatch.setattr(catalog,"SEED_PORTIONS",{})
                catalog._seed_catalog(connection)
                updated=update_library_food(1,"合成改名",VALUES,source="local")
                assert updated["id"]==1 and updated["kcal"]==180
                catalog._seed_catalog(connection)
                assert connection.execute(text("SELECT COUNT(*) FROM food_nutrition")).scalar()==1
                connection.execute(text("INSERT INTO user_custom_food(user_id,food_name) VALUES('alice','合成沙拉')"))
                personal=update_library_food(1,"合成新沙拉",VALUES,source="custom",user_id="alice")
                assert personal["name"]=="合成新沙拉" and personal["protein"]==12
            finally:
                transaction.rollback()
    finally:
        engine.dispose()
