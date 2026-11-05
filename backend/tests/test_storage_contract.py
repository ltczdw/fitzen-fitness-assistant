import pytest
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker, Session
from app.db import session as db
from app.db.models import WorkoutRecordModel, NutritionLogModel
from app.db.workout_session_models import WorkoutSessionModel, WorkoutExerciseModel, WorkoutSetModel
from app.db.repositories import create_workout
from app.db.nutrition_repository import create_nutrition
from app.db.storage import StorageUnavailableError, CommitUnknownError

@pytest.fixture
def sql_store(tmp_path, monkeypatch):
    engine = create_engine("sqlite:///" + str(tmp_path / "storage.db"))
    for model in (WorkoutRecordModel, NutritionLogModel, WorkoutSessionModel, WorkoutExerciseModel, WorkoutSetModel):
        model.__table__.create(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "SessionLocal", factory)
    monkeypatch.setattr(db, "DB_READY", True)
    yield factory
    engine.dispose()

def write(key="same-key"):
    return create_workout("fixture", "squat", 3, 8, 40, idempotency_key=key)

def test_same_key_returns_same_record_and_session(sql_store):
    first, second = write(), write()
    assert first["id"] == second["id"]
    assert first["session_id"] == second["session_id"]
    with sql_store() as s:
        assert s.scalar(select(func.count()).select_from(WorkoutRecordModel)) == 1
        assert s.scalar(select(func.count()).select_from(WorkoutSessionModel)) == 1

def test_refresh_failure_recovers_committed_record(sql_store, monkeypatch):
    monkeypatch.setattr(Session, "refresh", lambda *a, **k: (_ for _ in ()).throw(OSError("refresh disconnected")))
    result = write()
    assert result["id"] and result["session_id"]

def test_commit_response_lost_is_verified(sql_store, monkeypatch):
    commit = Session.commit
    def lost(self):
        commit(self)
        raise OSError("commit response lost")
    monkeypatch.setattr(Session, "commit", lost)
    assert write()["id"]

def test_commit_without_persisted_row_stays_unknown(sql_store, monkeypatch):
    monkeypatch.setattr(Session, "commit", lambda *a: (_ for _ in ()).throw(OSError("commit lost")))
    with pytest.raises(CommitUnknownError): write()

def test_no_key_does_not_guess_success(sql_store, monkeypatch):
    monkeypatch.setattr(Session, "refresh", lambda *a, **k: (_ for _ in ()).throw(OSError("refresh lost")))
    with pytest.raises(CommitUnknownError): write(None)
    with pytest.raises(CommitUnknownError): create_nutrition("fixture", "lunch", "eggs")

def test_failure_before_commit_is_typed_and_rolls_back(sql_store, monkeypatch):
    monkeypatch.setattr("app.db.workout_session_repository._replace_exercises", lambda *a: (_ for _ in ()).throw(RuntimeError("invalid child")))
    with pytest.raises(StorageUnavailableError): write()
    with sql_store() as s:
        assert s.scalar(select(func.count()).select_from(WorkoutRecordModel)) == 0
        assert s.scalar(select(func.count()).select_from(WorkoutSessionModel)) == 0


@pytest.mark.parametrize("operation", [
    lambda: write(),
    lambda: create_nutrition("fixture", "lunch", "eggs"),
    lambda: __import__("app.db.food_repository", fromlist=["log_food_items"]).log_food_items("fixture", [{}], {}, "meal", "key"),
    lambda: __import__("app.db.repositories", fromlist=["list_workouts"]).list_workouts(),
    lambda: __import__("app.db.nutrition_repository", fromlist=["list_nutrition"]).list_nutrition(),
    lambda: __import__("app.db.food_log_listing", fromlist=["list_food_logs"]).list_food_logs(),
    lambda: __import__("app.db.food_summary", fromlist=["food_nutrition_summary"]).food_nutrition_summary(),
])
def test_unavailable_never_reads_or_writes_json(operation, monkeypatch):
    from pathlib import Path
    def forbidden(*args, **kwargs):
        raise AssertionError("Online operation accessed a JSON file")
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    with pytest.raises(StorageUnavailableError): operation()


def test_read_api_returns_503_not_json():
    import asyncio, httpx
    from app.main import app
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            for path in ("/api/workouts", "/api/nutrition", "/api/workout-sessions"):
                response = await client.get(path)
                assert response.status_code == 503, (path, response.text)
                assert response.json()["code"] == "storage_unavailable"
    asyncio.run(check())


def test_remote_dispatched_write_never_becomes_local_fallback(monkeypatch):
    import asyncio
    from app.services.mcp.client import MCPClientBridge
    from app.core.config import settings
    class Remote:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def call_tool(self, *args): raise TimeoutError("response lost")
    bridge = MCPClientBridge()
    monkeypatch.setattr(settings, "mcp_url", "http://test")
    monkeypatch.setattr(bridge, "_new_client", lambda: Remote())
    with pytest.raises(CommitUnknownError):
        asyncio.run(bridge.call_tool("save_workout_record", {}))


@pytest.mark.parametrize("found", [[21, 22], None])
def test_food_commit_recovery_returns_only_confirmed_batch(sql_store, monkeypatch, found):
    from app.db import food_repository as food
    monkeypatch.setattr(food, "_insert_food_items", lambda *a: (_ for _ in ()).throw(CommitUnknownError("lost")))
    monkeypatch.setattr(food, "_find_food_items", lambda *a: found)
    if found is None:
        with pytest.raises(CommitUnknownError):
            food.log_food_items("fixture", [{}, {}], {}, "meal", "key")
    else:
        assert food.log_food_items("fixture", [{}, {}], {}, "meal", "key") == found


def test_commit_verification_connection_failure_stays_unknown(sql_store, monkeypatch):
    from app.db.storage import recover_committed
    with pytest.raises(CommitUnknownError):
        recover_committed(lambda: (_ for _ in ()).throw(OSError("database down")), CommitUnknownError("lost"))
