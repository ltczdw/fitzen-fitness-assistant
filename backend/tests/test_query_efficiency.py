"""Query growth and result regressions, using isolated SQLite / opt-in PostgreSQL."""
import asyncio
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import event, insert, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.core import clock
from app.db import session as db
from app.db import nutrition_repository as nutrition
from app.db import workout_session_repository as workouts
from app.db import food_log_mutations as meals
from app.db import exercise_library_repository as library
from app.db.food_log_listing import query_food_catalog
from app.db.models import NutritionLogModel, UserProfileModel, WorkoutRecordModel
from app.db.storage import BusinessValidationError, StorageUnavailableError, WorkoutAssociationError
from app.db.workout_session_models import WorkoutSessionModel, WorkoutExerciseModel, WorkoutSetModel, UserCustomExerciseModel
from tests.react_phase_c1_support import isolated
from tests.react_phase_c2_2_support import ensure_catalog

TODAY = date(2026, 10, 26)
NOW = datetime(2026, 10, 26, 12, 0, 0, 123456, tzinfo=timezone.utc)


@pytest.fixture(params=["sqlite", "postgres"])
def data(request, tmp_path, monkeypatch):
    with isolated(monkeypatch, tmp_path / "query-cost", request.param) as engine:
        ensure_catalog(engine)
        queries, fetched = [], []

        class MappingRows:
            def __init__(self, result):
                self.result = result

            def all(self):
                rows = self.result.all()
                fetched.append(len(rows))
                return rows

            def __getattr__(self, name):
                return getattr(self.result, name)

        class ReadResult:
            def __init__(self, result):
                self.result = result

            def mappings(self):
                return MappingRows(self.result.mappings())

            def __getattr__(self, name):
                return getattr(self.result, name)

        class CountingSession(Session):
            def execute(self, *args, **kwargs):
                result = super().execute(*args, **kwargs)
                return ReadResult(result) if getattr(result, "returns_rows", True) else result

        def capture(conn, cursor, statement, params, context, executemany):
            queries.append((statement, executemany))

        factory = sessionmaker(bind=engine, class_=CountingSession, expire_on_commit=False)
        monkeypatch.setattr(db, "SessionLocal", factory)
        monkeypatch.setattr(clock, "today", lambda: TODAY)
        monkeypatch.setattr(clock, "utc_now", lambda: NOW)
        monkeypatch.setattr(library, "date", SimpleNamespace(today=lambda: TODAY))
        with factory.begin() as session:
            session.add(UserProfileModel(user_id="owner", body_weight_kg=70))
            session.add(UserProfileModel(user_id="other", body_weight_kg=90))
        event.listen(engine, "before_cursor_execute", capture)
        yield SimpleNamespace(engine=engine, factory=factory, queries=queries, fetched=fetched)


@pytest.mark.parametrize("count", [1, 30])
def test_session_list_has_constant_query_count(data, count):
    with data.factory.begin() as session:
        for index in range(count):
            row = WorkoutSessionModel(user_id="owner", session_date=TODAY, title=f"session-{index}")
            session.add(row)
            session.flush()
            exercise = WorkoutExerciseModel(session_id=row.id, exercise_name="bench")
            session.add(exercise)
            session.flush()
            session.add_all([WorkoutSetModel(exercise_id=exercise.id, set_index=i, reps=i + 5) for i in (1, 0)])
        session.add(WorkoutSessionModel(user_id="other", session_date=TODAY, title="private"))
    data.queries.clear()
    rows = workouts.list_sessions("owner", limit=count)
    assert len(rows) == count
    assert [row["title"] for row in rows] == [f"session-{i}" for i in reversed(range(count))]
    assert all([item["reps"] for item in row["exercises"][0]["sets"]] == [5, 6] for row in rows)
    assert len(data.queries) <= 3
    print({"sessions": count, "selects": len(data.queries)})


@pytest.mark.parametrize("all_today", [False, True])
def test_nutrition_aggregates_without_loading_or_truncating_logs(data, monkeypatch, all_today):
    values = [{"user_id": "owner", "log_date": TODAY if all_today else TODAY - timedelta(days=60),
               "description": "synthetic", "calories": 1, "protein_g": None} for _ in range(1005)]
    if not all_today:
        values.append({"user_id": "owner", "log_date": TODAY, "description": "today", "calories": 1})
    values.append({"user_id": "other", "log_date": TODAY, "description": "private", "calories": 999})
    with data.factory.begin() as session:
        session.execute(insert(NutritionLogModel), values)
    monkeypatch.setattr(nutrition, "list_nutrition", lambda *a, **k: pytest.fail("summary must not load individual logs"))
    data.queries.clear()
    result = nutrition.nutrition_summary("owner", 1)
    assert result["entries"] == result["calories"] == (1005 if all_today else 1)
    assert result["protein_g"] == 0 and result["calories_target"] == 2100
    assert len(data.queries) == 2  # aggregate + profile, regardless of history size
    assert "sum(" in data.queries[0][0].lower() and "limit" not in data.queries[0][0].lower()


def test_relation_validation_reuses_the_record_and_keeps_ownership_check(data):
    with data.factory.begin() as session:
        record = WorkoutRecordModel(user_id="owner", workout_date=TODAY, exercise="bench", sets=1, reps=8, weight_kg=60)
        session.add(record)
        session.flush()
        row = WorkoutSessionModel(user_id="owner", source_record_id=record.id, session_date=TODAY)
        session.add(row)
        session.flush()
        session.add(WorkoutExerciseModel(session_id=row.id, source_record_id=record.id, exercise_name="bench"))
        record_id, session_id = record.id, row.id
    with data.factory() as session:
        row = session.get(WorkoutSessionModel, session_id)
        data.queries.clear()
        assert workouts._session_relation_record(session, row, "owner").id == record_id
        assert len(data.queries) == 2
        with pytest.raises(WorkoutAssociationError):
            workouts._session_relation_record(session, row, "other")


def test_context_profile_is_reused_only_within_one_context_load(data, monkeypatch):
    from app.db import nutrition_sync
    from app.services.agent.multi_agent_graph import FitnessMultiAgentGraph
    from app.services.mcp.tools import FitnessMCPTools
    monkeypatch.setattr(nutrition_sync, "food_nutrition_summary", lambda *a, **k: {})
    tools = FitnessMCPTools()

    async def call(name, args):
        return getattr(tools, name)(**args)

    data.queries.clear()
    context = asyncio.run(FitnessMultiAgentGraph._load_context_node(
        SimpleNamespace(_call_mcp=call), {"user_id": "owner", "raw_message": "分析我的训练和营养"}))
    profile_reads = [sql for sql, _ in data.queries if "from user_profiles" in sql.lower()]
    assert len(profile_reads) == 1
    assert context["nutrition_summary"]["calories_target"] == 2100
    with data.factory.begin() as session:
        session.get(UserProfileModel, "owner").body_weight_kg = 80
    assert nutrition.nutrition_summary("owner", 1)["calories_target"] == 2400
    with nutrition.nutrition_profile_context("owner", context["profile"]):
        assert nutrition.nutrition_summary("other", 1)["calories_target"] == 2700


def _food_log(data, count=1, *, owner="owner", group="meal", when=NOW, food_id=None):
    with data.engine.begin() as connection:
        connection.execute(text("""INSERT INTO user_food_log
            (user_id,meal_group_id,record_time,meal_type,food_id,raw_food_name,food_name,weight_g,
             total_kcal,total_protein,total_carb,total_fat,total_fiber,is_estimated,
             weight_source,match_method,idempotency_key,calculation_version)
            VALUES (:owner,:group,:when,'午餐',:food_id,'synthetic','synthetic',100,
                    100,10,10,1,0,FALSE,'user_input','user_custom',:key,'fixture')"""),
            [{"owner": owner, "group": group,
              "when": when.isoformat() if data.engine.dialect.name == "sqlite" and isinstance(when, datetime) else when,
              "food_id": food_id, "key": f"{group}-{owner}-{index}"} for index in range(count)])


def test_food_catalog_limits_in_sql_with_exact_timestamp_and_user_scope(data):
    with data.engine.connect() as connection:
        food_id = connection.execute(text("SELECT id FROM food_nutrition WHERE food_name='鸡蛋'")).scalar_one()
    cutoff = NOW - timedelta(days=90)
    _food_log(data, 500, food_id=food_id)
    _food_log(data, group="before", when=cutoff - timedelta(microseconds=1), food_id=food_id)
    _food_log(data, group="equal", when=cutoff, food_id=food_id)
    _food_log(data, group="offset", when=cutoff.astimezone(timezone(timedelta(hours=8))).isoformat(), food_id=food_id)
    _food_log(data, 7, owner="other", food_id=food_id)
    data.queries.clear()
    data.fetched.clear()
    with data.factory() as session:
        result = query_food_catalog(session, "owner", cutoff, 1)
    assert len(result) == 1 and result[0]["id"] == food_id and result[0]["usage_count"] == 502
    assert data.fetched == [1] and len(data.queries) == 1
    assert "LIMIT" in data.queries[0][0]
    print({"food_logs": 510, "catalog_rows_fetched": data.fetched[0]})


def test_exercise_search_and_ranking_fetch_only_top_rows(data):
    with data.factory.begin() as session:
        session.execute(insert(UserCustomExerciseModel), [
            {"user_id": "owner", "exercise_name": f"exercise-{i:04}"} for i in range(1000)])
        session.execute(insert(WorkoutRecordModel), [{"user_id": "owner", "workout_date": TODAY,
            "exercise": "exercise-0999", "sets": 1, "reps": 8, "weight_kg": 60} for _ in range(3)])
        session.add(UserCustomExerciseModel(user_id="other", exercise_name="private-exercise"))
    data.queries.clear()
    data.fetched.clear()
    assert [row["name"] for row in library.list_exercises("owner", "exercise-0999", 1)] == ["exercise-0999"]
    assert data.fetched == [1] and len(data.queries) == 1
    data.fetched.clear()
    first = library.list_exercises("owner", limit=1)[0]
    assert first["name"] == "exercise-0999" and first["usage_count"] == first["recent_use_count"] == 3
    assert data.fetched == [1]
    assert library.list_exercises("owner", "private-exercise") == []


def test_exercise_unicode_literal_search_and_default_override(data):
    with data.factory.begin() as session:
        session.add_all([UserCustomExerciseModel(user_id="owner", exercise_name=name, category="", target_muscles="")
                         for name in ("Straße", "İncline", "100%_effort", "杠铃卧推")])
    assert library.list_exercises("owner", "STRASSE")[0]["name"] == "Straße"
    assert library.list_exercises("owner", "i\u0307ncline")[0]["name"] == "İncline"
    assert [row["name"] for row in library.list_exercises("owner", "%_")] == ["100%_effort"]
    overridden = [row for row in library.list_exercises("owner", "杠铃卧推") if row["name"] == "杠铃卧推"]
    assert len(overridden) == 1 and overridden[0]["source"] == "custom" and overridden[0]["category"] == "胸部"


def test_meal_edits_batch_writes_reuse_rows_and_preserve_other_users(data, monkeypatch):
    _food_log(data, 20)
    _food_log(data, owner="other")
    with data.engine.connect() as connection:
        ids = list(connection.execute(text("SELECT id FROM user_food_log WHERE user_id='owner' ORDER BY id")).scalars())
    monkeypatch.setattr(meals, "resolve_food", lambda *a: pytest.fail("unchanged names already have nutrition data"))
    items = [{"id": i, "food_name": "synthetic", "weight_g": 200} for i in ids[:10]]
    items += [{"food_name": "new", "weight_g": 100, "source": "custom", "nutrition_per_100g": {"kcal": 50}} for _ in range(5)]
    data.queries.clear()
    result = meals.update_food_meal("owner", "meal", items=items)
    assert result["updated"] == 15 and result["deleted"] == 10
    assert len(data.queries) == 4  # read, delete IN, executemany update, executemany insert
    assert sum(many for _, many in data.queries) == 2
    with data.engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*),SUM(total_kcal) FROM user_food_log WHERE user_id='owner'")).one() == (15, 2250)
        assert connection.execute(text("SELECT COUNT(*),SUM(total_kcal) FROM user_food_log WHERE user_id='other'")).one() == (1, 100)
        assert connection.execute(text("SELECT COUNT(DISTINCT idempotency_key) FROM user_food_log WHERE user_id='owner'")).scalar_one() == 15


def test_meal_batch_failure_rolls_back_deletes_and_updates(data):
    _food_log(data, 2)
    with data.engine.connect() as connection:
        ids = list(connection.execute(text("SELECT id FROM user_food_log ORDER BY id")).scalars())
    items = [{"id": ids[0], "weight_g": 200}, {"food_name": "new", "weight_g": 100,
             "nutrition_per_100g": {"kcal": 50}}]

    def fail(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("INSERT INTO USER_FOOD_LOG"):
            raise RuntimeError("synthetic insert failure after delete/update")

    event.listen(data.engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(StorageUnavailableError):
            meals.update_food_meal("owner", "meal", items=items)
    finally:
        event.remove(data.engine, "before_cursor_execute", fail)
    with data.engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*),SUM(weight_g),SUM(total_kcal) FROM user_food_log")).one() == (2, 200, 200)
    _food_log(data, owner="other")
    data.queries.clear()
    with pytest.raises(BusinessValidationError):
        meals.update_food_meal("other", "meal", items=[{"id": ids[0], "weight_g": 200}])
    assert not any(sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")) for sql, _ in data.queries)


def test_meal_repeated_new_name_is_resolved_once(data, monkeypatch):
    from app.db.food_repository import resolve_food
    with data.engine.connect() as connection:
        food_id = connection.execute(text("SELECT id FROM food_nutrition WHERE food_name='鸡蛋'")).scalar_one()
    _food_log(data, 2, food_id=food_id)
    calls = []

    def lookup(name):
        calls.append(name)
        return resolve_food(name)

    monkeypatch.setattr(meals, "resolve_food", lookup)
    with data.engine.connect() as connection:
        ids = list(connection.execute(text("SELECT id FROM user_food_log ORDER BY id")).scalars())
    result = meals.update_food_meal("owner", "meal", items=[
        *[{"id": i, "food_name": "熟大米", "weight_g": 100} for i in ids],
        {"food_name": "熟大米", "weight_g": 100}])
    assert result["updated"] == 3 and calls == ["熟大米"]
