from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from app.db.food_log_listing import food_record_datetime, _meal_type
from app.api.dashboard import build_dashboard
from tests.react_phase_c1_support import isolated
from tests.react_phase_c2_2_support import ensure_catalog


def test_timestamp_representations_and_invalid_data():
    aware = datetime(2026, 10, 13, 17, tzinfo=timezone.utc)
    assert food_record_datetime(aware) is aware
    assert food_record_datetime("2026-10-13T17:00:00Z") == aware
    assert food_record_datetime("2026-10-13 17:00:00") == aware
    assert food_record_datetime(datetime(2026, 10, 13, 17)) == aware
    assert _meal_type("2026-10-14T04:00:00+00:00") == '午餐'
    with pytest.raises(ValueError):
        food_record_datetime('broken timestamp')


@pytest.mark.parametrize('backend', ['sqlite', 'postgres'])
def test_real_food_read_respects_local_day(monkeypatch, tmp_path, backend):
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings
    monkeypatch.setattr(settings, 'app_timezone', 'Asia/Shanghai')
    with isolated(monkeypatch, tmp_path / 'food-read', backend) as engine:
        ensure_catalog(engine)
        with engine.begin() as conn:
            # Both boundaries at full stored precision; equivalent offset
            # representation inside the day. No server-local timezone reliance.
            start = datetime(2026, 10, 13, 16, tzinfo=timezone.utc)
            end = start + timedelta(days=1)
            timestamps = (start-timedelta(microseconds=1), start,
                          '2026-10-14T01:00:00+08:00',
                          end-timedelta(microseconds=1), end)
            for n, timestamp in enumerate(timestamps):
                if backend=='sqlite' and isinstance(timestamp,datetime):
                    timestamp=timestamp.isoformat()
                conn.execute(text("""INSERT INTO user_food_log
                    (user_id,meal_group_id,record_time,meal_type,raw_food_name,food_name,weight_g,
                     total_kcal,total_protein,total_carb,total_fat,total_fiber,is_estimated,
                     weight_source,match_method,idempotency_key,calculation_version)
                    VALUES ('demo-user',:group,:timestamp,'午餐','synthetic','synthetic',100,
                            100,10,10,1,0,FALSE,'user_input','manual',:key,'fixture')"""),
                             {'group': f'timestamp-group-{n}', 'timestamp': timestamp, 'key': f'timestamp-key-{n}'})
        with sessionmaker(bind=engine)() as session:
            if backend=='postgres':
                session.execute(text("SET TIME ZONE 'America/Los_Angeles'"))
                sample=session.execute(text('SELECT record_time FROM user_food_log ORDER BY id LIMIT 1')).scalar_one()
                assert isinstance(sample,datetime) and sample.tzinfo is not None
            else:
                sample=session.execute(text('SELECT record_time FROM user_food_log ORDER BY id LIMIT 1')).scalar_one()
                assert isinstance(sample,str)
            result = build_dashboard(session, 'demo-user', 1, today=date(2026, 10, 14))
        assert {row['id'] for row in result['nutrition']} == {'timestamp-group-1','timestamp-group-2','timestamp-group-3'}
        assert all(row['date_key'] == '2026-10-14' for row in result['nutrition'])
        assert result['today']['meal_count'] == 3
        assert result['today']['calories'] == 300
