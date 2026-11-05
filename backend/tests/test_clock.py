from datetime import datetime, date, timezone
import pytest
from app.core import clock
from app.core.config import settings
from app.services.agent.workout_extractor import _extract_record_date
from app.services.agent.food_parser import parse_food_message
from app.services.agent.multi_agent_graph import _record_time_from_entries

@pytest.fixture
def fixed_clock(monkeypatch):
    monkeypatch.setattr(settings, "app_timezone", "Asia/Shanghai")
    monkeypatch.setattr(clock, "utc_now", lambda: datetime(2026,9,26,16,30,tzinfo=timezone.utc))

@pytest.mark.parametrize("word,expected", [("昨天",26),("昨晚",26),("今天",27),("明天",28)])
def test_shared_date_basis(fixed_clock,word,expected):
    assert _extract_record_date(word) == date(2026,9,expected)
    entries = parse_food_message(word+"记录两个鸡蛋")
    assert entries and entries[0].record_date == date(2026,9,expected).isoformat()
    assert _record_time_from_entries(entries).day == expected

def test_utc_midnight_and_configured_zone(fixed_clock,monkeypatch):
    assert clock.today() == date(2026,9,27)
    monkeypatch.setattr(settings,"app_timezone","UTC")
    assert clock.today() == date(2026,9,26)
    monkeypatch.setattr(clock,"utc_now",lambda:datetime(2026,9,27,0,30,tzinfo=timezone.utc))
    assert clock.today() == date(2026,9,27)

def test_weekdays_and_invalid_dates(fixed_clock):
    assert clock.parse_date("下周一") == date(2026,9,28)
    assert clock.parse_date("周一") == date(2026,9,21)
    with pytest.raises(ValueError): clock.parse_date("2026-02-30")
    with pytest.raises(ValueError): clock.parse_date("昨天早餐鸡蛋，今天晚餐米饭")


def test_unrecognized_or_multiple_entry_dates_require_clarification(fixed_clock):
    from dataclasses import replace
    entry = parse_food_message("今天记录两个鸡蛋")[0]
    with pytest.raises(ValueError):
        _record_time_from_entries([replace(entry, record_date="invalid")])
    with pytest.raises(ValueError):
        _record_time_from_entries([entry, replace(entry, record_date="2026-09-26")])
