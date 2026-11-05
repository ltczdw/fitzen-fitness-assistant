"""One application clock. Database server timestamps remain database-owned."""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import re
from app.core.config import settings


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def now() -> datetime:
    return utc_now().astimezone(ZoneInfo(settings.app_timezone))


def today() -> date:
    return now().date()


def parse_date(text: str, *, base_date: date | None = None) -> date | None:
    """Resolve a single date; bare weekday means the current configured week."""
    dates = set()
    for match in re.finditer(r"(20\d{2})[年/\-](\d{1,2})[月/\-](\d{1,2})", text):
        dates.add(date(*(int(v) for v in match.groups())))
    offsets = {"前天": -2, "昨天": -1, "昨日": -1, "昨晚": -1, "今天": 0, "今日": 0, "今晚": 0, "明天": 1, "明日": 1, "明晚": 1, "翌日": 1, "后天": 2}
    base = base_date if base_date is not None else today()
    for word, offset in offsets.items():
        if word in text:
            dates.add(base + timedelta(days=offset))
    for match in re.finditer(r"(上|下|本|这)?(?:周|星期)([一二三四五六日天])", text):
        week, day = match.groups()
        weekday = "一二三四五六日".index("日" if day == "天" else day)
        dates.add(base - timedelta(days=base.weekday()) + timedelta(days=weekday + {"上": -7, "下": 7}.get(week, 0)))
    if len(dates) > 1:
        raise ValueError("Multiple or contradictory dates require separate records")
    return next(iter(dates), None)


def record_time(value: str | None) -> datetime | None:
    if not value:
        return None
    day = parse_date(value)
    if day is None:
        raise ValueError("Unrecognized record date")
    current = now()
    return datetime.combine(day, current.timetz())
