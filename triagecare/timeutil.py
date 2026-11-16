"""Time helpers. The database stores naive UTC; people see local time."""
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

UTC = timezone.utc


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@lru_cache
def _zone(name: str) -> ZoneInfo:
    return ZoneInfo(name)


def local_tz() -> ZoneInfo:
    from triagecare.config import get_settings

    return _zone(get_settings().timezone)


def to_local(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC).astimezone(local_tz())


def to_utc(local_dt: datetime) -> datetime:
    """Interpret a naive datetime as local time and convert to naive UTC."""
    return local_dt.replace(tzinfo=local_tz()).astimezone(UTC).replace(tzinfo=None)


def local_today() -> date:
    return datetime.now(local_tz()).date()


def local_day_bounds_utc(day: date) -> tuple[datetime, datetime]:
    """UTC start/end of a local calendar day, for range queries."""
    start = to_utc(datetime.combine(day, time.min))
    return start, start + timedelta(days=1)


def minutes_since(dt: datetime | None, now: datetime | None = None) -> int:
    if dt is None:
        return 0
    return max(0, int(((now or utcnow()) - dt).total_seconds() // 60))


def humanize_minutes(minutes: int) -> str:
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min"
    hours, mins = divmod(minutes, 60)
    if hours < 24:
        return f"{hours} h {mins:02d} min"
    return f"{hours // 24} d {hours % 24} h"
