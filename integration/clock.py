"""Timestamp handling shared by the stages.

The forecaster's hour-of-day features come from the meal's local clock time (the training
meals carry CGMacros local timestamps). The prototype's users are Indonesian, so every
timezone-aware timestamp is converted to Asia/Jakarta (WIB) and a naive timestamp is taken as
WIB already. The browser sends `toISOString()` (UTC), hence the conversion.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from integration.errors import InvalidInput

LOCAL_TZ = ZoneInfo("Asia/Jakarta")


def parse_timestamp(value: Any) -> datetime:
    """ISO-8601 string (or datetime) -> timezone-aware datetime in WIB."""
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise InvalidInput("Format waktu tidak valid (gunakan ISO 8601).") from exc
    if dt.tzinfo is None:
        return dt.replace(tzinfo=LOCAL_TZ)
    return dt.astimezone(LOCAL_TZ)


def minutes_between(earlier: datetime, later: datetime) -> float:
    return (later - earlier).total_seconds() / 60.0
