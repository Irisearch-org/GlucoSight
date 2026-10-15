from __future__ import annotations

from typing import Sequence

import pandas as pd


DEFAULT_SCHEDULE = ((7, 0), (12, 0), (20, 0))


def sparse_history(
    readings: pd.DataFrame,
    event_time,
    schedule: Sequence[tuple[int, int]] = DEFAULT_SCHEDULE,
    tolerance_minutes: int = 15,
) -> pd.DataFrame:
    """Select the nearest earlier Dexcom observation at fixed daily times.

    This deterministic schedule approximates waking/pre-meal/bedtime
    fingersticks. It is a missing-at-random simulation, not observed
    fingerstick behavior, and every selected value must precede the event.
    """
    if tolerance_minutes < 0:
        raise ValueError("tolerance_minutes must be non-negative")
    event_ts = pd.Timestamp(event_time)
    if event_ts.tzinfo is not None:
        raise ValueError("event timestamps must use one consistent timezone")

    frame = readings.loc[:, ["Timestamp", "Dexcom GL"]].copy()
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    frame["Dexcom GL"] = pd.to_numeric(frame["Dexcom GL"], errors="coerce")
    frame = frame.dropna(subset=["Timestamp", "Dexcom GL"])
    frame = frame.loc[frame["Timestamp"] < event_ts].sort_values("Timestamp")

    selected = []
    used_timestamps = set()
    selection_counts = {}
    for day in sorted(frame["Timestamp"].dt.normalize().unique()):
        day = pd.Timestamp(day)
        day_readings = frame.loc[
            (frame["Timestamp"] >= day)
            & (frame["Timestamp"] < day + pd.Timedelta(days=1))
        ]
        for hour, minute in schedule:
            if selection_counts.get(day, 0) >= len(schedule):
                break
            scheduled = day + pd.Timedelta(hours=hour, minutes=minute)
            candidates = day_readings.loc[day_readings["Timestamp"] <= scheduled]
            if candidates.empty:
                continue
            candidate = candidates.iloc[-1]
            age = (scheduled - candidate["Timestamp"]).total_seconds() / 60.0
            key = candidate["Timestamp"]
            if 0 <= age <= tolerance_minutes and key < event_ts and key not in used_timestamps:
                selected.append(candidate)
                used_timestamps.add(key)
                selection_counts[day] = selection_counts.get(day, 0) + 1

    if not selected:
        return frame.iloc[0:0].copy()
    result = pd.DataFrame(selected).sort_values("Timestamp").reset_index(drop=True)
    if not result.empty and not (result["Timestamp"] < event_ts).all():
        raise AssertionError("sparsification admitted a post-event observation")
    return result
