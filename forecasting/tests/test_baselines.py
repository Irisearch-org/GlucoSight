from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.data.sparsify import sparse_history
from forecasting.evaluate.baselines import evaluate_baselines


def test_sparse_history_uses_only_fixed_schedule_observations_before_event():
    times = pd.date_range("2025-01-01 06:45", "2025-01-01 12:15", freq="15min")
    readings = pd.DataFrame({"Timestamp": times, "Dexcom GL": np.arange(len(times), dtype=float)})
    event_time = pd.Timestamp("2025-01-01 12:00")
    selected = sparse_history(readings, event_time, tolerance_minutes=15)
    assert selected["Timestamp"].tolist() == [
        pd.Timestamp("2025-01-01 07:00"),
        pd.Timestamp("2025-01-01 12:00") - pd.Timedelta(minutes=15),
    ]
    assert (selected["Timestamp"] < event_time).all()
    for timestamp in selected["Timestamp"]:
        daily_minutes = timestamp.hour * 60 + timestamp.minute
        assert min(abs(daily_minutes - target) for target in (420, 720, 1200)) <= 15


def test_sparse_history_never_reuses_a_reading_and_caps_at_three_per_day():
    timestamps = pd.date_range("2025-01-01", periods=48 * 60, freq="min")
    readings = pd.DataFrame({"Timestamp": timestamps, "Dexcom GL": 100.0})
    selected = sparse_history(
        readings,
        pd.Timestamp("2025-01-03 23:00"),
        tolerance_minutes=0,
    )
    assert selected["Timestamp"].is_unique
    assert selected.groupby(selected["Timestamp"].dt.date).size().max() <= 3
    mutated = readings.copy()
    mutated.loc[mutated["Timestamp"] >= pd.Timestamp("2025-01-03 23:00"), "Dexcom GL"] = 999.0
    after_future_mutation = sparse_history(
        mutated, pd.Timestamp("2025-01-03 23:00"), tolerance_minutes=0
    )
    pd.testing.assert_frame_equal(selected, after_future_mutation)


def _evaluation_events():
    rows = []
    for participant in range(6):
        for event in range(4):
            g0 = 100.0 + participant + event
            target60 = g0 + 10.0 + participant
            target120 = g0 + 5.0 + participant
            rows.append(
                {
                    "event_id": f"p{participant}-e{event}",
                    "participant_id": f"p{participant}",
                    "g0_mgdl": g0,
                    "g0_age_minutes": 15.0,
                    "history_n": 3.0,
                    "hour_of_day_sin": 0.0,
                    "hour_of_day_cos": 1.0,
                    "carbs_g": 40.0 + event,
                    "protein_g": 10.0,
                    "fat_g": 5.0,
                    "fiber_g": 2.0,
                    "amount_consumed_pct": 100.0,
                    "target_T+60": {"target_mgdl": target60},
                    "target_T+120": {"target_mgdl": target120},
                    "isolated_T+60": True,
                    "isolated_T+120": True,
                }
            )
    return pd.DataFrame(rows)


def test_evaluator_keeps_all_events_and_history_events_separate():
    events = _evaluation_events()
    events.loc[0, "g0_mgdl"] = np.nan
    report = evaluate_baselines(events, n_splits=3, include_meal_features=False)
    horizon = report["horizons"]["T+60"]
    all_rows = horizon["all_target_events"]["rows"]
    history_rows = horizon["history_available_events"]["rows"]
    assert set(all_rows) == {"B0"}
    assert set(history_rows) == {
        "B0",
        "B1",
        "B2",
        "Ridge_history_only",
    }
    assert horizon["all_target_events"]["n_events"] > horizon["history_available_events"]["n_events"]
    assert len({row["n_events"] for row in history_rows.values()}) == 1


def test_evaluator_excludes_stale_causal_readings_from_history_baselines():
    events = _evaluation_events()
    events["g0_age_minutes"] = 500.0
    report = evaluate_baselines(events, n_splits=3)
    cohort = report["horizons"]["T+60"]["history_available_events"]
    assert cohort["n_events"] == 0
    assert cohort["rows"] == {}


def test_evaluator_rejects_fewer_than_two_participants():
    events = _evaluation_events().query("participant_id == 'p0'")
    with pytest.raises(ValueError, match="at least two participants"):
        evaluate_baselines(events)
