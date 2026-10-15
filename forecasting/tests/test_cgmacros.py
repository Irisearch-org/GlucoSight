from __future__ import annotations

import pandas as pd
import pytest

from forecasting.data.cgmacros import (
    DatasetFormatError,
    audit_dataset,
    build_history,
    build_targets,
    extract_events,
    join_participant_metadata,
    load_participant,
)


def _participant_rows():
    times = pd.date_range("2025-01-01 07:00", periods=181, freq="min")
    frame = pd.DataFrame(
        {
            "Timestamp": times,
            "Dexcom GL": [100.0 + i / 10 for i in range(len(times))],
            "Meal Type": [None] * len(times),
            "Calories": [None] * len(times),
            "Carbs": [None] * len(times),
            "Protein": [None] * len(times),
            "Fat": [None] * len(times),
            "Fiber": [None] * len(times),
            "Amount Consumed": [None] * len(times),
            "Image path": [None] * len(times),
        }
    )
    frame.loc[60, ["Meal Type", "Calories", "Carbs", "Protein", "Fat", "Fiber"]] = [
        " Breakfast ", 450, 60, 20, 10, 5
    ]
    frame.loc[60, "Amount Consumed"] = 100
    frame.loc[60, "Image path"] = "photos/meal.jpg"
    return frame


def _write_dataset(root):
    frame = _participant_rows()
    frame.to_csv(root / "CGMacros-001.csv", index=False)
    pd.DataFrame({"subject": [1], "Age": [35]}).to_csv(root / "bio.csv", index=False)
    pd.DataFrame({"subject": [1], "Metabolic Fitness": [2]}).to_csv(
        root / "gut_health_test.csv", index=False
    )
    return frame


def test_load_participant_normalizes_meal_alias_but_preserves_source_label(tmp_path):
    _write_dataset(tmp_path)
    frame = load_participant(tmp_path, "CGMacros-001.csv")
    event = frame.loc[frame["event_candidate"]].iloc[0]
    assert event["meal_type"] == "breakfast"
    assert event["source_event_label"] == " Breakfast "
    assert event["participant_id"] == "cgmacros-001"


def test_metadata_joins_by_subject_and_rejects_missing_subject(tmp_path):
    bio = pd.DataFrame({"subject": [2, 1], "age": [42, 35]})
    gut = pd.DataFrame({"subject": [1, 2], "score": [1, 2]})
    joined = join_participant_metadata([1, 2], bio, gut)
    assert joined.set_index("subject").loc[1, "age"] == 35
    assert joined.set_index("subject").loc[2, "score"] == 2

    with pytest.raises(DatasetFormatError, match="missing cohort subjects"):
        join_participant_metadata([1, 2, 3], bio, gut)


def test_metadata_join_rejects_duplicate_subject_ids():
    bio = pd.DataFrame({"subject": [1, 1]})
    gut = pd.DataFrame({"subject": [1]})
    with pytest.raises(DatasetFormatError, match="duplicate subject"):
        join_participant_metadata([1], bio, gut)


def test_targets_use_exact_observed_readings_and_retain_actual_offset(tmp_path):
    frame = _write_dataset(tmp_path)
    loaded = load_participant(tmp_path, "CGMacros-001.csv")
    event = extract_events(loaded)
    targets, counts = build_targets(loaded, event)
    target = targets.iloc[0]
    assert counts == {
        "source_events": 1,
        "T+60": 1,
        "T+120": 1,
        "paired": 1,
        "isolated_T+60": 1,
        "isolated_T+120": 1,
        "isolated_paired": 1,
    }
    assert target["T+60"]["target_mgdl"] == frame.loc[120, "Dexcom GL"]
    assert target["T+60"]["actual_delta_t_minutes"] == 60
    assert target["T+120"]["target_mgdl"] == frame.loc[180, "Dexcom GL"]


def test_targets_do_not_interpolate_when_exact_observation_is_missing():
    frame = _participant_rows()
    frame = frame.drop(index=120).reset_index(drop=True)
    frame["source_row_id"] = range(len(frame))
    frame["participant_id"] = "cgmacros-001"
    frame["source_event_label"] = frame["Meal Type"]
    labels = frame["Meal Type"].astype("string").str.strip().str.casefold()
    frame["meal_type"] = labels.map({"breakfast": "breakfast"})
    frame["event_candidate"] = labels.notna() & labels.ne("")
    frame["unknown_meal_type"] = frame["event_candidate"] & frame["meal_type"].isna()
    event = extract_events(frame)
    targets, counts = build_targets(frame, event)
    assert pd.isna(targets.iloc[0]["T+60"])
    assert targets.iloc[0]["T+120"] is not None
    assert counts["T+60"] == 0


def test_intervening_event_is_flagged_but_target_is_retained():
    frame = _participant_rows()
    frame.loc[70, "Meal Type"] = "snack"
    frame["participant_id"] = "cgmacros-001"
    frame["source_row_id"] = range(len(frame))
    frame["source_event_label"] = frame["Meal Type"]
    normalized = frame["Meal Type"].astype("string").str.strip().str.casefold()
    frame["meal_type"] = normalized.map({"breakfast": "breakfast", "snack": "snack"})
    frame["event_candidate"] = normalized.notna() & normalized.ne("")
    frame["unknown_meal_type"] = frame["event_candidate"] & frame["meal_type"].isna()

    events = extract_events(frame)
    targets, counts = build_targets(frame, events)
    first = targets.sort_values("event_timestamp").iloc[0]

    assert first["T+60"] is not None
    assert not bool(first["isolated_T+60"])
    assert counts["T+60"] == 2
    assert counts["isolated_T+60"] == 1


def test_history_excludes_readings_at_and_after_event_time():
    frame = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                ["2025-01-01 07:58", "2025-01-01 07:59", "2025-01-01 08:00", "2025-01-01 08:01"]
            ),
            "Dexcom GL": [100.0, 101.0, 500.0, 600.0],
        }
    )
    history = build_history(frame, "2025-01-01 08:00")
    assert history["Dexcom GL"].tolist() == [100.0, 101.0]
    assert history["Timestamp"].max() < pd.Timestamp("2025-01-01 08:00")


def test_audit_report_accounts_for_real_participant_and_event_rows(tmp_path):
    _write_dataset(tmp_path)
    report = audit_dataset(tmp_path)
    assert report.participant_files == 1
    assert report.participant_ids == (1,)
    assert report.source_events == 1
    assert report.candidate_targets == {"T+60": 1, "T+120": 1}
    assert report.paired_candidate_events == 1
    assert report.isolated_candidate_targets == {"T+60": 1, "T+120": 1}
    assert report.isolated_paired_events == 1
    assert report.csv_checksums["CGMacros-001.csv"]


def test_audit_counts_optional_source_columns_as_missing(tmp_path):
    frame = _participant_rows().drop(columns=["Fiber", "Amount Consumed", "Image path"])
    frame.to_csv(tmp_path / "CGMacros-001.csv", index=False)
    pd.DataFrame({"subject": [1]}).to_csv(tmp_path / "bio.csv", index=False)
    pd.DataFrame({"subject": [1]}).to_csv(tmp_path / "gut_health_test.csv", index=False)

    report = audit_dataset(tmp_path)
    assert report.missingness["Fiber"] == 1
    assert report.missingness["Amount Consumed"] == 1
    assert report.missingness["Image path"] == 1
    assert report.range_violations["Carbs"] == {"below_minimum": 0, "above_maximum": 0}
