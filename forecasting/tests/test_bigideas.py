from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting.data.bigideas import (
    DatasetFormatError,
    build_meal_table,
    glucose_before,
    glucose_near,
    load_cohort,
    load_dexcom,
    load_food_log,
)
from forecasting.evaluate.bigideas_response import (
    carbohydrate_sensitivity,
    carbohydrate_variance_decomposition,
    crossing_point,
    history_baselines,
    loo_predict_linear,
)

DEXCOM_HEADER = (
    "Index,Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Event Subtype,Patient Info,"
    "Device Info,Source Device ID,Glucose Value (mg/dL),Insulin Value (u),"
    "Carb Value (grams),Duration (hh:mm:ss),Glucose Rate of Change (mg/dL/min),"
    "Transmitter Time (Long Integer)"
)
FOOD_LOG_HEADER = (
    "date,time,time_begin,time_end,logged_food,amount,unit,searched_food,calorie,"
    "total_carb,dietary_fiber,sugar,protein,total_fat"
)


def write_dexcom(path: Path, start: str, readings, step_minutes: int = 5) -> Path:
    """Write a minimal Dexcom Clarity export: preamble, then EGV rows."""
    times = pd.date_range(start, periods=len(readings), freq=f"{step_minutes}min")
    lines = [DEXCOM_HEADER, "1,,FirstName,,2019,,,,,,,,"]
    for index, (time, value) in enumerate(zip(times, readings), start=2):
        lines.append(
            f"{index},{time:%Y-%m-%d %H:%M:%S},EGV,,,,iPhone G6,{value},,,,,11101.0"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def write_food_log(path: Path, rows) -> Path:
    lines = [FOOD_LOG_HEADER]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n")
    return path


def food_row(when: str, food: str, carbs: float, amount: float = 1.0,
             unit: str = "serving", protein: float = 5.0) -> list:
    date, clock = when.split(" ")
    return [date, clock, when, "", food, amount, unit, food, 200.0, carbs,
            2.0, 5.0, protein, 4.0]


# ----------------------------------------------------------------------
# Leave-one-out estimator
# ----------------------------------------------------------------------

def test_loo_predict_linear_matches_refitting_with_each_point_held_out():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(8, 40))
        design = np.hstack([np.ones((n, 1)), rng.normal(size=(n, 2))])
        y = rng.normal(size=n)

        expected = np.empty(n)
        for i in range(n):
            keep = np.ones(n, bool)
            keep[i] = False
            beta, *_ = np.linalg.lstsq(design[keep], y[keep], rcond=None)
            expected[i] = design[i] @ beta

        assert np.allclose(loo_predict_linear(design, y), expected, atol=1e-9)


def test_loo_predict_linear_with_only_an_intercept_is_the_loo_mean():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    design = np.ones((4, 1))
    # Each prediction is the mean of the OTHER three points.
    assert np.allclose(loo_predict_linear(design, y),
                       [3.0, 8 / 3, 7 / 3, 2.0])


def test_crossing_point_interpolates_between_the_two_bracketing_points():
    assert crossing_point([0.0, 1.0], [-1.0, 1.0]) == pytest.approx(0.5)
    assert crossing_point([0.0, 1.0], [1.0, 2.0]) is None
    assert crossing_point([0.0, 1.0, 2.0], [-1.0, -1.0, 1.0]) == pytest.approx(1.5)


# ----------------------------------------------------------------------
# Food log parsing
# ----------------------------------------------------------------------

def test_headerless_food_log_is_rebuilt_and_verified(tmp_path):
    path = tmp_path / "Food_Log_003.csv"
    # 003's export: no header row, and 11 columns instead of 14.
    path.write_text(
        "2020-02-22,10:30:00,2020-02-22 10:30:00,Chicken Nuggets,8.0,piece,"
        "Chicken Nuggets,393.0,19.0,0.1,20.0\n"
        "2020-02-22,14:30:00,2020-02-22 14:30:00,Dots,11.0,piece,Dots,130.0,"
        "33.0,21.0,0.0\n"
    )
    food_log = load_food_log(path)
    assert food_log.food.tolist() == ["Chicken Nuggets", "Dots"]
    assert food_log.carbs_g.tolist() == [19.0, 33.0]
    assert food_log.t0.tolist() == [
        pd.Timestamp("2020-02-22 10:30"), pd.Timestamp("2020-02-22 14:30"),
    ]
    # Columns absent from a short export must be missing, never zero.
    assert food_log.protein_g.isna().all()
    assert food_log.fat_g.isna().all()


def test_headerless_food_log_with_an_unexpected_layout_is_rejected(tmp_path):
    path = tmp_path / "Food_Log_099.csv"
    path.write_text("apple,banana,cherry,date,elderberry,fig,grape\n")
    with pytest.raises(DatasetFormatError, match="no header row"):
        load_food_log(path)


def test_food_log_accepts_the_time_of_day_column_used_by_some_participants(tmp_path):
    path = tmp_path / "Food_Log_007.csv"
    path.write_text(
        "date,time_of_day,time_begin,time_end,logged_food,amount,unit,searched_food,"
        "calorie,total_carb,dietary_fiber,sugar,protein,total_fat\n"
        "2020-03-14,13:44:00,,,Rice,1,cup,Rice,200,45,1,0,4,0.5\n"
    )
    food_log = load_food_log(path)
    assert food_log.t0.tolist() == [pd.Timestamp("2020-03-14 13:44")]


# ----------------------------------------------------------------------
# Dexcom parsing and causal alignment
# ----------------------------------------------------------------------

def test_glucose_before_never_returns_a_reading_taken_after_the_meal(tmp_path):
    path = write_dexcom(tmp_path / "Dexcom_001.csv", "2020-02-13 17:00",
                        [100, 110, 120, 130, 140])
    dexcom = load_dexcom(path)
    # 17:12 falls between the 17:10 and 17:15 readings; only the earlier one
    # may be used, and only if it is recent enough.
    assert glucose_before(dexcom, pd.Timestamp("2020-02-13 17:12")) == 120.0
    assert np.isnan(glucose_before(dexcom, pd.Timestamp("2020-02-13 18:30")))
    assert np.isnan(glucose_before(dexcom, pd.Timestamp("2020-02-13 16:00")))


def test_glucose_near_respects_its_tolerance(tmp_path):
    path = write_dexcom(tmp_path / "Dexcom_001.csv", "2020-02-13 17:00",
                        [100, 110, 120, 130, 140])
    dexcom = load_dexcom(path)
    assert glucose_near(dexcom, pd.Timestamp("2020-02-13 17:16")) == 130.0
    assert np.isnan(glucose_near(dexcom, pd.Timestamp("2020-02-13 17:40")))


def test_dexcom_export_without_the_expected_column_is_rejected(tmp_path):
    path = tmp_path / "Dexcom_002.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(DatasetFormatError, match="Clarity export"):
        load_dexcom(path)


# ----------------------------------------------------------------------
# Meal assembly
# ----------------------------------------------------------------------

def test_items_sharing_a_timestamp_become_one_meal(tmp_path):
    path = write_food_log(tmp_path / "Food_Log_001.csv", [
        food_row("2020-02-13 18:00", "Chicken Leg", 0.0),
        food_row("2020-02-13 18:00", "Asparagus", 2.5),
        food_row("2020-02-13 20:30", "Smoothie", 85.0),
    ])
    food_log = load_food_log(path)
    dexcom = load_dexcom(write_dexcom(
        tmp_path / "Dexcom_001.csv", "2020-02-13 17:50", [100.0] * 60))

    meals = build_meal_table(dexcom, food_log, horizons=(60,))
    assert len(meals) == 2
    dinner = meals.iloc[0]
    assert dinner.n_items == 2
    assert dinner.carbs_g == pytest.approx(2.5)
    assert "Chicken Leg" in dinner.item_names
    assert "Asparagus" in dinner.item_names


def test_a_meal_with_no_nutrient_data_stays_missing_not_zero(tmp_path):
    # An export whose protein/total_fat columns are blank for every row.
    path = tmp_path / "Food_Log_001.csv"
    path.write_text(
        FOOD_LOG_HEADER + "\n"
        "2020-02-13,18:00:00,2020-02-13 18:00:00,,Rice,1,cup,Rice,200,45,,,,\n"
    )
    food_log = load_food_log(path)
    dexcom = load_dexcom(write_dexcom(
        tmp_path / "Dexcom_001.csv", "2020-02-13 17:50", [100.0] * 60))

    meals = build_meal_table(dexcom, food_log, horizons=(60,))
    assert meals.carbs_g.iloc[0] == pytest.approx(45.0)
    assert np.isnan(meals.protein_g.iloc[0])
    assert np.isnan(meals.fat_g.iloc[0])


# ----------------------------------------------------------------------
# Cohort assembly
# ----------------------------------------------------------------------

def test_participant_whose_windows_do_not_overlap_is_reported_not_silently_kept(
        tmp_path):
    write_food_log(tmp_path / "Food_Log_001.csv",
                   [food_row("2020-02-13 18:00", "Rice", 45.0)])
    write_dexcom(tmp_path / "Dexcom_001.csv", "2020-02-13 17:00", [100.0] * 60)

    # Participant 015 in the real release: food log and CGM are two weeks apart.
    write_food_log(tmp_path / "Food_Log_015.csv",
                   [food_row("2020-07-05 10:00", "Rice", 45.0)])
    write_dexcom(tmp_path / "Dexcom_015.csv", "2020-07-19 17:00", [100.0] * 60)

    cohort, problems = load_cohort(tmp_path, horizons=(60,))
    assert cohort.participant_id.unique().tolist() == ["001"]
    assert len(problems) == 1
    assert "015" in problems[0]
    assert "Dexcom window" in problems[0]


def test_load_cohort_raises_when_nothing_is_usable(tmp_path):
    write_food_log(tmp_path / "Food_Log_001.csv",
                   [food_row("2020-02-13 18:00", "Rice", 45.0)])
    write_dexcom(tmp_path / "Dexcom_001.csv", "2020-06-01 17:00", [100.0] * 60)
    with pytest.raises(DatasetFormatError, match="no usable participants"):
        load_cohort(tmp_path)


# ----------------------------------------------------------------------
# Variance decomposition
# ----------------------------------------------------------------------

def _cohort_with_items(tmp_path, rows) -> pd.DataFrame:
    write_food_log(tmp_path / "Food_Log_001.csv", rows)
    write_dexcom(tmp_path / "Dexcom_001.csv", "2020-02-13 17:00", [100.0] * 200)
    cohort, _ = load_cohort(tmp_path, horizons=(60,))
    return cohort


def test_portion_share_is_near_zero_when_a_dish_is_always_logged_the_same_size(
        tmp_path):
    rows = []
    for day, hour in [("2020-02-13", "18:00"), ("2020-02-14", "18:00"),
                      ("2020-02-15", "18:00")]:
        rows.append(food_row(f"{day} {hour}", "BigPasta", 80.0))
    for day, hour in [("2020-02-13", "12:00"), ("2020-02-14", "12:00"),
                      ("2020-02-15", "12:00")]:
        rows.append(food_row(f"{day} {hour}", "SmallSalad", 5.0))
    cohort = _cohort_with_items(tmp_path, rows)

    decomposition = carbohydrate_variance_decomposition(tmp_path, cohort)
    assert decomposition["distinct_dishes"] == 2
    # Identical sizes within each dish: all variance is between dishes.
    assert decomposition["portion_share"] == pytest.approx(0.0, abs=1e-9)
    assert decomposition["dish_share"] == pytest.approx(1.0, abs=1e-9)


def test_portion_share_is_positive_when_the_same_dish_is_sized_differently(
        tmp_path):
    rows = []
    for day, amount, carbs in [("2020-02-13", 3.0, 100.0),
                               ("2020-02-14", 0.75, 25.0),
                               ("2020-02-15", 1.5, 50.0)]:
        rows.append(food_row(f"{day} 18:00", "BigPasta", carbs, amount=amount))
    cohort = _cohort_with_items(tmp_path, rows)

    decomposition = carbohydrate_variance_decomposition(tmp_path, cohort)
    assert decomposition["portion_share"] > 0.0
    assert decomposition["portion_share"] + decomposition["dish_share"] == \
        pytest.approx(1.0)


def test_portion_share_is_bracketed_by_the_varying_amount_figure(tmp_path):
    """Half the repeated dishes are logged at one amount, pinning their
    within-dish variance to zero. The varying-amount figure must be the
    softer of the two, never the smaller."""
    rows = []
    for day, amount, carbs in [("2020-02-13", 2.0, 60.0),
                               ("2020-02-14", 2.0, 60.0),
                               ("2020-02-15", 2.0, 60.0)]:
        rows.append(food_row(f"{day} 18:00", "FixedRice", carbs, amount=amount))
    for day, amount, carbs in [("2020-02-13", 1.0, 10.0),
                               ("2020-02-14", 4.0, 40.0),
                               ("2020-02-15", 9.0, 90.0)]:
        rows.append(food_row(f"{day} 12:00", "VariablePasta", carbs,
                             amount=amount))
    cohort = _cohort_with_items(tmp_path, rows)

    decomposition = carbohydrate_variance_decomposition(tmp_path, cohort)
    assert decomposition["portion_share"] < \
        decomposition["portion_share_variable_amount_only"]
    assert decomposition["repeated_dishes_logged_at_one_amount"] == \
        pytest.approx(0.5)


# ----------------------------------------------------------------------
# Carbohydrate sensitivity
# ----------------------------------------------------------------------

def synthetic_cohort(n=15, slope=2.0, intercept=5.0, g0_effect=0.0, seed=0,
                      fried_effect=0.0, item_names=None):
    rng = np.random.default_rng(seed)
    g0 = 100.0 + rng.normal(0, 10, n)
    carbs = 40.0 + rng.normal(0, 20, n)
    if item_names is None:
        item_names = ["grilled chicken"] * n
    is_fried = np.array([1 if "fried" in name.lower() else 0 for name in item_names])
    delta = intercept + g0_effect * g0 + slope * carbs + fried_effect * is_fried
    return pd.DataFrame({
        "participant_id": ["001"] * n,
        "g0_mgdl": g0,
        "carbs_g": carbs,
        "item_names": item_names,
        "glucose_60_mgdl": g0 + delta,
        "glucose_120_mgdl": g0 + delta,
        "delta_60_mgdl": delta,
        "delta_120_mgdl": delta,
    })


def test_carbohydrate_sensitivity_recovers_a_known_slope():
    cohort = synthetic_cohort(slope=2.0)
    sensitivity = carbohydrate_sensitivity(cohort, 60)
    assert sensitivity["median_mgdl_per_g"] == pytest.approx(2.0, abs=1e-9)
    assert sensitivity["glucose_shift_for_30g_median_mgdl"] == \
        pytest.approx(60.0, abs=1e-9)
    assert sensitivity["participants"] == 1


def test_carbohydrate_sensitivity_is_not_fooled_by_the_g0_slope():
    """g0 dominates the rise and is correlated with meal size, so a fit that
    omitted it would misattribute the mean reversion to the carbohydrate."""
    cohort = synthetic_cohort(slope=1.5, g0_effect=-0.6, seed=3)
    sensitivity = carbohydrate_sensitivity(cohort, 60)
    assert sensitivity["median_mgdl_per_g"] == pytest.approx(1.5, abs=1e-9)


# ----------------------------------------------------------------------
# history_baselines — B4 adds is_fried_cooking (ADR 0004)
# ----------------------------------------------------------------------

def test_history_baselines_b4_recovers_a_known_fried_effect():
    """Half the cohort's meals are fried, with a real, known glucose effect
    B3 (g0 + slope + carbs) cannot see — B4 adds is_fried_cooking, derived
    from item_names via the same rule-based matcher the NLP track already
    ships (utils/cooking_method.py::is_fried), and should recover it."""
    n = 16
    names = (["fried chicken"] * (n // 2)) + (["grilled chicken"] * (n // 2))
    cohort = synthetic_cohort(n=n, slope=2.0, fried_effect=15.0,
                               item_names=names, seed=1)
    summary, _ = history_baselines(cohort, 60)
    by_model = summary.set_index("model")["mae_mgdl"]
    assert "B4 g0 + slope + carbs + is_fried" in by_model.index
    # Deterministic synthetic data, no noise: B4 should fit it almost
    # exactly, and clearly beat B3, which has no way to see the fried term.
    assert by_model["B4 g0 + slope + carbs + is_fried"] < 1e-6
    assert by_model["B4 g0 + slope + carbs + is_fried"] < by_model["B3 g0 + slope + carbs"]


def test_history_baselines_b4_is_a_noop_when_nothing_is_fried():
    """No fried meals in the cohort: B4 should match B3, not invent a signal."""
    cohort = synthetic_cohort(n=16, slope=2.0, fried_effect=0.0, seed=2)
    summary, _ = history_baselines(cohort, 60)
    by_model = summary.set_index("model")["mae_mgdl"]
    assert by_model["B4 g0 + slope + carbs + is_fried"] == pytest.approx(
        by_model["B3 g0 + slope + carbs"], abs=1e-6
    )
