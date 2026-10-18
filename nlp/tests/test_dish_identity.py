"""Tests for the dish-identity method test (Path A, `nlp/eval_dish_identity.py`).

Three kinds of test live here.

* **Shape and plumbing** — the join, the design matrices, the fold helpers.
* **A positive control** — synthetic meals whose T+60 rise is generated *from
  dish identity* and from nothing else. The dish arm must win there, or the
  harness cannot detect the thing it claims to measure.
* **A null control** — synthetic meals whose rise is generated from `g0` and
  noise, with dish labels attached at random. The dish arm must *not* win
  there, or the gain on real data is capacity rather than signal.

The controls are the point. Finding F15 recorded a metric that improved while
the model learned nothing, and the real-data run is only interpretable next to
these.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nlp.eval_dish_identity import (
    ARMS,
    DEFAULT_LEGACY,
    DEFAULT_RESPONSE,
    dish_matrix,
    feature_diagnostics,
    load_meals,
    repeated_ladder,
    run_ladder,
    shuffled_folds,
)
from nlp.data.splits import grouped_folds

DISHES = [f"dish{index}" for index in range(12)]
#: Evenly spread, so a meal's dish identity moves its target by several mg/dL.
EFFECT = {name: (index - 5.5) * 5.0 for index, name in enumerate(DISHES)}


def synthetic(*, dish_driven: bool, n_subjects: int = 60, meals_each: int = 10,
              seed: int = 0) -> pd.DataFrame:
    """Meals with the columns `run_ladder` needs, and a known generating rule."""
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(n_subjects):
        for meal in range(meals_each):
            g0 = float(rng.normal(130.0, 25.0))
            chosen = tuple(sorted(
                rng.choice(DISHES, size=int(rng.integers(1, 4)), replace=False).tolist()
            ))
            noise = float(rng.normal(0.0, 12.0))
            if dish_driven:
                rise = np.mean([EFFECT[name] for name in chosen]) + noise
            else:
                rise = -0.4 * (g0 - 130.0) + noise
            rows.append({
                "meal_id": f"S{subject:03d}M{meal}",
                "participant_id": f"{subject:04d}_0_20201230",
                "subject_id": f"{subject:04d}",
                "g0_mgdl": g0,
                "total_grams": float(rng.uniform(100.0, 600.0)),
                "is_fried_cooking": int(rng.random() < 0.12),
                "is_large_portion": int(rng.random() < 0.27),
                "rise_t60": rise,
                "valid_t60": 1,
                "insulin_near_meal": 0,
                "dish_set": chosen,
            })
    return pd.DataFrame(rows).reset_index(drop=True)


# --------------------------------------------------------------------------
# the committed tables
# --------------------------------------------------------------------------

def test_committed_inputs_exist():
    assert DEFAULT_RESPONSE.exists()
    assert DEFAULT_LEGACY.exists()


def test_load_meals_joins_the_two_committed_tables():
    frame = load_meals()
    assert len(frame) == 3045
    assert frame["meal_id"].is_unique
    for column in ("g0_mgdl", "total_grams", "is_fried_cooking", "is_large_portion", "rise_t60"):
        assert column in frame.columns
        assert frame[column].notna().all()


def test_load_meals_only_keeps_meals_with_a_t60_rise():
    frame = load_meals()
    assert (frame["valid_t60"] == 1).all()


def test_load_meals_carries_a_subject_column_separate_from_the_period():
    frame = load_meals()
    assert frame["subject_id"].nunique() == 100
    assert frame["participant_id"].nunique() > frame["subject_id"].nunique()


# --------------------------------------------------------------------------
# design matrices
# --------------------------------------------------------------------------

def test_dish_matrix_marks_presence():
    frame = load_meals().head(50)
    matrix, vocabulary = dish_matrix(frame)
    assert matrix.shape == (50, len(vocabulary))
    assert set(matrix.ravel().tolist()) <= {0.0, 1.0}
    assert vocabulary == sorted(vocabulary)
    assert (matrix.sum(axis=1) > 0).all()


def test_dish_matrix_repeated_dish_stays_one():
    frame = pd.DataFrame({"dish_set": [("rice", "rice", "egg")]})
    matrix, vocabulary = dish_matrix(frame)
    assert matrix.tolist() == [[1.0, 1.0]]
    assert vocabulary == ["egg", "rice"]


# --------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------

def test_shuffled_folds_do_not_split_a_subject():
    frame = synthetic(dish_driven=True)
    subjects = frame["subject_id"]
    for train, test in shuffled_folds(frame, seed=3):
        assert not (set(subjects.iloc[train]) & set(subjects.iloc[test]))


# --------------------------------------------------------------------------
# the positive control
# --------------------------------------------------------------------------

def test_dish_identity_is_detected_when_it_is_the_only_signal():
    frame = synthetic(dish_driven=True)
    scored = run_ladder(frame, grouped_folds(frame, n_splits=5))
    assert scored["M3_dish"]["mae"] < scored["M2_shipped"]["mae"] - 1.0


def test_grams_and_g0_cannot_recover_a_dish_driven_rise():
    """The baseline arm is genuinely blind here, so the control has teeth."""
    frame = synthetic(dish_driven=True)
    scored = run_ladder(frame, grouped_folds(frame, n_splits=5))
    gain_over_constant = scored["M0_constant"]["mae"] - scored["M2_shipped"]["mae"]
    assert gain_over_constant < 0.5


# --------------------------------------------------------------------------
# the null control
# --------------------------------------------------------------------------

def test_permuted_dishes_earn_nothing_when_the_rise_is_dish_blind():
    frame = synthetic(dish_driven=False, seed=7)
    scored = run_ladder(frame, grouped_folds(frame, n_splits=5))
    assert scored["M3p_permuted"]["mae"] >= scored["M2_shipped"]["mae"] - 0.75


def test_real_dishes_earn_little_when_the_rise_is_dish_blind():
    frame = synthetic(dish_driven=False, seed=7)
    scored = run_ladder(frame, grouped_folds(frame, n_splits=5))
    assert scored["M3_dish"]["mae"] >= scored["M2_shipped"]["mae"] - 0.75


# --------------------------------------------------------------------------
# the ladder's own consistency
# --------------------------------------------------------------------------

def test_every_arm_is_scored_and_finite():
    frame = synthetic(dish_driven=True)
    scored = run_ladder(frame, grouped_folds(frame, n_splits=5))
    for arm in ARMS:
        assert np.isfinite(scored[arm]["mae"])
        assert np.isfinite(scored[arm]["rmse"])
        assert len(scored[arm]["fold_mae"]) == 5


def test_constant_arm_is_the_training_mean():
    frame = synthetic(dish_driven=True)
    scored = run_ladder(frame, grouped_folds(frame, n_splits=5))
    spread = frame["rise_t60"].sub(frame["rise_t60"].mean()).abs().mean()
    assert scored["M0_constant"]["mae"] == pytest.approx(spread, abs=2.0)


def test_diagnostics_report_the_fold_vocabulary_and_coverage():
    frame = synthetic(dish_driven=True)
    scored = run_ladder(frame, grouped_folds(frame, n_splits=5))
    diagnostics = scored["_diagnostics"]
    assert diagnostics["vocabulary_size"] == len(DISHES)
    assert len(diagnostics["modelled_dishes_per_fold"]) == 5
    assert 0.0 <= diagnostics["share_test_meals_with_a_modelled_dish"] <= 1.0


def test_repeated_ladder_reports_a_spread_for_every_delta():
    frame = synthetic(dish_driven=True, n_subjects=24, meals_each=8)
    spread = repeated_ladder(frame, seeds=3)
    assert set(spread["_deltas"]) == {
        "M3_minus_M2", "M3p_minus_M2", "M2_minus_M1", "M2_minus_M0"
    }
    for name, stat in spread["_deltas"].items():
        assert np.isfinite(stat["mean"]), name
        assert stat["sd"] >= 0.0, name
        assert len(stat["per_seed"]) == 3


def test_repeated_ladder_finds_the_dish_signal():
    frame = synthetic(dish_driven=True, n_subjects=24, meals_each=8)
    spread = repeated_ladder(frame, seeds=3)
    assert spread["_deltas"]["M3_minus_M2"]["mean"] < -1.0


# --------------------------------------------------------------------------
# the diagnostics block (the numbers docs/METRICS.md quotes)
# --------------------------------------------------------------------------

def test_feature_diagnostics_reports_the_quoted_correlations():
    diagnostics = feature_diagnostics(load_meals())
    correlations = diagnostics["correlation_with_target"]
    assert set(correlations) == {
        "g0_mgdl", "total_grams", "is_fried_cooking", "is_large_portion"
    }
    # g0 is the only one of the four that carries signal on this cohort.
    assert correlations["g0_mgdl"] == pytest.approx(-0.296, abs=0.005)
    assert abs(correlations["total_grams"]) < 0.02


def test_feature_diagnostics_reports_the_ols_ladder():
    diagnostics = feature_diagnostics(load_meals())
    r2 = diagnostics["ols_r2"]
    assert r2["constant"] < 0.0
    assert r2["g0"] > r2["constant"]
    assert r2["g0_plus_shipped"] > r2["g0"]
    assert r2["g0_plus_frequent_dishes"] > r2["g0_plus_shipped"]
    assert diagnostics["frequent_dish_count"] == 23
    assert diagnostics["comparable_to_ridge_ladder"] is False


def test_feature_diagnostics_per_subject_counts_only_subjects_past_the_floor():
    diagnostics = feature_diagnostics(load_meals(), min_meals_per_subject=10)
    assert diagnostics["per_subject_correlation"]["g0_mgdl"]["n_subjects"] == 87
    assert diagnostics["min_meals_per_subject"] == 10


def test_feature_diagnostics_finds_a_dish_driven_signal():
    """On dish-driven data the dish R^2 must beat the shipped-feature R^2."""
    frame = synthetic(dish_driven=True)
    diagnostics = feature_diagnostics(frame, min_dish_meals=10)
    r2 = diagnostics["ols_r2"]
    assert r2["g0_plus_frequent_dishes"] > r2["g0_plus_shipped"]
