"""Contract v1.2 enforcement — the four silent-failure rules.

Each of these failures produces a clean run and a plausible loss curve.
Nothing errors on its own. That is why they are tested rather than reviewed.
"""

import numpy as np
import pytest

from integration import contract, features, fusion
from integration.adapters import cv_to_contract, nlp_to_contract, ppg_to_contract


def _cv_ok():
    return {
        "sample_id": "m1", "modality": "vision", "food_top1": "nasi_goreng",
        "food_top3": ["nasi_goreng"], "carbs_g": 65.2, "protein_g": 12.1,
        "fat_g": 8.3, "fiber_g": 2.1, "calories_kcal": 400, "confidence": 0.87,
        "feature_status": "ok", "model_version": "cv-baseline-v0.1",
        "created_at": "2026-01-15T19:00:00+07:00",
    }


def _nlp_ok():
    return {
        "is_fried_cooking": 1, "is_large_portion": 0, "nlp_confidence": 0.91,
        "nlp_present": 1, "nlp_feature_source": "rule_based",
        "nlp_model_version": "nlp-baseline-v0.1",
    }


def _ppg_ok():
    return {
        "ppg_pulse_rate_bpm": 82.3, "ppg_hrv_rmssd": 28.4,
        "ppg_signal_quality": 0.74, "ppg_perfusion_index": 3.1,
        "ppg_present": 1, "ppg_clip_fraction": 0.02, "ppg_n_windows": 3,
        "ppg_model_version": "ppg-baseline-v0.1",
    }


def _assemble(**kw):
    base = dict(
        meal_id="a3f9", participant_id="P042",
        t0_timestamp="2026-01-15T19:30:00", source_dataset="cgmacros",
        delta_t_minutes=62.0, cv_record=_cv_ok(), nlp_record=_nlp_ok(),
        ppg_record=_ppg_ok(), portion_reported=2,
    )
    base.update(kw)
    return fusion.assemble(**base)


# ---------------------------------------------------------------------
# C2 — causality
# ---------------------------------------------------------------------

def test_c2_history_at_or_after_t0_is_rejected():
    with pytest.raises(contract.ContractViolation, match="LABEL LEAKAGE"):
        contract.assert_causal_history(
            [{"timestamp": "2026-01-15T19:30:00", "value": 150.0}],
            "2026-01-15T19:30:00",
        )


def test_c2_filter_drops_noncausal_and_keeps_the_rest():
    history = [
        {"timestamp": "2026-01-15T07:30:00", "value": 118.0},
        {"timestamp": "2026-01-15T12:00:00", "value": 142.0},
        {"timestamp": "2026-01-15T20:30:00", "value": 190.0},   # T+60 TARGET
        {"timestamp": "2026-01-15T21:30:00", "value": 165.0},   # T+120 TARGET
    ]
    kept = contract.filter_causal_history(history, "2026-01-15T19:30:00")
    assert len(kept) == 2
    assert all(e["value"] in (118.0, 142.0) for e in kept)


def test_c2_assemble_silently_drops_nothing_it_does_not_report():
    meal = _assemble(glucose_history=[
        {"timestamp": "2026-01-15T18:00:00", "value": 120.0},
        {"timestamp": "2026-01-15T20:30:00", "value": 190.0},
    ])
    assert meal["_provenance"]["n_history_readings_dropped_as_noncausal"] == 1
    assert meal["g0"] == 120.0


def test_c2_derived_features_use_only_causal_readings():
    """The whole point: g0 must never be a post-t0 reading."""
    meal = _assemble(glucose_history=[
        {"timestamp": "2026-01-15T19:00:00", "value": 110.0},
        {"timestamp": "2026-01-15T20:30:00", "value": 250.0},
    ])
    assert meal["g0"] == 110.0, "g0 took a post-meal reading — C2 violated"


def test_c2_no_causal_reading_yields_none_not_zero():
    meal = _assemble(glucose_history=[
        {"timestamp": "2026-01-15T20:30:00", "value": 190.0},
    ])
    assert meal["g0"] is None, "absent history must not encode as a glucose value"


# ---------------------------------------------------------------------
# C1 — timing
# ---------------------------------------------------------------------

def test_c1_delta_t_is_carried_not_assumed():
    meal = _assemble(delta_t_minutes=83.0)
    assert meal["delta_t_minutes"] == 83.0
    assert "delta_t_minutes" in features.feature_names(features.TIER_DEPLOYABLE)


def test_c1_delta_t_out_of_contract_range_is_rejected():
    with pytest.raises(contract.ContractViolation):
        _assemble(delta_t_minutes=999.0)


# ---------------------------------------------------------------------
# H1 — missing != negative, down-weight != exclude
# ---------------------------------------------------------------------

def test_h1_absent_modality_keeps_every_column():
    full = _assemble()
    none_at_all = _assemble(cv_record=None, nlp_record=None, ppg_record=None)
    X_full, names_full = features.vectorize([full])
    X_none, names_none = features.vectorize([none_at_all])
    assert names_full == names_none
    assert X_full.shape == X_none.shape, (
        "a missing modality changed input dimensionality — this is exactly "
        "what down-weight-never-exclude forbids"
    )


def test_h1_present_mask_is_separate_from_confidence():
    absent = _assemble(cv_record=None)
    assert absent["cv_present"] == 0
    assert absent["cv_confidence"] == 0.0

    present_but_useless = _assemble(cv_record={**_cv_ok(), "confidence": 0.01})
    assert present_but_useless["cv_present"] == 1
    assert present_but_useless["cv_confidence"] == 0.01
    # The two states must be distinguishable downstream.
    assert absent["cv_present"] != present_but_useless["cv_present"]


def test_h1_contradictory_mask_and_confidence_is_a_violation():
    bad = {"cv_present": 0, "cv_confidence": 0.9}
    errors = contract.validate_meal_input(bad, strict=False)
    assert any("different states" in e for e in errors)


def test_h1_weight_is_never_zero():
    """A zero weight is indistinguishable from a genuine zero value."""
    absent = _assemble(ppg_record=None)
    w = contract.modality_weight(absent, "ppg")
    assert w > 0.0
    assert w == contract.MIN_MODALITY_WEIGHT


def test_h1_low_confidence_downweights_monotonically():
    weights = []
    for conf in (0.0, 0.1, 0.2, 0.3, 0.9):
        meal = _assemble(nlp_record={**_nlp_ok(), "nlp_confidence": conf})
        weights.append(contract.modality_weight(meal, "nlp"))
    assert weights == sorted(weights)
    assert weights[-1] == 1.0


def test_h1_quality_scores_are_not_scaled_by_themselves():
    """Scaling ppg_signal_quality by the weight derived from it is circular."""
    low = _assemble(ppg_record={**_ppg_ok(), "ppg_signal_quality": 0.1})
    X, names = features.vectorize([low])
    idx = names.index("ppg_signal_quality")
    assert X[0, idx] == pytest.approx(0.1)


# ---------------------------------------------------------------------
# H7 — one-hot, not integer
# ---------------------------------------------------------------------

def test_h7_gi_category_is_one_hot():
    names = features.feature_names(features.TIER_DEPLOYABLE)
    assert "gi_category" not in names
    assert {"gi_category__0", "gi_category__1", "gi_category__2"} <= set(names)


def test_h7_portion_reported_is_one_hot():
    names = features.feature_names(features.TIER_DEPLOYABLE)
    assert "portion_reported" not in names
    assert {"portion_reported__0", "portion_reported__1",
            "portion_reported__2"} <= set(names)


def test_h7_one_hot_encodes_the_right_column():
    meal = _assemble(portion_reported=2)
    X, names = features.vectorize([meal])
    assert X[0, names.index("portion_reported__2")] == 1.0
    assert X[0, names.index("portion_reported__0")] == 0.0


# ---------------------------------------------------------------------
# E1 — the ppg_glucose_estimate ring-fence
# ---------------------------------------------------------------------

def test_e1_ringfenced_field_is_not_in_any_default_tier():
    for tier in features.TIERS:
        assert "ppg_glucose_estimate" not in features.feature_names(tier), (
            f"tier {tier} leaked the ring-fenced PPG glucose estimate into the "
            f"default feature set (E1)"
        )


def test_e1_ablation_can_opt_in_explicitly():
    names = features.feature_names(features.TIER_DEPLOYABLE, include_ringfenced=True)
    assert "ppg_glucose_estimate" in names


def test_e1_absent_capture_carries_no_glucose_estimate():
    from integration.adapters.ppg import ppg_to_contract as p2c
    block = p2c({**_ppg_ok(), "ppg_present": 0,
                 "ppg_glucose_estimate": 145.0, "ppg_n_windows": 0}, "m1")
    assert block["ppg_glucose_estimate"] is None


# ---------------------------------------------------------------------
# Normalization leakage rule
# ---------------------------------------------------------------------

def test_scaler_refuses_a_second_fit():
    names = features.feature_names(features.TIER_DEPLOYABLE)
    sc = features.TrainOnlyScaler(names)
    sc.fit(np.zeros((10, len(names))))
    with pytest.raises(RuntimeError, match="fit called twice"):
        sc.fit(np.zeros((10, len(names))))


def test_scaler_refuses_transform_before_fit():
    names = features.feature_names(features.TIER_DEPLOYABLE)
    sc = features.TrainOnlyScaler(names)
    with pytest.raises(RuntimeError, match="before fit"):
        sc.transform(np.zeros((3, len(names))))


def test_scaler_leaves_binary_and_onehot_untouched():
    names = features.feature_names(features.TIER_DEPLOYABLE)
    sc = features.TrainOnlyScaler(names)
    rng = np.random.default_rng(0)
    X = rng.normal(5, 3, (50, len(names)))
    for col in ("cv_present", "gi_category__1", "is_fried_cooking"):
        X[:, names.index(col)] = rng.integers(0, 2, 50)
    sc.fit(X)
    Xt = sc.transform(X)
    for col in ("cv_present", "gi_category__1", "is_fried_cooking"):
        i = names.index(col)
        assert np.allclose(Xt[:, i], X[:, i]), f"{col} was z-scored"


def test_scaler_round_trips_through_disk_format():
    names = features.feature_names(features.TIER_DEPLOYABLE)
    sc = features.TrainOnlyScaler(names)
    sc.fit(np.random.default_rng(1).normal(0, 1, (30, len(names))))
    restored = features.TrainOnlyScaler.from_dict(sc.to_dict())
    X = np.random.default_rng(2).normal(0, 1, (5, len(names)))
    assert np.allclose(sc.transform(X), restored.transform(X))


# ---------------------------------------------------------------------
# Feature tiers
# ---------------------------------------------------------------------

def test_oracle_tier_is_a_strict_superset_of_deployable():
    dep = set(features.feature_names(features.TIER_DEPLOYABLE))
    orc = set(features.feature_names(features.TIER_ORACLE))
    assert dep < orc


def test_no_lab_or_gut_feature_reaches_the_deployable_tier():
    dep = set(features.feature_names(features.TIER_DEPLOYABLE))
    for name in features.ORACLE_ONLY_CONTINUOUS + features.ORACLE_ONLY_BINARY:
        assert name not in dep, (
            f"{name!r} is in the deployable tier but cannot be obtained from a "
            f"phone: {features.ORACLE_JUSTIFICATION.get(name)}"
        )


def test_every_oracle_feature_has_a_stated_justification():
    for name in features.ORACLE_ONLY_CONTINUOUS + features.ORACLE_ONLY_BINARY:
        assert features.ORACLE_JUSTIFICATION.get(name), (
            f"{name} is excluded from the deployable tier without a reason"
        )
