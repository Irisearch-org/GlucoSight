"""ExperimentalFusionPredictor: the one thing that must never silently
regress is "absent modality never excludes, never crashes, never produces
a number outside the contract range" -- see CLAUDE.md rule 9 and the
2026-10-20 grill (bypasses the evidence gate by explicit decision; this
test is not a gate, it is a sanity check that the bypass still behaves)."""
from __future__ import annotations

import json

import pytest

from integration import predictor


ARTIFACT = predictor.EXPERIMENTAL_FUSION_ARTIFACT


def _blob():
    with open(ARTIFACT) as fh:
        return json.load(fh)


def _meal(**overrides):
    base = {
        "g0_mgdl": 130.0, "g0": 130.0, "g0_age_minutes": 5.0, "g0_trust": 0.95,
        "g0_is_direct_measurement": True,
        "glucose_history": [{"timestamp": "2026-10-20T06:00:00", "value": 128.0}],
        "carbs_g": 60.0, "protein_g": 20.0, "fat_g": 10.0, "fiber_g": 5.0,
        "cv_confidence": 0.9, "cv_present": 1,
        "is_fried_cooking": 1, "is_large_portion": 0,
        "nlp_confidence": 0.8, "nlp_present": 1,
        "ppg_pulse_rate_bpm": 80.0, "ppg_signal_quality": 0.7, "ppg_present": 1,
        "hour_of_day_sin": 0.5, "hour_of_day_cos": 0.5,
        "delta_t_minutes": 60.0,
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def loaded():
    import os
    if not os.path.exists(ARTIFACT):
        pytest.skip("run forecasting.evaluate.fit_experimental_fusion first")
    return predictor.ExperimentalFusionPredictor.load()


def test_labelled_experimental_and_ungated(loaded):
    assert loaded.label == "experimental_ungated"
    assert loaded.is_trained_model is True


def test_present_modalities_move_the_number_away_from_pure_baseline(loaded):
    present = loaded.predict(_meal())
    absent = loaded.predict(_meal(
        cv_present=0, cv_confidence=0.0, nlp_present=0, nlp_confidence=0.0,
        ppg_present=0, ppg_signal_quality=0.0,
    ))
    assert present["glucose_t60"] != absent["glucose_t60"]


def test_absent_modality_down_weights_never_excludes(loaded):
    """Dropping a modality must change the WEIGHT, never the feature count
    or crash the predictor (H1: down-weight, never exclude)."""
    out = loaded.predict(_meal(
        cv_present=0, cv_confidence=0.0, carbs_g=None, protein_g=None,
        fat_g=None, fiber_g=None,
    ))
    assert 70.0 <= out["glucose_t60"] <= 400.0


def test_out_of_distribution_feature_does_not_blow_up(loaded):
    """Regression for the exact failure found while building this class:
    perfusion_index=95.7 against a training scale of 0.0075 extrapolated
    to glucose_t60=312 before z-scores were clipped. Same shape as F9."""
    out = loaded.predict(_meal(
        ppg_perfusion_index=95.7, ppg_pulse_rate_bpm=77.85,
        ppg_hrv_rmssd=4.24, ppg_signal_quality=1.0,
    ))
    assert 70.0 <= out["glucose_t60"] <= 400.0


def test_ppg_estimates_g0_only_when_no_direct_measurement(loaded):
    """The actual target: PPG supplies the CURRENT glucose (g0), which then
    feeds T+60 = g0 + excursion -- the same role a fingerstick plays -- not
    a separate pull on the T+60 output. It must only kick in when nothing
    better (fingerstick/CGM) is available."""
    no_direct = _meal(g0_is_direct_measurement=False, g0_source="population_fallback")
    out = loaded.predict(no_direct)
    assert out["g0_source"] == "ppg_estimate"
    assert "PPG estimate" in out["basis"]

    has_direct = _meal(g0_is_direct_measurement=True)
    out2 = loaded.predict(has_direct)
    assert out2["g0_source"] != "ppg_estimate"


def test_t120_has_no_trained_arm_and_falls_back_to_b2(loaded):
    out = loaded.predict(_meal(delta_t_minutes=120.0))
    b2 = predictor.BaselineB2Predictor().predict(_meal(delta_t_minutes=120.0))
    assert out["glucose_t120"] == b2["glucose_t120"]


def test_never_becomes_the_shipped_default():
    assert not isinstance(predictor.default_predictor(), predictor.ExperimentalFusionPredictor)
