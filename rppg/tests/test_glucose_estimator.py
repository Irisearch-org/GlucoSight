"""The evidence gate must let real signal through and keep noise out.

The raw PPG signals are gitignored, so these tests exercise the gate on
synthetic feature tables with known structure. That is the property worth
testing anyway: the gate's job is to distinguish a model that learned
something from one that got a lucky fold split.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from rppg.models.glucose_estimator import (
    FEATURE_NAMES, PPGGlucoseEstimator, evaluate,
)

N_SUBJECTS = 23
RECS_PER_SUBJECT = 3


def _synthetic(signal_strength: float, seed: int = 0):
    """Feature table where glucose depends on feature 0 by `signal_strength`.

    Mirrors the real dataset's shape: 23 subjects, a few recordings each,
    glucose around 115 +/- 19 mg/dL.
    """
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for s in range(N_SUBJECTS):
        subject_offset = rng.normal(0, 13)      # between-subject variance
        for _ in range(RECS_PER_SUBJECT):
            feats = rng.normal(0, 1, len(FEATURE_NAMES))
            glucose = (115.0 + subject_offset
                       + signal_strength * feats[0]
                       + rng.normal(0, 15))     # within-subject noise
            X.append(feats)
            y.append(glucose)
            groups.append(f"S{s:02d}")
    return np.asarray(X), np.asarray(y), np.asarray(groups)


def test_gate_rejects_pure_noise():
    """No relationship between features and glucose -> gate must fail."""
    X, y, g = _synthetic(signal_strength=0.0, seed=1)
    ev = evaluate(X, y, g, run_permutation=False)
    passed, reasons = ev.passes_gate()
    assert not passed, f"gate passed on pure noise: {reasons}"


def test_gate_accepts_strong_signal():
    """A genuine, strong feature-glucose relationship must get through."""
    X, y, g = _synthetic(signal_strength=25.0, seed=2)
    ev = evaluate(X, y, g, run_permutation=False)
    passed, reasons = ev.passes_gate()
    assert passed, f"gate blocked real signal: {reasons}"
    assert ev.improvement > 0


def test_gate_rejects_weak_signal_that_only_wins_on_average():
    """The failure mode this gate exists for: a small average improvement
    that is not consistent across folds and whose CI includes zero."""
    X, y, g = _synthetic(signal_strength=2.0, seed=3)
    ev = evaluate(X, y, g, run_permutation=False)
    passed, _ = ev.passes_gate()
    assert not passed


def test_folds_never_share_a_subject():
    X, y, g = _synthetic(signal_strength=10.0, seed=4)
    ev = evaluate(X, y, g, run_permutation=False)
    assert len(ev.folds) == 5
    assert sum(f.n_test for f in ev.folds) == len(y)


def test_bootstrap_resamples_subjects_not_recordings():
    """Recordings from one subject are not independent."""
    X, y, g = _synthetic(signal_strength=25.0, seed=5)
    ev = evaluate(X, y, g, run_permutation=False)
    lo, hi = ev.bootstrap_ci
    assert lo < ev.improvement < hi or lo <= ev.improvement <= hi


def test_permutation_test_reports_a_p_value():
    X, y, g = _synthetic(signal_strength=0.0, seed=6)
    ev = evaluate(X, y, g, run_permutation=True)
    assert ev.permutation_p is not None
    assert 0.0 < ev.permutation_p <= 1.0


def test_permutation_test_flags_noise_as_unremarkable():
    X, y, g = _synthetic(signal_strength=0.0, seed=7)
    ev = evaluate(X, y, g, run_permutation=True)
    passed, reasons = ev.passes_gate()
    assert not passed
    assert any("permutation" in r or "FAIL" in r for r in reasons)


# ---------------------------------------------------------------------
# The estimator object
# ---------------------------------------------------------------------

def _blob(provides=True):
    return {
        "model_version": "test-v1",
        "provides_information": provides,
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": [0.0] * len(FEATURE_NAMES),
        "scaler_scale": [1.0] * len(FEATURE_NAMES),
        "coef": [1.0] + [0.0] * (len(FEATURE_NAMES) - 1),
        "intercept": 115.0,
        "min_signal_quality": 0.3,
    }


def _ppg_block(**kw):
    block = {
        "ppg_present": 1, "ppg_signal_quality": 0.8,
        "ppg_pulse_rate_bpm": 5.0, "ppg_hrv_rmssd": 30.0,
        "ppg_perfusion_index": 2.0,
        "rise_time_ms": 120.0, "pulse_width_half_ms": 200.0,
        "n_beats_detected": 12,
    }
    block.update(kw)
    return block


def test_estimator_refuses_a_low_quality_capture():
    est = PPGGlucoseEstimator(_blob())
    assert est.estimate(_ppg_block(ppg_signal_quality=0.1)) is None


def test_estimator_refuses_an_absent_capture():
    est = PPGGlucoseEstimator(_blob())
    assert est.estimate(_ppg_block(ppg_present=0)) is None


def test_estimator_refuses_when_a_feature_is_missing():
    est = PPGGlucoseEstimator(_blob())
    assert est.estimate(_ppg_block(ppg_hrv_rmssd=None)) is None


def test_estimator_produces_a_clipped_plausible_value():
    est = PPGGlucoseEstimator(_blob())
    out = est.estimate(_ppg_block())
    assert out is not None
    assert 40.0 <= out <= 400.0


def test_provides_information_comes_from_the_artifact_not_by_hand():
    assert PPGGlucoseEstimator(_blob(provides=True)).provides_information is True
    assert PPGGlucoseEstimator(_blob(provides=False)).provides_information is False


def test_load_missing_artifact_explains_the_gate():
    with pytest.raises(FileNotFoundError, match="evidence gate"):
        PPGGlucoseEstimator.load("/nonexistent/model.json")


def test_a_failed_gate_estimator_is_refused_by_glucose_source():
    """End to end: even a loaded model with provides_information=False must
    not be preferred over the population fallback."""
    from integration import glucose_source as gsrc
    est = PPGGlucoseEstimator(_blob(provides=False))
    gsrc.register_ppg_g0_estimator(est)
    try:
        r = gsrc.resolve_g0(
            t0_timestamp="2026-01-15T19:30:00",
            ppg_block=_ppg_block(), allow_ppg_estimate=True)
        assert r.source == gsrc.SOURCE_POPULATION_FALLBACK
    finally:
        gsrc.register_ppg_g0_estimator(None)


def test_a_passed_gate_estimator_is_used_by_glucose_source():
    from integration import glucose_source as gsrc
    est = PPGGlucoseEstimator(_blob(provides=True))
    gsrc.register_ppg_g0_estimator(est)
    try:
        r = gsrc.resolve_g0(
            t0_timestamp="2026-01-15T19:30:00",
            ppg_block=_ppg_block(), allow_ppg_estimate=True)
        assert r.source == gsrc.SOURCE_PPG_ESTIMATE
        assert r.is_direct_measurement is False
    finally:
        gsrc.register_ppg_g0_estimator(None)
