"""Where g0 comes from — fingerstick, CGM, PPG estimate, or nothing.

The product question: a user with a glucometer types their reading; a user
without one gets the finger scan. These tests pin what each path is
actually allowed to do.
"""

import pytest

from integration import contract, fusion
from integration import glucose_source as gsrc

T0 = "2026-01-15T19:30:00"


@pytest.fixture(autouse=True)
def _clean_registry():
    gsrc.register_ppg_g0_estimator(None)
    yield
    gsrc.register_ppg_g0_estimator(None)


def _ppg_present():
    return {"ppg_present": 1, "ppg_signal_quality": 0.8,
            "ppg_pulse_rate_bpm": 78.0, "ppg_n_windows": 3}


# ---------------------------------------------------------------------
# Fingerstick — the path that works today
# ---------------------------------------------------------------------

def test_fingerstick_is_used_when_no_history_exists():
    r = gsrc.resolve_g0(t0_timestamp=T0, causal_history=[],
                        fingerstick_mgdl=132.0,
                        fingerstick_timestamp="2026-01-15T19:10:00")
    assert r.source == gsrc.SOURCE_FINGERSTICK
    assert r.value_mgdl == 132.0
    assert r.age_minutes == 20.0
    assert r.is_direct_measurement is True
    assert r.trust == 1.0


def test_fresher_fingerstick_beats_older_cgm():
    """Both are real measurements; recency decides."""
    r = gsrc.resolve_g0(
        t0_timestamp=T0,
        causal_history=[{"timestamp": "2026-01-15T17:00:00", "value": 110.0}],
        fingerstick_mgdl=145.0, fingerstick_timestamp="2026-01-15T19:20:00")
    assert r.source == gsrc.SOURCE_FINGERSTICK
    assert r.value_mgdl == 145.0


def test_fresher_cgm_beats_older_fingerstick():
    r = gsrc.resolve_g0(
        t0_timestamp=T0,
        causal_history=[{"timestamp": "2026-01-15T19:25:00", "value": 110.0}],
        fingerstick_mgdl=145.0, fingerstick_timestamp="2026-01-15T16:00:00")
    assert r.source == gsrc.SOURCE_CGM
    assert r.value_mgdl == 110.0


def test_fingerstick_after_t0_is_rejected_as_a_target():
    with pytest.raises(contract.ContractViolation, match="C2"):
        gsrc.resolve_g0(t0_timestamp=T0, fingerstick_mgdl=200.0,
                        fingerstick_timestamp="2026-01-15T20:30:00")


def test_fingerstick_without_a_timestamp_is_rejected():
    with pytest.raises(contract.ContractViolation, match="needs a timestamp"):
        gsrc.resolve_g0(t0_timestamp=T0, fingerstick_mgdl=132.0)


def test_implausible_fingerstick_is_rejected_with_a_readable_message():
    with pytest.raises(contract.ContractViolation, match="mmol/L"):
        gsrc.resolve_g0(t0_timestamp=T0, fingerstick_mgdl=7.4,
                        fingerstick_timestamp="2026-01-15T19:10:00")


def test_a_genuine_extreme_is_accepted():
    """Rejecting a true hypo is worse than accepting a typo."""
    r = gsrc.resolve_g0(t0_timestamp=T0, fingerstick_mgdl=42.0,
                        fingerstick_timestamp="2026-01-15T19:10:00")
    assert r.value_mgdl == 42.0


def test_non_numeric_fingerstick_is_rejected():
    with pytest.raises(contract.ContractViolation, match="not a number"):
        gsrc.resolve_g0(t0_timestamp=T0, fingerstick_mgdl="banyak",
                        fingerstick_timestamp="2026-01-15T19:10:00")


def test_stale_fingerstick_falls_back_and_says_why():
    r = gsrc.resolve_g0(t0_timestamp=T0, fingerstick_mgdl=132.0,
                        fingerstick_timestamp="2026-01-14T19:10:00")
    assert r.source == gsrc.SOURCE_POPULATION_FALLBACK
    assert any("rejected" in msg for msg in r.rejected)


# ---------------------------------------------------------------------
# PPG estimate — gated twice
# ---------------------------------------------------------------------

def test_ppg_path_is_off_by_default():
    r = gsrc.resolve_g0(t0_timestamp=T0, ppg_block=_ppg_present())
    assert r.source == gsrc.SOURCE_POPULATION_FALLBACK
    assert any("E1.2" in m for m in r.rejected)


def test_enabling_without_a_registered_estimator_still_falls_back():
    r = gsrc.resolve_g0(t0_timestamp=T0, ppg_block=_ppg_present(),
                        allow_ppg_estimate=True)
    assert r.source == gsrc.SOURCE_POPULATION_FALLBACK
    assert any("no estimator is registered" in m for m in r.rejected)


def test_a_constant_estimator_is_refused_as_no_better_than_the_fallback():
    """The mean baseline predicts one number for everyone. Labelling that a
    'measurement' is exactly what this module exists to prevent."""
    gsrc.register_ppg_g0_estimator(gsrc.MeanBaselineG0Estimator())
    r = gsrc.resolve_g0(t0_timestamp=T0, ppg_block=_ppg_present(),
                        allow_ppg_estimate=True)
    assert r.source == gsrc.SOURCE_POPULATION_FALLBACK
    assert any("constant predictor" in m for m in r.rejected)


def test_an_informative_estimator_is_used_and_marked_not_a_measurement():
    class Informative:
        model_version = "ppg-glucose-ridge-v1"
        provides_information = True
        def estimate(self, block):
            return 138.0

    gsrc.register_ppg_g0_estimator(Informative())
    r = gsrc.resolve_g0(t0_timestamp=T0, ppg_block=_ppg_present(),
                        allow_ppg_estimate=True)
    assert r.source == gsrc.SOURCE_PPG_ESTIMATE
    assert r.value_mgdl == 138.0
    assert r.is_direct_measurement is False, (
        "a PPG inference must never be labelled a direct measurement")
    assert "NOT a glucose measurement" in r.detail


def test_a_direct_measurement_always_beats_a_ppg_estimate():
    class Informative:
        model_version = "ppg-glucose-ridge-v1"
        provides_information = True
        def estimate(self, block):
            return 138.0

    gsrc.register_ppg_g0_estimator(Informative())
    r = gsrc.resolve_g0(
        t0_timestamp=T0, ppg_block=_ppg_present(), allow_ppg_estimate=True,
        fingerstick_mgdl=121.0, fingerstick_timestamp="2026-01-15T19:15:00")
    assert r.source == gsrc.SOURCE_FINGERSTICK


def test_registering_a_non_conforming_estimator_is_refused():
    class Bad:
        pass
    with pytest.raises(TypeError, match="model_version"):
        gsrc.register_ppg_g0_estimator(Bad())


def test_ppg_absent_means_no_estimate_even_when_enabled():
    class Informative:
        model_version = "v1"
        provides_information = True
        def estimate(self, block):
            return 138.0

    gsrc.register_ppg_g0_estimator(Informative())
    r = gsrc.resolve_g0(t0_timestamp=T0, allow_ppg_estimate=True,
                        ppg_block={"ppg_present": 0})
    assert r.source == gsrc.SOURCE_POPULATION_FALLBACK


# ---------------------------------------------------------------------
# Trust ordering
# ---------------------------------------------------------------------

def test_trust_is_ordered_measurement_above_inference_above_nothing():
    t = gsrc.SOURCE_TRUST
    assert (t[gsrc.SOURCE_FINGERSTICK] >= t[gsrc.SOURCE_CGM]
            > t[gsrc.SOURCE_PPG_ESTIMATE] > t[gsrc.SOURCE_POPULATION_FALLBACK])


# ---------------------------------------------------------------------
# Through fusion
# ---------------------------------------------------------------------

def _assemble(**kw):
    base = dict(meal_id="m1", participant_id="P1", t0_timestamp=T0,
                source_dataset="live_capture", delta_t_minutes=60.0)
    base.update(kw)
    return fusion.assemble(**base)


def test_fusion_reports_the_source_on_every_meal():
    meal = _assemble(fingerstick_mgdl=132.0,
                     fingerstick_timestamp="2026-01-15T19:10:00")
    assert meal["g0"] == 132.0
    assert meal["g0_source"] == gsrc.SOURCE_FINGERSTICK
    assert meal["g0_is_direct_measurement"] is True


def test_fusion_leaves_slope_undefined_for_a_lone_fingerstick():
    """One point defines no slope. Zero would assert a flat trend."""
    meal = _assemble(fingerstick_mgdl=132.0,
                     fingerstick_timestamp="2026-01-15T19:10:00")
    assert meal["glucose_slope_30min"] is None


def test_fusion_keeps_slope_when_a_cgm_trace_is_present():
    meal = _assemble(glucose_history=[
        {"timestamp": "2026-01-15T19:00:00", "value": 100.0},
        {"timestamp": "2026-01-15T19:25:00", "value": 125.0},
    ])
    assert meal["glucose_slope_30min"] is not None
    assert meal["g0_source"] == gsrc.SOURCE_CGM


def test_fusion_falls_back_and_records_it_when_nothing_is_available():
    meal = _assemble()
    assert meal["g0"] is None
    assert meal["g0_mgdl"] == 120.0
    assert meal["g0_source"] == gsrc.SOURCE_POPULATION_FALLBACK
