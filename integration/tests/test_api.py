"""End-to-end tests for the MVP endpoint.

The degradation cases matter more than the happy path here: a demo that
works only when all three modalities are present is a demo that will break
in front of an audience, and the contract's whole missingness design exists
for the cases below.
"""

import math

import pytest
from fastapi.testclient import TestClient

from integration.api import app

client = TestClient(app)


def _payload(**kw):
    base = {
        "participant_id": "P042",
        "t0_timestamp": "2026-01-15T19:30:00",
        "delta_t_minutes": 60.0,
        "portion_reported": 2,
        "glucose_history": [
            {"timestamp": "2026-01-15T12:00:00", "value": 142.0},
            {"timestamp": "2026-01-15T18:45:00", "value": 118.0},
        ],
    }
    base.update(kw)
    return base


def _ppg(seconds=33.1, fs=30, bpm=78):
    n = int(seconds * fs)
    return {
        "samples": [0.5 + 0.2 * math.sin(2 * math.pi * (bpm / 60) * (i / fs))
                    for i in range(n)],
        "timestamps_seconds": [i / fs for i in range(n)],
        "sampling_rate_hz": fs,
    }


def test_health_declares_whether_a_model_is_being_served():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["schema_version"] == "1.2"
    assert "is_trained_model" in body
    assert "Not a medical device" in body["disclaimer"]


def test_contract_endpoint_exposes_the_live_spec():
    body = client.get("/contract").json()
    assert body["schema_version"] == "1.2"
    names = {f["name"] for f in body["cv"]}
    assert {"carbs_source", "cv_present", "portion_reported"} <= names
    assert body["ringfenced"] == ["ppg_glucose_estimate"]


def test_predict_with_no_modalities_at_all_still_returns_a_valid_forecast():
    """The most important degradation case: nothing but a glucose history."""
    r = client.post("/predict", json=_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    for m in ("cv", "nlp", "ppg"):
        assert body["modalities"][m]["present"] == 0
        # Down-weighted, never dropped.
        assert body["modalities"][m]["weight"] > 0
    assert 70 <= body["forecast"]["glucose_t60"] <= 400


def test_predict_with_a_note_only():
    r = client.post("/predict", json=_payload(
        note_text="nasi goreng porsi besar, digoreng"))
    assert r.status_code == 200, r.text
    nlp = r.json()["modalities"]["nlp"]
    assert nlp["present"] == 1
    assert nlp["confidence"] == 0.0
    assert nlp["calibrated"] is False
    assert nlp["calibration_status"] == "unavailable"
    assert nlp["weight"] > 0


def test_predict_with_a_timestamped_ppg_capture_only():
    r = client.post("/predict", json=_payload(ppg_capture=_ppg()))
    assert r.status_code == 200, r.text
    body = r.json()
    ppg = body["modalities"]["ppg"]
    assert ppg["present"] == 1
    assert 40 <= ppg["pulse_rate_bpm"] <= 200
    assert body["modality_processing"]["ppg"]["n_valid_windows"] == 3
    assert 0.0 <= ppg["perfusion_index"] <= 20.0


def test_ppg_processing_diagnostics_disclose_limited_scalar_capture_quality():
    body = client.post("/predict", json=_payload(ppg_capture=_ppg())).json()
    status = body["modality_processing"]["ppg"]
    assert status["status"] == "processed"
    assert 0.0 <= status["clip_fraction_proxy"] <= 1.0
    assert "RGB camera channel clipping" in status["limitation"]


def test_invalid_ppg_timestamping_is_reported_and_marked_absent():
    capture = _ppg()
    capture["timestamps_seconds"] = list(reversed(capture["timestamps_seconds"]))
    response = client.post("/predict", json=_payload(ppg_capture=capture))
    assert response.status_code == 200
    body = response.json()
    assert body["modalities"]["ppg"]["present"] == 0
    assert body["modality_processing"]["ppg"]["status"] == "invalid"
    assert "strictly increasing" in body["modality_processing"]["ppg"]["detail"]


def test_response_never_contains_the_ringfenced_estimate():
    """E1 rule 2: never surfaced to a user."""
    r = client.post("/predict", json=_payload(
        ppg_capture=_ppg(), note_text="nasi goreng"))
    assert "ppg_glucose_estimate" not in r.text


def test_noncausal_history_is_dropped_and_reported_not_used():
    r = client.post("/predict", json=_payload(glucose_history=[
        {"timestamp": "2026-01-15T18:45:00", "value": 118.0},
        {"timestamp": "2026-01-15T20:30:00", "value": 205.0},   # T+60 TARGET
        {"timestamp": "2026-01-15T21:30:00", "value": 180.0},   # T+120 TARGET
    ]))
    assert r.status_code == 200, r.text
    used = r.json()["glucose_history_used"]
    assert used["n_causal_readings"] == 1
    assert used["n_dropped_as_noncausal"] == 2
    assert used["g0"] == 118.0, "a post-t0 reading reached g0 — C2 violated"


def test_empty_history_reports_the_fallback_rather_than_hiding_it():
    r = client.post("/predict", json=_payload(glucose_history=[]))
    body = r.json()
    assert body["glucose_history_used"]["g0"] is None
    assert body["forecast"]["used_fallback_g0"] is True
    assert body["forecast"]["prediction_confidence"] <= 0.15


def test_baseline_is_labelled_as_a_baseline():
    body = client.post("/predict", json=_payload()).json()
    f = body["forecast"]
    assert f["is_trained_model"] is False
    assert "B2 baseline" in f["basis"]
    assert f["model_version"] == "baseline-b2-v1"


def test_prediction_confidence_is_capped_for_a_baseline():
    body = client.post("/predict", json=_payload(
        note_text="nasi goreng", ppg_capture=_ppg())).json()
    assert body["forecast"]["prediction_confidence"] <= 0.45


def test_demo_mode_reports_modality_features_do_not_change_baseline_glucose():
    base = client.post("/predict", json=_payload()).json()
    with_modalities = client.post("/predict", json=_payload(
        note_text="digoreng porsi besar",
        ppg_capture=_ppg(),
    )).json()
    assert base["prediction_mode"] == "baseline_demo"
    assert with_modalities["prediction_mode"] == "baseline_demo"
    assert base["forecast"]["glucose_t60"] == with_modalities["forecast"]["glucose_t60"]
    assert base["forecast"]["glucose_t120"] == with_modalities["forecast"]["glucose_t120"]
    assert with_modalities["forecast_input_usage"]["not_used_for_glucose_value"] == [
        "cv", "nlp", "ppg", "meal_context"
    ]


def test_b2_horizon_selector_is_reported_without_changing_both_fixed_outputs():
    t60 = client.post("/predict", json=_payload(delta_t_minutes=60)).json()
    t120 = client.post("/predict", json=_payload(delta_t_minutes=120)).json()
    assert t60["forecast"]["requested_horizon_minutes"] == 60
    assert t120["forecast"]["requested_horizon_minutes"] == 120
    assert t60["forecast"]["target_horizons_minutes"] == {
        "glucose_t60": 60, "glucose_t120": 120
    }
    assert t60["forecast"]["glucose_t60"] == t120["forecast"]["glucose_t60"]
    assert t60["forecast"]["glucose_t120"] == t120["forecast"]["glucose_t120"]
    assert t120["forecast"]["delta_t_minutes"] == 120


def test_b2_rejects_non_supported_horizons():
    for delta in (62.0, 95.0, 200.0):
        r = client.post("/predict", json=_payload(delta_t_minutes=delta))
        assert r.status_code == 422


def test_trained_mode_fails_closed_when_artifact_is_not_configured():
    r = client.post("/predict", json=_payload(prediction_mode="trained_forecast"))
    assert r.status_code == 409
    assert "No trained forecasting artifact" in r.json()["detail"]


def test_health_reports_nlp_dish_identity_mode_status():
    """nlp/models/dish_identity_ridge.json has been trained against the real
    ShanghaiT2DM data and passed its gate, so this mode is live."""
    body = client.get("/health").json()
    assert "nlp_dish_identity_mode" in body
    assert "available" in body["nlp_dish_identity_mode"]
    assert "nlp-dish-identity-ridge-v1" in body["nlp_dish_identity_mode"]


def test_nlp_dish_identity_mode_returns_a_trained_prediction():
    r = client.post("/predict", json=_payload(
        prediction_mode="nlp_dish_identity_forecast",
        note_text="Rice 150 g | Egg 60 g",
    ))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction_mode"] == "nlp_dish_identity_forecast"
    assert body["forecast"]["is_trained_model"] is True
    assert body["forecast"]["model_version"] == "nlp-dish-identity-ridge-v1"
    assert 70 <= body["forecast"]["glucose_t60"] <= 400
    assert body["forecast_input_usage"]["used_for_glucose_value"] == [
        "g0_mgdl", "nlp_total_grams", "nlp_dish_names",
        "is_fried_cooking", "is_large_portion",
    ]


def test_nlp_dish_identity_mode_t120_still_falls_back_to_b2():
    """No T+120 arm exists for this model anywhere in the repo."""
    trained = client.post("/predict", json=_payload(
        prediction_mode="nlp_dish_identity_forecast", delta_t_minutes=120.0,
    )).json()
    baseline = client.post("/predict", json=_payload(delta_t_minutes=120.0)).json()
    assert trained["forecast"]["glucose_t120"] == baseline["forecast"]["glucose_t120"]
    assert "T+120" in trained["forecast"]["basis"]


def test_out_of_range_delta_t_is_rejected_by_validation():
    r = client.post("/predict", json=_payload(delta_t_minutes=999.0))
    assert r.status_code == 422


def test_glucose_range_matches_the_contract_thresholds():
    body = client.post("/predict", json=_payload()).json()
    f = body["forecast"]
    t60 = f["glucose_t60"]
    expected = "normal" if t60 < 140 else ("borderline" if t60 < 200 else "high")
    assert f["glucose_range"] == expected


def test_invalid_image_data_url_is_reported_as_invalid():
    r = client.post("/predict", json=_payload(meal_image_data_url="file:///etc/passwd"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["modalities"]["cv"]["present"] == 0
    assert body["modality_processing"]["cv"]["status"] == "invalid"
    assert "data URL" in body["modality_processing"]["cv"]["detail"]


def test_demo_page_is_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "GlucoSight" in r.text


# ---------------------------------------------------------------------
# Staleness — the mirror image of C2, found by running the demo
# ---------------------------------------------------------------------

def test_stale_but_causal_reading_is_rejected_not_used():
    """A reading from months ago is before t0, so C2 passes it. It is still
    not a pre-meal glucose value, and persistence from it is not a baseline."""
    r = client.post("/predict", json=_payload(glucose_history=[
        {"timestamp": "2026-01-15T12:00:00", "value": 142.0},
    ], t0_timestamp="2026-09-10T19:30:00"))
    assert r.status_code == 200, r.text
    used = r.json()["glucose_history_used"]
    assert used["n_causal_readings"] == 1, "the reading is causal and must be counted"
    assert used["g0"] is None, "a 238-day-old reading was used as pre-meal glucose"
    assert used["g0_rejected_as_stale"] is True
    assert r.json()["forecast"]["used_fallback_g0"] is True


def test_a_fresh_reading_is_not_rejected():
    r = client.post("/predict", json=_payload())
    used = r.json()["glucose_history_used"]
    assert used["g0"] == 118.0
    assert used["g0_rejected_as_stale"] is False


def test_staleness_boundary_is_the_contract_constant():
    from integration import contract as c
    inside = client.post("/predict", json=_payload(
        t0_timestamp="2026-01-15T19:30:00",
        glucose_history=[{"timestamp": "2026-01-15T14:00:00", "value": 130.0}],
    )).json()
    assert inside["glucose_history_used"]["g0"] == 130.0

    outside = client.post("/predict", json=_payload(
        t0_timestamp="2026-01-15T19:30:00",
        glucose_history=[{"timestamp": "2026-01-15T13:00:00", "value": 130.0}],
    )).json()
    assert outside["glucose_history_used"]["g0"] is None
    assert c.MAX_G0_AGE_MINUTES == 360.0


# ---------------------------------------------------------------------
# Dual g0 path: fingerstick, or the finger scan
# ---------------------------------------------------------------------

def test_fingerstick_path_end_to_end():
    r = client.post("/predict", json=_payload(
        glucose_history=[],
        fingerstick_mgdl=132.0,
        fingerstick_timestamp="2026-01-15T19:10:00"))
    assert r.status_code == 200, r.text
    gs = r.json()["glucose_source"]
    assert gs["source"] == "fingerstick"
    assert gs["value_mgdl"] == 132.0
    assert gs["is_direct_measurement"] is True
    assert r.json()["forecast"]["used_fallback_g0"] is False


def test_fingerstick_has_higher_baseline_confidence_than_population_fallback():
    with_fs = client.post("/predict", json=_payload(
        glucose_history=[], fingerstick_mgdl=132.0,
        fingerstick_timestamp="2026-01-15T19:10:00")).json()
    without = client.post("/predict", json=_payload(glucose_history=[])).json()
    assert (with_fs["forecast"]["prediction_confidence"]
            > without["forecast"]["prediction_confidence"])


def test_bad_fingerstick_returns_a_readable_422():
    r = client.post("/predict", json=_payload(
        glucose_history=[], fingerstick_mgdl=7.4,
        fingerstick_timestamp="2026-01-15T19:10:00"))
    assert r.status_code == 422
    assert "mmol/L" in r.json()["detail"]


def test_ppg_g0_path_is_refused_and_explains_why():
    """The user has no glucometer and opts for the scan. Today that falls
    back, and the response says which rule and which evidence blocked it."""
    r = client.post("/predict", json=_payload(
        glucose_history=[], ppg_capture=_ppg(), allow_ppg_g0_estimate=True))
    assert r.status_code == 200, r.text
    gs = r.json()["glucose_source"]
    assert gs["source"] == "population_fallback"
    assert gs["is_direct_measurement"] is False
    assert any("estimator" in m for m in gs["rejected_candidates"])


def test_ppg_capture_without_opting_in_cites_the_ringfence():
    r = client.post("/predict", json=_payload(
        glucose_history=[], ppg_capture=_ppg()))
    gs = r.json()["glucose_source"]
    assert any("E1.2" in m for m in gs["rejected_candidates"])


def test_health_declares_which_g0_sources_are_live():
    body = client.get("/health").json()
    assert body["g0_sources"]["fingerstick"].startswith("available")
    assert body["g0_sources"]["ppg_estimate"].startswith("UNAVAILABLE")


def test_forecast_carries_the_g0_source_for_stratified_reporting():
    body = client.post("/predict", json=_payload(
        glucose_history=[], fingerstick_mgdl=132.0,
        fingerstick_timestamp="2026-01-15T19:10:00")).json()
    assert body["forecast"]["g0_source"] == "fingerstick"
    assert body["forecast"]["g0_is_direct_measurement"] is True
