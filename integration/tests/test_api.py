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
        "delta_t_minutes": 62.0,
        "portion_reported": 2,
        "glucose_history": [
            {"timestamp": "2026-01-15T12:00:00", "value": 142.0},
            {"timestamp": "2026-01-15T18:45:00", "value": 118.0},
        ],
    }
    base.update(kw)
    return base


def _ppg(seconds=30, fs=30, bpm=78):
    n = int(seconds * fs)
    return [1.0 + 0.05 * math.sin(2 * math.pi * (bpm / 60) * (i / fs))
            for i in range(n)]


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
    assert nlp["weight"] > 0


def test_predict_with_a_ppg_capture_only():
    r = client.post("/predict", json=_payload(ppg_signal=_ppg()))
    assert r.status_code == 200, r.text
    ppg = r.json()["modalities"]["ppg"]
    assert ppg["present"] == 1
    assert 40 <= ppg["pulse_rate_bpm"] <= 200


def test_response_never_contains_the_ringfenced_estimate():
    """E1 rule 2: never surfaced to a user."""
    r = client.post("/predict", json=_payload(
        ppg_signal=_ppg(), note_text="nasi goreng"))
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
        note_text="nasi goreng", ppg_signal=_ppg())).json()
    assert body["forecast"]["prediction_confidence"] <= 0.45


def test_delta_t_is_echoed_not_replaced_by_60():
    body = client.post("/predict", json=_payload(delta_t_minutes=95.0)).json()
    assert body["forecast"]["delta_t_minutes"] == 95.0


def test_delta_t_far_from_the_horizon_lowers_confidence():
    near = client.post("/predict", json=_payload(delta_t_minutes=60.0)).json()
    far = client.post("/predict", json=_payload(delta_t_minutes=200.0)).json()
    assert (far["forecast"]["prediction_confidence"]
            < near["forecast"]["prediction_confidence"])


def test_out_of_range_delta_t_is_rejected_by_validation():
    r = client.post("/predict", json=_payload(delta_t_minutes=999.0))
    assert r.status_code == 422


def test_glucose_range_matches_the_contract_thresholds():
    body = client.post("/predict", json=_payload()).json()
    f = body["forecast"]
    t60 = f["glucose_t60"]
    expected = "normal" if t60 < 140 else ("borderline" if t60 < 200 else "high")
    assert f["glucose_range"] == expected


def test_a_broken_track_degrades_to_absent_rather_than_500():
    """A nonexistent image path must not take the service down."""
    r = client.post("/predict", json=_payload(
        meal_image_path="/nonexistent/definitely_not_a_photo.jpg"))
    assert r.status_code == 200, r.text
    assert r.json()["modalities"]["cv"]["present"] == 0


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


def test_fingerstick_beats_the_population_fallback_in_confidence():
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
        glucose_history=[], ppg_signal=_ppg(), allow_ppg_g0_estimate=True))
    assert r.status_code == 200, r.text
    gs = r.json()["glucose_source"]
    assert gs["source"] == "population_fallback"
    assert gs["is_direct_measurement"] is False
    assert any("estimator" in m for m in gs["rejected_candidates"])


def test_ppg_capture_without_opting_in_cites_the_ringfence():
    r = client.post("/predict", json=_payload(
        glucose_history=[], ppg_signal=_ppg()))
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
