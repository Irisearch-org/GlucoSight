"""Contract v2.0: the schema, nothing else."""

import pytest
from pydantic import ValidationError

from integration import contract
from integration.contract import PredictRequestV2, glucose_range

BASE = {
    "t0_timestamp": "2026-01-15T12:00:00Z",
    "glucose": {"source": "meter", "value_mgdl": 123, "timestamp": "2026-01-15T11:50:00Z"},
    "meal": {"items": [{"row_id": "X", "grams": 100, "source": "manual"}], "vlm_model_version": None},
    "profile": {"age_years": None, "sex": None, "height_cm": None, "weight_kg": None,
                "diabetes_type": "Tipe 2"},
}


@pytest.mark.parametrize("value, expected", [
    (60, "normal"), (139.9, "normal"), (140, "borderline"), (199.9, "borderline"),
    (200, "high"), (350, "high")])
def test_glucose_range_bands(value, expected):
    assert glucose_range(value) == expected


def test_schema_version_and_horizons():
    assert contract.SCHEMA_VERSION == "2.0"
    assert contract.HORIZONS == (60, 120)


def test_request_accepts_meter_and_ppg_glucose_variants():
    assert PredictRequestV2.model_validate(BASE).glucose.source == "meter"
    ppg = {**BASE, "glucose": {"source": "ppg", "reading": {
        "value_mgdl": 110, "source": "ppg", "timestamp": "2026-01-15T11:55:00Z", "age_minutes": 0,
        "error_range_mgdl": 25, "heart_rate_bpm": 70, "capture_valid": True,
        "ppg_model_version": "v"}}}
    assert PredictRequestV2.model_validate(ppg).glucose.source == "ppg"


@pytest.mark.parametrize("patch", [
    {"profile": {"diabetes_type": "Tipe 9"}},
    {"profile": {"diabetes_type": "Tipe 2", "sex": "X"}},
    {"glucose": {"source": "population", "value_mgdl": 120, "timestamp": "2026-01-15T11:50:00Z"}},
    {"meal": {"items": [{"row_id": "X", "grams": -5, "source": "manual"}]}},
    {"meal": {"items": [{"row_id": "X", "portion": 3, "source": "manual"}]}},
    {"meal": {"items": [{"row_id": "X", "grams": 5, "source": "nlp"}]}},
])
def test_invalid_requests_are_rejected(patch):
    with pytest.raises(ValidationError):
        PredictRequestV2.model_validate({**BASE, **patch})


def test_v1_fields_are_gone():
    for dropped in ("CONFIDENCE_FLOOR", "MIN_MODALITY_WEIGHT", "modality_weight", "apply_fallbacks",
                    "validate_meal_input", "filter_causal_history", "PPG_FIELDS"):
        assert not hasattr(contract, dropped)
