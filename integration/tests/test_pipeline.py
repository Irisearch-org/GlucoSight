"""Pipeline end to end: real forecaster, real PPG ridge, fakes only at Embedder / VisionNamer."""

import base64

import pytest

from integration.contract import PredictRequestV2
from integration.errors import InvalidInput, ModelUnavailable
from integration.pipeline import (
    decode_image_data_url,
    forecast_meal,
    photo_items,
    scan_glucose,
    search_dishes,
)
from integration.stages.glucose import InvalidCapture, StaleReading
from integration.stages.meal import MealError

from .conftest import NOW, FakeNamer, ppg_capture

PROFILE = {"age_years": 52, "sex": "F", "height_cm": 160, "weight_kg": 70, "diabetes_type": "Tipe 2"}


def _request(**patch) -> PredictRequestV2:
    body = {
        "t0_timestamp": "2026-01-15T12:00:00Z",
        "glucose": {"source": "meter", "value_mgdl": 123, "timestamp": "2026-01-15T11:50:00Z"},
        "meal": {
            "items": [
                {"row_id": "TST01", "grams": 150, "source": "manual"},
                {"row_id": "TST03", "grams": 50, "source": "manual"},
            ],
            "vlm_model_version": None,
        },
        "profile": PROFILE,
    }
    body.update(patch)
    return PredictRequestV2.model_validate(body)


def test_meter_and_two_manual_items_end_to_end(deps, table):
    r = forecast_meal(_request(), deps)
    assert r.schema_version == "2.0"
    assert (r.g0.source, r.g0.value_mgdl, r.g0.error_range_mgdl) == ("meter", 123.0, 18.45)
    assert r.g0.age_minutes == pytest.approx(10.0)
    # nasi 150 g + tempe 50 g from the synthetic table
    t = r.meal.totals
    assert (t.carbs_g, t.protein_g, t.fat_g, t.fibre_g, t.kcal) == (66.0, 13.5, 10.75, 1.75, 420.0)
    assert [i.grams for i in r.meal.items] == [150.0, 50.0]
    assert r.meal.items[0].carbs_g == 60.0
    assert r.meal.dish_table_build == table.build_hash
    assert r.forecast.t60.range in ("normal", "borderline", "high")
    assert r.forecast.t60.error_band_mgdl > 0 and r.forecast.t120.error_band_mgdl > 0
    assert r.outside_training is False
    assert "bukan alat medis" in r.disclaimer


def test_item_with_unknown_fibre_makes_the_total_null_not_zero(deps):
    meal = {"items": [{"row_id": "TST01", "grams": 100, "source": "manual"},
                      {"row_id": "TST02", "portion": 1, "source": "manual"}],
            "vlm_model_version": None}
    r = forecast_meal(_request(meal=meal), deps)
    assert r.meal.totals.fibre_g is None
    assert r.meal.items[1].fibre_g is None and r.meal.items[1].grams == 80.0  # 1.0 x serving_g


def test_photo_items_keep_their_match_evidence_and_vlm_version(deps):
    meal = {"items": [{"row_id": "TST01", "portion": 2, "source": "photo",
                       "similarity": 0.91, "margin": 0.12}],
            "vlm_model_version": "fake-vlm"}
    r = forecast_meal(_request(meal=meal), deps)
    assert r.meal.vlm_model_version == "fake-vlm"
    assert r.meal.items[0].source == "photo" and r.meal.items[0].grams == 150.0


def test_tipe_1_is_flagged_outside_training(deps):
    r = forecast_meal(_request(profile={**PROFILE, "diabetes_type": "Tipe 1"}), deps)
    assert r.outside_training is True


def test_stale_reading_is_refused_not_defaulted(deps):
    glucose = {"source": "cgm", "value_mgdl": 130, "timestamp": "2026-01-15T10:30:00Z"}  # 90 min
    with pytest.raises(StaleReading):
        forecast_meal(_request(glucose=glucose), deps)


def test_scan_then_predict_uses_the_ppg_source_and_band(deps, served):
    reading = scan_glucose(ppg_capture(), deps)
    assert reading.source == "ppg" and reading.error_range_mgdl == 25.0
    r = forecast_meal(_request(glucose={"source": "ppg", "reading": reading.model_dump()}), deps)
    assert r.g0.source == "ppg"
    bands = served["forecaster"]["error_bands_mgdl"]
    assert r.forecast.t60.error_band_mgdl == bands["t60"]["ppg"]
    assert r.forecast.t120.error_band_mgdl == bands["t120"]["ppg"]


def test_invalid_scan_is_refused(deps):
    with pytest.raises(InvalidCapture):
        scan_glucose(ppg_capture(torch_verified=False), deps)


def test_unknown_row_is_a_meal_error(deps):
    with pytest.raises(MealError):
        forecast_meal(_request(meal={"items": [{"row_id": "NOPE", "grams": 10, "source": "manual"}]}), deps)


def test_item_without_grams_or_portion_is_a_meal_error(deps):
    with pytest.raises(MealError):
        forecast_meal(_request(meal={"items": [{"row_id": "TST01", "source": "manual"}]}), deps)


def test_missing_forecaster_or_table_is_model_unavailable(deps):
    deps.forecaster = None
    with pytest.raises(ModelUnavailable):
        forecast_meal(_request(), deps)
    deps.table = None
    with pytest.raises(ModelUnavailable):
        search_dishes("nasi", 5, deps)


def test_photo_accepts_the_confident_item_and_sends_the_ambiguous_one_to_manual_entry(deps, table):
    data_url = "data:image/jpeg;base64," + base64.b64encode(b"not-decoded-by-the-fake").decode()
    r = photo_items(data_url, deps)
    assert r.dish_table_build == table.build_hash and r.vlm_model_version == "fake-vlm"
    sure, unsure = r.items
    assert sure.accepted and sure.row.id == "TST01"
    assert not unsure.accepted and unsure.row is None and unsure.vlm_name == "gado-gado"
    assert {s.id for s in unsure.suggestions} >= {"TST01", "TST02"}
    assert unsure.similarity < 0.80


def test_photo_naming_nothing_returns_no_items(deps):
    deps.namer = FakeNamer([])
    data_url = "data:image/png;base64," + base64.b64encode(b"x").decode()
    assert photo_items(data_url, deps).items == []


@pytest.mark.parametrize("data_url", [
    "not-a-data-url", "data:image/gif;base64,AAAA", "data:image/jpeg;base64,@@@", "data:image/jpeg;base64,"])
def test_bad_image_data_urls_are_invalid_input(data_url):
    with pytest.raises(InvalidInput):
        decode_image_data_url(data_url)


def test_search_dishes_autocompletes(deps):
    assert [r.id for r in search_dishes("nas", 5, deps)] == ["TST01"]
