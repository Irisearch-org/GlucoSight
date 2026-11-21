"""Step 3: the two served LightGBM models, loaded from the real artifacts."""

import math
from datetime import datetime

import pytest

from forecasting.evaluate.gbdt_forecaster import encode_inference
from integration.clock import LOCAL_TZ
from integration.contract import ForecastResult, Profile, glucose_range
from integration.stages.forecast import Forecaster, bmi_of
from integration.stages.glucose import G0Reading
from integration.stages.meal import resolve_meal

from .conftest import NOW

T0 = datetime(2026, 1, 15, 19, 0, tzinfo=LOCAL_TZ)
FULL = Profile(age_years=52, sex="F", height_cm=160, weight_kg=70, diabetes_type="Tipe 2")


def _g0(value=130.0, source="meter") -> G0Reading:
    return G0Reading(value, source, NOW, 5.0, None, None, None, None)


@pytest.fixture
def meal(table):
    return resolve_meal(
        [{"row_id": "TST01", "grams": 150, "source": "manual"},
         {"row_id": "TST03", "grams": 50, "source": "manual"}],
        table, vlm_model_version=None)


def test_fixed_input_gives_forecasts_matching_a_hand_built_row(real_forecaster, meal, served):
    r = real_forecaster.predict(_g0(), meal, FULL, T0)
    assert isinstance(r, ForecastResult)
    bands = served["forecaster"]["error_bands_mgdl"]
    for key, h in (("t60", r.t60), ("t120", r.t120)):
        assert 40.0 < h.glucose_mgdl < 400.0
        assert h.range == glucose_range(h.glucose_mgdl)
        assert h.error_band_mgdl == bands[key]["meter"]
        assert h.model_version == served["forecaster"][key]["version"]
        # Independent check of the whole encode -> predict path: the documented column order,
    # filled by hand, straight into each booster.
    row = [130.0, 1.0, 0.0, math.sin(2 * math.pi * 19 / 24), math.cos(2 * math.pi * 19 / 24),
           66.0, 13.5, 10.75, 1.75, 52.0, 1.0, 70 / 1.6 ** 2, 2.0]
    for key, h in ((60, r.t60), (120, r.t120)):
        direct = real_forecaster._h[key].booster.predict([row])[0]
        assert h.glucose_mgdl == pytest.approx(direct, abs=0.06)


def test_prediction_is_deterministic(real_forecaster, meal):
    a = real_forecaster.predict(_g0(), meal, FULL, T0)
    b = real_forecaster.predict(_g0(), meal, FULL, T0)
    assert a == b


def test_encoder_columns_match_the_served_models(real_forecaster):
    frame = encode_inference(120, "meter", 19.0, 50, 10, 10, 2.0, 50, 1.0, 25.0, 2.0)
    for h in (60, 120):
        assert list(frame.columns) == real_forecaster._h[h].columns


@pytest.mark.parametrize("diabetes_type, outside", [
    ("Tipe 2", False), ("Prediabetes", False), ("Tipe 1", True), ("Lainnya", True)])
def test_only_type_1_and_other_are_flagged_outside_training(
        real_forecaster, meal, diabetes_type, outside):
    p = Profile(52, "M", 170, 70, diabetes_type)
    assert real_forecaster.predict(_g0(), meal, p, T0).outside_training is outside


def test_missing_profile_values_are_nan_not_zero(real_forecaster, meal):
    empty = Profile(None, None, None, None, "Tipe 2")
    r = real_forecaster.predict(_g0(), meal, empty, T0)
    assert math.isfinite(r.t60.glucose_mgdl) and math.isfinite(r.t120.glucose_mgdl)
    assert bmi_of(empty) is None
    assert bmi_of(Profile(40, "M", 170, None, "Tipe 2")) is None


def test_bmi_comes_from_height_and_weight():
    assert bmi_of(FULL) == pytest.approx(70 / 1.6 ** 2)


def test_unknown_fibre_is_passed_as_missing(real_forecaster, table):
    # TST02 has no fibre value: the meal fibre is None and the forecast still runs.
    meal = resolve_meal([{"row_id": "TST02", "grams": 100, "source": "manual"}], table,
                        vlm_model_version=None)
    assert meal.fibre_g is None
    assert math.isfinite(real_forecaster.predict(_g0(), meal, FULL, T0).t60.glucose_mgdl)


@pytest.mark.parametrize("source", ["meter", "cgm", "ppg"])
def test_error_band_follows_the_g0_source(real_forecaster, meal, served, source):
    r = real_forecaster.predict(_g0(source=source), meal, FULL, T0)
    bands = served["forecaster"]["error_bands_mgdl"]
    assert r.t60.error_band_mgdl == bands["t60"][source]
    assert r.t120.error_band_mgdl == bands["t120"][source]
