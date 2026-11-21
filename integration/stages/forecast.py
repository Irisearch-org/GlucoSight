"""Step 3: forecast glucose at T+60 and T+120 with the two served LightGBM models.

The feature frame is built by `forecasting.evaluate.gbdt_forecaster.encode_inference`, the same
`design` function the trial trained and validated with, so serving cannot drift from training.
Error bands are the 90th percentile of absolute held-out error per horizon and g0 source from the
trial's nested CV (see integration/models/served.json).

The models were trained on meter and CGM g0 only: a PPG-started forecast runs them on a
near-constant g0 and claims no benefit (spec section 3).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import lightgbm as lgb

from forecasting.evaluate.gbdt_forecaster import DM_BAND, encode_inference
from integration.contract import (
    ForecastResult,
    HorizonForecast,
    Profile,
    glucose_range,
)
from integration.errors import ModelUnavailable
from integration.stages.glucose import G0Reading

if TYPE_CHECKING:
    from integration.stages.meal import Meal


def _dm_band(diabetes_type: str) -> tuple[float, bool]:
    """(dm_band code, outside_training). Tipe 1 / Lainnya have no training counterpart."""
    if diabetes_type == "Tipe 2":
        return DM_BAND["t2d"], False
    if diabetes_type == "Prediabetes":
        return DM_BAND["prediabetes"], False
    return DM_BAND["t2d"], True


def bmi_of(profile: Profile) -> float | None:
    if profile.height_cm is None or profile.weight_kg is None:
        return None
    m = profile.height_cm / 100.0
    return profile.weight_kg / (m * m)


class _Horizon:
    def __init__(self, horizon: int, spec: Mapping[str, Any], bands: Mapping[str, float], root: Path):
        model_path = root / spec["path"]
        meta_path = model_path.with_suffix(".json")
        for p in (model_path, meta_path):
            if not p.is_file():
                raise ModelUnavailable(f"Model prakiraan T+{horizon} belum tersedia ({p}).")
        self.booster = lgb.Booster(model_file=str(model_path))
        meta = json.loads(meta_path.read_text())
        self.offset = bool(meta["params"]["offset"])
        self.version = str(spec["version"])
        self.bands = {k: float(v) for k, v in bands.items()}
        self.columns = self.booster.feature_name()


class Forecaster:
    def __init__(self, horizons: Mapping[int, _Horizon]):
        self._h = dict(horizons)

    @classmethod
    def load(cls, spec: Mapping[str, Any], root: Path) -> "Forecaster":
        """`spec` is served.json["forecaster"]; model paths are relative to `root` (the repo)."""
        bands = spec["error_bands_mgdl"]
        return cls({
            60: _Horizon(60, spec["t60"], bands["t60"], root),
            120: _Horizon(120, spec["t120"], bands["t120"], root),
        })

    def predict(self, g0: G0Reading, meal: "Meal", profile: Profile, t0: datetime) -> ForecastResult:
        dm_band, outside = _dm_band(profile.diabetes_type)
        female = None if profile.sex is None else (1.0 if profile.sex == "F" else 0.0)
        hour = t0.hour + t0.minute / 60.0
        features = encode_inference(
            g0.value_mgdl, g0.source, hour,
            meal.carbs_g, meal.protein_g, meal.fat_g, meal.fibre_g,
            profile.age_years, female, bmi_of(profile), dm_band,
        )
        out: dict[int, HorizonForecast] = {}
        for horizon, h in self._h.items():
            pred = float(h.booster.predict(features[h.columns])[0])
            if h.offset:
                pred += g0.value_mgdl
            pred = round(pred, 1)
            out[horizon] = HorizonForecast(
                glucose_mgdl=pred,
                range=glucose_range(pred),
                error_band_mgdl=h.bands[g0.source],
                model_version=h.version,
            )
        return ForecastResult(t60=out[60], t120=out[120], outside_training=outside)
