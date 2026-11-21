"""GlucoSight data contract v2.0: the schema the three stages exchange. Nothing else.

Two layers, both defined here and nowhere else:

* Domain dataclasses handed between stages in Python (`Profile`, `HorizonForecast`,
  `ForecastResult`; `G0Reading` lives in `integration.stages.glucose` and `Meal` in
  `integration.stages.meal`, next to the code that builds them).
* The HTTP bodies (pydantic) for `integration/api.py`.

Resolution of the contract ticket (#23): missing != zero (a null nutrient stays null),
no fallback macros (F6), horizons fixed at 60 and 120 so there is no delta_t (F7),
no population fallback for g0 (F12: a reading older than 60 minutes is refused).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "2.0"
HORIZONS = (60, 120)

DISCLAIMER = (
    "Prototipe riset — bukan alat medis. Perkiraan ini tidak boleh dipakai untuk "
    "keputusan pengobatan, dosis insulin, atau pengaturan obat."
)

G0_MIN_MGDL = 40.0
G0_MAX_MGDL = 400.0
G0_MAX_AGE_MINUTES = 60.0

DiabetesType = Literal["Tipe 2", "Tipe 1", "Prediabetes", "Lainnya"]
G0Source = Literal["meter", "cgm", "ppg"]
GlucoseRange = Literal["normal", "borderline", "high"]


def glucose_range(value_mgdl: float) -> GlucoseRange:
    """<140 normal, 140-199 borderline, >=200 high."""
    if value_mgdl < 140.0:
        return "normal"
    if value_mgdl < 200.0:
        return "borderline"
    return "high"


# ---------------------------------------------------------------------------
# Domain dataclasses (stage <-> stage)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Profile:
    """User profile as typed in the app. Any numeric field may be missing (None)."""
    age_years: Optional[float]
    sex: Optional[Literal["M", "F"]]
    height_cm: Optional[float]
    weight_kg: Optional[float]
    diabetes_type: DiabetesType


@dataclass(frozen=True)
class HorizonForecast:
    glucose_mgdl: float
    range: GlucoseRange
    error_band_mgdl: float
    model_version: str


@dataclass(frozen=True)
class ForecastResult:
    t60: HorizonForecast
    t120: HorizonForecast
    outside_training: bool


# ---------------------------------------------------------------------------
# HTTP bodies (api <-> frontend)
# ---------------------------------------------------------------------------

class DishRowJSON(BaseModel):
    """Nutrients per 100 g edible portion."""
    id: str
    name: str
    kind: Literal["dish", "ingredient"]
    serving_g: Optional[float]
    energy_kcal: float
    carbs_g: float
    protein_g: float
    fat_g: float
    fibre_g: Optional[float]
    confidence: Literal["exact", "approximate", "computed"]


class G0ReadingJSON(BaseModel):
    value_mgdl: float
    source: G0Source
    timestamp: str
    age_minutes: float
    error_range_mgdl: Optional[float]
    heart_rate_bpm: Optional[float]
    capture_valid: Optional[bool]
    ppg_model_version: Optional[str]


class PpgCapture(BaseModel):
    samples: list[float]
    fs: float
    timestamps: Optional[list[float]] = None
    torch_verified: bool
    clip_fraction: float


class PpgRequest(BaseModel):
    capture: PpgCapture


class PhotoRequest(BaseModel):
    image_data_url: str


class PhotoItemJSON(BaseModel):
    vlm_name: str
    accepted: bool
    row: Optional[DishRowJSON]
    suggestions: list[DishRowJSON]
    similarity: float
    margin: float


class PhotoResponse(BaseModel):
    items: list[PhotoItemJSON]
    vlm_model_version: str
    dish_table_build: str


class DishesResponse(BaseModel):
    items: list[DishRowJSON]


class MeterCgmGlucose(BaseModel):
    source: Literal["meter", "cgm"]
    value_mgdl: float
    timestamp: str


class PpgGlucose(BaseModel):
    source: Literal["ppg"]
    reading: G0ReadingJSON


class ItemIn(BaseModel):
    row_id: str
    grams: Optional[float] = Field(default=None, gt=0)
    portion: Optional[Literal[0, 1, 2]] = None
    source: Literal["photo", "manual"]
    similarity: Optional[float] = None
    margin: Optional[float] = None


class MealIn(BaseModel):
    items: list[ItemIn]
    vlm_model_version: Optional[str] = None


class ProfileIn(BaseModel):
    age_years: Optional[float] = Field(default=None, ge=0, le=120)
    sex: Optional[Literal["M", "F"]] = None
    height_cm: Optional[float] = Field(default=None, gt=0, le=250)
    weight_kg: Optional[float] = Field(default=None, gt=0, le=400)
    diabetes_type: DiabetesType


class PredictRequestV2(BaseModel):
    t0_timestamp: str
    glucose: MeterCgmGlucose | PpgGlucose
    meal: MealIn
    profile: ProfileIn


class MealItemOut(BaseModel):
    row: DishRowJSON
    grams: float
    source: Literal["photo", "manual"]
    carbs_g: float
    protein_g: float
    fat_g: float
    fibre_g: Optional[float]
    kcal: float


class MealTotals(BaseModel):
    carbs_g: float
    protein_g: float
    fat_g: float
    fibre_g: Optional[float]
    kcal: float


class MealOut(BaseModel):
    items: list[MealItemOut]
    totals: MealTotals
    dish_table_build: str
    vlm_model_version: Optional[str]


class HorizonOut(BaseModel):
    glucose_mgdl: float
    range: GlucoseRange
    error_band_mgdl: float
    model_version: str


class ForecastOut(BaseModel):
    t60: HorizonOut
    t120: HorizonOut


class ForecastResponse(BaseModel):
    schema_version: Literal["2.0"] = SCHEMA_VERSION
    g0: G0ReadingJSON
    meal: MealOut
    forecast: ForecastOut
    outside_training: bool
    disclaimer: str = DISCLAIMER
