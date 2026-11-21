"""The three flow steps behind one set of functions: used by the API and by evaluation scripts.

    current glucose -> meal -> forecast_meal(request, deps)

`Deps` carries the served components (dish table, embedder, VLM namer, forecaster, PPG model),
built once at app start from `integration/models/served.json`. Tests inject fakes at the two real
seams (`Embedder`, `VisionNamer`); the forecaster and PPG model are always the real artifacts.
A component whose artifact is missing is None and surfaces as `ModelUnavailable` (HTTP 503) when
a request needs it, so the app still starts on a fresh clone.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from integration.clock import parse_timestamp
from integration.contract import (
    DishRowJSON,
    ForecastOut,
    ForecastResponse,
    G0ReadingJSON,
    HorizonOut,
    MealItemOut,
    MealOut,
    MealTotals,
    PhotoItemJSON,
    PhotoResponse,
    PredictRequestV2,
    Profile,
)
from integration.dishes import DishRow, DishTable, Embedder, GeminiEmbedder
from integration.errors import InvalidInput, ModelUnavailable
from integration.stages.forecast import Forecaster
from integration.stages.glucose import G0Reading, PPGModel, current_glucose
from integration.stages.meal import (
    GeminiKeyError,
    GeminiVisionNamer,
    MealItem,
    VisionNamer,
    identify_photo,
    load_gemini_key,
    resolve_meal,
)

log = logging.getLogger("glucosight.pipeline")

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVED_PATH = Path(__file__).resolve().parent / "models" / "served.json"

MAX_IMAGE_BYTES = 10 * 1024 * 1024
_IMAGE_MIME = {
    "data:image/jpeg;base64": "image/jpeg",
    "data:image/png;base64": "image/png",
    "data:image/webp;base64": "image/webp",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Deps:
    served: dict[str, Any]
    table: Optional[DishTable] = None
    embedder: Optional[Embedder] = None
    namer: Optional[VisionNamer] = None
    forecaster: Optional[Forecaster] = None
    ppg_model: Optional[PPGModel] = None
    now: Callable[[], datetime] = field(default=_utcnow)

    @classmethod
    def from_served(cls, served: dict[str, Any], root: Path = REPO_ROOT) -> "Deps":
        """Build every component whose artifact exists; log (not hide) the ones that do not."""
        deps = cls(served=served)

        ppg = served["ppg_model"]
        try:
            deps.ppg_model = PPGModel.load(
                root / ppg["path"], ppg["version"], ppg["error_range_mgdl"]
            )
        except ModelUnavailable as exc:
            log.warning("PPG model unavailable: %s", exc.detail)

        try:
            deps.forecaster = Forecaster.load(served["forecaster"], root)
        except ModelUnavailable as exc:
            log.warning("Forecaster unavailable: %s", exc.detail)

        table_path = root / served["dish_table"]["path"]
        try:
            deps.table = DishTable.load(table_path)
        except FileNotFoundError:
            log.warning("Dish table not built: %s", table_path)

        vlm = served["vlm"]
        try:
            key = load_gemini_key(root / vlm["key_file"])
        except GeminiKeyError as exc:
            log.warning("Gemini key unavailable: %s", exc)
        else:
            deps.embedder = GeminiEmbedder(key, vlm["embedding_model"])
            deps.namer = GeminiVisionNamer(key, vlm["model"])
        return deps


def load_served(path: Path = SERVED_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def _need(component: Any, what: str) -> Any:
    if component is None:
        raise ModelUnavailable(f"{what} belum tersedia di server.")
    return component


# ---------------------------------------------------------------------------
# Serialisation (domain -> HTTP bodies)
# ---------------------------------------------------------------------------

def dish_row_json(row: DishRow) -> DishRowJSON:
    return DishRowJSON(
        id=row.id, name=row.name, kind=row.kind, serving_g=row.serving_g,
        energy_kcal=row.energy_kcal, carbs_g=row.carbs_g, protein_g=row.protein_g,
        fat_g=row.fat_g, fibre_g=row.fibre_g, confidence=row.confidence,
    )


def g0_json(g0: G0Reading) -> G0ReadingJSON:
    return G0ReadingJSON(
        value_mgdl=round(g0.value_mgdl, 1), source=g0.source,
        timestamp=g0.timestamp.isoformat(), age_minutes=round(g0.age_minutes, 1),
        error_range_mgdl=g0.error_range_mgdl, heart_rate_bpm=g0.heart_rate_bpm,
        capture_valid=g0.capture_valid, ppg_model_version=g0.ppg_model_version,
    )


# ---------------------------------------------------------------------------
# Step operations used by the endpoints
# ---------------------------------------------------------------------------

def scan_glucose(capture: dict[str, Any], deps: Deps) -> G0ReadingJSON:
    g0 = current_glucose({"source": "ppg", "capture": capture}, deps.now(), ppg_model=deps.ppg_model)
    return g0_json(g0)


def search_dishes(q: str, limit: int, deps: Deps) -> list[DishRowJSON]:
    table = _need(deps.table, "Tabel makanan")
    return [dish_row_json(r) for r in table.search(q, limit)]


def decode_image_data_url(data_url: str) -> tuple[bytes, str]:
    header, separator, encoded = data_url.partition(",")
    if not separator or header not in _IMAGE_MIME:
        raise InvalidInput("Foto harus berupa data URL base64 JPEG, PNG, atau WebP.")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidInput("Data foto bukan base64 yang valid.") from exc
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise InvalidInput("Ukuran foto harus antara 1 byte dan 10 MiB.")
    return content, _IMAGE_MIME[header]


def photo_items(data_url: str, deps: Deps) -> PhotoResponse:
    table = _need(deps.table, "Tabel makanan")
    embedder = _need(deps.embedder, "Layanan pencocokan makanan")
    namer = _need(deps.namer, "Layanan pengenalan foto")
    image, mime = decode_image_data_url(data_url)
    items = identify_photo(image, mime, namer, table, embedder)
    return PhotoResponse(
        items=[
            PhotoItemJSON(
                vlm_name=i.vlm_name, accepted=i.accepted,
                row=dish_row_json(i.row) if i.row is not None else None,
                suggestions=[dish_row_json(r) for r in i.suggestions],
                similarity=i.similarity, margin=i.margin,
            )
            for i in items
        ],
        vlm_model_version=namer.model_version,
        dish_table_build=table.build_hash,
    )


def _item_out(item: MealItem) -> MealItemOut:
    m = item.macros()
    return MealItemOut(
        row=dish_row_json(item.row), grams=item.grams, source=item.source,
        carbs_g=round(m["carbs_g"], 2), protein_g=round(m["protein_g"], 2),
        fat_g=round(m["fat_g"], 2),
        fibre_g=None if m["fibre_g"] is None else round(m["fibre_g"], 2),
        kcal=round(m["kcal"], 1),
    )


def forecast_meal(request: PredictRequestV2, deps: Deps) -> ForecastResponse:
    now = deps.now()
    t0 = parse_timestamp(request.t0_timestamp)

    glucose_in = request.glucose.model_dump()
    g0 = current_glucose(glucose_in, now, ppg_model=deps.ppg_model)

    table = _need(deps.table, "Tabel makanan")
    items = [i.model_dump(exclude_none=True) for i in request.meal.items]
    meal = resolve_meal(items, table, vlm_model_version=request.meal.vlm_model_version)

    p = request.profile
    profile = Profile(
        age_years=p.age_years, sex=p.sex, height_cm=p.height_cm,
        weight_kg=p.weight_kg, diabetes_type=p.diabetes_type,
    )
    result = _need(deps.forecaster, "Model prakiraan").predict(g0, meal, profile, t0)

    def horizon(h) -> HorizonOut:
        return HorizonOut(
            glucose_mgdl=h.glucose_mgdl, range=h.range,
            error_band_mgdl=h.error_band_mgdl, model_version=h.model_version,
        )

    return ForecastResponse(
        g0=g0_json(g0),
        meal=MealOut(
            items=[_item_out(i) for i in meal.items],
            totals=MealTotals(
                carbs_g=round(meal.carbs_g, 2), protein_g=round(meal.protein_g, 2),
                fat_g=round(meal.fat_g, 2),
                fibre_g=None if meal.fibre_g is None else round(meal.fibre_g, 2),
                kcal=round(meal.kcal, 1),
            ),
            dish_table_build=meal.dish_table_build, vlm_model_version=meal.vlm_model_version,
        ),
        forecast=ForecastOut(t60=horizon(result.t60), t120=horizon(result.t120)),
        outside_training=result.outside_training,
    )
