"""GlucoSight v2 HTTP layer. Transport only: every rule lives in a stage or in `pipeline`.

    GET  /health        what is being served (integration/models/served.json) and what loaded
    GET  /dishes        dish-table autocomplete
    POST /glucose/ppg   finger-scan capture -> g0 reading (409 when the capture is not a pulse)
    POST /meal/photo    photo -> dish items, each accepted or needing manual entry
    POST /predict       g0 + meal + profile -> T+60 / T+120 forecast
    /auth/*, /user/*    accounts and history (unchanged)

Errors are {"detail": "<Bahasa message>"}: 422 invalid input, 409 stale reading or invalid scan,
503 model / table / VLM unavailable.

Research prototype. Not a medical device; the result carries that disclaimer.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from integration import auth, pipeline, user_data
from integration.contract import (
    DISCLAIMER,
    SCHEMA_VERSION,
    DishesResponse,
    ForecastResponse,
    G0ReadingJSON,
    PhotoRequest,
    PhotoResponse,
    PpgRequest,
    PredictRequestV2,
)
from integration.dishes import GeminiUnavailable
from integration.errors import InvalidInput, ModelUnavailable
from integration.pipeline import Deps
from integration.stages.glucose import GlucoseError, InvalidCapture, StaleReading
from integration.stages.meal import GeminiKeyError, MealError, VisionUnavailable

log = logging.getLogger("glucosight.api")

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
VISION_UNAVAILABLE = (
    "Layanan pengenalan foto sedang tidak tersedia. Isi makanan secara manual."
)

app = FastAPI(title="GlucoSight API", version=SCHEMA_VERSION, description=DISCLAIMER)
app.include_router(auth.router)
app.include_router(user_data.router)
app.state.deps = Deps.from_served(pipeline.load_served())


def get_deps(request: Request) -> Deps:
    return request.app.state.deps


def _detail(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": message})


@app.exception_handler(RequestValidationError)
async def _invalid_body(_: Request, exc: RequestValidationError) -> JSONResponse:
    fields = "; ".join(
        f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()
    )
    return _detail(422, f"Masukan tidak valid — {fields}")


@app.exception_handler(StaleReading)
@app.exception_handler(InvalidCapture)
async def _glucose_conflict(_: Request, exc: GlucoseError) -> JSONResponse:
    return _detail(409, exc.detail)


@app.exception_handler(GlucoseError)
@app.exception_handler(InvalidInput)
async def _glucose_invalid(_: Request, exc: GlucoseError | InvalidInput) -> JSONResponse:
    return _detail(422, exc.detail)


@app.exception_handler(MealError)
async def _meal_invalid(_: Request, exc: MealError) -> JSONResponse:
    return _detail(422, str(exc))


@app.exception_handler(ModelUnavailable)
async def _unavailable(_: Request, exc: ModelUnavailable) -> JSONResponse:
    return _detail(503, exc.detail)


@app.exception_handler(VisionUnavailable)
@app.exception_handler(GeminiUnavailable)
@app.exception_handler(GeminiKeyError)
async def _vision_unavailable(_: Request, exc: Exception) -> JSONResponse:
    log.warning("vision/embedding unavailable: %s", exc)
    return _detail(503, VISION_UNAVAILABLE)


@app.get("/health")
def health(deps: Deps = Depends(get_deps)) -> dict:
    return {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "served": deps.served,
        "dish_table_build": deps.table.build_hash if deps.table is not None else None,
        "components": {
            "ppg_model": deps.ppg_model is not None,
            "forecaster": deps.forecaster is not None,
            "dish_table": deps.table is not None,
            "photo": deps.namer is not None and deps.embedder is not None,
        },
    }


@app.get("/dishes", response_model=DishesResponse)
def dishes(
    q: str = "", limit: int = Query(10, ge=1, le=50), deps: Deps = Depends(get_deps)
) -> DishesResponse:
    return DishesResponse(items=pipeline.search_dishes(q, limit, deps))


@app.post("/glucose/ppg", response_model=G0ReadingJSON)
def glucose_ppg(body: PpgRequest, deps: Deps = Depends(get_deps)) -> G0ReadingJSON:
    return pipeline.scan_glucose(body.capture.model_dump(), deps)


@app.post("/meal/photo", response_model=PhotoResponse)
def meal_photo(body: PhotoRequest, deps: Deps = Depends(get_deps)) -> PhotoResponse:
    return pipeline.photo_items(body.image_data_url, deps)


@app.post("/predict", response_model=ForecastResponse)
def predict(body: PredictRequestV2, deps: Deps = Depends(get_deps)) -> ForecastResponse:
    return pipeline.forecast_meal(body, deps)


# Last, so no route above is shadowed. Absent until `npm run build` has run.
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
