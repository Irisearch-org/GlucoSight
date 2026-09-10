"""GlucoSight MVP inference endpoint.

    POST /predict     meal photo + Bahasa note + PPG capture -> prediction
    GET  /contract    the live contract v1.2 field spec
    GET  /health      liveness + what the service is actually serving
    GET  /            the demo page

Two things this service will not do, both deliberate:

1. **It never returns `ppg_glucose_estimate`.** Ring-fenced under E1: never
   a sole prediction, never surfaced to a user. `strip_ringfenced` runs on
   every response body.

2. **It never presents a baseline as a model.** The response carries
   `is_trained_model`, `model_version` and a plain-language `basis` string,
   and the demo page renders a banner from them. No trained forecasting
   estimator is committed to this repo (see `integration/predictor.py`), so
   today every response says so.

This is a research prototype. It is not a medical device and must not be
used to make a treatment decision.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field as PField

from integration import contract, fusion, glucose_source as gsrc, predictor
from integration.adapters.ppg import strip_ringfenced

log = logging.getLogger("glucosight.api")

DISCLAIMER = (
    "Research prototype. Not a medical device. Do not use for any treatment "
    "decision."
)

app = FastAPI(
    title="GlucoSight Integration API",
    version="0.1.0",
    description=DISCLAIMER,
)

_predictor = predictor.default_predictor()

# Pick up a trained PPG glucose estimator if one has been persisted. No
# artifact exists unless `rppg.models.glucose_estimator`'s evidence gate
# passed, so this is a no-op until a model has earned the finger-scan path.
_ppg_g0 = gsrc.autoload_ppg_g0_estimator()
_WEB = Path(__file__).parent / "web"


class GlucoseReading(BaseModel):
    timestamp: str
    value: float


class PredictRequest(BaseModel):
    """One meal. Every modality is optional; missingness is a real state."""

    participant_id: str
    t0_timestamp: str = PField(
        ..., description="FIRST BITE, ISO8601, logged to the second (C1)"
    )
    delta_t_minutes: float = PField(
        60.0, ge=0, le=240,
        description="Actual horizon to the label. Never assumed to be 60 (C1).",
    )
    meal_id: Optional[str] = None
    source_dataset: str = "live_capture"

    # Track inputs, each optional.
    meal_image_path: Optional[str] = None
    note_text: Optional[str] = PField(None, description="Bahasa Indonesia meal note")
    ppg_signal: Optional[List[float]] = PField(
        None, description="Contact PPG samples from the finger capture"
    )
    ppg_sampling_rate: float = 30.0

    # In-app question, not a CV output.
    portion_reported: int = PField(1, ge=0, le=2, description="0=kecil 1=sedang 2=besar")

    glucose_history: List[GlucoseReading] = PField(
        default_factory=list,
        description="CGM trace or prior readings, strictly before t0 (C2)",
    )

    # Option 1: the user owns a glucometer and types the reading in.
    fingerstick_mgdl: Optional[float] = PField(
        None, description="User-entered glucometer reading, mg/dL")
    fingerstick_timestamp: Optional[str] = PField(
        None, description="When the fingerstick was taken. Required with it.")

    # Option 2: the user has no glucometer, so estimate g0 from the finger
    # scan. Disabled by default — see integration/glucose_source.py. Using a
    # PPG estimate as the sole driver of a user-visible number conflicts
    # with ring-fence rule E1.2 until contract v2.0 resolves it.
    allow_ppg_g0_estimate: bool = PField(
        False,
        description="Estimate pre-meal glucose from the PPG capture. "
                    "Conflicts with ring-fence E1.2; off by default.",
    )

    time_since_last_meal_hours: Optional[float] = None


def _run_cv(image_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not image_path:
        return None
    try:
        from cv.cv_baseline.predict import predict as cv_predict
        return cv_predict({"sample_id": "api", "meal_image_path": image_path})
    except Exception as exc:
        # A track failing is a missing modality, not a 500. The contract has
        # a fallback for exactly this; the mask records that it happened.
        log.warning("CV track unavailable, treating photo as absent: %s", exc)
        return None


def _run_nlp(text: Optional[str], participant_id: str) -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None
    try:
        from nlp.nlp_baseline.predict import predict as nlp_predict
        return nlp_predict({"dietary_intake": text, "participant_id": participant_id})
    except Exception as exc:
        log.warning("NLP track unavailable, treating note as absent: %s", exc)
        return None


def _run_ppg(signal: Optional[List[float]], fs: float) -> Optional[Dict[str, Any]]:
    if not signal:
        return None
    try:
        import numpy as np
        from rppg.features.extractor import extract_features, to_contract_dict
        arr = np.asarray(signal, dtype=float)
        # The contract's capture protocol: discard the first 2-3 seconds
        # (LED thermal settling) before any analysis.
        skip = int(3 * fs)
        if arr.size <= skip + int(fs):
            log.warning("PPG capture too short after settling discard")
            return None
        feats = extract_features(arr[skip:], fs)
        n_windows = min(3, max(1, int((arr.size - skip) / (10 * fs))))
        return to_contract_dict(feats, glucose_estimate=None,
                                present=1, n_windows=n_windows)
    except Exception as exc:
        log.warning("PPG track unavailable, treating capture as absent: %s", exc)
        return None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "schema_version": contract.SCHEMA_VERSION,
        "model_version": _predictor.model_version,
        "is_trained_model": _predictor.is_trained_model,
        "serving": (
            "baseline B2 (persistence + cohort mean excursion) — no trained "
            "forecasting model is committed to this repository"
            if not _predictor.is_trained_model else "trained model"
        ),
        "g0_sources": {
            "fingerstick": "available — user enters a glucometer reading",
            "cgm": "available — supplied via glucose_history",
            "ppg_estimate": (
                "UNAVAILABLE — no informative estimator registered; the only "
                "committed PPG-glucose model is a constant predictor "
                "(MAE 14.51, n=23 subjects). Also conflicts with ring-fence "
                "E1.2 until contract v2.0."
                if (gsrc.registered_ppg_g0_estimator() is None
                    or not gsrc.registered_ppg_g0_estimator().provides_information)
                else f"available — {gsrc.registered_ppg_g0_estimator().model_version}"
            ),
            "population_fallback": "always available, not personalised",
        },
        "disclaimer": DISCLAIMER,
    }


@app.get("/contract")
def contract_spec() -> Dict[str, Any]:
    """The live field spec, so a track can check itself against the validator."""
    def dump(fields):
        return [
            {"name": f.name, "type": f.kind.__name__, "required": f.required,
             "fallback": f.fallback, "min": f.lo, "max": f.hi,
             "allowed": sorted(f.allowed) if f.allowed else None,
             "note": f.note}
            for f in fields
        ]

    return {
        "schema_version": contract.SCHEMA_VERSION,
        "envelope": dump(contract.ENVELOPE_FIELDS),
        "cv": dump(contract.CV_FIELDS),
        "nlp": dump(contract.NLP_FIELDS),
        "ppg": dump(contract.PPG_FIELDS),
        "forecast": dump(contract.FORECAST_FIELDS),
        "ringfenced": ["ppg_glucose_estimate"],
        "confidence_floor": contract.CONFIDENCE_FLOOR,
        "min_modality_weight": contract.MIN_MODALITY_WEIGHT,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> JSONResponse:
    meal_id = req.meal_id or str(uuid.uuid4())

    cv_record = _run_cv(req.meal_image_path)
    nlp_record = _run_nlp(req.note_text, req.participant_id)
    ppg_record = _run_ppg(req.ppg_signal, req.ppg_sampling_rate)

    try:
        meal = fusion.assemble(
            meal_id=meal_id,
            participant_id=req.participant_id,
            t0_timestamp=req.t0_timestamp,
            source_dataset=req.source_dataset,
            delta_t_minutes=req.delta_t_minutes,
            cv_record=cv_record,
            nlp_record=nlp_record,
            ppg_record=ppg_record,
            glucose_history=[g.model_dump() for g in req.glucose_history],
            portion_reported=req.portion_reported,
            time_since_last_meal_hours=req.time_since_last_meal_hours,
            fingerstick_mgdl=req.fingerstick_mgdl,
            fingerstick_timestamp=req.fingerstick_timestamp,
            allow_ppg_g0_estimate=req.allow_ppg_g0_estimate,
        )
    except contract.ContractViolation as exc:
        # A contract violation is the caller's input being wrong, not a
        # server fault, and the message names the rule that was broken.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    forecast = _predictor.predict(meal)
    provenance = meal.pop("_provenance", {})

    body = {
        "meal_id": meal_id,
        "forecast": forecast,
        "modalities": {
            "cv": {
                "present": meal["cv_present"],
                "confidence": meal["cv_confidence"],
                "food_class": meal.get("food_class"),
                "carbs_g": meal["carbs_g"],
                "protein_g": meal["protein_g"],
                "fat_g": meal["fat_g"],
                "fiber_g": meal["fiber_g"],
                "gi_category": meal["gi_category"],
                "carbs_source": meal["carbs_source"],
                "weight": forecast["modality_weights"]["cv"],
            },
            "nlp": {
                "present": meal["nlp_present"],
                "confidence": meal["nlp_confidence"],
                "is_fried_cooking": meal["is_fried_cooking"],
                "is_large_portion": meal["is_large_portion"],
                "feature_source": meal["nlp_feature_source"],
                "calibrated": provenance.get("nlp_calibrated"),
                "weight": forecast["modality_weights"]["nlp"],
            },
            "ppg": {
                "present": meal["ppg_present"],
                "signal_quality": meal["ppg_signal_quality"],
                "pulse_rate_bpm": meal["ppg_pulse_rate_bpm"],
                "hrv_rmssd": meal["ppg_hrv_rmssd"],
                "perfusion_index": meal["ppg_perfusion_index"],
                "clip_fraction": meal["ppg_clip_fraction"],
                "n_windows": meal["ppg_n_windows"],
                "weight": forecast["modality_weights"]["ppg"],
            },
        },
        "glucose_history_used": {
            "n_causal_readings": len(meal.get("glucose_history", [])),
            "n_dropped_as_noncausal": provenance.get(
                "n_history_readings_dropped_as_noncausal", 0),
            "g0": meal.get("g0"),
            "g0_age_minutes": meal.get("g0_age_minutes"),
            "g0_rejected_as_stale": meal.get("g0_rejected_as_stale", False),
            "max_g0_age_minutes": contract.MAX_G0_AGE_MINUTES,
        },
        "glucose_source": {
            "value_mgdl": meal.get("g0_mgdl"),
            "source": meal.get("g0_source"),
            "is_direct_measurement": meal.get("g0_is_direct_measurement"),
            "trust": meal.get("g0_trust"),
            "detail": meal.get("g0_detail"),
            "rejected_candidates": meal.get("g0_rejected_candidates", []),
        },
        "schema_version": contract.SCHEMA_VERSION,
        "disclaimer": DISCLAIMER,
        "generated_at": datetime.now().isoformat(),
    }

    # E1: nothing ring-fenced leaves the service.
    body["modalities"]["ppg"] = strip_ringfenced(body["modalities"]["ppg"])
    return JSONResponse(content=body)


@app.get("/")
def index():
    page = _WEB / "index.html"
    if not page.exists():
        return JSONResponse(content={"detail": "demo page not installed"},
                            status_code=404)
    return FileResponse(page)
