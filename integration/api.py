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

import base64
import binascii
import logging
import tempfile
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

# Same shape, for the gated NLP dish-identity forecaster (ADR 0004). No
# artifact exists unless nlp.models.dish_identity_ridge's gate passed, so
# this is a no-op until someone has trained one against the real data.
# Reachable only via the explicit prediction_mode below -- never swaps
# out `_predictor`, which stays B2 (see integration/predictor.py).
_nlp_dish_predictor = predictor.autoload_nlp_dish_identity_predictor()

# Experimental, ungated CV+NLP+PPG blend (2026-10-20 product decision to
# bypass the evidence gate for this mode only -- see
# integration/predictor.py::ExperimentalFusionPredictor). Explicit opt-in,
# never swaps out `_predictor`; `baseline_demo` stays the real default.
_experimental_fusion_predictor = predictor.autoload_experimental_fusion_predictor()

_UNAVAILABLE_FORECAST_DETAIL = (
    "No trained forecasting artifact is configured; only baseline_demo is available."
)

_WEB = Path(__file__).parent / "web"


def _active_predictor(prediction_mode: str):
    """Which Predictor serves this request, or None if the mode is
    unknown or its artifact hasn't earned its gate yet."""
    if prediction_mode == "baseline_demo":
        return _predictor
    if prediction_mode == "nlp_dish_identity_forecast" and _nlp_dish_predictor is not None:
        return _nlp_dish_predictor
    if prediction_mode == "experimental_fusion" and _experimental_fusion_predictor is not None:
        return _experimental_fusion_predictor
    return None


class GlucoseReading(BaseModel):
    timestamp: str
    value: float


class PredictRequest(BaseModel):
    """One meal. Every modality is optional; missingness is a real state."""

    participant_id: str
    t0_timestamp: str = PField(
        ..., description="Event timestamp, ISO8601. First-bite semantics must be established by the source."
    )
    delta_t_minutes: float = PField(
        60.0, ge=60, le=120,
        description="B2 demo supports only fixed T+60 or T+120 horizons.",
    )
    prediction_mode: str = PField(
        "baseline_demo", description="Currently only baseline_demo is available; it does not consume meal modalities."
    )
    meal_id: Optional[str] = None
    source_dataset: str = "live_capture"

    # Demo inputs are optional; all are processed independently and their
    # results are diagnostic only for the B2 baseline.
    meal_image_data_url: Optional[str] = PField(
        None, description="Base64 data URL for a supported meal image"
    )
    note_text: Optional[str] = PField(None, description="Meal note; current rules are not clinically validated for Bahasa")
    ppg_capture: Optional[Dict[str, Any]] = PField(
        None, description="{samples, timestamps_seconds, sampling_rate_hz}; scalar array must be normalized to [0,1]"
    )

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


def _decode_image_data_url(data_url: str) -> tuple[bytes, str]:
    header, separator, encoded = data_url.partition(",")
    mime_to_suffix = {
        "data:image/jpeg;base64": ".jpg",
        "data:image/png;base64": ".png",
        "data:image/webp;base64": ".webp",
    }
    if not separator or header not in mime_to_suffix:
        raise ValueError("meal image must be a base64 JPEG, PNG, or WebP data URL")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("meal image data URL has invalid base64") from exc
    if not content or len(content) > 10 * 1024 * 1024:
        raise ValueError("decoded meal image must be between 1 byte and 10 MiB")
    return content, mime_to_suffix[header]


def _run_cv(data_url: Optional[str]) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if not data_url:
        return None, {"status": "absent", "detail": "no image supplied"}
    temporary_path = None
    try:
        content, suffix = _decode_image_data_url(data_url)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(content)
            temporary_path = temporary.name
        from cv.cv_baseline.predict import predict as cv_predict
        record = cv_predict({"sample_id": "api", "meal_image_path": temporary_path})
        status = str(record.get("feature_status", "unknown"))
        return record, {"status": status, "detail": None}
    except Exception as exc:
        log.exception("CV track failed")
        return None, {"status": "invalid" if isinstance(exc, ValueError) else "error", "detail": str(exc)}
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


def _run_nlp(text: Optional[str], participant_id: str) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if not text or not text.strip():
        return None, {"status": "absent", "detail": "no text supplied"}
    try:
        from nlp.nlp_baseline.predict import predict as nlp_predict
        record = nlp_predict({"dietary_intake": text, "participant_id": participant_id})
        return record, {
            "status": "processed",
            "detail": None,
            "feature_source": record.get("nlp_feature_source"),
            "calibrated": False,
            "limitation": "rule-based labels are not validated for Bahasa clinical use",
        }
    except Exception as exc:
        log.exception("NLP track failed")
        return None, {"status": "error", "detail": str(exc)}


def _run_ppg(capture: Optional[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if capture is None:
        return None, {"status": "absent", "detail": "no PPG capture supplied"}
    try:
        import numpy as np
        from rppg.features.extractor import extract_features, to_contract_dict
        samples = np.asarray(capture.get("samples", []), dtype=float)
        timestamps = np.asarray(capture.get("timestamps_seconds", []), dtype=float)
        if samples.ndim != 1 or timestamps.ndim != 1 or samples.size != timestamps.size:
            raise ValueError("PPG requires equal-length 1D samples and timestamps_seconds")
        if samples.size < 2 or not np.isfinite(samples).all() or not np.isfinite(timestamps).all():
            raise ValueError("PPG samples and timestamps must be finite with at least two observations")
        if np.any(samples < 0.0) or np.any(samples > 1.0):
            raise ValueError("PPG scalar samples must be normalized to [0, 1]")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("PPG timestamps must be strictly increasing")
        fs = float(capture.get("sampling_rate_hz", 0))
        if not np.isfinite(fs) or fs < 10 or fs > 240:
            raise ValueError("PPG sampling_rate_hz must be in [10, 240]")
        intervals = np.diff(timestamps)
        observed_fs = 1.0 / float(np.median(intervals))
        if abs(observed_fs - fs) / fs > 0.10:
            raise ValueError("timestamp-derived PPG frame rate disagrees with sampling_rate_hz by over 10%")
        # The existing extractor assumes uniform samples. Reject high jitter
        # rather than silently applying an FFT to nonuniform timestamps.
        if float(np.std(intervals) / np.mean(intervals)) > 0.05:
            raise ValueError("PPG timestamp jitter exceeds 5%; uniform resampling is not implemented")
        start = float(timestamps[0])
        capture_duration = float(timestamps[-1] - timestamps[0])
        skip = 3.0
        usable_start = start + skip
        boundaries = []
        cursor = usable_start
        while cursor + 10.0 <= float(timestamps[-1]) + 1e-9 and len(boundaries) < 3:
            boundaries.append((cursor, cursor + 10.0))
            cursor += 10.0
        if not boundaries:
            return None, {"status": "invalid", "detail": "no valid 10-second PPG window after 3-second settling discard"}
        features = []
        clips = []
        for window_start, window_end in boundaries:
            mask = (timestamps >= window_start) & (timestamps < window_end)
            window = samples[mask]
            if len(window) < int(fs * 8):
                continue
            # A normalized signal is expected. Saturation proxy at either
            # endpoint is reported; raw RGB-channel clipping needs a future
            # camera-capture schema.
            clips.append(float(np.mean((window <= 0.0) | (window >= 1.0))))
            features.append(extract_features(window, fs))
        if not features:
            return None, {"status": "invalid", "detail": "all PPG windows were too short"}
        keys = features[0].keys()
        median_features = {
            key: float(np.nanmedian([feature[key] for feature in features]))
            for key in keys
        }
        # The extractor reports AC/DC as a percentage; contract v1.2 defines
        # perfusion_index as a ratio. Convert units, never clamp the value.
        median_features["perfusion_index"] /= 100.0
        record = to_contract_dict(
            median_features,
            glucose_estimate=None,
            present=1,
            n_windows=len(features),
            clip_fraction=float(np.median(clips)) if clips else 1.0,
        )
        return record, {
            "status": "processed",
            "detail": None,
            "capture_duration_seconds": capture_duration,
            "observed_sampling_rate_hz": observed_fs,
            "timestamp_jitter_fraction": float(np.std(intervals) / np.mean(intervals)),
            "n_valid_windows": len(features),
            "clip_fraction_proxy": record["ppg_clip_fraction"],
            "limitation": "clipping is a normalized-signal proxy; RGB camera channel clipping is not available",
        }
    except Exception as exc:
        log.exception("PPG track failed")
        return None, {"status": "invalid", "detail": str(exc)}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "prediction_mode": "baseline_demo",
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
                "(MAE 14.54, n=23 subjects). Also conflicts with ring-fence "
                "E1.2 until contract v2.0."
                if (gsrc.registered_ppg_g0_estimator() is None
                    or not gsrc.registered_ppg_g0_estimator().provides_information)
                else f"available — {gsrc.registered_ppg_g0_estimator().model_version}"
            ),
            "population_fallback": "always available, not personalised",
        },
        "nlp_dish_identity_mode": (
            "UNAVAILABLE — nlp.models.dish_identity_ridge's gate has not "
            "passed (no artifact configured); prediction_mode="
            "'nlp_dish_identity_forecast' stays off"
            if _nlp_dish_predictor is None
            else f"available as prediction_mode='nlp_dish_identity_forecast' "
                 f"— {_nlp_dish_predictor.model_version}"
        ),
        "experimental_fusion_mode": (
            "UNAVAILABLE — no artifact at integration/models/experimental_fusion.json; "
            "run `python -m forecasting.evaluate.fit_experimental_fusion`"
            if _experimental_fusion_predictor is None
            else f"available as prediction_mode='experimental_fusion' — "
                 f"{_experimental_fusion_predictor.model_version} — "
                 f"EXPERIMENTAL, bypasses the evidence gate by explicit "
                 f"product decision (2026-10-20); not validated"
        ),
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
    active_predictor = _active_predictor(req.prediction_mode)
    if active_predictor is None:
        raise HTTPException(status_code=409, detail=_UNAVAILABLE_FORECAST_DETAIL)
    if req.delta_t_minutes not in (60.0, 120.0):
        raise HTTPException(
            status_code=422,
            detail="baseline_demo supports only delta_t_minutes=60 or 120",
        )
    meal_id = req.meal_id or str(uuid.uuid4())

    cv_record, cv_status = _run_cv(req.meal_image_data_url)
    nlp_record, nlp_status = _run_nlp(req.note_text, req.participant_id)
    ppg_record, ppg_status = _run_ppg(req.ppg_capture)

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

    forecast = active_predictor.predict(meal)
    provenance = meal.pop("_provenance", {})

    if active_predictor is _predictor:
        forecast_input_usage = {
            "used_for_glucose_value": ["g0_mgdl", "versioned_B2_mean_excursion"],
            "not_used_for_glucose_value": ["cv", "nlp", "ppg", "meal_context"],
            "message": "CV/NLP/PPG outputs are diagnostics only; they do not affect this baseline glucose value.",
        }
    elif active_predictor is _experimental_fusion_predictor:
        forecast_input_usage = {
            "used_for_glucose_value": [
                "g0_mgdl", "carbs_g", "protein_g", "fat_g", "fiber_g",
                "is_fried_cooking", "ppg_pulse_rate_bpm",
            ],
            "not_used_for_glucose_value": [],
            "message": (
                "EXPERIMENTAL, UNGATED: T+60 blends a CGMacros-trained carbs/"
                "macro adjustment and a BIG IDEAs-trained is_fried_cooking/"
                "PPG adjustment onto the B2 baseline, each weighted by that "
                "modality's own confidence. Neither adjustment has cleared "
                "the evidence gate the rest of this repo requires -- see "
                "forecast.basis and docs/INTEGRATION_AUDIT.md. T+120 has no "
                "trained arm and falls back to B2 unchanged."
            ),
        }
    else:
        forecast_input_usage = {
            "used_for_glucose_value": [
                "g0_mgdl", "nlp_total_grams", "nlp_dish_names",
                "is_fried_cooking", "is_large_portion",
            ],
            "not_used_for_glucose_value": ["cv", "ppg"],
            "message": (
                "T+60 uses the gated NLP dish-identity model "
                f"({forecast['model_version']}); T+120 still falls back to "
                "the B2 baseline constant — no T+120 arm exists for it."
            ),
        }

    body = {
        "meal_id": meal_id,
        "prediction_mode": req.prediction_mode,
        "forecast": forecast,
        "forecast_input_usage": forecast_input_usage,
        "modality_processing": {"cv": cv_status, "nlp": nlp_status, "ppg": ppg_status},
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
                "calibration_status": provenance.get("nlp_calibration_status", "unavailable"),
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
            # Sourced from `forecast`, not `meal`: a predictor
            # (ExperimentalFusionPredictor) may resolve g0 differently from
            # fusion.assemble()'s own pass, and this must report what
            # actually drove the returned number.
            "value_mgdl": forecast.get("g0_mgdl", meal.get("g0_mgdl")),
            "source": forecast.get("g0_source", meal.get("g0_source")),
            "is_direct_measurement": forecast.get("g0_is_direct_measurement", meal.get("g0_is_direct_measurement")),
            "trust": forecast.get("g0_trust", meal.get("g0_trust")),
            "detail": forecast.get("g0_detail", meal.get("g0_detail")),
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


@app.get("/ppg")
def ppg_capture():
    """Camera-based contact PPG capture, producing a /predict ppg_capture.

    Requires a secure context: http://127.0.0.1 and https qualify, a LAN IP
    over http does not, so getUserMedia will be unavailable there.
    """
    page = _WEB / "ppg-capture.html"
    if not page.exists():
        return JSONResponse(content={"detail": "capture page not installed"},
                            status_code=404)
    return FileResponse(page)
