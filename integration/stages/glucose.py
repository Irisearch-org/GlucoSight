"""Step 1: current glucose (g0) from a glucometer, a CGM reading, or a finger PPG scan.

    current_glucose({"source": "meter"|"cgm", "value_mgdl": 123, "timestamp": ...}, now)
    current_glucose({"source": "ppg", "capture": {...}}, now, ppg_model=model)   # scan
    current_glucose({"source": "ppg", "reading": {...}}, now, ppg_model=model)   # re-check at /predict

Rules (spec section 1): value 40-400 mg/dL, reading age <= 60 min, no population fallback.
A PPG scan is only a capture-validity gate plus the served ridge model; it never claims the
estimate is accurate (ADR 0005): the reading carries a +-25 mg/dL error range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import numpy as np

from integration.clock import minutes_between, parse_timestamp
from integration.contract import G0_MAX_AGE_MINUTES, G0_MAX_MGDL, G0_MIN_MGDL
from integration.errors import InvalidInput, ModelUnavailable, StageError

# ISO 15197:2013: +-15 mg/dL below 100 mg/dL, +-15 % at or above.
ISO_ABS_MGDL = 15.0
ISO_REL = 0.15
ISO_SWITCH_MGDL = 100.0

# Capture validation (ported from the pre-v2 api._run_ppg; spec section 1).
FS_MIN_HZ = 10.0
FS_MAX_HZ = 240.0
FS_TOLERANCE = 0.10          # timestamp-derived rate vs declared fs
MAX_JITTER = 0.05            # std/mean of frame intervals
SETTLE_S = 3.0
WINDOW_S = 10.0
MAX_WINDOWS = 3
MIN_WINDOW_FILL = 0.8        # a window needs >= 80 % of fs * WINDOW_S samples
MAX_CLIP_FRACTION = 0.10     # share of clipped frames the server tolerates
HR_MIN_BPM = 40.0
HR_MAX_BPM = 180.0

STALE_MESSAGE = "Hasil pengukuran sudah lebih dari 60 menit. Silakan ukur ulang."
INVALID_SCAN_MESSAGE = "Pemindaian tidak valid. Ulangi scan."


class GlucoseError(StageError, ValueError):
    """Base of every error `current_glucose` raises on purpose."""


class StaleReading(GlucoseError):
    """The reading is older than 60 minutes (HTTP 409)."""


class InvalidCapture(GlucoseError):
    """The PPG capture is not a usable pulse recording (HTTP 409)."""


class GlucoseOutOfRange(GlucoseError):
    """Missing, non-finite or out-of-range input (HTTP 422)."""


@dataclass(frozen=True)
class G0Reading:
    value_mgdl: float
    source: Literal["meter", "cgm", "ppg"]
    timestamp: datetime
    age_minutes: float
    error_range_mgdl: Optional[float]      # ppg: served constant; meter: ISO 15197 band; cgm: None
    heart_rate_bpm: Optional[float]
    capture_valid: Optional[bool]
    ppg_model_version: Optional[str]


def meter_error_range(value_mgdl: float) -> float:
    """Half-width (mg/dL) of the ISO 15197:2013 accuracy band at this reading."""
    return ISO_ABS_MGDL if value_mgdl < ISO_SWITCH_MGDL else ISO_REL * value_mgdl


# ---------------------------------------------------------------------------
# Served PPG model
# ---------------------------------------------------------------------------

class PPGModel:
    """The served ridge (scaler + ridge on the 7 handcrafted features), loaded from a joblib file.

    `rppg/models/weights/ppg_served_model.pt` (written by
    `rppg/experiments/ppg_served_model/serve.py`) only unpickles through torch; the sklearn
    estimator inside is re-saved once with `rppg/experiments/ppg_served_model/export_joblib.py`
    so the server does not depend on torch.
    """

    def __init__(self, estimator: Any, version: str, error_range_mgdl: float):
        self._estimator = estimator
        self.version = version
        self.error_range_mgdl = float(error_range_mgdl)

    @classmethod
    def load(cls, path: Path, version: str, error_range_mgdl: float) -> "PPGModel":
        import joblib

        if not path.is_file():
            raise ModelUnavailable(
                f"Model glukosa PPG belum tersedia ({path}). Jalankan export_joblib.py."
            )
        blob = joblib.load(path)
        return cls(blob["estimator"], version, error_range_mgdl)

    def predict(self, features: Mapping[str, float]) -> float:
        from rppg.models.glucose_estimator import FEATURE_NAMES

        x = np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)
        return float(self._estimator.predict(x)[0])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def current_glucose(
    inp: Mapping[str, Any], now: datetime, *, ppg_model: Optional[PPGModel] = None
) -> G0Reading:
    """Validate step-1 input and return the g0 reading. Raises `GlucoseError` subclasses."""
    source = inp.get("source")
    if source in ("meter", "cgm"):
        return _typed_reading(inp, source, now)
    if source == "ppg":
        if ppg_model is None:
            raise ModelUnavailable("Model glukosa PPG belum tersedia.")
        if inp.get("capture") is not None:
            return _scan(inp["capture"], now, ppg_model)
        if inp.get("reading") is not None:
            return _recheck(inp["reading"], now, ppg_model)
        raise GlucoseOutOfRange("Data pemindaian PPG tidak ada.")
    raise GlucoseOutOfRange("Sumber glukosa harus meter, cgm, atau ppg.")


def _check_value(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise GlucoseOutOfRange("Nilai glukosa harus berupa angka.") from exc
    if not math.isfinite(v) or not (G0_MIN_MGDL <= v <= G0_MAX_MGDL):
        raise GlucoseOutOfRange(
            f"Nilai glukosa harus antara {G0_MIN_MGDL:.0f} dan {G0_MAX_MGDL:.0f} mg/dL."
        )
    return v


def _timestamp(value: Any) -> datetime:
    try:
        return parse_timestamp(value)
    except InvalidInput as exc:
        raise GlucoseOutOfRange(exc.detail) from exc


def _age_minutes(timestamp: datetime, now: datetime) -> float:
    age = minutes_between(timestamp, now)
    if age < -2.0:  # tolerate a little clock skew, nothing more
        raise GlucoseOutOfRange("Waktu pengukuran tidak boleh di masa depan.")
    age = max(age, 0.0)
    if age > G0_MAX_AGE_MINUTES:
        raise StaleReading(STALE_MESSAGE)
    return age


def _typed_reading(inp: Mapping[str, Any], source: str, now: datetime) -> G0Reading:
    value = _check_value(inp.get("value_mgdl"))
    ts = _timestamp(inp.get("timestamp"))
    age = _age_minutes(ts, now)
    return G0Reading(
        value_mgdl=value,
        source=source,  # type: ignore[arg-type]
        timestamp=ts,
        age_minutes=age,
        error_range_mgdl=meter_error_range(value) if source == "meter" else None,
        heart_rate_bpm=None,
        capture_valid=None,
        ppg_model_version=None,
    )


def _recheck(reading: Mapping[str, Any], now: datetime, model: PPGModel) -> G0Reading:
    """A PPG reading returned by the scan, sent back with /predict: only its age is re-validated."""
    if reading.get("capture_valid") is not True:
        raise InvalidCapture(INVALID_SCAN_MESSAGE)
    value = _check_value(reading.get("value_mgdl"))
    ts = _timestamp(reading.get("timestamp"))
    age = _age_minutes(ts, now)
    hr = reading.get("heart_rate_bpm")
    return G0Reading(
        value_mgdl=value,
        source="ppg",
        timestamp=ts,
        age_minutes=age,
        error_range_mgdl=model.error_range_mgdl,
        heart_rate_bpm=float(hr) if hr is not None else None,
        capture_valid=True,
        ppg_model_version=model.version,
    )


# ---------------------------------------------------------------------------
# PPG capture -> g0
# ---------------------------------------------------------------------------

def _scan(capture: Mapping[str, Any], now: datetime, model: PPGModel) -> G0Reading:
    from rppg.features.extractor import extract_features
    from rppg.models.glucose_estimator import FEATURE_NAMES

    samples, timestamps, fs = _validated_signal(capture)

    if capture.get("torch_verified") is not True:
        raise InvalidCapture("Lampu senter tidak terverifikasi menyala. Ulangi scan.")
    try:
        clip = float(capture.get("clip_fraction"))
    except (TypeError, ValueError) as exc:
        raise InvalidCapture("Data saturasi kamera tidak ada. Ulangi scan.") from exc
    if not math.isfinite(clip) or clip < 0.0 or clip > MAX_CLIP_FRACTION:
        raise InvalidCapture("Gambar kamera terlalu terang (saturasi). Ulangi scan.")

    windows = _windows(samples, timestamps, fs)
    per_window = []
    for window in windows:
        try:
            per_window.append(extract_features(window, fs))
        except ValueError as exc:
            raise InvalidCapture("Sinyal denyut tidak dapat dianalisis. Ulangi scan.") from exc

    features: dict[str, float] = {}
    for name in FEATURE_NAMES:
        finite = [f[name] for f in per_window if f.get(name) is not None and math.isfinite(f[name])]
        if not finite:
            raise InvalidCapture("Denyut nadi tidak terdeteksi dengan jelas. Ulangi scan.")
        features[name] = float(np.median(finite))

    hr = features["pulse_rate_bpm"]
    if not (HR_MIN_BPM <= hr <= HR_MAX_BPM):
        raise InvalidCapture("Denyut nadi di luar rentang 40–180 bpm. Ulangi scan.")

    value = _check_value(model.predict(features))
    return G0Reading(
        value_mgdl=value,
        source="ppg",
        timestamp=now,
        age_minutes=0.0,
        error_range_mgdl=model.error_range_mgdl,
        heart_rate_bpm=hr,
        capture_valid=True,
        ppg_model_version=model.version,
    )


def _validated_signal(capture: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        samples = np.asarray(capture.get("samples", []), dtype=float)
        fs = float(capture.get("fs", 0))
        raw_ts = capture.get("timestamps")
        timestamps = None if raw_ts is None else np.asarray(raw_ts, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidCapture("Data pemindaian tidak valid. Ulangi scan.") from exc

    if not math.isfinite(fs) or fs < FS_MIN_HZ or fs > FS_MAX_HZ:
        raise InvalidCapture("Laju sampel harus antara 10 dan 240 Hz. Ulangi scan.")
    if samples.ndim != 1 or samples.size < 2 or not np.isfinite(samples).all():
        raise InvalidCapture("Sampel PPG harus berupa deret angka hingga yang cukup panjang.")
    if np.any(samples < 0.0) or np.any(samples > 1.0):
        raise InvalidCapture("Sampel PPG harus ternormalisasi pada rentang 0–1.")

    if timestamps is None:
        timestamps = np.arange(samples.size, dtype=float) / fs
    if timestamps.ndim != 1 or timestamps.size != samples.size or not np.isfinite(timestamps).all():
        raise InvalidCapture("Waktu sampel harus sama panjang dengan sampel dan berupa angka hingga.")
    intervals = np.diff(timestamps)
    if np.any(intervals <= 0):
        raise InvalidCapture("Waktu sampel harus naik secara ketat.")
    observed_fs = 1.0 / float(np.median(intervals))
    if abs(observed_fs - fs) / fs > FS_TOLERANCE:
        raise InvalidCapture("Laju frame dari waktu sampel berbeda lebih dari 10% dari laju yang dilaporkan.")
    if float(np.std(intervals) / np.mean(intervals)) > MAX_JITTER:
        raise InvalidCapture("Jitter waktu frame melebihi 5%. Ulangi scan dan jaga ponsel tetap stabil.")
    return samples, timestamps, fs


def _windows(samples: np.ndarray, timestamps: np.ndarray, fs: float) -> list[np.ndarray]:
    """Up to three 10 s windows after the 3 s settle; each must hold at least 80 % of its samples."""
    end = float(timestamps[-1])
    cursor = float(timestamps[0]) + SETTLE_S
    out: list[np.ndarray] = []
    while cursor + WINDOW_S <= end + 1e-9 and len(out) < MAX_WINDOWS:
        mask = (timestamps >= cursor) & (timestamps < cursor + WINDOW_S)
        window = samples[mask]
        if len(window) >= int(fs * WINDOW_S * MIN_WINDOW_FILL):
            out.append(window)
        cursor += WINDOW_S
    if not out:
        raise InvalidCapture(
            "Rekaman terlalu pendek: butuh minimal 10 detik setelah 3 detik penyesuaian. Ulangi scan."
        )
    return out
