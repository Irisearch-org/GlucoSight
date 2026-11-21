"""Step 1: current glucose. Stale and invalid captures are refused, never defaulted."""

from datetime import timedelta

import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from integration.errors import ModelUnavailable
from integration.stages import glucose
from integration.stages.glucose import (
    G0Reading,
    GlucoseError,
    GlucoseOutOfRange,
    InvalidCapture,
    PPGModel,
    StaleReading,
    current_glucose,
    meter_error_range,
)

from .conftest import NOW, ppg_capture


def _iso(minutes_ago: float) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


@pytest.fixture(scope="module")
def ppg_model() -> PPGModel:
    """A real ridge pipeline (not the served weights) so capture validation runs offline."""
    import numpy as np

    rng = np.random.default_rng(0)
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1000.0)).fit(
        rng.normal(size=(20, 7)), 100.0 + rng.normal(size=20))
    return PPGModel(pipe, "test-ppg-v1", 25.0)


# ---------------------------------------------------------------- meter / CGM

def test_meter_reading_carries_the_iso_15197_band():
    r = current_glucose({"source": "meter", "value_mgdl": 85, "timestamp": _iso(5)}, NOW)
    assert isinstance(r, G0Reading)
    assert (r.value_mgdl, r.source, r.error_range_mgdl) == (85.0, "meter", 15.0)
    assert r.age_minutes == pytest.approx(5.0)
    assert r.capture_valid is None and r.ppg_model_version is None


@pytest.mark.parametrize("value, band", [(99.9, 15.0), (100.0, 15.0), (200.0, 30.0), (400.0, 60.0)])
def test_iso_band_switches_from_absolute_to_relative_at_100(value, band):
    assert meter_error_range(value) == pytest.approx(band)


def test_cgm_reading_has_no_error_range():
    r = current_glucose({"source": "cgm", "value_mgdl": 130, "timestamp": _iso(10)}, NOW)
    assert r.source == "cgm" and r.error_range_mgdl is None


@pytest.mark.parametrize("value", [39.9, 400.1, float("nan"), "abc", None])
def test_value_outside_40_400_is_refused(value):
    with pytest.raises(GlucoseOutOfRange):
        current_glucose({"source": "meter", "value_mgdl": value, "timestamp": _iso(1)}, NOW)


def test_reading_older_than_60_minutes_asks_to_remeasure():
    with pytest.raises(StaleReading) as exc:
        current_glucose({"source": "meter", "value_mgdl": 120, "timestamp": _iso(61)}, NOW)
    assert exc.value.detail == "Hasil pengukuran sudah lebih dari 60 menit. Silakan ukur ulang."


def test_reading_exactly_60_minutes_old_is_still_valid():
    r = current_glucose({"source": "cgm", "value_mgdl": 120, "timestamp": _iso(60)}, NOW)
    assert r.age_minutes == pytest.approx(60.0)


def test_reading_from_the_future_is_refused():
    with pytest.raises(GlucoseOutOfRange):
        current_glucose({"source": "meter", "value_mgdl": 120, "timestamp": _iso(-10)}, NOW)


def test_naive_timestamp_is_local_jakarta_time():
    # NOW is 19:00 WIB; a naive 18:30 is 30 minutes ago, not 7 h 30 min.
    r = current_glucose({"source": "meter", "value_mgdl": 120, "timestamp": "2026-01-15T18:30:00"}, NOW)
    assert r.age_minutes == pytest.approx(30.0)


def test_unparseable_timestamp_and_unknown_source_are_glucose_errors():
    with pytest.raises(GlucoseError):
        current_glucose({"source": "meter", "value_mgdl": 120, "timestamp": "kemarin"}, NOW)
    with pytest.raises(GlucoseError):
        current_glucose({"source": "tebak", "value_mgdl": 120, "timestamp": _iso(1)}, NOW)


# ------------------------------------------------------------------- PPG scan

def test_valid_scan_returns_the_served_estimate_with_its_error_range(ppg_model):
    r = current_glucose({"source": "ppg", "capture": ppg_capture()}, NOW, ppg_model=ppg_model)
    assert r.source == "ppg" and r.capture_valid is True
    assert 40.0 <= r.value_mgdl <= 400.0
    assert r.error_range_mgdl == 25.0 and r.ppg_model_version == "test-ppg-v1"
    assert r.heart_rate_bpm == pytest.approx(78.0, abs=3.0)
    assert r.age_minutes == 0.0 and r.timestamp == NOW


def test_scan_without_timestamps_assumes_the_declared_rate(ppg_model):
    cap = ppg_capture()
    del cap["timestamps"]
    r = current_glucose({"source": "ppg", "capture": cap}, NOW, ppg_model=ppg_model)
    assert r.capture_valid is True


def test_scan_needs_the_served_model():
    with pytest.raises(ModelUnavailable):
        current_glucose({"source": "ppg", "capture": ppg_capture()}, NOW, ppg_model=None)


@pytest.mark.parametrize("label, override", [
    ("torch not verified", {"torch_verified": False}),
    ("clipping over the limit", {"clip_fraction": 0.5}),
    ("clip fraction missing", {"clip_fraction": None}),
    ("fs below 10 Hz", {"fs": 5.0}),
    ("fs above 240 Hz", {"fs": 300.0}),
    ("samples above 1", {"samples": [1.5] * 990}),
    ("samples below 0", {"samples": [-0.1] * 990}),
    ("recording too short", {"__seconds": 12.0}),
])
def test_invalid_capture_is_refused(ppg_model, label, override):
    seconds = override.pop("__seconds", 33.0)
    cap = ppg_capture(seconds=seconds, **override)
    with pytest.raises(InvalidCapture):
        current_glucose({"source": "ppg", "capture": cap}, NOW, ppg_model=ppg_model)


def test_timestamp_jitter_over_5_percent_is_refused(ppg_model):
    cap = ppg_capture()
    ts = cap["timestamps"]
    for i in range(0, len(ts), 3):
        ts[i] += 0.012  # a third of the frames late by ~36 % of a frame; the median rate is unchanged
    with pytest.raises(InvalidCapture, match="Jitter"):
        current_glucose({"source": "ppg", "capture": cap}, NOW, ppg_model=ppg_model)


def test_timestamp_rate_disagreeing_with_fs_is_refused(ppg_model):
    cap = ppg_capture(fs=60.0)  # timestamps still say 30 Hz
    cap["timestamps"] = [i / 30.0 for i in range(len(cap["samples"]))]
    with pytest.raises(InvalidCapture):
        current_glucose({"source": "ppg", "capture": cap}, NOW, ppg_model=ppg_model)


def test_heart_rate_outside_40_180_is_refused(ppg_model, monkeypatch):
    import rppg.features.extractor as extractor

    real = extractor.extract_features
    monkeypatch.setattr(
        extractor, "extract_features", lambda x, fs: {**real(x, fs), "pulse_rate_bpm": 200.0})
    with pytest.raises(InvalidCapture, match="40–180"):
        current_glucose({"source": "ppg", "capture": ppg_capture()}, NOW, ppg_model=ppg_model)


# ------------------------------------------------ PPG reading re-sent to /predict

def _reading(minutes_ago: float = 5, **kw) -> dict:
    base = {
        "value_mgdl": 112.0, "source": "ppg", "timestamp": _iso(minutes_ago), "age_minutes": 0.0,
        "error_range_mgdl": 999.0, "heart_rate_bpm": 77.0, "capture_valid": True,
        "ppg_model_version": "forged",
    }
    base.update(kw)
    return base


def test_resent_reading_is_revalidated_by_age_and_takes_served_metadata(ppg_model):
    r = current_glucose({"source": "ppg", "reading": _reading(5)}, NOW, ppg_model=ppg_model)
    assert r.age_minutes == pytest.approx(5.0)
    # the client's error range / version are not trusted
    assert r.error_range_mgdl == 25.0 and r.ppg_model_version == "test-ppg-v1"
    assert r.heart_rate_bpm == 77.0


def test_resent_reading_older_than_60_minutes_is_stale(ppg_model):
    with pytest.raises(StaleReading):
        current_glucose({"source": "ppg", "reading": _reading(61)}, NOW, ppg_model=ppg_model)


def test_resent_reading_must_come_from_a_valid_capture(ppg_model):
    with pytest.raises(InvalidCapture):
        current_glucose(
            {"source": "ppg", "reading": _reading(capture_valid=False)}, NOW, ppg_model=ppg_model)


def test_the_served_ppg_artifact_reproduces_the_trial_prediction(real_ppg_model):
    # Training-set-like feature vector -> the trial's pred ~ 113.4 + 0.010 * true, so near the mean.
    feats = dict(zip(
        ("pulse_rate_bpm", "hrv_rmssd_ms", "perfusion_index", "signal_quality",
         "rise_time_ms", "pulse_width_half_ms", "n_beats_detected"),
        (75.0, 60.0, 1.0, 0.8, 150.0, 250.0, 12.0)))
    assert 80.0 < real_ppg_model.predict(feats) < 120.0
