import numpy as np
import pytest

from rppg.data import loader
from rppg.features import extractor


def _synthetic_pulse_train(fs, duration_s, bpm, noise_std=0.0, seed=0):
    """A repeating asymmetric bump (fast rise, slow fall) at a known bpm — a
    crude but adequate stand-in for a real PPG pulse waveform."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration_s)) / fs
    period = 60.0 / bpm
    phase = (t % period) / period
    # fast rise (0 -> 1 over first 30% of the cycle), slower decay after
    pulse = np.where(phase < 0.3, phase / 0.3, np.exp(-(phase - 0.3) * 4))
    baseline = 500.0
    sig = baseline + 20.0 * pulse
    if noise_std > 0:
        sig = sig + rng.normal(0, noise_std, size=sig.shape)
    return sig


def test_estimate_pulse_rate_bpm_on_synthetic_signal():
    fs = 2190
    bpm_true = 72.0
    sig = _synthetic_pulse_train(fs, 10, bpm_true)
    bpm_est = extractor.estimate_pulse_rate_bpm(sig, fs)
    assert abs(bpm_est - bpm_true) < 6.0  # within one Welch frequency bin


def test_estimate_pulse_rate_bpm_on_all_real_recordings_is_always_finite_and_plausible():
    records = loader.load_all_recordings()
    for r in records:
        bpm = extractor.estimate_pulse_rate_bpm(r['signal'], loader.FS_NATIVE)
        assert np.isfinite(bpm), f"non-finite bpm for subject {r['subject_id']} rec {r['recording_id']}"
        assert 40.0 <= bpm <= 130.0


def test_detect_beats_and_rmssd_on_clean_synthetic_signal():
    fs = 2190
    sig = _synthetic_pulse_train(fs, 10, 72.0, noise_std=0.05)
    bpm = extractor.estimate_pulse_rate_bpm(sig, fs)
    beats, _ = extractor.detect_beats(sig, fs, bpm)
    assert len(beats) >= 10  # ~12 beats expected in 10s at 72bpm
    rmssd = extractor.compute_hrv_rmssd(beats, fs)
    # a perfectly regular synthetic pulse train should have near-zero RMSSD
    assert rmssd < 15.0


def test_compute_hrv_rmssd_returns_nan_for_too_few_beats():
    assert np.isnan(extractor.compute_hrv_rmssd(np.array([10.0, 20.0]), 2190))
    assert np.isnan(extractor.compute_hrv_rmssd(np.array([]), 2190))


def test_compute_signal_quality_is_bounded():
    fs = 2190
    sig = _synthetic_pulse_train(fs, 10, 72.0)
    ac = extractor._bandpass(sig, fs, *extractor.AC_BAND_HZ)
    q = extractor.compute_signal_quality(ac, fs)
    assert 0.0 <= q <= 1.0
    noise = np.random.default_rng(0).normal(500, 1.0, size=fs * 10)
    ac_noise = extractor._bandpass(noise, fs, *extractor.AC_BAND_HZ)
    q_noise = extractor.compute_signal_quality(ac_noise, fs)
    assert q_noise < q  # clean periodic signal must score higher than pure noise


def test_extract_features_returns_all_expected_keys_on_real_data():
    records = loader.load_all_recordings()
    feats = extractor.extract_features(records[0]['signal'], loader.FS_NATIVE)
    expected_keys = {
        'pulse_rate_bpm', 'hrv_rmssd_ms', 'perfusion_index',
        'signal_quality', 'rise_time_ms', 'pulse_width_half_ms', 'n_beats_detected',
    }
    assert set(feats.keys()) == expected_keys
    assert np.isfinite(feats['pulse_rate_bpm'])
    assert 0.0 <= feats['signal_quality'] <= 1.0


def test_to_contract_dict_substitutes_fallbacks_for_missing_values():
    feats = {
        'pulse_rate_bpm': float('nan'), 'hrv_rmssd_ms': float('nan'),
        'perfusion_index': float('nan'), 'signal_quality': 0.0,
        'rise_time_ms': float('nan'), 'pulse_width_half_ms': float('nan'),
        'n_beats_detected': 0,
    }
    d = extractor.to_contract_dict(feats, glucose_estimate=float('nan'))
    assert d['ppg_pulse_rate_bpm'] == 75.0
    assert d['ppg_hrv_rmssd'] == 30.0
    assert d['ppg_perfusion_index'] == 2.0
    assert d['ppg_glucose_estimate'] is None


def test_to_contract_dict_passes_through_real_values():
    feats = {
        'pulse_rate_bpm': 82.3, 'hrv_rmssd_ms': 28.4, 'perfusion_index': 3.1,
        'signal_quality': 0.74, 'rise_time_ms': 150.0, 'pulse_width_half_ms': 110.0,
        'n_beats_detected': 12,
    }
    d = extractor.to_contract_dict(feats, glucose_estimate=118.5, n_windows=1)
    assert d['ppg_pulse_rate_bpm'] == 82.3
    assert d['ppg_glucose_estimate'] == 118.5
    assert d['ppg_n_windows'] == 1
    assert d['ppg_present'] == 1


def test_validate_contract_output_accepts_valid_dict():
    feats = {
        'pulse_rate_bpm': 82.3, 'hrv_rmssd_ms': 28.4, 'perfusion_index': 3.1,
        'signal_quality': 0.74, 'rise_time_ms': 150.0, 'pulse_width_half_ms': 110.0,
        'n_beats_detected': 12,
    }
    d = extractor.to_contract_dict(feats, glucose_estimate=118.5)
    problems = extractor.validate_contract_output(d)
    assert problems == []


def test_validate_contract_output_flags_out_of_range_value():
    feats = {
        'pulse_rate_bpm': 999.0, 'hrv_rmssd_ms': 28.4, 'perfusion_index': 3.1,
        'signal_quality': 0.74, 'rise_time_ms': 150.0, 'pulse_width_half_ms': 110.0,
        'n_beats_detected': 12,
    }
    d = extractor.to_contract_dict(feats, glucose_estimate=118.5)
    problems = extractor.validate_contract_output(d)
    assert any('ppg_pulse_rate_bpm' in p for p in problems)
