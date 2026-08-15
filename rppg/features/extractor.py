"""Per-recording PPG feature extraction.

See docs/superpowers/plans/2026-08-15-ppg-glucose-pipeline.md Task 4 for why
pulse rate is estimated from a Welch PSD peak rather than time-domain beat
counting, and why beat-level features (HRV, morphology) are best-effort.
"""
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

F0_BAND_HZ = (0.7, 2.0)    # 42-120 bpm: resting-plausible pulse-rate search band
AC_BAND_HZ = (0.5, 8.0)    # general pulsatile-component band (perfusion, quality)
MIN_BEATS_FOR_HRV = 3
MIN_BEATS_FOR_MORPHOLOGY = 2
IBI_OUTLIER_LOW = 0.7
IBI_OUTLIER_HIGH = 1.4
PPG_MODEL_VERSION = 'ppg-baseline-ridge-v0.1'


def _bandpass(signal, fs, low_hz, high_hz, order=3):
    """Second-order-sections Butterworth bandpass. SOS (not transfer-function
    b/a coefficients) is used because narrow relative bandwidths — as used
    here for beat detection around a specific fundamental frequency — make
    the b/a representation numerically unstable; filtfilt on b/a can blow up
    to absurd magnitudes on sharp/asymmetric waveforms even though the same
    filter is fine in SOS form. Caught by test_detect_beats_and_rmssd_on_clean_synthetic_signal
    during development: filt values reached ~1e10 with b/a on a realistic
    asymmetric pulse shape."""
    nyq = fs / 2.0
    low_hz = max(low_hz, 1e-3)
    high_hz = min(high_hz, nyq * 0.99)
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, signal)


def estimate_pulse_rate_bpm(signal, fs, band_hz=F0_BAND_HZ):
    s = signal - signal.mean()
    freqs, psd = welch(s, fs=fs, nperseg=len(s), nfft=len(s) * 4, detrend='linear')
    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(mask):
        return float('nan')
    f0 = freqs[mask][np.argmax(psd[mask])]
    return float(f0 * 60.0)


def _parabolic_interp(y, i):
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return float(i)
    return i + 0.5 * (y0 - y2) / denom


def detect_beats(signal, fs, pulse_rate_bpm):
    """Narrowband-filter around the PSD-estimated fundamental (suppressing the
    2nd cardiac harmonic) then peak-pick with sub-sample parabolic refinement."""
    if not np.isfinite(pulse_rate_bpm) or pulse_rate_bpm <= 0:
        return np.array([]), np.zeros_like(signal)
    f0 = pulse_rate_bpm / 60.0
    lo = max(0.4, f0 * 0.5)
    hi = min(fs / 2.0 * 0.95, f0 * 1.8)
    filt = _bandpass(signal, fs, lo, hi)
    min_dist = max(1, int(fs * 0.5 / f0))
    prominence = 0.4 * np.std(filt)
    peaks, _ = find_peaks(filt, distance=min_dist, prominence=prominence)
    refined = np.array([_parabolic_interp(filt, p) for p in peaks])
    return refined, filt


def compute_hrv_rmssd(beat_positions, fs):
    if len(beat_positions) < MIN_BEATS_FOR_HRV:
        return float('nan')
    ibi_ms = np.diff(beat_positions) / fs * 1000.0
    median_ibi = np.median(ibi_ms)
    valid = (ibi_ms > median_ibi * IBI_OUTLIER_LOW) & (ibi_ms < median_ibi * IBI_OUTLIER_HIGH)
    diffs = [ibi_ms[i + 1] - ibi_ms[i]
             for i in range(len(ibi_ms) - 1) if valid[i] and valid[i + 1]]
    if len(diffs) < 1:
        return float('nan')
    return float(np.sqrt(np.mean(np.square(diffs))))


def compute_perfusion_index(ac_signal, raw_signal):
    dc = raw_signal.mean()
    if dc == 0:
        return float('nan')
    ac_pp = ac_signal.max() - ac_signal.min()
    return float(ac_pp / dc * 100.0)


def compute_signal_quality(ac_signal, fs, min_bpm=40, max_bpm=200):
    """Autocorrelation-based periodicity SQI (Elgendi-style): the strength of
    the strongest autocorrelation peak within the physiological IBI lag
    range, normalized by zero-lag energy. 1.0 = perfectly periodic, near 0 =
    aperiodic/noisy."""
    s = ac_signal - ac_signal.mean()
    autocorr = np.correlate(s, s, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    if autocorr[0] == 0:
        return 0.0
    autocorr = autocorr / autocorr[0]
    min_lag = int(fs * 60 / max_bpm)
    max_lag = min(int(fs * 60 / min_bpm), len(autocorr) - 1)
    if min_lag >= max_lag:
        return 0.0
    return float(np.clip(autocorr[min_lag:max_lag].max(), 0.0, 1.0))


def compute_morphology(beat_positions, ac_signal, fs, search_frac=0.3):
    """Rise time and pulse width at half amplitude, averaged across beats.

    Measured on the wideband AC signal (not the narrowband beat-detection
    signal) so the measured shape reflects the true waveform rather than the
    detector's own filter response.
    """
    if len(beat_positions) < MIN_BEATS_FOR_MORPHOLOGY:
        return float('nan'), float('nan')
    rise_times, widths = [], []
    for i in range(1, len(beat_positions)):
        prev_beat, this_beat = beat_positions[i - 1], beat_positions[i]
        ibi_samples = this_beat - prev_beat
        window_start = max(int(this_beat - search_frac * ibi_samples), int(prev_beat))
        peak_idx = int(round(this_beat))
        if window_start >= peak_idx or peak_idx >= len(ac_signal):
            continue
        window = ac_signal[window_start:peak_idx + 1]
        if len(window) < 2:
            continue
        foot_idx = window_start + int(np.argmin(window))
        foot_val, peak_val = ac_signal[foot_idx], ac_signal[peak_idx]
        if peak_val <= foot_val:
            continue
        rise_times.append((peak_idx - foot_idx) / fs * 1000.0)

        half_amp = foot_val + 0.5 * (peak_val - foot_val)
        left = peak_idx
        while left > foot_idx and ac_signal[left] > half_amp:
            left -= 1
        right = peak_idx
        max_right = min(peak_idx + (peak_idx - foot_idx), len(ac_signal) - 1)
        while right < max_right and ac_signal[right] > half_amp:
            right += 1
        if right > left:
            widths.append((right - left) / fs * 1000.0)

    rise = float(np.mean(rise_times)) if rise_times else float('nan')
    width = float(np.mean(widths)) if widths else float('nan')
    return rise, width


def extract_features(signal, fs):
    """The per-recording feature vector consumed by the notebook's regressor."""
    pulse_rate_bpm = estimate_pulse_rate_bpm(signal, fs)
    ac_signal = _bandpass(signal, fs, *AC_BAND_HZ)
    beats, _ = detect_beats(signal, fs, pulse_rate_bpm)
    hrv_rmssd = compute_hrv_rmssd(beats, fs)
    perfusion_index = compute_perfusion_index(ac_signal, signal)
    signal_quality = compute_signal_quality(ac_signal, fs)
    rise_time_ms, pulse_width_half_ms = compute_morphology(beats, ac_signal, fs)
    return {
        'pulse_rate_bpm': pulse_rate_bpm,
        'hrv_rmssd_ms': hrv_rmssd,
        'perfusion_index': perfusion_index,
        'signal_quality': signal_quality,
        'rise_time_ms': rise_time_ms,
        'pulse_width_half_ms': pulse_width_half_ms,
        'n_beats_detected': len(beats),
    }


def _fallback(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, (int, float)) and not np.isfinite(value):
        return fallback
    return float(value)


def to_contract_dict(feature_dict, glucose_estimate, present=1, n_windows=1,
                      model_version=PPG_MODEL_VERSION, clip_fraction=0.0):
    """Map an extract_features() output (+ a glucose estimate) onto Interface
    Contract v1.1's PPG schema, substituting the contract's documented
    fallback values for anything missing/non-finite."""
    quality = feature_dict.get('signal_quality', 0.0)
    return {
        'ppg_pulse_rate_bpm': _fallback(feature_dict.get('pulse_rate_bpm'), 75.0),
        'ppg_hrv_rmssd': _fallback(feature_dict.get('hrv_rmssd_ms'), 30.0),
        'ppg_signal_quality': float(np.clip(quality, 0.0, 1.0)) if np.isfinite(quality) else 0.0,
        'ppg_perfusion_index': _fallback(feature_dict.get('perfusion_index'), 2.0),
        'ppg_glucose_estimate': _fallback(glucose_estimate, None),
        'ppg_present': int(present),
        'ppg_clip_fraction': float(clip_fraction),
        'ppg_n_windows': int(n_windows),
        'ppg_model_version': model_version,
    }


# (type, low, high, required) — low/high are None for non-numeric or unbounded fields
CONTRACT_SPEC = {
    'ppg_pulse_rate_bpm': (float, 40.0, 200.0, True),
    'ppg_hrv_rmssd': (float, 0.0, 200.0, False),
    'ppg_signal_quality': (float, 0.0, 1.0, True),
    'ppg_perfusion_index': (float, 0.0, 20.0, False),
    'ppg_glucose_estimate': (float, 70.0, 400.0, False),
    'ppg_present': (int, 0, 1, True),
    'ppg_clip_fraction': (float, 0.0, 1.0, True),
    'ppg_n_windows': (int, 0, 3, True),
    'ppg_model_version': (str, None, None, True),
}


def validate_contract_output(d):
    """Check an output dict against Interface Contract v1.1's PPG schema.
    Returns a list of problem strings; empty list means valid."""
    problems = []
    for key, (expected_type, lo, hi, required) in CONTRACT_SPEC.items():
        if key not in d:
            problems.append(f'{key}: missing')
            continue
        value = d[key]
        if value is None:
            if required:
                problems.append(f'{key}: required but None')
            continue
        if expected_type in (float, int):
            if not isinstance(value, (int, float, np.integer, np.floating)):
                problems.append(f'{key}: expected {expected_type.__name__}, got {type(value).__name__}')
            elif lo is not None and not (lo <= value <= hi):
                problems.append(f'{key}: {value} outside [{lo}, {hi}]')
        elif expected_type is str:
            if not isinstance(value, str):
                problems.append(f'{key}: expected str, got {type(value).__name__}')
    return problems
