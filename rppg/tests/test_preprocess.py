import numpy as np
import pytest

from rppg.data import preprocess


def test_bandpass_filter_removes_dc_and_keeps_pulsatile_component():
    fs = 2190
    t = np.arange(fs * 10) / fs
    sig = 1000 + 5 * np.sin(2 * np.pi * 1.2 * t)  # ~72 bpm tone on a large DC offset
    filt = preprocess.bandpass_filter(sig, fs)
    assert abs(filt.mean()) < 1.0
    assert filt.std() > 1.0


def test_decimate_to_30hz_exact_length():
    fs = 2190
    sig = np.random.randn(fs * 10)
    dec, fs_out = preprocess.decimate_to_30hz(sig, fs_in=fs)
    assert fs_out == 30
    assert dec.shape[0] == 300


def test_decimate_to_30hz_rejects_non_integer_ratio():
    sig = np.random.randn(100)
    with pytest.raises(ValueError):
        preprocess.decimate_to_30hz(sig, fs_in=100, fs_out=30)


def test_decimate_preserves_low_frequency_tone():
    fs = 2190
    t = np.arange(fs * 10) / fs
    sig = np.sin(2 * np.pi * 1.2 * t)  # 72 bpm, far below the 15 Hz Nyquist at 30 Hz
    dec, fs_out = preprocess.decimate_to_30hz(sig, fs_in=fs)
    t_dec = np.arange(len(dec)) / fs_out
    expected = np.sin(2 * np.pi * 1.2 * t_dec)
    corr = np.corrcoef(dec, expected)[0, 1]
    assert corr > 0.99
