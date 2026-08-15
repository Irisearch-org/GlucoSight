"""Signal conditioning: bandpass filtering and 30 Hz deployment-rate decimation.

2190 / 30 = 73 exactly, so decimate_to_30hz uses a single-stage
scipy.signal.decimate with an FIR anti-alias filter (ftype='fir') rather than
the default IIR — FIR is more numerically stable at large decimation factors
and this is a one-shot offline operation, so its extra cost is irrelevant.
Naive slicing (signal[::73]) would alias high-frequency content back into the
pulse band and is exactly what this function exists to avoid.
"""
import numpy as np
from scipy.signal import butter, decimate, filtfilt

FS_DEPLOY = 30


def bandpass_filter(signal, fs, low_hz=0.5, high_hz=8.0, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype='band')
    return filtfilt(b, a, signal)


def decimate_to_30hz(signal, fs_in, fs_out=FS_DEPLOY):
    if fs_in % fs_out != 0:
        raise ValueError(f'fs_in={fs_in} must be an integer multiple of fs_out={fs_out}')
    q = fs_in // fs_out
    decimated = decimate(signal, q, ftype='fir', zero_phase=True)
    return decimated, fs_out
