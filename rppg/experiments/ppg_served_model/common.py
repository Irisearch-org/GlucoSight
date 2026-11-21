"""Shared loaders for the PPG served-model trial.

Inputs and outputs:
  * raw recordings: PPG_RAW_DIR (absolute path to rppg/PPG_Dataset/RawData,
    gitignored, so it is not in a fresh worktree)
  * PaPaGei-S weights: PAPAGEI_WEIGHTS (papagei_s.pt, gitignored)
  * PaPaGei-S code: PAPAGEI_REPO (clone of Nokia-Bell-Labs/papagei-foundation-model)
  * feature cache: PPG_CACHE (default /tmp/ppg_served_cache/features.npz)
"""
import os
import sys
from math import gcd

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

FS_NATIVE = 2190          # rppg/PPG_Dataset raw files: 10 s at 2190 Hz
FS_DEPLOY = 30            # phone-camera rate, sensitivity only
PAPAGEI_FS = 125          # PaPaGei-S input rate
PAPAGEI_LEN = 1250        # 10 s at 125 Hz

RAW_DIR = os.environ.get("PPG_RAW_DIR")
PAPAGEI_REPO = os.environ.get("PAPAGEI_REPO", "/tmp/papagei")
PAPAGEI_WEIGHTS = os.environ.get(
    "PAPAGEI_WEIGHTS", os.path.join(REPO_ROOT, "rppg", "models", "weights", "papagei_s.pt"))
CACHE_PATH = os.environ.get("PPG_CACHE", "/tmp/ppg_served_cache/features.npz")


def load_recordings():
    from rppg.data.loader import load_all_recordings, RAW_DIR as DEFAULT_RAW
    return load_all_recordings(raw_dir=RAW_DIR or DEFAULT_RAW)


def handcrafted(raw, fs=FS_NATIVE):
    """The 7 features of rppg/models/glucose_estimator.FEATURE_NAMES, plus SQI."""
    from rppg.features.extractor import extract_features
    from rppg.models.glucose_estimator import FEATURE_NAMES
    feats = extract_features(raw, fs)
    vec = np.array([feats.get(n, np.nan) for n in FEATURE_NAMES], dtype=float)
    return vec, float(feats.get("signal_quality", np.nan))


def papagei_window(raw, fs):
    """Raw recording -> 1250-sample PaPaGei-S input window at 125 Hz.

    Z-score, polyphase resample to PaPaGei's 125 Hz training rate, then
    PaPaGei's pyPPG band-pass (0.5-12 Hz, order 4) at 125 Hz, then
    centre-pad/crop to 1250 samples. Order differs from the notebook
    (which filters at the source rate): pyPPG's ba-form IIR filter with a
    0.5 Hz corner at 2190 Hz overflows (|x| ~ 1e134), so filtering is done
    at the model's own rate, where the same 0.5-12 Hz band is passed.
    """
    import pyPPG.preproc as PP
    from dotmap import DotMap
    from scipy.signal import resample_poly

    x = np.asarray(raw, dtype=float).ravel()
    x = (x - x.mean()) / x.std()
    fs_int = int(round(fs))
    g = gcd(PAPAGEI_FS, fs_int)
    x = resample_poly(x, PAPAGEI_FS // g, fs_int // g)

    prep = PP.Preprocess(fL=0.5, fH=12, order=4,
                         sm_wins={"ppg": 50, "vpg": 10, "apg": 10, "jpg": 10})
    sig = DotMap()
    sig.v = x
    sig.fs = PAPAGEI_FS
    sig.filtering = True
    ppg = np.asarray(prep.get_signals(sig)[0], dtype=float)

    pad = PAPAGEI_LEN - len(ppg)
    if pad > 0:
        left = pad // 2
        ppg = np.pad(ppg, (left, pad - left))
    return ppg[:PAPAGEI_LEN]


def load_papagei_encoder(weights=PAPAGEI_WEIGHTS):
    import torch
    if PAPAGEI_REPO not in sys.path:
        sys.path.insert(0, PAPAGEI_REPO)
    from models.resnet import ResNet1DMoE   # from the PaPaGei repo
    model = ResNet1DMoE(in_channels=1, base_filters=32, kernel_size=3, stride=2,
                        groups=1, n_block=18, n_classes=512, n_experts=3)
    ckpt = torch.load(weights, map_location="cpu")
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def papagei_embed(model, windows):
    """(N, 1250) windows -> (N, 512) PaPaGei-S embeddings (output index 0)."""
    import torch
    x = torch.tensor(np.asarray(windows, dtype=np.float32)).unsqueeze(1)
    with torch.inference_mode():
        return model(x)[0].cpu().numpy()
