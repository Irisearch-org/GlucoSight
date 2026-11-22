"""Per-recording features for every BUT PPG recording that carries glycaemia (3,840).

For each recording: brightest channel (highest mean, as frontend/src/ppg.ts), red, green;
handcrafted features (rppg.features.extractor, fs = 30, the 7 served FEATURE_NAMES) on the
signal as recorded ('raw', what the app sends) and polarity-flipped ('inv'); PaPaGei-S
embeddings through the 125 Hz path of ppg_served_model.common.papagei_window.
"""
import os
import sys
import time

import numpy as np

from common import CACHE_PATH, FS, load_meta, read_ppg, served_common
from rppg.features.extractor import extract_features
from rppg.models.glucose_estimator import FEATURE_NAMES

CH = {"r": 0, "g": 1, "b": 2}


def invert(x):
    return (x.max() + x.min()) - x


def hand(sig):
    try:
        f = extract_features(sig, FS)
        return np.array([f.get(n, np.nan) for n in FEATURE_NAMES], dtype=float)
    except Exception:
        return np.full(len(FEATURE_NAMES), np.nan)


def main():
    sc = served_common()
    enc = sc.load_papagei_encoder()
    meta = load_meta()
    n = len(meta)
    out = {k: np.full((n, 7), np.nan) for k in
           ("h_bright_raw", "h_bright_inv", "h_red_raw", "h_green_raw")}
    emb_keys = ("e_bright_inv", "e_bright_raw", "e_red_inv")
    win = {k: np.zeros((n, 1250), dtype=np.float32) for k in emb_keys}
    win_ok = {k: np.zeros(n, dtype=bool) for k in emb_keys}
    bright = np.zeros(n, dtype=int)
    means = np.zeros((n, 3))
    clip = np.zeros(n)
    t0 = time.time()
    for i, rid in enumerate(meta["ID"]):
        x = read_ppg(rid)
        assert x.shape == (300, 3), (rid, x.shape)
        means[i] = x.mean(0)
        b = int(np.argmax(means[i]))
        bright[i] = b
        clip[i] = float((x[:, b] >= 250 / 255).mean())   # frontend CLIP_LEVEL
        xb, xr, xg = x[:, b], x[:, 0], x[:, 1]
        out["h_bright_raw"][i] = hand(xb)
        out["h_bright_inv"][i] = hand(invert(xb))
        out["h_red_raw"][i] = hand(xr)
        out["h_green_raw"][i] = hand(xg)
        for key, sig in (("e_bright_inv", invert(xb)), ("e_bright_raw", xb), ("e_red_inv", invert(xr))):
            if np.std(sig) > 1e-9:
                win[key][i] = sc.papagei_window(sig, FS)
                win_ok[key][i] = True
        if i % 500 == 0:
            print(f"{i}/{n} {time.time() - t0:.0f}s", flush=True)
    emb = {}
    for key in emb_keys:
        e = np.full((n, 512), np.nan)
        ok = np.flatnonzero(win_ok[key])
        for s in range(0, len(ok), 256):
            idx = ok[s:s + 256]
            e[idx] = sc.papagei_embed(enc, win[key][idx])
        emb[key] = e
        print(key, "embedded", len(ok), flush=True)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        CACHE_PATH, ID=meta["ID"].to_numpy(), subj=meta["subj"].to_numpy(), rec=meta["rec"].to_numpy(),
        glucose=meta["glucose"].to_numpy(), site=meta["site"].to_numpy(), still=meta["still"].to_numpy(),
        session_id=meta["session_id"].to_numpy(), sess_order=meta["sess_order"].to_numpy(),
        bright=bright, means=means, clip=clip, feature_names=np.array(FEATURE_NAMES),
        **out, **emb)
    print("wrote", CACHE_PATH, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
