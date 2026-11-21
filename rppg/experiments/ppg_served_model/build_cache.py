"""Compute and cache per-recording features once: handcrafted (native rate),
SQI, and PaPaGei-S embeddings at native and 30 Hz. Run with /tmp/ppg-venv (needs pyPPG).

    PPG_RAW_DIR=... PAPAGEI_WEIGHTS=... /tmp/ppg-venv/bin/python build_cache.py
"""
import os

import numpy as np

from common import (CACHE_PATH, FS_DEPLOY, FS_NATIVE, handcrafted, load_papagei_encoder,
                    load_recordings, papagei_embed, papagei_window)


def main():
    from rppg.data.preprocess import decimate_to_30hz

    records = load_recordings()
    model = load_papagei_encoder()

    keep, subj, rec, y, hand, sqi, win_n, win_30, dropped = [], [], [], [], [], [], [], [], []
    for r in records:
        raw = np.asarray(r["signal"], dtype=float).ravel()
        hvec, q = handcrafted(raw, FS_NATIVE)
        w_n = papagei_window(raw, FS_NATIVE)
        sig30, fs30 = decimate_to_30hz(raw, FS_NATIVE, FS_DEPLOY)
        w_30 = papagei_window(sig30, fs30)
        ok = np.all(np.isfinite(hvec)) and np.all(np.isfinite(w_n)) and np.all(np.isfinite(w_30))
        if not ok:
            dropped.append(f"{r['subject_id']}/{r['recording_id']}")
            continue
        subj.append(r["subject_id"])
        rec.append(r["recording_id"])
        y.append(r["glucose_mgdl"])
        hand.append(hvec)
        sqi.append(q)
        win_n.append(w_n)
        win_30.append(w_30)

    emb_n = papagei_embed(model, np.stack(win_n))
    emb_30 = papagei_embed(model, np.stack(win_30))

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez(
        CACHE_PATH,
        subject_id=np.array(subj),
        recording_id=np.array(rec),
        glucose=np.array(y, dtype=float),
        hand=np.stack(hand),
        sqi=np.array(sqi, dtype=float),
        papagei_native=emb_n,
        papagei_30hz=emb_30,
        dropped=np.array(dropped),
    )
    print(f"recordings kept={len(y)} subjects={len(set(subj))} dropped={dropped}")
    print(f"emb native {emb_n.shape}, emb 30Hz {emb_30.shape}, cache={CACHE_PATH}")


if __name__ == "__main__":
    main()
