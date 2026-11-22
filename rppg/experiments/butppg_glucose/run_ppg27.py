"""Ticket #27: cross-person glucose from smartphone-camera PPG (BUT PPG v2).

    python run_ppg27.py primary [N_PERM_RIDGE] [N_PERM_RF]   # finger + still, gated
    python run_ppg27.py sens                                  # sensitivities, ungated
Writes results/ppg27_primary.json / ppg27_sens.json and OOF predictions for #28.
"""
import json
import os
import sys
import time

import numpy as np

from common import RESULTS_DIR, load_cache
from eval_lib import FITS, N_JOBS, N_REPEATS, run_cv, summarize

FEATS = {
    "hand_raw": "h_bright_raw", "hand_inv": "h_bright_inv", "hand_red": "h_red_raw",
    "hand_green": "h_green_raw", "pap_inv": "e_bright_inv", "pap_raw": "e_bright_raw",
    "pap_red_inv": "e_red_inv",
}
PRIMARY = [("hand_raw", "ridge"), ("pap_inv", "ridge"), ("hand_raw", "rf")]


def subset_mask(c, name):
    site, still = c["site"], c["still"]
    return {"finger_still": (site == "finger") & still, "finger_all": site == "finger",
            "ear_still": (site == "ear") & still, "all": np.ones(len(site), bool)}[name]


def prepare(c, subset, feats):
    m = subset_mask(c, subset)
    for f in ("hand_raw", "pap_inv") + tuple(feats):
        m &= np.isfinite(c[FEATS[f]]).all(1)
    return m


def one(c, subset, feat, model, n_perm=0, loso=False, n_jobs=N_JOBS):
    m = prepare(c, subset, (feat,))
    X = c[FEATS[feat]][m]
    y, g, s, so = c["glucose"][m], c["subj"][m], c["session_id"][m], c["sess_order"][m]
    t0 = time.time()
    res, pred, base = summarize(FITS[model], X, y, g, s, so, n_perm=n_perm, loso=loso, n_jobs=n_jobs)
    # between- vs within-person tracking of the repeat-averaged OOF predictions
    pm = pred.mean(0)
    subs = np.unique(g)
    sm_p = np.array([pm[g == k].mean() for k in subs]); sm_y = np.array([y[g == k].mean() for k in subs])
    ids, inv = np.unique(s, return_inverse=True)
    sp = np.bincount(inv, weights=pm) / np.bincount(inv); sy = np.bincount(inv, weights=y) / np.bincount(inv)
    sg = np.zeros(len(ids), int); sg[inv] = g
    dp = sp - np.array([sp[sg == k].mean() for k in sg]); dy = sy - np.array([sy[sg == k].mean() for k in sg])
    res["tracking"] = {
        "r_between_subject_means": round(float(np.corrcoef(sm_p, sm_y)[0, 1]), 3),
        "r_within_subject_session_means": round(float(np.corrcoef(dp, dy)[0, 1]), 3) if dy.std() > 0 else None,
    }
    res["runtime_s"] = round(time.time() - t0, 1)
    res["subset"], res["features"], res["model"] = subset, feat, model
    return res, pred, base, m


def main():
    mode = sys.argv[1]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    c = load_cache()
    out = {}
    jpath = os.path.join(RESULTS_DIR, f"ppg27_{'primary' if mode == 'primary' else 'sens'}.json")
    if os.path.exists(jpath):          # resume: keep finished runs
        out = json.load(open(jpath))
    if mode == "primary":
        n_ridge = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        n_rf = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        for feat, model in PRIMARY:
            key = f"{feat}+{model}"
            if key in out:
                continue
            res, pred, base, m = one(c, "finger_still", feat, model, n_perm=n_ridge if model == "ridge" else n_rf)
            out[key] = res
            print(key, res["recording"], res.get("gate_recording", {}).get("passes"), res["runtime_s"], flush=True)
            np.savez_compressed(os.path.join(RESULTS_DIR, f"oof_{key.replace('+', '_')}.npz"),
                                idx=np.flatnonzero(m), pred=pred.astype(np.float32), base=base.astype(np.float32))
            json.dump(out, open(os.path.join(RESULTS_DIR, "ppg27_primary.json"), "w"), indent=1)
    else:
        jobs = []
        for feat, model in [("hand_inv", "ridge"), ("hand_inv", "rf"), ("hand_red", "ridge"), ("hand_green", "ridge"),
                            ("pap_raw", "ridge"), ("pap_red_inv", "ridge")]:
            jobs.append(("finger_still", feat, model, False))
        for feat, model in PRIMARY:
            jobs.append(("finger_still", feat, model, True))      # LOSO
            for subset in ("finger_all", "ear_still", "all"):
                jobs.append((subset, feat, model, False))
        for subset, feat, model, loso in jobs:
            key = f"{subset}|{feat}+{model}" + ("|LOSO" if loso else "")
            if key in out:
                continue
            res, *_ = one(c, subset, feat, model, loso=loso)
            out[key] = res
            print(key, res["recording"]["mae"], res["recording"]["baseline_mae"], res["recording"]["gain"],
                  res["recording"]["ci95"], res["runtime_s"], flush=True)
            json.dump(out, open(os.path.join(RESULTS_DIR, "ppg27_sens.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
