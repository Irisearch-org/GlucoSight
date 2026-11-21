"""Fit the served candidate (selected in ppg_served_model.json) on all 23 subjects
and save it as a gitignored artifact (*.pt). Runs a round-trip check: the raw-signal
path must reproduce the cached-feature prediction.

    /tmp/ppg-venv/bin/python serve.py --out <path>.pt
    load: PPGGlucoseServed.from_artifact(<path>.pt).predict(raw_2190hz_signal)
"""
import argparse
import hashlib
import json
import os

import numpy as np

from common import (CACHE_PATH, FS_NATIVE, HERE, PAPAGEI_WEIGHTS, handcrafted,
                    load_papagei_encoder, load_recordings, papagei_embed, papagei_window)
from candidates import CANDIDATES, get_X

JSON_PATH = os.path.join(HERE, "ppg_served_model.json")


class PPGGlucoseServed:
    """Served estimator: raw 2190 Hz recording (10 s) -> glucose mg/dL estimate."""

    def __init__(self, estimator, feature_key):
        self.estimator = estimator
        self.feature_key = feature_key
        self._encoder = None

    @classmethod
    def from_artifact(cls, path):
        import torch
        blob = torch.load(path, weights_only=False)
        return cls(blob["estimator"], blob["feature_key"])

    def _features(self, raw, fs=FS_NATIVE):
        if self.feature_key == "hand":
            vec, sqi = handcrafted(raw, fs)
            return vec[None, :], sqi
        if self._encoder is None:
            self._encoder = load_papagei_encoder()
        return papagei_embed(self._encoder, papagei_window(raw, fs)[None, :]), None

    def predict(self, raw, fs=FS_NATIVE):
        X, _ = self._features(raw, fs)
        return float(self.estimator.predict(X)[0])

    def predict_with_sqi(self, raw, fs=FS_NATIVE):
        X, sqi = self._features(raw, fs)
        return float(self.estimator.predict(X)[0]), sqi


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(JSON_PATH) as fh:
        res = json.load(fh)
    name = res["selected_served"]
    feat, fit, desc = CANDIDATES[name]
    cache = np.load(CACHE_PATH)
    y = cache["glucose"]
    groups = cache["subject_id"]
    X = get_X(cache, feat)
    est = fit(X, y, groups)
    served = PPGGlucoseServed(est, feat)

    # Round-trip: raw-signal path vs cached-feature prediction for every training recording.
    raws = {(str(r["subject_id"]), str(r["recording_id"])): np.asarray(r["signal"], float).ravel()
            for r in load_recordings()}
    diffs = []
    for i, (s, r) in enumerate(zip(cache["subject_id"], cache["recording_id"])):
        cached_pred = float(est.predict(X[i:i + 1])[0])
        raw_pred = served.predict(raws[(str(s), str(r))])
        diffs.append(abs(cached_pred - raw_pred))
    max_diff = float(np.max(diffs))
    print(f"round-trip max |raw - cached| = {max_diff:.6f} mg/dL over {len(diffs)} recordings")

    sha = hashlib.sha256(open(PAPAGEI_WEIGHTS, "rb").read()).hexdigest() \
        if feat.startswith("papagei") else None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "estimator": est,
        "name": name,
        "feature_key": feat,
        "description": desc,
        "cv_mae_gate_protocol": res["candidates"][name]["mae"],
        "fs_native": FS_NATIVE,
        "train_n_recordings": int(len(y)),
        "train_n_subjects": int(len(np.unique(groups))),
        "roundtrip_max_abs_diff": max_diff,
        "papagei_s_weights_sha256": sha,
    }, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
