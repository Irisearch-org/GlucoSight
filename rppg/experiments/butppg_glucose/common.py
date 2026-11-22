"""Shared loaders and statistics for the BUT PPG v2 glucose experiments (tickets #27, #28).

Raw data (CC BY 4.0, PhysioNet) lives outside git: BUTPPG_DIR (default
/Users/auliamnaufal/Downloads/butppg-2.0.0). Features cache: BUTPPG_CACHE
(default /tmp/butppg_cache/features.npz). Run with /tmp/ppg-venv/bin/python.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

DATA_DIR = os.environ.get("BUTPPG_DIR", "/Users/auliamnaufal/Downloads/butppg-2.0.0")
CACHE_PATH = os.environ.get("BUTPPG_CACHE", "/tmp/butppg_cache/features.npz")
RESULTS_DIR = os.path.join(HERE, "results")

FS = 30                     # BUT PPG camera frame rate
MGDL_PER_MMOL = 18.016
ADC_FULL_SCALE = 255.0      # channels are 8-bit pixel means; the app sends 0-1 samples
RANDOM_STATE = 42

GLYC = "Glycaemia [mmol/l]"


def load_meta():
    """subject-info.csv for the 3,840 recordings that carry glycaemia.

    Columns added: subj, rec, site ('finger'|'ear'), still (Motion == 0), glucose (mg/dL),
    session_id (subject_glucose, the unit the label belongs to) and sess_order (0 = the
    earlier session, i.e. lower record numbers; 1 = later), per subject.
    """
    d = pd.read_csv(os.path.join(DATA_DIR, "subject-info.csv"))
    d["ID"] = d["ID"].astype(str)
    d["subj"] = d["ID"].str[:3].astype(int)
    d["rec"] = d["ID"].str[3:].astype(int)
    d = d.dropna(subset=[GLYC]).copy()
    d["site"] = np.where(d["Ear/finger"] == 1, "finger", "ear")
    d["still"] = d["Motion"].astype(str) == "0"
    d["glucose"] = d[GLYC] * MGDL_PER_MMOL
    d["gl_mmol"] = d[GLYC].round(1)
    d = d.sort_values(["subj", "rec"]).reset_index(drop=True)
    d["session_id"] = d["subj"].astype(str) + "_" + d["gl_mmol"].astype(str)
    first_rec = d.groupby("session_id")["rec"].transform("min")
    order = {}
    for s, g in d.groupby("subj"):
        ids = g.groupby("session_id")["rec"].min().sort_values().index.tolist()
        for k, sid in enumerate(ids):
            order[sid] = k
    d["sess_order"] = d["session_id"].map(order)
    return d


def read_ppg(rec_id):
    """(300, 3) array of the R, G, B channel means, scaled to 0-1 like the app's samples."""
    import wfdb
    sid = str(rec_id)
    r = wfdb.rdrecord(os.path.join(DATA_DIR, sid[:6], f"{sid}_PPG"))
    return np.asarray(r.p_signal, dtype=float) / ADC_FULL_SCALE


def served_common():
    """rppg/experiments/ppg_served_model/common.py (PaPaGei window + encoder), loaded by path."""
    path = os.path.join(REPO_ROOT, "rppg", "experiments", "ppg_served_model", "common.py")
    spec = importlib.util.spec_from_file_location("served_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_cache():
    z = np.load(CACHE_PATH, allow_pickle=True)
    return {k: z[k] for k in z.files}
