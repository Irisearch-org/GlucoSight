"""GBDT CGM forecaster trial on the full CGMacros release (wayfinder #17).

Two LightGBM regressors, one per horizon (T+60, T+120). Target: the exact Dexcom
reading at meal time + horizon (no interpolation). Input g0: the last Dexcom
reading strictly before the meal (C2). Non-CGM inputs are produced by noise
augmentation:

* meter: ISO 15197:2013 accuracy envelope. Per reading, error ~ N(0, sd) with
  sd chosen so that 95% of readings fall inside the limit (+-15 mg/dL below
  100 mg/dL, +-15% at or above 100 mg/dL). Inputs clipped to [20, 600] mg/dL.
* ppg: g0_ppg = a + b * g0 + N(0, sigma), with (a, b, sigma) fitted on the served PPG
  model's held-out recordings (pred vs true, separate cohort from CGMacros) when a CSV is supplied.

Evaluation (ADR 0004 gate, C3/C4):
* Outer loop: N_REPEATS repetitions of a participant-grouped N_SPLITS-fold split
  (participants reshuffled per repetition). Every test participant is held out.
* Inner loop (training participants only): grouped CV selects the feature set
  and GBDT configuration. The outer test folds never influence selection.
* B0/B1/B2 are computed on the same outer folds and the same g0 inputs.
* Gate: positive pooled gain vs B2, >= MIN_FOLD_WINS wins in every repetition,
  participant-cluster bootstrap CI excluding zero, and within-participant macro
  permutation p <= MAX_PERMUTATION_P (selected configuration held fixed).
* Clarke Error Grid via utils/clarke_grid.py, per-participant median/IQR, and
  zone A by reference range.

Usage (from repo root):
    python -m forecasting.evaluate.gbdt_forecaster \
        --data-root /Users/auliamnaufal/Downloads/CGMacros_full/CGMacros \
        --out forecasting/gbdt_results [--horizons 60 120] \
        [--ppg-residuals PATH.csv] [--carb-error-pool PATH.csv] \
        [--models-dir DIR] [--permutations 200]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from utils.clarke_grid import (
    clarke_error_grid,
    per_participant_zone_a,
    zone_a_by_reference_range,
)

HORIZONS = (60, 120)
N_SPLITS = 5            # outer folds per repetition (matches rppg/models/glucose_estimator.py)
N_REPEATS = 5           # outer repetitions with reshuffled participant folds
INNER_SPLITS = 4
RANDOM_STATE = 42
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 200    # p resolution 1/201
MIN_FOLD_WINS = 4       # of N_SPLITS within every repetition (ADR 0004)
MAX_PERMUTATION_P = 0.05
ATWATER_TOL = 0.30      # #12 primary rule
SENSITIVITY_TOLS = (0.20, 0.30, 0.50)
N_METER_DRAWS = 3       # training noise draws per meal per noisy source
N_EVAL_DRAWS = 5        # evaluation noise draws per test meal per noisy source
LEARNING_RATE = 0.05
RANGES = ((0, 100), (100, 140), (140, 180), (180, 250), (250, 10000))
MEAL_TYPES = {"Breakfast", "Lunch", "Dinner", "Snacks"}

# ISO 15197:2013 accuracy limit: 95% of readings within +-15 mg/dL (reference < 100
# mg/dL) or +-15% (reference >= 100 mg/dL). Treated as a zero-mean normal error whose
# 95% two-sided half-width equals the limit: sd = limit / 1.96.
ISO_ABS_MGDL = 15.0
ISO_REL = 0.15
ISO_Z = 1.959963984540054
METER_CLIP = (20.0, 600.0)

BASE = ["g0_mgdl", "src_meter", "src_ppg", "hour_sin", "hour_cos"]
MACRO = ["carbs_g", "protein_g", "fat_g", "fibre_g"]
PROFILE = ["age_y", "female", "bmi", "dm_band"]
FEATURE_SETS = {
    "g0_hour": BASE,
    "g0_hour_macros": BASE + MACRO,
    "g0_hour_profile": BASE + PROFILE,
    "full": BASE + MACRO + PROFILE,
}
GRID = [
    {"offset": o, "num_leaves": nl, "min_child_samples": mc, "n_estimators": ne}
    for o in (False, True)
    for nl in (4, 8)
    for mc in (20, 60)
    for ne in (100, 300)
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _participant_csvs(root: Path):
    for d in sorted(root.glob("CGMacros-*")):
        m = re.fullmatch(r"CGMacros-(\d{3})", d.name)
        path = d / f"{d.name}.csv"
        if m and path.exists():
            yield int(m.group(1)), path


def _dm_band(a1c: pd.Series) -> pd.Series:
    """HbA1c (%): <5.7 normal (0), 5.7-6.4 prediabetes (1), >=6.5 T2D (2)."""
    band = pd.Series(np.nan, index=a1c.index)
    band[a1c < 5.7] = 0.0
    band[(a1c >= 5.7) & (a1c < 6.5)] = 1.0
    band[a1c >= 6.5] = 2.0
    return band


def build_dataset(root: Path):
    """One row per meal event with g0, exact Dexcom targets and cleaned nutrients."""
    bio = pd.read_csv(root / "bio.csv")
    bio.columns = bio.columns.str.strip()
    bio["subject"] = bio["subject"].astype(int)
    bio = bio.set_index("subject")

    meal_rows = []
    for pid, path in _participant_csvs(root):
        raw = pd.read_csv(path)
        raw.columns = raw.columns.str.strip()
        raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
        dexcom = pd.to_numeric(raw["Dexcom GL"], errors="coerce")
        dex = (
            raw.assign(dexcom=dexcom)
            .dropna(subset=["dexcom"])
            .drop_duplicates("Timestamp", keep="last")
            .set_index("Timestamp")["dexcom"]
            .sort_index()
        )

        meal_mask = raw["Meal Type"].notna()
        for _, r in raw.loc[meal_mask].iterrows():
            t0 = r["Timestamp"]
            label = str(r["Meal Type"]).strip().lower()
            meal_type = "Snacks" if label.startswith("snack") else label.title()
            if meal_type not in MEAL_TYPES:
                raise ValueError(f"unknown meal type {meal_type!r} in CGMacros-{pid:03d}")

            i = dex.index.searchsorted(t0, side="left") - 1  # last index strictly < t0
            if i >= 0:
                g0, g0_ts = float(dex.iloc[i]), dex.index[i]
                assert g0_ts < t0, "C2 violation: g0 at or after t0"
                g0_age = (t0 - g0_ts).total_seconds() / 60.0
            else:
                g0, g0_ts, g0_age = np.nan, pd.NaT, np.nan

            def target(h_min: int) -> float:
                ts = t0 + pd.Timedelta(minutes=h_min)
                return float(dex.loc[ts]) if ts in dex.index else np.nan

            meal_rows.append({
                "subject": pid,
                "t0": t0,
                "meal_type": meal_type,
                "calories": r["Calories"],
                "carbs_g": r["Carbs"],
                "protein_g": r["Protein"],
                "fat_g": r["Fat"],
                "fibre_g": r["Fiber"] if "Fiber" in r.index else np.nan,
                "g0_mgdl": g0,
                "g0_ts": g0_ts,
                "g0_age_min": g0_age,
                "y60": target(60),
                "y120": target(120),
            })

    meals = pd.DataFrame(meal_rows)
    for c in ["calories", "carbs_g", "protein_g", "fat_g", "fibre_g"]:
        meals[c] = pd.to_numeric(meals[c], errors="coerce")
    meals = meals.reset_index(drop=True)
    meals.insert(0, "meal_id", np.arange(len(meals)))
    meals, cleaning = clean_meals(meals)

    a1c = pd.to_numeric(bio["A1c PDL (Lab)"], errors="coerce")
    profile = pd.DataFrame({
        "age_y": pd.to_numeric(bio["Age"], errors="coerce"),
        "female": bio["Gender"].astype(str).str.strip().str.upper().map({"F": 1.0, "M": 0.0}),
        "bmi": pd.to_numeric(bio["BMI"], errors="coerce"),
        "dm_band": _dm_band(a1c),
    })
    for c in profile.columns:
        meals[c] = meals["subject"].map(profile[c])

    hour = meals["t0"].dt.hour + meals["t0"].dt.minute / 60.0
    meals["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    meals["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    cleaning["n_participants_with_meals"] = int(meals["subject"].nunique())
    return meals, cleaning, profile


def clean_meals(meals: pd.DataFrame):
    """#12 rules: all-zero markers out; Atwater check at ATWATER_TOL; fibre>carbs -> NaN."""
    rep: dict = {"meal_events_total": int(len(meals))}
    core = ["calories", "carbs_g", "protein_g", "fat_g"]
    rep["fibre_gt_carbs_raw_rows"] = int((meals["fibre_g"] > meals["carbs_g"]).sum())

    zero = (meals[core] == 0).all(axis=1)
    rep["dropped_all_zero_marker"] = int(zero.sum())
    meals = meals[~zero]

    missing = meals[core].isna().any(axis=1)
    rep["dropped_missing_composition"] = int(missing.sum())
    meals = meals[~missing].copy()

    atw = 4 * meals["carbs_g"] + 4 * meals["protein_g"] + 9 * meals["fat_g"]
    rep["atwater_kcal_zero_rows"] = int((meals["calories"] == 0).sum())
    ratio = np.where(meals["calories"] > 0, atw / meals["calories"].where(meals["calories"] > 0), np.inf)
    dev = np.abs(ratio - 1.0)
    rep["atwater_deviation_counts"] = {
        f"gt_{int(t * 100)}pct": int((dev > t).sum()) for t in SENSITIVITY_TOLS
    }
    drop = dev > ATWATER_TOL
    rep["dropped_atwater_primary"] = int(drop.sum())
    rep["dropped_atwater_primary_participants"] = int(meals.loc[drop, "subject"].nunique())
    meals = meals[~drop].copy()
    rep["median_atwater_ratio_kept"] = float(np.median(
        (4 * meals["carbs_g"] + 4 * meals["protein_g"] + 9 * meals["fat_g"]) / meals["calories"]
    ))

    fib = meals["fibre_g"] > meals["carbs_g"]
    rep["fibre_gt_carbs_set_nan"] = int(fib.sum())
    rep["fibre_gt_carbs_by_participant"] = {
        str(k): int(v) for k, v in meals.loc[fib, "subject"].value_counts().items()
    }
    meals.loc[fib, "fibre_g"] = np.nan

    rep["meal_events_kept"] = int(len(meals))
    rep["participants_kept"] = int(meals["subject"].nunique())
    return meals.reset_index(drop=True), rep


def load_ppg_calibration(path: Path | None) -> dict | None:
    """Served-PPG calibration map fitted on held-out PPG recordings: pred = a + b * true + N(0, sigma).

    The PPG study is a separate cohort from CGMacros, so the map cannot leak test participants.
    The served model compresses predictions (pred SD ~4 vs truth SD ~19); a linear map captures that.
    """
    if path is None:
        return None
    df = pd.read_csv(path)
    need = {"glucose_true", "glucose_pred"}
    if not need <= set(df.columns):
        raise ValueError(f"{path} needs columns {sorted(need)}")
    df = df.dropna(subset=list(need))
    x = df["glucose_true"].to_numpy(dtype=float)
    y = df["glucose_pred"].to_numpy(dtype=float)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    return {
        "a": float(a),
        "b": float(b),
        "sigma": float(resid.std(ddof=2)),
        "n_rows": int(len(df)),
        "n_recordings": int(df["recording_id"].nunique()) if "recording_id" in df.columns else None,
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else None,
        "corr_true_pred": float(np.corrcoef(x, y)[0, 1]),
    }


def load_carb_error_pool(path: Path | None) -> np.ndarray | None:
    """Empirical dish-ID carb errors (g, estimate - true), resampled per CGMacros test meal.

    Uses the dish-ID test split, embedding matcher, non-abstained rows. Abstained rows are
    manual entries and carry no estimation error, so they are excluded. The file is
    TKPI-derived and is read from its external path; it is never committed.
    """
    if path is None:
        return None
    df = pd.read_csv(path)
    need = {"split", "arm", "abstained", "carb_error_g"}
    if not need <= set(df.columns):
        raise ValueError(f"{path} needs columns {sorted(need)}")
    keep = df[(df["split"] == "test") & (df["arm"] == "embed") & (df["abstained"] == 0)]
    vals = keep["carb_error_g"].dropna().to_numpy(dtype=float)
    if vals.size == 0:
        raise ValueError(f"{path} has no test/embed/non-abstained carb errors")
    return vals


def apply_carb_errors(meals: pd.DataFrame, pool: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Sensitivity only: each test meal's carbs shifted by an error resampled from `pool`.

    Not a per-meal measurement: CGMacros meals cannot be linked to dish-ID images.
    """
    err = rng.choice(pool, size=len(meals), replace=True)
    out = meals.copy()
    out["carbs_g"] = np.clip(meals["carbs_g"].to_numpy(dtype=float) + err, 0.0, None)
    return out


# ---------------------------------------------------------------------------
# Noise models, rows, features, model
# ---------------------------------------------------------------------------

def iso15197_sd(g: np.ndarray) -> np.ndarray:
    """Per-reading SD (mg/dL): 95% of readings within the ISO 15197:2013 limit."""
    return np.where(g < 100.0, ISO_ABS_MGDL / ISO_Z, ISO_REL * g / ISO_Z)


def iso_meter_reading(g0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.clip(g0 + rng.normal(0.0, iso15197_sd(g0)), *METER_CLIP)


def expand(meals: pd.DataFrame, calib: dict | None, rng: np.random.Generator,
           n_meter: int, n_ppg: int) -> pd.DataFrame:
    """CGM row (true Dexcom g0) plus noise-augmented meter rows and calibrated PPG rows per meal.

    PPG rows: g0_ppg = a + b * g0 + N(0, sigma), the served-PPG map fitted on the PPG study.
    """
    g0 = meals["g0_mgdl"].to_numpy(dtype=float)
    parts = [meals.assign(g0_input=g0, g0_source="CGM", draw=0)]
    for d in range(n_meter):
        parts.append(meals.assign(g0_input=iso_meter_reading(g0, rng), g0_source="meter", draw=d))
    if calib is not None:
        for d in range(n_ppg):
            mean = calib["a"] + calib["b"] * g0
            g_ppg = np.clip(mean + rng.normal(0.0, calib["sigma"], size=len(g0)), *METER_CLIP)
            parts.append(meals.assign(g0_input=g_ppg, g0_source="ppg", draw=d))
    return pd.concat(parts, ignore_index=True)


def design(rows: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=rows.index)
    X["g0_mgdl"] = rows["g0_input"].to_numpy(dtype=float)
    X["src_meter"] = (rows["g0_source"] == "meter").astype(float).to_numpy()
    X["src_ppg"] = (rows["g0_source"] == "ppg").astype(float).to_numpy()
    X["hour_sin"] = rows["hour_sin"].to_numpy(dtype=float)
    X["hour_cos"] = rows["hour_cos"].to_numpy(dtype=float)
    for c in MACRO + PROFILE:
        X[c] = rows[c].to_numpy(dtype=float)
    return X


def fit_rows(rows: pd.DataFrame, horizon: int, cols: list[str], p: dict) -> lgb.LGBMRegressor:
    X = design(rows)[cols]
    y = rows[f"y{horizon}"].to_numpy(dtype=float)
    g0 = rows["g0_input"].to_numpy(dtype=float)
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=p["n_estimators"],
        learning_rate=LEARNING_RATE,
        num_leaves=p["num_leaves"],
        min_child_samples=p["min_child_samples"],
        random_state=RANDOM_STATE,
        deterministic=True,
        force_row_wise=True,
        n_jobs=4,
        verbose=-1,
    )
    model.fit(X, y - g0 if p["offset"] else y)
    return model


def predict_rows(model, rows: pd.DataFrame, cols: list[str], p: dict) -> np.ndarray:
    X = design(rows)[cols]
    out = model.predict(X)
    return out + rows["g0_input"].to_numpy(dtype=float) if p["offset"] else out


# ---------------------------------------------------------------------------
# Selection (inner loop, training participants only) and outer evaluation
# ---------------------------------------------------------------------------

def inner_select(meals: pd.DataFrame, horizon: int, ppg_pool, seed: int):
    """Choose (feature set, GBDT config) by grouped inner CV over the given participants.

    Scores the mean absolute error over CGM and noise-augmented rows of each inner
    validation fold. Only `meals` (training participants of the caller) are used.
    """
    y_col = f"y{horizon}"
    keys = [(name, i) for name in FEATURE_SETS for i in range(len(GRID))]
    errs = {k: [] for k in keys}
    gkf = GroupKFold(n_splits=INNER_SPLITS)
    for f, (tr, va) in enumerate(gkf.split(meals, groups=meals["subject"].to_numpy())):
        tr_m, va_m = meals.iloc[tr], meals.iloc[va]
        ppg = ppg_pool
        rng = np.random.default_rng(seed * 100 + f)
        rows_tr = expand(tr_m, ppg, rng, N_METER_DRAWS, N_METER_DRAWS)
        rows_va = expand(va_m, ppg, rng, N_METER_DRAWS, N_METER_DRAWS)
        y_va = rows_va[y_col].to_numpy(dtype=float)
        for name, i in keys:
            cols, p = FEATURE_SETS[name], GRID[i]
            model = fit_rows(rows_tr, horizon, cols, p)
            errs[(name, i)].append(np.abs(predict_rows(model, rows_va, cols, p) - y_va))
    scores = {f"{name}|{i}": float(np.concatenate(errs[(name, i)]).mean()) for name, i in keys}
    best_name, best_i = min(keys, key=lambda k: scores[f"{k[0]}|{k[1]}"])
    return best_name, GRID[best_i], scores


def outer_folds(subjects: np.ndarray, repeat: int) -> list[np.ndarray]:
    """Participant-level folds for one repetition (participants shuffled per repetition)."""
    rng = np.random.default_rng(RANDOM_STATE + 1000 * repeat)
    perm = rng.permutation(np.sort(np.unique(subjects)))
    return np.array_split(perm, N_SPLITS)


def fit_predict_fold(tr_m: pd.DataFrame, te_m: pd.DataFrame, horizon: int, ppg_pool,
                     name: str, p: dict, seed: int, n_eval: int, carb_err=None) -> pd.DataFrame:
    """Fit on training participants; predict test meals; B0/B1/B2 on the same g0 inputs."""
    y_col = f"y{horizon}"
    cols = FEATURE_SETS[name]
    ppg = ppg_pool
    rows_tr = expand(tr_m, ppg, np.random.default_rng(seed + 5000), N_METER_DRAWS, N_METER_DRAWS)
    model = fit_rows(rows_tr, horizon, cols, p)
    b0 = float(tr_m[y_col].mean())
    excursion = float((tr_m[y_col] - tr_m["g0_mgdl"]).mean())

    rows = expand(te_m, ppg, np.random.default_rng(seed + 9000), n_eval, n_eval)
    out = rows.assign(
        y=rows[y_col].to_numpy(dtype=float),
        gbdt=predict_rows(model, rows, cols, p),
        b0=b0,
        b1=rows["g0_input"].to_numpy(dtype=float),
        b2=rows["g0_input"].to_numpy(dtype=float) + excursion,
    )
    if carb_err is not None:
        pert = apply_carb_errors(te_m, carb_err, np.random.default_rng(seed + 9100))
        rows_c = expand(pert, ppg, np.random.default_rng(seed + 9000), n_eval, n_eval)
        out["gbdt_carb"] = predict_rows(model, rows_c, cols, p)
    return out


def nested_cv(cohort: pd.DataFrame, horizon: int, ppg_pool, carb_err=None):
    """N_REPEATS x N_SPLITS participant-grouped outer folds; selection inside each training set."""
    subjects = cohort["subject"].to_numpy()
    frames, selections = [], []
    for r in range(N_REPEATS):
        for k, te_subs in enumerate(outer_folds(subjects, r)):
            te_mask = np.isin(subjects, te_subs)
            tr_m, te_m = cohort[~te_mask], cohort[te_mask]
            assert not (set(tr_m["subject"]) & set(te_m["subject"])), "participant leak"
            fold = r * N_SPLITS + k + 1
            seed = RANDOM_STATE + 100 * r + k
            name, p, scores = inner_select(tr_m, horizon, ppg_pool, seed)
            frame = fit_predict_fold(tr_m, te_m, horizon, ppg_pool, name, p, seed,
                                     N_EVAL_DRAWS, carb_err)
            frames.append(frame.assign(repeat=r + 1, fold=fold))
            selections.append({
                "repeat": r + 1,
                "fold": fold,
                "seed": seed,
                "test_subjects": sorted(int(s) for s in te_subs),
                "n_train_participants": int(tr_m["subject"].nunique()),
                "selected_set": name,
                "params": p,
                "inner_mae_scores": scores,
            })
    res = pd.concat(frames, ignore_index=True)
    return res, selections


def permute_macros(meals: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle meal macro vectors among each participant's own meals (within-participant null)."""
    vals = meals[MACRO].to_numpy(dtype=float)
    new = vals.copy()
    subj = meals["subject"].to_numpy()
    for s in np.unique(subj):
        idx = np.flatnonzero(subj == s)
        perm = rng.permutation(len(idx))
        new[idx] = vals[idx][perm]
    out = meals.copy()
    out[MACRO] = new
    return out


def permutation_p(cohort: pd.DataFrame, horizon: int, ppg_pool, selections: list,
                  observed: float, n_perm: int) -> tuple[float, np.ndarray]:
    """Macro-permutation null. Selected configuration per outer fold is held fixed."""
    subjects = cohort["subject"].to_numpy()

    def pooled_gain(data: pd.DataFrame) -> float:
        frames = []
        for s in selections:
            te_mask = np.isin(data["subject"].to_numpy(), s["test_subjects"])
            frames.append(fit_predict_fold(data[~te_mask], data[te_mask], horizon, ppg_pool,
                                           s["selected_set"], s["params"], s["seed"], 0))
        f = pd.concat(frames, ignore_index=True)
        return _mae(f["y"], f["b2"]) - _mae(f["y"], f["gbdt"])

    identity = pooled_gain(cohort)
    if abs(identity - observed) > 1e-9:
        raise RuntimeError(f"identity check failed: {identity} vs observed {observed}")
    assert subjects.size == cohort.shape[0]
    gains = np.empty(n_perm)
    for i in range(n_perm):
        rng = np.random.default_rng(RANDOM_STATE + 50_000 + i)
        gains[i] = pooled_gain(permute_macros(cohort, rng))
    p = (int(np.sum(gains >= observed)) + 1) / (n_perm + 1)
    return p, gains


def fit_final(cohort: pd.DataFrame, horizon: int, ppg_pool, models_dir: Path) -> dict:
    """Select on all participants (grouped inner CV), refit, save the model and metadata."""
    name, p, scores = inner_select(cohort, horizon, ppg_pool, seed=RANDOM_STATE + horizon)
    cols = FEATURE_SETS[name]
    ppg = ppg_pool
    rows = expand(cohort, ppg, np.random.default_rng(RANDOM_STATE + horizon + 7000),
                  N_METER_DRAWS, N_METER_DRAWS)
    model = fit_rows(rows, horizon, cols, p)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"gbdt_cgmacros_T{horizon}.txt"
    model.booster_.save_model(str(model_path))
    meta = {
        "horizon_min": horizon,
        "selected_feature_set": name,
        "feature_columns": cols,
        "parameterisation": "offset (target = input - g0)" if p["offset"] else "direct (target = Dexcom)",
        "params": p,
        "learning_rate": LEARNING_RATE,
        "random_state": RANDOM_STATE,
        "n_meals": int(len(cohort)),
        "n_participants": int(cohort["subject"].nunique()),
        "training_rows_incl_augmentation": int(len(rows)),
        "g0_definition": "last Dexcom GL reading strictly before the meal timestamp (C2)",
        "g0_sources": ["CGM", "meter"] + (["ppg"] if ppg is not None else []),
        "meter_noise_model": {
            "standard": "ISO 15197:2013 accuracy limit",
            "sd_below_100_mgdl": ISO_ABS_MGDL / ISO_Z,
            "sd_relative_at_or_above_100_mgdl": ISO_REL / ISO_Z,
            "clip_mgdl": list(METER_CLIP),
        },
        "inner_mae_mgdl": float(scores[f"{name}|{GRID.index(p)}"]),
        "target": "exact Dexcom GL reading at meal time + horizon",
        "dataset": "CGMacros full release (45 participant folders)",
    }
    meta_path = models_dir / f"gbdt_cgmacros_T{horizon}.json"
    meta_path.write_text(json.dumps(_clean(meta), indent=2))
    return {"model": str(model_path), "meta": str(meta_path), "selected_feature_set": name,
            "params": p}


# ---------------------------------------------------------------------------
# Metrics and gate
# ---------------------------------------------------------------------------

def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def _rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


MODELS = ("gbdt", "b0", "b1", "b2")


def metric_block(df: pd.DataFrame) -> dict:
    return {m: {"mae": _mae(df["y"], df[m]), "rmse": _rmse(df["y"], df[m])} for m in MODELS}


def subject_bootstrap_improvement(rows: pd.DataFrame, base: str, rng: np.random.Generator):
    """Participant-cluster bootstrap of MAE(base) - MAE(gbdt) over all rows of each participant."""
    subs = np.sort(rows["subject"].unique())
    e_g = (rows["y"] - rows["gbdt"]).abs()
    e_b = (rows["y"] - rows[base]).abs()
    grp = rows["subject"]
    sg = e_g.groupby(grp).sum().reindex(subs).to_numpy()
    sb = e_b.groupby(grp).sum().reindex(subs).to_numpy()
    nn = grp.groupby(grp).size().reindex(subs).to_numpy().astype(float)
    idx = rng.integers(0, len(subs), size=(N_BOOTSTRAP, len(subs)))
    n_tot = nn[idx].sum(axis=1)
    diffs = sb[idx].sum(axis=1) / n_tot - sg[idx].sum(axis=1) / n_tot
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def fold_wins(rows: pd.DataFrame, base: str):
    """Per (repeat, fold) win flag of GBDT vs base on mean absolute error."""
    d = rows.assign(eg=(rows["y"] - rows["gbdt"]).abs(), eb=(rows["y"] - rows[base]).abs())
    m = d.groupby(["repeat", "fold"])[["eg", "eb"]].mean()
    win = m["eg"] < m["eb"]
    return win, win.groupby(level="repeat").sum()


def gate_vs(rows: pd.DataFrame, base: str, rng: np.random.Generator, perm_p: float | None):
    imp = _mae(rows["y"], rows[base]) - _mae(rows["y"], rows["gbdt"])
    win, by_repeat = fold_wins(rows, base)
    min_wins = int(by_repeat.min())
    lo, hi = subject_bootstrap_improvement(rows, base, rng)
    checks = {
        "positive_gain": imp > 0,
        "fold_wins_every_repeat": min_wins >= MIN_FOLD_WINS,
        "bootstrap_ci_excludes_zero": lo > 0,
    }
    if perm_p is not None:
        checks["permutation_p"] = perm_p <= MAX_PERMUTATION_P
    reasons = [
        f"{'pass' if checks['positive_gain'] else 'FAIL'}: MAE gain {imp:.2f} mg/dL vs {base}",
        f"{'pass' if checks['fold_wins_every_repeat'] else 'FAIL'}: GBDT beat {base} in "
        f"{by_repeat.tolist()} folds per repetition (min {MIN_FOLD_WINS} each)",
        f"{'pass' if checks['bootstrap_ci_excludes_zero'] else 'FAIL'}: participant bootstrap 95% CI [{lo:.2f}, {hi:.2f}]",
    ]
    if perm_p is not None:
        reasons.append(
            f"{'pass' if checks['permutation_p'] else 'FAIL'}: within-participant macro permutation p = {perm_p:.3f}"
        )
    return {
        "improvement_mgdl": imp,
        "fold_wins_by_repeat": [int(x) for x in by_repeat.tolist()],
        "fold_wins_total": int(win.sum()),
        "n_folds": int(win.size),
        "bootstrap_ci_95": [lo, hi],
        "permutation_p": perm_p,
        "checks": checks,
        "passes": bool(all(checks.values())),
        "reasons": reasons,
    }


def _per_participant_or_none(y, p, subj, min_meals: int):
    """None when no participant reaches min_meals (helper raises in that case)."""
    try:
        return per_participant_zone_a(y, p, subj, min_meals=min_meals)
    except ValueError:
        return None


def clarke_block(df: pd.DataFrame) -> dict:
    out = {}
    y = df["y"].to_numpy(dtype=float)
    subj = df["subject"].to_numpy()
    for m in MODELS:
        p = df[m].to_numpy(dtype=float)
        zones, pct = clarke_error_grid(y, p)
        pp = _per_participant_or_none(y, p, subj, min_meals=5)
        by_range = zone_a_by_reference_range(y, p, ranges=RANGES)
        per_range = {}
        for lo, hi in RANGES:
            mask = (y >= lo) & (y < hi)
            per_range[f"{lo}-{hi}"] = _per_participant_or_none(y[mask], p[mask], subj[mask], min_meals=3)
        out[m] = {
            "zones_count": zones,
            "zones_pct": pct,
            "per_participant_zone_a": pp,
            "zone_a_by_reference_range": by_range,
            "per_participant_zone_a_by_range_min3": per_range,
        }
    return out


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_horizon(meals: pd.DataFrame, horizon: int, ppg_pool, n_perm: int, carb_err):
    cohort = meals[meals[f"y{horizon}"].notna() & meals["g0_mgdl"].notna()].reset_index(drop=True)
    res, selections = nested_cv(cohort, horizon, ppg_pool, carb_err)
    cgm = res[res["g0_source"] == "CGM"].reset_index(drop=True)
    meter = res[res["g0_source"] == "meter"].reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_STATE + horizon)
    observed = _mae(cgm["y"], cgm["b2"]) - _mae(cgm["y"], cgm["gbdt"])
    perm_p, null_gains = permutation_p(cohort, horizon, ppg_pool, selections, observed, n_perm)

    gates = {
        "vs_b2": gate_vs(cgm, "b2", rng, perm_p),
        "vs_b1": gate_vs(cgm, "b1", rng, None),
        "vs_b0": gate_vs(cgm, "b0", rng, None),
        "meter_vs_b2": gate_vs(meter, "b2", rng, None),
    }
    if ppg_pool is not None:
        gates["ppg_vs_b2"] = gate_vs(res[res["g0_source"] == "ppg"], "b2", rng, None)
    t2d = cgm[cgm["dm_band"] == 2.0]
    set_counts = Counter(s["selected_set"] for s in selections)
    param_counts = Counter(json.dumps(s["params"], sort_keys=True) for s in selections)

    carb_block = None
    if carb_err is not None:
        carb_cgm = cgm.assign(gbdt=cgm["gbdt_carb"])
        lo, hi = subject_bootstrap_improvement(carb_cgm, "b2", rng)
        carb_block = {
            "n_test_meals_perturbed_per_repeat": int((res["g0_source"] == "CGM").sum() // N_REPEATS),
            "mae_gbdt_carb": _mae(cgm["y"], cgm["gbdt_carb"]),
            "mae_gbdt_clean": _mae(cgm["y"], cgm["gbdt"]),
            "mae_b2": _mae(cgm["y"], cgm["b2"]),
            "gain_vs_b2_mgdl": _mae(cgm["y"], cgm["b2"]) - _mae(cgm["y"], cgm["gbdt_carb"]),
            "bootstrap_ci_95": [lo, hi],
        }

    result = {
        "horizon_min": horizon,
        "n_meals_cohort": int(len(cohort)),
        "n_participants_cohort": int(cohort["subject"].nunique()),
        "n_participants_eval_cgm": int(cgm["subject"].nunique()),
        "n_t2d_participants": int(t2d["subject"].nunique()),
        "n_repeats": N_REPEATS,
        "n_outer_folds": N_REPEATS * N_SPLITS,
        "outer_fold_test_sizes": {
            int(f): int(g["subject"].nunique()) for f, g in cgm.groupby("fold")
        },
        "selection_frequency": dict(set_counts),
        "selection_params_frequency": dict(param_counts),
        "selections": selections,
        "metrics_by_source": {src: metric_block(res[res["g0_source"] == src])
                              for src in res["g0_source"].unique()},
        "metrics_cgm": metric_block(cgm),
        "metrics_t2d_cgm": metric_block(t2d) if len(t2d) else None,
        "clarke_cgm": clarke_block(cgm),
        "clarke_t2d_cgm": clarke_block(t2d) if len(t2d) else None,
        "clarke_meter": clarke_block(meter),
        "gate": gates,
        "gate_passes_vs_b2": gates["vs_b2"]["passes"],
        "permutation": {
            "n": int(n_perm),
            "observed_gain_mgdl": observed,
            "null_gain_mean": float(null_gains.mean()) if n_perm else None,
            "null_gain_p95": float(np.percentile(null_gains, 95)) if n_perm else None,
            "p": perm_p,
        },
        "carb_error_sensitivity": carb_block,
    }
    return result, res


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return o


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(_clean(obj), indent=2, default=str))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--ppg-residuals", type=Path, default=None)
    ap.add_argument("--carb-error-pool", type=Path, default=None)
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    meals, cleaning, _profile = build_dataset(args.data_root)
    ppg_calib = load_ppg_calibration(args.ppg_residuals)
    carb_df = load_carb_error_pool(args.carb_error_pool)
    _write(args.out / "cleaning.json", cleaning)

    for h in args.horizons:
        result, res = run_horizon(meals, h, ppg_calib, args.permutations, carb_df)
        if result["gate_passes_vs_b2"] and args.models_dir is not None:
            cohort = meals[meals[f"y{h}"].notna() & meals["g0_mgdl"].notna()].reset_index(drop=True)
            result["final_model"] = fit_final(cohort, h, ppg_calib, args.models_dir)
        _write(args.out / f"T{h}.json", result)
        res.to_csv(args.out / f"_predictions_T{h}.csv.gz", index=False, compression="gzip")

    summary = {}
    for f in sorted(args.out.glob("T*.json")):
        summary[f.stem] = json.loads(f.read_text())
    summary["cleaning"] = cleaning
    summary["noise_model"] = {
        "meter": "ISO 15197:2013 limit as zero-mean normal; sd_below_100=15/1.96, "
                 "sd_at_or_above_100=0.15*g/1.96; clip [20, 600] mg/dL",
    }
    summary["ppg_calibration"] = ("[NOT RUN] no PPG residuals file supplied" if ppg_calib is None
                                  else ppg_calib)
    summary["carb_errors"] = ("[NOT RUN] no carb-error pool supplied" if carb_df is None
                              else f"sensitivity: {len(carb_df)} dish-ID test errors resampled per test meal")
    _write(args.out / "summary.json", summary)
    print(json.dumps(_clean({k: {
        "n_meals_cohort": v.get("n_meals_cohort"),
        "gate_passes_vs_b2": v.get("gate_passes_vs_b2"),
        "gate_vs_b2": v.get("gate", {}).get("vs_b2", {}).get("reasons"),
        "selection_frequency": v.get("selection_frequency"),
        "permutation": v.get("permutation"),
    } for k, v in summary.items() if k.startswith("T")}), indent=2, default=str))


if __name__ == "__main__":
    main()
