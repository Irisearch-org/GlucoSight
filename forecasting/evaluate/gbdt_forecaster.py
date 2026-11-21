"""GBDT CGM forecaster trial on the full CGMacros release (wayfinder #17).

Two LightGBM regressors, one per horizon (T+60, T+120). Target: the exact Dexcom
reading at meal time + horizon (no interpolation). Input g0: the last Dexcom
reading strictly before the meal (C2). g0_source in {CGM, meter, ppg} is produced
by noise augmentation: meter noise is resampled from Contour-vs-Dexcom
differences in bio.csv; ppg noise is resampled from the served PPG model's
held-out residuals when a residuals CSV is supplied.

Evaluation: participant-grouped folds (GroupKFold over subjects), B0/B1/B2 on the
same folds and the same g0 inputs, ADR 0004 gate (fold wins, participant-cluster
bootstrap CI, within-participant macro permutation), Clarke Error Grid from
utils/clarke_grid.py, feature ablation, and final models fit on all participants.

Usage (from repo root):
    python -m forecasting.evaluate.gbdt_forecaster \
        --data-root /Users/auliamnaufal/Downloads/CGMacros_full/CGMacros \
        --out forecasting/gbdt_results \
        [--ppg-residuals PATH.csv] [--models-dir DIR] [--permutations 1000]
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
N_SPLITS = 5            # matches rppg/models/glucose_estimator.py
INNER_SPLITS = 4
RANDOM_STATE = 42
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 200   # reduced from 1000 for runtime; p resolution 1/201
MIN_FOLD_WINS = 4       # of N_SPLITS, matches glucose_estimator.py
MAX_PERMUTATION_P = 0.05
ATWATER_TOL = 0.30      # #12 primary rule
SENSITIVITY_TOLS = (0.20, 0.30, 0.50)
N_METER_DRAWS = 3       # training noise draws per meal per noisy source
N_EVAL_DRAWS = 5        # evaluation noise draws per test meal per noisy source
LEARNING_RATE = 0.05
RANGES = ((0, 100), (100, 140), (140, 180), (180, 250), (250, 10000))
MEAL_TYPES = {"Breakfast", "Lunch", "Dinner", "Snacks"}
CONTOUR_PAIRS = (
    ("#1 Contour Fingerstick GLU", "Time (t)"),
    ("#2 Contour Fingerstick GLU", "Time (t).1"),
    ("#3 Contour Fingerstick GLU", "Time (t).2"),
)

BASE = ["g0_mgdl", "src_meter", "src_ppg", "hour_sin", "hour_cos"]
MACRO = ["carbs_g", "protein_g", "fat_g", "fibre_g"]
PROFILE = ["age_y", "female", "bmi", "dm_band"]
FEATURE_SETS = {
    "g0_hour": BASE,
    "g0_hour_macros": BASE + MACRO,
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

    meal_rows, contour_rows = [], []
    dex_by_subject: dict[int, pd.Series] = {}
    unparsed_contour_times = 0

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
        dex_by_subject[pid] = dex

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

        # Contour readings carry no date; pair each with Dexcom at the same clock
        # time on every day of this participant's record (see report, section 2).
        days = pd.DatetimeIndex(dex.index.normalize().unique())
        for reading, (vc, tc) in enumerate(CONTOUR_PAIRS, start=1):
            if pid not in bio.index:
                continue
            val = pd.to_numeric(pd.Series([bio.at[pid, vc]]), errors="coerce").iloc[0]
            tt = bio.at[pid, tc]
            if pd.isna(val) or pd.isna(tt):
                continue
            try:
                clock = pd.to_datetime(str(tt).strip(), format="%H:%M")
            except (ValueError, TypeError):
                unparsed_contour_times += 1
                continue
            offset = pd.Timedelta(hours=clock.hour, minutes=clock.minute)
            pairs = []
            for day in days:
                ts = day + offset
                if ts in dex.index:
                    pairs.append((ts, float(val) - float(dex.loc[ts])))
            best = min(pairs, key=lambda z: abs(z[1])) if pairs else None
            for ts, res in pairs:
                contour_rows.append({
                    "subject": pid,
                    "reading": reading,
                    "clock": str(tt).strip(),
                    "day": ts.normalize(),
                    "contour_mgdl": float(val),
                    "dexcom_mgdl": float(dex.loc[ts]),
                    "residual_mgdl": res,
                    "is_min_abs_day": best is not None and ts == best[0],
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

    contour = pd.DataFrame(contour_rows)
    cleaning["contour_unparsed_times"] = unparsed_contour_times
    cleaning["n_participants_with_meals"] = int(meals["subject"].nunique())
    return meals, cleaning, contour, profile


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


def load_ppg_residuals(path: Path | None) -> pd.DataFrame | None:
    """Held-out residuals of the served PPG model: estimate - reference (mg/dL)."""
    if path is None:
        return None
    df = pd.read_csv(path)
    if "residual_mgdl" not in df.columns:
        raise ValueError(f"{path} needs a residual_mgdl column (PPG estimate - reference)")
    df = df.dropna(subset=["residual_mgdl"]).copy()
    df.attrs["global"] = "subject" not in df.columns
    return df


def pool_for(pool: pd.DataFrame | None, subjects) -> np.ndarray | None:
    """Residuals usable for a set of training participants (C4: never test participants)."""
    if pool is None:
        return None
    if pool.attrs.get("global"):
        vals = pool["residual_mgdl"].to_numpy(dtype=float)
    else:
        vals = pool.loc[pool["subject"].isin(list(subjects)), "residual_mgdl"].to_numpy(dtype=float)
    return vals if vals.size else None


def meter_pool_from_contour(contour: pd.DataFrame) -> pd.DataFrame:
    """All-day Contour-minus-Dexcom residuals (date of each Contour reading is not in the release)."""
    return pd.DataFrame({
        "subject": contour["subject"].to_numpy(),
        "residual_mgdl": contour["residual_mgdl"].to_numpy(dtype=float),
    })


# ---------------------------------------------------------------------------
# Rows, features, model
# ---------------------------------------------------------------------------

def expand(meals: pd.DataFrame, meter: np.ndarray | None, ppg: np.ndarray | None,
           rng: np.random.Generator, n_meter: int, n_ppg: int) -> pd.DataFrame:
    """CGM row (true Dexcom g0) plus noise-augmented meter / ppg rows per meal."""
    parts = [meals.assign(g0_input=meals["g0_mgdl"], g0_source="CGM", draw=0)]
    for src, pool, k in (("meter", meter, n_meter), ("ppg", ppg, n_ppg)):
        if pool is None or k == 0:
            continue
        for d in range(k):
            noise = rng.choice(pool, size=len(meals), replace=True)
            parts.append(meals.assign(
                g0_input=meals["g0_mgdl"].to_numpy() + noise, g0_source=src, draw=d,
            ))
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


def tune(meals: pd.DataFrame, horizon: int, cols: list[str], meter_pool, ppg_pool, seed: int):
    """Grid search with inner GroupKFold over participants; scores mean MAE over all source rows."""
    gkf = GroupKFold(n_splits=INNER_SPLITS)
    subjects = meals["subject"].to_numpy()
    errs = {i: [] for i in range(len(GRID))}
    for k, (tr, va) in enumerate(gkf.split(meals, groups=subjects)):
        tr_m, va_m = meals.iloc[tr], meals.iloc[va]
        subs = set(tr_m["subject"])
        mp, pp = pool_for(meter_pool, subs), pool_for(ppg_pool, subs)
        rng = np.random.default_rng(seed * 100 + k)
        rows_tr = expand(tr_m, mp, pp, rng, N_METER_DRAWS, N_METER_DRAWS)
        rows_va = expand(va_m, mp, pp, rng, N_METER_DRAWS, N_METER_DRAWS)
        y_va = rows_va[f"y{horizon}"].to_numpy(dtype=float)
        for i, p in enumerate(GRID):
            model = fit_rows(rows_tr, horizon, cols, p)
            errs[i].append(np.abs(predict_rows(model, rows_va, cols, p) - y_va))
    scores = [float(np.concatenate(errs[i]).mean()) for i in range(len(GRID))]
    best = int(np.argmin(scores))
    return GRID[best], scores


def cross_validate(meals: pd.DataFrame, horizon: int, cols: list[str], meter_pool, ppg_pool,
                   fixed: dict | None = None, n_meter_eval: int = N_EVAL_DRAWS,
                   n_ppg_eval: int = N_EVAL_DRAWS, seed: int = RANDOM_STATE):
    """GroupKFold(N_SPLITS) by participant. B0/B1/B2 use the same folds and same g0 inputs."""
    y_col = f"y{horizon}"
    gkf = GroupKFold(n_splits=N_SPLITS)
    out, params_by_fold = [], {}
    for k, (tr, te) in enumerate(gkf.split(meals, groups=meals["subject"].to_numpy()), start=1):
        tr_m, te_m = meals.iloc[tr], meals.iloc[te]
        subs = set(tr_m["subject"])
        assert not (subs & set(te_m["subject"])), "participant leak across folds"
        mp, pp = pool_for(meter_pool, subs), pool_for(ppg_pool, subs)
        rng = np.random.default_rng(seed + k)

        if fixed is None:
            p, _ = tune(tr_m, horizon, cols, meter_pool, ppg_pool, seed + k)
        else:
            p = fixed[k]
        params_by_fold[k] = p

        rows_tr = expand(tr_m, mp, pp, rng, N_METER_DRAWS, N_METER_DRAWS)
        model = fit_rows(rows_tr, horizon, cols, p)

        b0 = float(tr_m[y_col].mean())
        excursion = float((tr_m[y_col] - tr_m["g0_mgdl"]).mean())
        rows_te = expand(te_m, mp, pp, rng, n_meter_eval, n_ppg_eval)
        rows_te = rows_te.assign(
            y=rows_te[y_col].to_numpy(dtype=float),
            gbdt=predict_rows(model, rows_te, cols, p),
            b0=b0,
            b1=rows_te["g0_input"].to_numpy(dtype=float),
            b2=rows_te["g0_input"].to_numpy(dtype=float) + excursion,
            fold=k,
        )
        out.append(rows_te)
    res = pd.concat(out, ignore_index=True)
    return res, params_by_fold


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


def subject_bootstrap_improvement(cgm: pd.DataFrame, base: str, rng: np.random.Generator):
    """Participant-cluster bootstrap of MAE(base) - MAE(gbdt), CGM rows."""
    subs = np.sort(cgm["subject"].unique())
    e_g = (cgm["y"] - cgm["gbdt"]).abs()
    e_b = (cgm["y"] - cgm[base]).abs()
    grp = cgm["subject"]
    sg = e_g.groupby(grp).sum().reindex(subs).to_numpy()
    sb = e_b.groupby(grp).sum().reindex(subs).to_numpy()
    nn = grp.groupby(grp).size().reindex(subs).to_numpy().astype(float)
    idx = rng.integers(0, len(subs), size=(N_BOOTSTRAP, len(subs)))
    n_tot = nn[idx].sum(axis=1)
    diffs = sb[idx].sum(axis=1) / n_tot - sg[idx].sum(axis=1) / n_tot
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def gate_vs(cgm: pd.DataFrame, base: str, rng: np.random.Generator, perm_p: float | None):
    y = cgm["y"].to_numpy()
    imp = _mae(y, cgm[base]) - _mae(y, cgm["gbdt"])
    wins = []
    for _, g in cgm.groupby("fold"):
        wins.append(bool(_mae(g["y"], g["gbdt"]) < _mae(g["y"], g[base])))
    n_wins = int(sum(wins))
    lo, hi = subject_bootstrap_improvement(cgm, base, rng)
    checks = {
        "positive_gain": imp > 0,
        "fold_wins": n_wins >= MIN_FOLD_WINS,
        "bootstrap_ci_excludes_zero": lo > 0,
    }
    if perm_p is not None:
        checks["permutation_p"] = perm_p <= MAX_PERMUTATION_P
    reasons = [
        f"{'pass' if checks['positive_gain'] else 'FAIL'}: MAE gain {imp:.2f} mg/dL vs {base}",
        f"{'pass' if checks['fold_wins'] else 'FAIL'}: GBDT beat {base} in {n_wins}/{len(wins)} folds (need {MIN_FOLD_WINS})",
        f"{'pass' if checks['bootstrap_ci_excludes_zero'] else 'FAIL'}: participant bootstrap 95% CI [{lo:.2f}, {hi:.2f}]",
    ]
    if perm_p is not None:
        reasons.append(
            f"{'pass' if checks['permutation_p'] else 'FAIL'}: within-participant macro permutation p = {perm_p:.3f}"
        )
    return {
        "improvement_mgdl": imp,
        "fold_wins": n_wins,
        "fold_win_flags": wins,
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
            key = f"{lo}-{hi}"
            per_range[key] = _per_participant_or_none(y[mask], p[mask], subj[mask], min_meals=3)
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

def permutation_p(cohort: pd.DataFrame, horizon: int, fixed: dict, meter_pool, ppg_pool,
                  observed_gain: float, n_perm: int, cols: list | None = None) -> tuple[float, np.ndarray]:
    """Shuffle macros within participant, keep fold-tuned params, compare gain vs B2 (CGM rows)."""
    gains = np.empty(n_perm)
    for i in range(n_perm):
        rng = np.random.default_rng(RANDOM_STATE + 10_000 + i)
        perm_meals = permute_macros(cohort, rng)
        res, _ = cross_validate(
            perm_meals, horizon, cols or FEATURE_SETS["full"], meter_pool, ppg_pool,
            fixed=fixed, n_meter_eval=0, n_ppg_eval=0,
        )
        gains[i] = _mae(res["y"], res["b2"]) - _mae(res["y"], res["gbdt"])
    p = (int(np.sum(gains >= observed_gain)) + 1) / (n_perm + 1)
    return p, gains


def run_horizon(meals: pd.DataFrame, horizon: int, meter_pool, ppg_pool, n_perm: int) -> tuple[dict, pd.DataFrame]:
    cohort = meals[meals[f"y{horizon}"].notna() & meals["g0_mgdl"].notna()].reset_index(drop=True)
    cols = FEATURE_SETS["full"]
    res, fixed = cross_validate(cohort, horizon, cols, meter_pool, ppg_pool)
    cgm = res[res["g0_source"] == "CGM"].reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_STATE + horizon)
    observed = _mae(cgm["y"], cgm["b2"]) - _mae(cgm["y"], cgm["gbdt"])
    perm_p, null_gains = permutation_p(cohort, horizon, fixed, meter_pool, ppg_pool, observed, n_perm)

    gates = {
        "vs_b2": gate_vs(cgm, "b2", rng, perm_p),
        "vs_b1": gate_vs(cgm, "b1", rng, None),
        "vs_b0": gate_vs(cgm, "b0", rng, None),
    }
    t2d = cgm[cgm["dm_band"] == 2.0]
    result = {
        "horizon_min": horizon,
        "n_meals_cohort": int(len(cohort)),
        "n_participants_cohort": int(cohort["subject"].nunique()),
        "n_participants_eval_cgm": int(cgm["subject"].nunique()),
        "n_t2d_participants": int(t2d["subject"].nunique()),
        "n_folds": N_SPLITS,
        "fold_sizes_participants": {
            int(k): int(g["subject"].nunique()) for k, g in cgm.groupby("fold")
        },
        "fold_params": fixed,
        "metrics_by_source": {src: metric_block(res[res["g0_source"] == src])
                              for src in res["g0_source"].unique()},
        "metrics_cgm": metric_block(cgm),
        "metrics_t2d_cgm": metric_block(t2d) if len(t2d) else None,
        "clarke_cgm": clarke_block(cgm),
        "clarke_t2d_cgm": clarke_block(t2d) if len(t2d) else None,
        "gate": gates,
        "gate_passes_vs_b2": gates["vs_b2"]["passes"],
        "permutation": {
            "n": int(n_perm),
            "observed_gain_mgdl": observed,
            "null_gain_mean": float(null_gains.mean()),
            "null_gain_p95": float(np.percentile(null_gains, 95)),
            "p": perm_p,
        },
    }
    return result, res


def run_ablation(cohort_meals: pd.DataFrame, horizon: int, meter_pool, ppg_pool) -> dict:
    out = {}
    for name, cols in FEATURE_SETS.items():
        cohort = cohort_meals[cohort_meals[f"y{horizon}"].notna() & cohort_meals["g0_mgdl"].notna()]
        cohort = cohort.reset_index(drop=True)
        res, params = cross_validate(cohort, horizon, cols, meter_pool, ppg_pool)
        cgm = res[res["g0_source"] == "CGM"].reset_index(drop=True)
        rng = np.random.default_rng(RANDOM_STATE + horizon)
        gate_b2 = gate_vs(cgm, "b2", rng, None)
        clarke = clarke_block(cgm)["gbdt"]
        out[name] = {
            "features": cols,
            "metrics_cgm": metric_block(cgm),
            "gbdt_zone_a_pooled_pct": clarke["zones_pct"].get("A"),
            "gbdt_zone_a_participant_median": clarke["per_participant_zone_a"]["median_zone_a"] if clarke["per_participant_zone_a"] else None,
            "improvement_vs_b2_mgdl": gate_b2["improvement_mgdl"],
            "fold_wins_vs_b2": gate_b2["fold_wins"],
            "fold_params": params,
        }
    return out


def fit_final(meals: pd.DataFrame, horizon: int, meter_pool, ppg_pool, models_dir: Path) -> dict:
    cohort = meals[meals[f"y{horizon}"].notna() & meals["g0_mgdl"].notna()].reset_index(drop=True)
    cols = FEATURE_SETS["full"]
    p, scores = tune(cohort, horizon, cols, meter_pool, ppg_pool, seed=RANDOM_STATE)
    all_subs = set(cohort["subject"])
    rng = np.random.default_rng(RANDOM_STATE + horizon)
    rows = expand(cohort, pool_for(meter_pool, all_subs), pool_for(ppg_pool, all_subs), rng,
                  N_METER_DRAWS, N_METER_DRAWS)
    model = fit_rows(rows, horizon, cols, p)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"gbdt_cgmacros_T{horizon}.txt"
    model.booster_.save_model(str(model_path))
    meta = {
        "horizon_min": horizon,
        "feature_columns": cols,
        "parameterisation": "offset (target = Dexcom - g0_input)" if p["offset"] else "direct (target = Dexcom)",
        "params": p,
        "learning_rate": LEARNING_RATE,
        "random_state": RANDOM_STATE,
        "n_meals": int(len(cohort)),
        "n_participants": int(cohort["subject"].nunique()),
        "training_rows_incl_augmentation": int(len(rows)),
        "g0_definition": "last Dexcom GL reading strictly before the meal timestamp (C2)",
        "g0_sources": ["CGM", "meter"] + (["ppg"] if ppg_pool is not None else []),
        "meter_residual_pool_mgdl": {
            "n": int(len(meter_pool)),
            "definition": "Contour fingerstick minus Dexcom at the same clock time, all days (date unknown)",
        },
        "ppg_residual_pool": None if ppg_pool is None else {"n": int(len(ppg_pool))},
        "tuning_inner_mae_mgdl": float(min(scores)),
        "target": "exact Dexcom GL reading at meal time + horizon",
        "dataset": "CGMacros full release (45 participant folders)",
    }
    meta_path = models_dir / f"gbdt_cgmacros_T{horizon}.json"
    meta_path.write_text(json.dumps(_clean(meta), indent=2))
    return {"model": str(model_path), "meta": str(meta_path), "params": p}


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


def posthoc_macros_gate(meals: pd.DataFrame, horizon: int, meter_pool, ppg_pool,
                        ablation_fold_params: dict, n_perm: int) -> dict:
    """EXPLORATORY: full ADR 0004 gate for the g0_hour_macros feature set.

    Feature set was chosen from the ablation on the same folds, so this is a
    post-hoc analysis, not a confirmatory claim. Fold params are the ablation's.
    """
    cohort = meals[meals[f"y{horizon}"].notna() & meals["g0_mgdl"].notna()].reset_index(drop=True)
    cols = FEATURE_SETS["g0_hour_macros"]
    fixed = {int(k): v for k, v in ablation_fold_params.items()}
    res, _ = cross_validate(cohort, horizon, cols, meter_pool, ppg_pool, fixed=fixed)
    cgm = res[res["g0_source"] == "CGM"].reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_STATE + horizon)
    observed = _mae(cgm["y"], cgm["b2"]) - _mae(cgm["y"], cgm["gbdt"])
    perm_p, null_gains = permutation_p(cohort, horizon, fixed, meter_pool, ppg_pool,
                                       observed, n_perm, cols=cols)
    return {
        "exploratory": True,
        "note": "post-hoc: feature set selected from ablation on the same folds",
        "features": cols,
        "fold_params": ablation_fold_params,
        "n_meals_cohort": int(len(cohort)),
        "n_participants_cohort": int(cohort["subject"].nunique()),
        "metrics_cgm": {m: {"mae": _mae(cgm["y"], cgm[m])} for m in ("gbdt", "b0", "b1", "b2")},
        "gate": {
            "vs_b2": gate_vs(cgm, "b2", rng, perm_p),
            "vs_b1": gate_vs(cgm, "b1", rng, None),
            "vs_b0": gate_vs(cgm, "b0", rng, None),
        },
        "permutation": {
            "n": n_perm,
            "observed_gain_mgdl": float(observed),
            "null_gain_mean": float(null_gains.mean()),
            "p": perm_p,
        },
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ppg-residuals", type=Path, default=None)
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--posthoc-macros", action="store_true",
                    help="only run exploratory gate for g0_hour_macros (needs ablation.json in --out)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    meals, cleaning, contour, _profile = build_dataset(args.data_root)
    meter_df = meter_pool_from_contour(contour)
    ppg_df = load_ppg_residuals(args.ppg_residuals)
    if args.posthoc_macros:
        abl = json.loads((args.out / "ablation.json").read_text())
        post = {
            f"T{h}": posthoc_macros_gate(meals, h, meter_df, ppg_df,
                                         abl[f"T{h}"]["g0_hour_macros"]["fold_params"],
                                         args.permutations)
            for h in HORIZONS
        }
        _write(args.out / "posthoc_g0_hour_macros.json", post)
        return

    cont_stats = {
        "n_residuals_all_days": int(len(contour)),
        "n_residuals_min_abs_day": int(contour["is_min_abs_day"].sum()),
        "all_days_mean": float(contour["residual_mgdl"].mean()),
        "all_days_sd": float(contour["residual_mgdl"].std()),
        "all_days_mad_abs": float(contour["residual_mgdl"].abs().median()),
        "min_abs_day_sd_optimistic": float(contour.loc[contour["is_min_abs_day"], "residual_mgdl"].std()),
        "per_participant_n": {str(k): int(v) for k, v in contour.groupby("subject").size().items()},
    }
    _write(args.out / "cleaning.json", cleaning)
    _write(args.out / "contour_pool.json", cont_stats)

    results, all_preds = {}, []
    for h in HORIZONS:
        res, preds = run_horizon(meals, h, meter_df, ppg_df, args.permutations)
        results[f"T{h}"] = res
        preds = preds.assign(horizon_min=h)
        all_preds.append(preds)
        _write(args.out / f"T{h}.json", res)

    ablation = {}
    for h in HORIZONS:
        ablation[f"T{h}"] = run_ablation(meals, h, meter_df, ppg_df)
    _write(args.out / "ablation.json", ablation)

    pd.concat(all_preds, ignore_index=True).to_csv(
        args.out / "predictions.csv.gz", index=False, compression="gzip")

    if args.models_dir is not None:
        results["final_models"] = {
            f"T{h}": fit_final(meals, h, meter_df, ppg_df, args.models_dir) for h in HORIZONS
        }
    results["ppg_residuals"] = "absent: [NOT RUN] awaiting #16" if ppg_df is None else f"{len(ppg_df)} residuals"
    results["cleaning"] = cleaning
    results["contour_pool"] = cont_stats
    _write(args.out / "summary.json", results)
    print(json.dumps(_clean({k: {
        "n_meals_cohort": v.get("n_meals_cohort"),
        "n_participants_cohort": v.get("n_participants_cohort"),
        "gate_passes_vs_b2": v.get("gate_passes_vs_b2"),
        "gate_vs_b2": v.get("gate", {}).get("vs_b2", {}).get("reasons"),
        "metrics_cgm": v.get("metrics_cgm"),
        "permutation": v.get("permutation"),
    } for k, v in results.items() if k.startswith("T")}), indent=2, default=str))


if __name__ == "__main__":
    main()
