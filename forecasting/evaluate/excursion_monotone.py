"""Excursion target + monotone carb constraint vs the current GBDT forecaster (wayfinder #30).

Reuses the #17 pipeline (`gbdt_forecaster`): same data, cleaning, features, noise models, outer folds
(5 repeats x GroupKFold(5) by participant), seeds, noise draws and inner GroupKFold(4) selection.
Every arm is fitted on the same augmented training rows and scored on the same test rows, so
comparisons between arms are paired.

Arms (decision rule is pre-registered in forecasting/excursion_monotone_results/REPORT.md):
  cur         full #17 grid (direct or excursion, unconstrained)       = the current procedure
  cur_direct  #17 grid restricted to the direct-target configs         = what the served models are
  exc         excursion target only, unconstrained                     = diagnostic
  mono_c      excursion target, monotone +1 on carbs_g                 = the candidate
  mono_cf     excursion, +1 on carbs_g and -1 on fibre_g               = exploratory second arm

Feature sets without the constrained columns have no constraint, so their fits (and inner scores)
are identical to the unconstrained ones and are reused.

Usage (from repo root, one thread per BLAS/OpenMP, 3 worker processes):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m forecasting.evaluate.excursion_monotone \
        --data-root ~/Downloads/CGMacros_full/CGMacros --out forecasting/excursion_monotone_results \
        --raw-out /tmp/em_raw --permutations 200 --workers 3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import time
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from forecasting.evaluate import gbdt_forecaster as G

# arm -> (allowed config predicate on GRID entry, monotone spec {feature: sign})
ARMS = {
    "cur": (lambda p: True, {}),
    "cur_direct": (lambda p: not p["offset"], {}),
    "exc": (lambda p: p["offset"], {}),
    "mono_c": (lambda p: p["offset"], {"carbs_g": 1}),
    "mono_cf": (lambda p: p["offset"], {"carbs_g": 1, "fibre_g": -1}),
}
ARM_NAMES = tuple(ARMS)
MONO_KINDS = ("mono_c", "mono_cf")


def mono_key(spec: dict) -> tuple:
    return tuple(sorted(spec.items()))


def effective_mono(spec: dict, cols: list[str]) -> tuple:
    """Constraint restricted to the columns present: the identity of the fitted model."""
    return mono_key({k: v for k, v in spec.items() if k in cols})


def fit(rows: pd.DataFrame, horizon: int, cols: list[str], p: dict, mono: tuple) -> lgb.LGBMRegressor:
    X = G.design(rows)[cols]
    y = rows[f"y{horizon}"].to_numpy(dtype=float)
    g0 = rows["g0_input"].to_numpy(dtype=float)
    spec = dict(mono)
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=p["n_estimators"],
        learning_rate=G.LEARNING_RATE,
        num_leaves=p["num_leaves"],
        min_child_samples=p["min_child_samples"],
        random_state=G.RANDOM_STATE,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
        monotone_constraints=[spec.get(c, 0) for c in cols] if spec else None,
    )
    model.fit(X, y - g0 if p["offset"] else y)
    return model


class FitCache:
    """Fits keyed by (set, config index, effective constraint) so identical models are fitted once."""

    def __init__(self, rows: pd.DataFrame, horizon: int):
        self.rows, self.horizon, self.models = rows, horizon, {}

    def get(self, name: str, i: int, mono: tuple):
        key = (name, i, mono)
        if key not in self.models:
            self.models[key] = fit(self.rows, self.horizon, G.FEATURE_SETS[name], G.GRID[i], mono)
        return self.models[key]


def predict(model, rows: pd.DataFrame, name: str, p: dict) -> np.ndarray:
    out = model.predict(G.design(rows)[G.FEATURE_SETS[name]])
    return out + rows["g0_input"].to_numpy(dtype=float) if p["offset"] else out


def inner_scores(meals: pd.DataFrame, horizon: int, seed: int) -> dict:
    """Inner-CV MAE for every distinct (set, config, effective constraint), training participants only."""
    y_col = f"y{horizon}"
    keys = set()
    for spec_arm in ARM_NAMES:
        pred, spec = ARMS[spec_arm]
        for name, cols in G.FEATURE_SETS.items():
            for i, p in enumerate(G.GRID):
                if pred(p):
                    keys.add((name, i, effective_mono(spec, cols)))
    keys = sorted(keys)
    errs = {k: [] for k in keys}
    gkf = GroupKFold(n_splits=G.INNER_SPLITS)
    for f, (tr, va) in enumerate(gkf.split(meals, groups=meals["subject"].to_numpy())):
        tr_m, va_m = meals.iloc[tr], meals.iloc[va]
        rng = np.random.default_rng(seed * 100 + f)
        rows_tr = G.expand(tr_m, None, rng, G.N_METER_DRAWS, G.N_METER_DRAWS)
        rows_va = G.expand(va_m, None, rng, G.N_METER_DRAWS, G.N_METER_DRAWS)
        y_va = rows_va[y_col].to_numpy(dtype=float)
        cache = FitCache(rows_tr, horizon)
        for name, i, mono in keys:
            m = cache.get(name, i, mono)
            errs[(name, i, mono)].append(np.abs(predict(m, rows_va, name, G.GRID[i]) - y_va))
    return {k: float(np.concatenate(v).mean()) for k, v in errs.items()}


def select(scores: dict) -> dict:
    """Per arm: (set, config index, effective constraint) with the lowest inner MAE (grid order breaks ties)."""
    out = {}
    for arm, (pred, spec) in ARMS.items():
        cand = []
        for name, cols in G.FEATURE_SETS.items():
            for i, p in enumerate(G.GRID):
                if pred(p):
                    cand.append((name, i, effective_mono(spec, cols)))
        out[arm] = min(cand, key=lambda k: scores[k])
    return out


# ---------------------------------------------------------------------------
# Outer fold worker
# ---------------------------------------------------------------------------

_COHORTS: dict = {}


def run_fold(task):
    horizon, r, k = task
    cohort = _COHORTS[horizon]
    subjects = cohort["subject"].to_numpy()
    te_subs = G.outer_folds(subjects, r)[k]
    te_mask = np.isin(subjects, te_subs)
    tr_m, te_m = cohort[~te_mask], cohort[te_mask]
    assert not (set(tr_m["subject"]) & set(te_m["subject"])), "participant leak"
    fold = r * G.N_SPLITS + k + 1
    seed = G.RANDOM_STATE + 100 * r + k
    t = time.time()
    scores = inner_scores(tr_m, horizon, seed)
    sel = select(scores)

    y_col = f"y{horizon}"
    rows_tr = G.expand(tr_m, None, np.random.default_rng(seed + 5000), G.N_METER_DRAWS, G.N_METER_DRAWS)
    rows_te = G.expand(te_m, None, np.random.default_rng(seed + 9000), G.N_EVAL_DRAWS, G.N_EVAL_DRAWS)
    cache = FitCache(rows_tr, horizon)
    out = rows_te[["meal_id", "subject", "g0_source", "draw", "g0_input", "g0_mgdl", "dm_band"]].copy()
    out["y"] = rows_te[y_col].to_numpy(dtype=float)
    for arm in ARM_NAMES:
        name, i, mono = sel[arm]
        out[arm] = predict(cache.get(name, i, mono), rows_te, name, G.GRID[i])
    out["b0"] = float(tr_m[y_col].mean())
    g0 = rows_te["g0_input"].to_numpy(dtype=float)
    out["b1"] = g0
    out["b2"] = g0 + float((tr_m[y_col] - tr_m["g0_mgdl"]).mean())
    out["repeat"], out["fold"] = r + 1, fold

    meta = {
        "repeat": r + 1, "fold": fold, "seed": seed, "test_subjects": sorted(int(s) for s in te_subs),
        "selection": {arm: {"set": s[0], "config_index": s[1], "params": G.GRID[s[1]],
                            "constraint": dict(s[2])} for arm, s in sel.items()},
        "inner_mae": {arm: scores[sel[arm]] for arm in ARM_NAMES},
        "seconds": round(time.time() - t, 1),
    }
    return horizon, out, meta


def run_perm(task):
    """One macro-permutation: pooled CGM gain vs B2 of every arm, selections held fixed (i = -1: identity)."""
    horizon, i, selections = task
    cohort = _COHORTS[horizon]
    data = cohort if i < 0 else G.permute_macros(cohort, np.random.default_rng(G.RANDOM_STATE + 50_000 + i))
    y_col = f"y{horizon}"
    ys, b2s = [], []
    preds = {a: [] for a in ARM_NAMES}
    for s in selections:
        te_mask = np.isin(data["subject"].to_numpy(), s["test_subjects"])
        tr_m, te_m = data[~te_mask], data[te_mask]
        rows_tr = G.expand(tr_m, None, np.random.default_rng(s["seed"] + 5000), G.N_METER_DRAWS, G.N_METER_DRAWS)
        rows_te = G.expand(te_m, None, np.random.default_rng(s["seed"] + 9000), 0, 0)
        cache = FitCache(rows_tr, horizon)
        ys.append(rows_te[y_col].to_numpy(dtype=float))
        b2s.append(rows_te["g0_input"].to_numpy(dtype=float) + float((tr_m[y_col] - tr_m["g0_mgdl"]).mean()))
        for arm in ARM_NAMES:
            sl = s["selection"][arm]
            preds[arm].append(predict(cache.get(sl["set"], sl["config_index"], mono_key(sl["constraint"])),
                                      rows_te, sl["set"], sl["params"]))
    y, b2 = np.concatenate(ys), np.concatenate(b2s)
    return i, {arm: G._mae(y, b2) - G._mae(y, np.concatenate(preds[arm])) for arm in ARM_NAMES}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def paired(rows: pd.DataFrame, cand: str, base: str, rng) -> dict:
    d = rows.assign(gbdt=rows[cand])
    g = G.gate_vs(d, base, rng, None)
    win, by_repeat = G.fold_wins(d, base)
    return {"gain_mgdl": g["improvement_mgdl"], "bootstrap_ci_95": g["bootstrap_ci_95"],
            "folds_won": int(win.sum()), "n_folds": int(win.size),
            "wins_by_repeat": [int(x) for x in by_repeat.tolist()]}


def band_block(df: pd.DataFrame, arm: str) -> dict:
    """90th-percentile absolute error (the app band) and its empirical coverage across subgroups."""
    err = (df["y"] - df[arm]).abs()
    band = float(np.percentile(err, 90))
    cov = (err <= band)
    sub = cov.groupby(df["subject"]).mean()
    g0_bins = pd.cut(df["g0_mgdl"], [0, 100, 140, 180, 1000], right=False).astype(str)
    return {
        "p90_abs_error": band,
        "mae": float(err.mean()),
        "median_abs_error": float(err.median()),
        "p50_p75_p95": [float(np.percentile(err, q)) for q in (50, 75, 95)],
        "coverage_overall": float(cov.mean()),
        "coverage_per_participant_median_min_max": [float(sub.median()), float(sub.min()), float(sub.max())],
        "participants_below_80pct_coverage": int((sub < 0.80).sum()),
        "coverage_by_true_g0_range": {str(k): float(v) for k, v in cov.groupby(g0_bins).mean().items()},
        "coverage_by_dm_band": {str(k): float(v) for k, v in cov.groupby(df["dm_band"]).mean().items()},
    }


def analyse(horizon: int, res: pd.DataFrame, metas: list, null: dict, observed: dict, n_perm: int) -> dict:
    rng = np.random.default_rng(G.RANDOM_STATE + horizon)
    out = {"horizon_min": horizon, "n_permutations": n_perm}
    cgm = res[res["g0_source"] == "CGM"].reset_index(drop=True)
    meter = res[res["g0_source"] == "meter"].reset_index(drop=True)
    out["n_meals"] = int(cgm["meal_id"].nunique())
    out["n_participants"] = int(cgm["subject"].nunique())
    arms = {}
    for arm in ARM_NAMES:
        d_cgm = cgm.assign(gbdt=cgm[arm])
        gains = np.asarray(null[arm]) if n_perm else np.array([])
        p = (int(np.sum(gains >= observed[arm])) + 1) / (n_perm + 1) if n_perm else None
        gate = G.gate_vs(d_cgm, "b2", rng, p)
        d_met = meter.assign(gbdt=meter[arm])
        sel = [m["selection"][arm] for m in metas]
        arms[arm] = {
            "mae_cgm": G._mae(cgm["y"], cgm[arm]),
            "mae_meter": G._mae(meter["y"], meter[arm]),
            "rmse_cgm": G._rmse(cgm["y"], cgm[arm]),
            "gate_cgm_vs_b2": gate,
            "meter_vs_b2": G.gate_vs(d_met, "b2", rng, None),
            "permutation": {"n": n_perm, "observed_gain": observed[arm],
                            "null_mean": float(gains.mean()) if n_perm else None,
                            "null_p95": float(np.percentile(gains, 95)) if n_perm else None, "p": p},
            "selection_set_frequency": dict(Counter(s["set"] for s in sel)),
            "selection_config_frequency": dict(Counter(json.dumps(s["params"], sort_keys=True) for s in sel)),
            "selection_set_config_frequency": dict(Counter(
                f"{s['set']}|{'exc' if s['params']['offset'] else 'direct'}|nl{s['params']['num_leaves']}"
                f"|mc{s['params']['min_child_samples']}|n{s['params']['n_estimators']}" for s in sel)),
            "bands_cgm": band_block(cgm, arm),
            "bands_meter": band_block(meter, arm),
        }
    out["arms"] = arms
    out["baselines"] = {b: {"mae_cgm": G._mae(cgm["y"], cgm[b]), "mae_meter": G._mae(meter["y"], meter[b])}
                        for b in ("b0", "b1", "b2")}
    out["paired_vs_cur"] = {}
    out["paired_vs_cur_direct"] = {}
    for arm in ARM_NAMES:
        if arm != "cur":
            out["paired_vs_cur"][arm] = {"cgm": paired(cgm, arm, "cur", rng), "meter": paired(meter, arm, "cur", rng)}
        if arm not in ("cur", "cur_direct"):
            out["paired_vs_cur_direct"][arm] = {"cgm": paired(cgm, arm, "cur_direct", rng),
                                                "meter": paired(meter, arm, "cur_direct", rng)}
    out["paired_cur_direct_vs_cur"] = {"cgm": paired(cgm, "cur_direct", "cur", rng)}
    return out


def decide(per_h: dict) -> dict:
    """Pre-registered rule applied to mono_c (CGM rows, paired vs cur)."""
    def ok(h):
        a, pr = per_h[h]["arms"]["mono_c"], per_h[h]["paired_vs_cur"]["mono_c"]["cgm"]
        crit = {
            "paired_gain_positive": pr["gain_mgdl"] > 0,
            "paired_ci_excludes_zero": pr["bootstrap_ci_95"][0] > 0,
            "folds_won_majority": pr["folds_won"] > pr["n_folds"] / 2,
            "passes_gate_vs_b2": bool(a["gate_cgm_vs_b2"]["passes"]),
        }
        return crit, all(crit.values()), pr["gain_mgdl"]
    res = {f"T{h}": ok(h) for h in per_h}
    detail = {k: {"criteria": c, "all_met": m, "paired_gain": g} for k, (c, m, g) in res.items()}
    met = [v["all_met"] for v in detail.values()]
    gains = [v["paired_gain"] for v in detail.values()]
    adopt = all(met) or (any(met) and all(g >= 0 for g in gains))
    return {"per_horizon": detail, "adopt_mono_c": bool(adopt)}


def _clean(o):
    return json.loads(json.dumps(G._clean(o), default=str))


# ---------------------------------------------------------------------------
# Final fit (only if adopted)
# ---------------------------------------------------------------------------

def fit_final(cohort: pd.DataFrame, horizon: int, arm: str, models_dir: Path) -> dict:
    seed = G.RANDOM_STATE + horizon          # same seed as the served models' selection
    scores = inner_scores(cohort, horizon, seed)
    name, i, mono = select(scores)[arm]
    p, cols = G.GRID[i], G.FEATURE_SETS[name]
    rows = G.expand(cohort, None, np.random.default_rng(G.RANDOM_STATE + horizon + 7000),
                    G.N_METER_DRAWS, G.N_METER_DRAWS)
    model = fit(rows, horizon, cols, p, mono)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"gbdt_cgmacros_T{horizon}_excursion.txt"
    model.booster_.save_model(str(path))
    meta = {
        "horizon_min": horizon, "arm": arm, "selected_feature_set": name, "feature_columns": cols,
        "parameterisation": "offset (target = Dexcom - g0); prediction = g0 + model output",
        "monotone_constraints": dict(mono), "params": p, "learning_rate": G.LEARNING_RATE,
        "random_state": G.RANDOM_STATE, "n_meals": int(len(cohort)),
        "n_participants": int(cohort["subject"].nunique()),
        "training_rows_incl_augmentation": int(len(rows)), "g0_sources": ["CGM", "meter"],
        "inner_mae_mgdl": scores[(name, i, mono)],
        "dataset": "CGMacros full release (45 participant folders)",
    }
    path.with_suffix(".json").write_text(json.dumps(_clean(meta), indent=2))
    return {"model": str(path), "selected_feature_set": name, "params": p, "constraint": dict(mono)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw-out", type=Path, required=True, help="predictions and cached dataset (not committed)")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(G.HORIZONS))
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--final-arm", choices=MONO_KINDS, default=None)
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--stage", choices=["folds", "perms", "all"], default="all")
    ap.add_argument("--perm-arms-note", default="")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    args.raw_out.mkdir(parents=True, exist_ok=True)

    cache = args.raw_out / "meals.pkl"
    if cache.exists():
        meals = pickle.loads(cache.read_bytes())
    else:
        meals, cleaning, _ = G.build_dataset(args.data_root)
        cache.write_bytes(pickle.dumps(meals))
        G._write(args.out / "cleaning.json", cleaning)
    for h in args.horizons:
        _COHORTS[h] = meals[meals[f"y{h}"].notna() & meals["g0_mgdl"].notna()].reset_index(drop=True)

    ctx = mp.get_context("fork")
    per_h = {}
    for h in args.horizons:
        t0 = time.time()
        fold_file, meta_file = args.raw_out / f"preds_T{h}.pkl", args.raw_out / f"metas_T{h}.pkl"
        if fold_file.exists():
            res, metas = pickle.loads(fold_file.read_bytes()), pickle.loads(meta_file.read_bytes())
        else:
            tasks = [(h, r, k) for r in range(G.N_REPEATS) for k in range(G.N_SPLITS)]
            with ctx.Pool(args.workers) as pool:
                outs = pool.map(run_fold, tasks, chunksize=1)
            res = pd.concat([o[1] for o in outs], ignore_index=True)
            metas = [o[2] for o in outs]
            fold_file.write_bytes(pickle.dumps(res))
            meta_file.write_bytes(pickle.dumps(metas))
            res.to_csv(args.raw_out / f"_predictions_T{h}.csv.gz", index=False, compression="gzip")
        print(f"T{h} folds done in {time.time() - t0:.0f}s", flush=True)
        if args.stage == "folds":
            continue

        cgm = res[res["g0_source"] == "CGM"]
        observed = {a: G._mae(cgm["y"], cgm["b2"]) - G._mae(cgm["y"], cgm[a]) for a in ARM_NAMES}
        null = {a: [] for a in ARM_NAMES}
        perm_file = args.raw_out / f"perm_T{h}_{args.permutations}.pkl"
        if args.permutations:
            if perm_file.exists():
                null = pickle.loads(perm_file.read_bytes())
            else:
                t1 = time.time()
                tasks = [(h, i, metas) for i in range(-1, args.permutations)]
                with ctx.Pool(args.workers) as pool:
                    outs = pool.map(run_perm, tasks, chunksize=1)
                ident = dict(outs)[-1]
                for a in ARM_NAMES:
                    if abs(ident[a] - observed[a]) > 1e-9:
                        raise RuntimeError(f"identity check failed ({a}): {ident[a]} vs {observed[a]}")
                for i, g in sorted(o for o in outs if o[0] >= 0):
                    for a in ARM_NAMES:
                        null[a].append(g[a])
                perm_file.write_bytes(pickle.dumps(null))
                print(f"T{h} perms done in {time.time() - t1:.0f}s", flush=True)
        per_h[h] = analyse(h, res, metas, null, observed, args.permutations)
        per_h[h]["fold_selections"] = metas
        G._write(args.out / f"T{h}.json", _clean(per_h[h]))

    if args.stage == "folds":
        return
    decision = decide(per_h)
    G._write(args.out / "decision.json", decision)
    print(json.dumps(decision, indent=2, default=str))
    if args.final_arm:
        if args.models_dir is None:
            raise SystemExit("--final-arm needs --models-dir")
        final = {f"T{h}": fit_final(_COHORTS[h], h, args.final_arm, args.models_dir) for h in args.horizons}
        G._write(args.out / "final_models.json", final)
        print(json.dumps(_clean(final), indent=2))


if __name__ == "__main__":
    main()
