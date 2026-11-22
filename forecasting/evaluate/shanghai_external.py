"""External check of the GBDT forecaster on ShanghaiT2DM (wayfinder #29).

(a) Transfer: the CGMacros-trained forecasters are applied to Shanghai meals with the inputs
    Shanghai has (g0 = last CGM reading at or before t0, clock hour, age/sex/BMI, diabetes = T2D;
    macros NaN). Arms: the served `full` models (+ a no-profile diagnostic) and two CGMacros
    models retrained without macros (`g0_hour`, `g0_hour_profile`, trial settings, config chosen by
    the trial's grouped inner CV on CGMacros only). Baselines B0/B2 use CGMacros cohort constants;
    B1 is persistence.
(b) Within Shanghai: participant-grouped nested CV GBDT (same 5 x 5 outer folds as (a)); feature
    set and hyper-parameters chosen in inner folds only. Feature sets: g0_hour, g0_hour_profile,
    and both plus `dish_score` (a text-derived per-meal score, see `text_scores`).

Everything reuses `forecasting.evaluate.gbdt_forecaster` (design, GRID, fit_rows, folds, gate,
Clarke blocks). Two in-process patches, no edit of that module: LightGBM runs single-threaded
(compute budget), and `design` also emits `dish_score` when the rows carry it.

Subcommands (python -m forecasting.evaluate.shanghai_external <cmd>): prep, transfer --h H,
within-cv, within-perm --n N [--first K], report.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import pickle
import re
import time
import types
import zipfile
from multiprocessing import get_context
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import forecasting.evaluate.gbdt_forecaster as G
from nlp.data.diet_items import meal_items
from nlp.eval_dish_identity import MIN_DISH_MEALS
from nlp.eval_dish_identity import _fit_and_predict as ridge_alpha_cv

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("SHANGHAI_WORK", "/tmp/shanghai-external-work"))
OUT = REPO / "forecasting" / "shanghai_results"
SHANGHAI_DIR = Path.home() / "Downloads" / "ShanghaiT2DM"
SHANGHAI_ZIP = Path.home() / "Downloads" / "ShanghaiT2DM_raw" / "data.zip"
CGMACROS_ROOT = Path.home() / "Downloads" / "CGMacros_full" / "CGMacros"
SERVED_DIR = Path("/Users/auliamnaufal/Downloads/glucosight/forecasting/models")
HORIZONS = (60, 120)
N_PERM = 200
CGM_LABEL = "cgm"

# ---------------------------------------------------------------------------
# In-process patches of the trial module (see module docstring)
# ---------------------------------------------------------------------------
_LGBM = lgb.LGBMRegressor


def _one_thread(**kw):
    kw["n_jobs"] = 1
    return _LGBM(**kw)


G.lgb = types.SimpleNamespace(LGBMRegressor=_one_thread, Booster=lgb.Booster)
_design0 = G.design


def _design_ext(rows):
    X = _design0(rows)
    if "dish_score" in rows.columns:
        X["dish_score"] = rows["dish_score"].to_numpy(dtype=float)
    return X


G.design = _design_ext

PROFILE_SH = ["age_y", "female", "bmi"]  # dm_band is constant (T2D) in Shanghai: dropped
WITHIN_SETS = {
    "g0_hour": ["g0_mgdl", "hour_sin", "hour_cos"],
    "g0_hour_profile": ["g0_mgdl", "hour_sin", "hour_cos"] + PROFILE_SH,
    "g0_hour_text": ["g0_mgdl", "hour_sin", "hour_cos", "dish_score"],
    "g0_hour_profile_text": ["g0_mgdl", "hour_sin", "hour_cos"] + PROFILE_SH + ["dish_score"],
}
NO_TEXT = ("g0_hour", "g0_hour_profile")
SET_NAMES = tuple(WITHIN_SETS)
INSULIN_RE = re.compile(
    r"insulin|novolin|humulin|gansulin|lantus|levemir|toujeo|tresiba|novorapid|lispro|aspart|"
    r"glargine|detemir|degludec|glulisine|mixtard|\b\d\dr\b|70/30|50/50|30r|50r|70r", re.I)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _save(name: str, obj) -> None:
    with open(WORK / name, "wb") as f:
        pickle.dump(obj, f)


def _load(name: str):
    with open(WORK / name, "rb") as f:
        return pickle.load(f)


def _mae(y, p) -> float:
    return G._mae(y, p)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_profile() -> pd.DataFrame:
    """Per-period profile from Shanghai_T2DM_Summary.xlsx (shipped inside the raw zip)."""
    path = WORK / "Shanghai_T2DM_Summary.xlsx"
    if not path.exists():
        with zipfile.ZipFile(SHANGHAI_ZIP) as z:
            member = [n for n in z.namelist() if n.endswith("Shanghai_T2DM_Summary.xlsx")]
            assert len(member) == 1, member
            path.write_bytes(z.read(member[0]))
    s = pd.read_excel(path)
    s.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in s.columns]
    return s.set_index("Patient Number")


def build_shanghai() -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    resp = pd.read_csv(WORK / "response.csv")
    committed = pd.read_csv(REPO / "nlp" / "features" / "shanghai_nlp_response.csv")
    legacy = pd.read_csv(REPO / "nlp" / "features" / "shanghai_nlp_features.csv")[
        ["meal_id", "total_grams", "is_fried_cooking", "is_large_portion"]]
    info = {"rebuild_identical_to_committed_response_csv": bool(resp.equals(committed))}
    sh = resp.merge(legacy, on="meal_id", how="left", validate="one_to_one")
    prof = load_profile()
    sh["subject"] = sh["subject_id"].astype(int)
    sh["age_y"] = sh["participant_id"].map(prof["Age (years)"]).astype(float)
    sh["female"] = sh["participant_id"].map(prof["Gender (Female=1, Male=2)"]).map({1: 1.0, 2: 0.0})
    sh["bmi"] = sh["participant_id"].map(prof["BMI (kg/m2)"]).astype(float)
    sh["dm_band"] = 2.0  # all participants are T2D (Type of Diabetes == T2DM for 109/109 periods)
    t0 = pd.to_datetime(sh["t0_timestamp_iso"], errors="coerce")
    hour = t0.dt.hour + t0.dt.minute / 60.0
    sh["hour"] = hour
    sh["hour_sin"], sh["hour_cos"] = G.hour_sin_cos(hour)
    for c in ("carbs_g", "protein_g", "fat_g", "fibre_g"):
        sh[c] = np.nan  # Shanghai has no weighed macros
    sh["g0_input"] = sh["g0_mgdl"]
    sh["g0_source"] = CGM_LABEL
    sh["ridx"] = np.arange(len(sh))
    for h in HORIZONS:
        sh[f"y{h}"] = np.where(sh[f"valid_t{h}"] == 1, sh[f"glucose_t{h}"], np.nan)
    # text features: dish multi-hot (weighted items only, as nlp/eval_dish_identity), grams, flags
    sets = [tuple(sorted({n for n, _ in meal_items(t)})) for t in sh["dietary_intake"]]
    vocab = sorted({n for s in sets for n in s})
    index = {n: i for i, n in enumerate(vocab)}
    D = np.zeros((len(sh), len(vocab)), dtype=np.float32)
    for r, s in enumerate(sets):
        for n in s:
            D[r, index[n]] = 1.0
    info["n_grams_missing"] = int(sh["total_grams"].isna().sum())
    sh["total_grams"] = sh["total_grams"].fillna(sh["total_grams"].median())
    for c in ("is_fried_cooking", "is_large_portion"):
        sh[c] = sh[c].fillna(0.0)
    info["vocabulary_size"] = len(vocab)
    info["meals_with_no_parsed_dish"] = int((D.sum(1) == 0).sum())
    return sh, D, vocab, info


def cohort(sh: pd.DataFrame, h: int) -> pd.DataFrame:
    keep = sh[f"y{h}"].notna() & sh["g0_mgdl"].notna() & sh["age_y"].notna() & sh["bmi"].notna() \
        & sh["female"].notna()
    c = sh[keep].copy()
    c["overlaps"] = c[f"overlaps_next_meal_t{h}"].astype(int)
    return c.reset_index(drop=True)


def profile_report(sh: pd.DataFrame, prof: pd.DataFrame) -> dict:
    """Describe the Shanghai profile and medication fields (reported; not model inputs)."""
    per = prof.copy()
    agents = per["Hypoglycemic Agents"].astype(str)
    insulin = agents.str.contains(INSULIN_RE)
    subj = sh.groupby("subject").first()
    out = {
        "n_periods": int(len(per)), "n_subjects": int(sh["subject"].nunique()),
        "type_of_diabetes": per["Type of Diabetes"].value_counts().to_dict(),
        "age_years": {"mean": float(per["Age (years)"].mean()), "sd": float(per["Age (years)"].std()),
                      "min": float(per["Age (years)"].min()), "max": float(per["Age (years)"].max())},
        "female_share_periods": float((per["Gender (Female=1, Male=2)"] == 1).mean()),
        "bmi": {"mean": float(per["BMI (kg/m2)"].mean()), "sd": float(per["BMI (kg/m2)"].std()),
                "min": float(per["BMI (kg/m2)"].min()), "max": float(per["BMI (kg/m2)"].max())},
        "diabetes_duration_years_median": float(per["Duration of diabetes (years)"].median()),
        "periods_with_insulin_in_agent_text_HEURISTIC_regex": int(insulin.sum()),
        "periods_with_metformin_in_agent_text": int(agents.str.contains("metformin", case=False).sum()),
        "periods_with_any_agent_listed": int((agents.str.lower() != "none").sum()),
        "meals_insulin_near_meal_within_30min": int(sh["insulin_near_meal"].sum()),
        "meals_total": int(len(sh)),
        "hba1c_mmol_per_mol_recorded_periods": int(
            pd.to_numeric(per["HbA1c (mmol/mol)"], errors="coerce").notna().sum()),
        "note": "medication / insulin / HbA1c are reported only; none enters a model input",
    }
    return out


# ---------------------------------------------------------------------------
# Text-derived meal feature
# ---------------------------------------------------------------------------

def _text_mats(ctx, ridx_tr, ridx_te):
    D, g = ctx["D"], ctx["grams"]
    cols = np.flatnonzero(D[ridx_tr].sum(axis=0) >= MIN_DISH_MEALS)
    mu, sd = g[ridx_tr].mean(), g[ridx_tr].std() or 1.0

    def mk(idx):
        return np.hstack([((g[idx] - mu) / sd)[:, None], ctx["flags"][idx], D[idx][:, cols]])
    return mk(ridx_tr), mk(ridx_te)


def _ridge(Xtr, ytr, gtr, Xte, alpha):
    if alpha is None:
        return ridge_alpha_cv(Xtr, ytr, gtr, Xte)
    return Ridge(alpha=alpha).fit(Xtr, ytr).predict(Xte), alpha


def text_scores(ctx, tr: pd.DataFrame, te: pd.DataFrame, h: int, alpha=None):
    """Ridge on (dish multi-hot with >= MIN_DISH_MEALS training meals, grams, 2 flags) -> glucose
    rise at horizon h. Returns (out-of-fold score for `tr` over participant-grouped folds,
    score for `te` from a fit on all of `tr`, alpha used). Uses no g0, hour or profile."""
    ytr = (tr[f"y{h}"] - tr["g0_mgdl"]).to_numpy(dtype=float)
    rtr, rte = tr["ridx"].to_numpy(), te["ridx"].to_numpy()
    gtr = tr["subject"].to_numpy()
    oof = np.empty(len(tr))
    for a, b in GroupKFold(5).split(rtr, groups=gtr):
        Xa, Xb = _text_mats(ctx, rtr[a], rtr[b])
        oof[b], _ = _ridge(Xa, ytr[a], gtr[a], Xb, alpha)
    Xa, Xb = _text_mats(ctx, rtr, rte)
    sc, used = _ridge(Xa, ytr, gtr, Xb, alpha)
    return oof, sc, used


# ---------------------------------------------------------------------------
# (b) Within-Shanghai nested CV
# ---------------------------------------------------------------------------
_CTX: dict = {}


def _init_worker():
    global _CTX
    _CTX = _load("ctx.pkl")


def _split(coh: pd.DataFrame, r: int, k: int):
    subjects = coh["subject"].to_numpy()
    te_subs = G.outer_folds(subjects, r)[k]
    m = np.isin(subjects, te_subs)
    tr, te = coh[~m].reset_index(drop=True), coh[m].reset_index(drop=True)
    assert not (set(tr["subject"]) & set(te["subject"])), "participant leak"
    return tr, te, sorted(int(s) for s in te_subs)


def _fit_pred(train_rows, test_rows, h, set_name, p):
    cols = WITHIN_SETS[set_name]
    model = G.fit_rows(train_rows, h, cols, p)
    return G.predict_rows(model, test_rows, cols, p)


def outer_fold_task(args):
    h, r, k = args
    ctx = _CTX
    coh = ctx["cohort"][h]
    tr, te, te_subs = _split(coh, r, k)
    t_start = time.time()
    y_col = f"y{h}"
    errs = {(s, i): [] for s in SET_NAMES for i in range(len(G.GRID))}
    for a, b in GroupKFold(G.INNER_SPLITS).split(tr, groups=tr["subject"].to_numpy()):
        tr_a, va_b = tr.iloc[a], tr.iloc[b]
        oof, sc, _ = text_scores(ctx, tr_a, va_b, h)
        ra, rb = tr_a.assign(dish_score=oof), va_b.assign(dish_score=sc)
        y_b = rb[y_col].to_numpy(dtype=float)
        for s in SET_NAMES:
            for i, p in enumerate(G.GRID):
                errs[(s, i)].append(np.abs(_fit_pred(ra, rb, h, s, p) - y_b))
    scores = {(s, i): float(np.concatenate(v).mean()) for (s, i), v in errs.items()}
    best_i = {s: min(range(len(G.GRID)), key=lambda i: scores[(s, i)]) for s in SET_NAMES}
    set_score = {s: scores[(s, best_i[s])] for s in SET_NAMES}
    s1 = min(NO_TEXT, key=lambda s: set_score[s])
    s2 = min(SET_NAMES, key=lambda s: set_score[s])
    oof_tr, sc_te, alpha = text_scores(ctx, tr, te, h)
    rtr, rte = tr.assign(dish_score=oof_tr), te.assign(dish_score=sc_te)
    preds = {s: _fit_pred(rtr, rte, h, s, G.GRID[best_i[s]]) for s in SET_NAMES}
    b0 = float(tr[y_col].mean())
    exc = float((tr[y_col] - tr["g0_mgdl"]).mean())
    g0 = te["g0_mgdl"].to_numpy(dtype=float)
    out = pd.DataFrame({
        "subject": te["subject"].to_numpy(), "meal_id": te["meal_id"].to_numpy(),
        "y": te[y_col].to_numpy(dtype=float), "b0": b0, "b1": g0, "b2": g0 + exc,
        "insulin_near_meal": te["insulin_near_meal"].to_numpy(), "overlaps": te["overlaps"].to_numpy(),
        "repeat": r + 1, "fold": r * G.N_SPLITS + k + 1,
        "gbdt_s1": preds[s1], "gbdt_s2": preds[s2],
        **{f"pred_{s}": preds[s] for s in SET_NAMES},
        "dish_score": sc_te,
    })
    sel = {"h": h, "repeat": r + 1, "k": k, "fold": r * G.N_SPLITS + k + 1, "test_subjects": te_subs,
           "n_train_participants": int(tr["subject"].nunique()), "s1": s1, "s2": s2,
           "best_params": {s: G.GRID[best_i[s]] for s in SET_NAMES}, "inner_set_mae": set_score,
           "alpha": float(alpha), "seconds": time.time() - t_start}
    return h, out, sel


def within_cv(workers: int) -> None:
    tasks = [(h, r, k) for h in HORIZONS for r in range(G.N_REPEATS) for k in range(G.N_SPLITS)]
    _log(f"within-cv: {len(tasks)} outer folds, {workers} workers")
    frames = {h: [] for h in HORIZONS}
    sels = {h: [] for h in HORIZONS}
    with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for n, (h, out, sel) in enumerate(pool.imap_unordered(outer_fold_task, tasks), 1):
            frames[h].append(out)
            sels[h].append(sel)
            _log(f"fold {n}/{len(tasks)} (h={h} fold={sel['fold']}) {sel['seconds']:.0f}s "
                 f"s1={sel['s1']} s2={sel['s2']}")
    for h in HORIZONS:
        res = pd.concat(frames[h], ignore_index=True).sort_values(["fold", "subject", "meal_id"])
        _save(f"within_res_T{h}.pkl", res.reset_index(drop=True))
        _save(f"within_sel_T{h}.pkl", sorted(sels[h], key=lambda s: s["fold"]))


def _perm_rise(coh: pd.DataFrame, h: int, rng) -> pd.DataFrame:
    """Within-participant shuffle of the rise (y - g0); g0, hour, profile, text stay with their meal."""
    out = coh.copy()
    y = out[f"y{h}"].to_numpy(dtype=float)
    g0 = out["g0_mgdl"].to_numpy(dtype=float)
    rise = y - g0
    new = rise.copy()
    for idx in _CTX["groups"][h]:
        new[idx] = rng.permutation(rise[idx])
    out[f"y{h}"] = g0 + new
    return out


def _pooled_gains(coh: pd.DataFrame, h: int, sels: list[dict], s1_only: bool = False) -> tuple[float, float, float]:
    """Pooled (over the 25 outer folds) MAE gains vs B2: S1, S2, and S1-minus-S2 (text increment).
    `s1_only` skips S2 (returns NaN for it and for the increment)."""
    ctx = _CTX
    y_col = f"y{h}"
    e_b2 = e1 = e2 = 0.0
    n = 0
    for s in sels:
        m = np.isin(coh["subject"].to_numpy(), s["test_subjects"])
        tr, te = coh[~m].reset_index(drop=True), coh[m].reset_index(drop=True)
        tr0, te0 = tr.assign(dish_score=0.0), te.assign(dish_score=0.0)
        p1 = s["best_params"][s["s1"]]
        p2 = s["best_params"][s["s2"]]
        pred1 = _fit_pred(tr0, te0, h, s["s1"], p1)
        if s1_only:
            pred2 = np.full(len(te), np.nan)
        elif s["s2"] == s["s1"] and p1 == p2:
            pred2 = pred1
        else:
            rtr, rte = tr0, te0
            if "text" in s["s2"]:
                oof, sc, _ = text_scores(ctx, tr, te, h, alpha=s["alpha"])
                rtr, rte = tr.assign(dish_score=oof), te.assign(dish_score=sc)
            pred2 = _fit_pred(rtr, rte, h, s["s2"], p2)
        y = te[y_col].to_numpy(dtype=float)
        g0 = te["g0_mgdl"].to_numpy(dtype=float)
        exc = float((tr[y_col] - tr["g0_mgdl"]).mean())
        e_b2 += np.abs(y - (g0 + exc)).sum()
        e1 += np.abs(y - pred1).sum()
        e2 += np.abs(y - pred2).sum()
        n += len(y)
    return (e_b2 - e1) / n, (e_b2 - e2) / n, (e1 - e2) / n


def perm_task(args):
    h, ids, identity, s1_only = args
    coh = _CTX["cohort"][h]
    sels = _CTX["sels"][h]
    res = []
    if identity:
        res.append((-1, _pooled_gains(coh, h, sels, s1_only)))
    for i in ids:
        rng = np.random.default_rng([G.RANDOM_STATE, 29, h, i])
        res.append((i, _pooled_gains(_perm_rise(coh, h, rng), h, sels, s1_only)))
    return h, res


def within_perm(workers: int, n: int, first: int, s1_only: bool) -> None:
    """Permutations first..n-1 (ids are stable, so a later call extends an earlier one)."""
    ctx = _load("ctx.pkl")
    ctx["sels"] = {h: _load(f"within_sel_T{h}.pkl") for h in HORIZONS}
    _save("ctx_perm.pkl", ctx)
    ids = list(range(first, n))
    chunk = 5
    tasks = []
    for h in HORIZONS:
        for j in range(0, len(ids), chunk):
            tasks.append((h, ids[j:j + chunk], first == 0 and j == 0, s1_only))
    _log(f"within-perm: ids {first}..{n - 1} x 2 horizons, {len(tasks)} tasks")
    out = {h: {} for h in HORIZONS}
    ident = {}
    with get_context("spawn").Pool(workers, initializer=_init_worker_perm) as pool:
        for m, (h, res) in enumerate(pool.imap_unordered(perm_task, tasks), 1):
            for i, g in res:
                if i == -1:
                    ident[h] = g
                else:
                    out[h][i] = g
            if m % 10 == 0:
                _log(f"perm tasks {m}/{len(tasks)}")
    name = f"within_perm_{first}_{n}{'_s1only' if s1_only else ''}.pkl"
    _save(name, {"null": {h: out[h] for h in HORIZONS}, "identity": ident})


def _init_worker_perm():
    global _CTX
    _CTX = _load("ctx_perm.pkl")


# ---------------------------------------------------------------------------
# (a) Transfer
# ---------------------------------------------------------------------------

def _features(coh: pd.DataFrame, use_profile: bool) -> pd.DataFrame:
    nan = float("nan")
    parts = []
    for r in coh.itertuples(index=False):
        parts.append(G.encode_inference(
            r.g0_mgdl, CGM_LABEL, r.hour, nan, nan, nan, None,
            r.age_y if use_profile else None, r.female if use_profile else None,
            r.bmi if use_profile else None, 2.0 if use_profile else nan))
    return pd.concat(parts, ignore_index=True)


def _frames_over_folds(coh: pd.DataFrame, preds: dict[str, np.ndarray], b0: float, exc: float, h: int):
    """Test-fold frames over the shared Shanghai outer folds (every meal once per repeat)."""
    subjects = coh["subject"].to_numpy()
    y = coh[f"y{h}"].to_numpy(dtype=float)
    g0 = coh["g0_mgdl"].to_numpy(dtype=float)
    exc_sh = float((y - g0).mean())
    out = []
    for r in range(G.N_REPEATS):
        for k, te_subs in enumerate(G.outer_folds(subjects, r)):
            m = np.isin(subjects, te_subs)
            out.append(pd.DataFrame({
                "subject": subjects[m], "y": y[m], "b0": b0, "b1": g0[m], "b2": g0[m] + exc,
                "b2_sh": g0[m] + exc_sh,
                "insulin_near_meal": coh["insulin_near_meal"].to_numpy()[m],
                "overlaps": coh["overlaps"].to_numpy()[m],
                "repeat": r + 1, "fold": r * G.N_SPLITS + k + 1,
                **{a: p[m] for a, p in preds.items()},
            }))
    return pd.concat(out, ignore_index=True)


def transfer(h: int) -> None:
    ctx = _load("ctx.pkl")
    coh = ctx["cohort"][h]
    cg_meals = _load("cgmacros.pkl")
    cg = cg_meals[cg_meals[f"y{h}"].notna() & cg_meals["g0_mgdl"].notna()].reset_index(drop=True)
    b0 = float(cg[f"y{h}"].mean())
    exc = float((cg[f"y{h}"] - cg["g0_mgdl"]).mean())
    _log(f"T{h}: CGMacros cohort {len(cg)} meals / {cg['subject'].nunique()} participants; "
         f"B0={b0:.2f} B2 excursion={exc:.2f}; Shanghai {len(coh)} meals / {coh['subject'].nunique()} participants")

    X_full = _features(coh, use_profile=True)
    X_nop = _features(coh, use_profile=False)
    g0 = coh["g0_mgdl"].to_numpy(dtype=float)
    preds, meta = {}, {}

    meta_json = json.loads((SERVED_DIR / f"gbdt_cgmacros_T{h}.json").read_text())
    booster = lgb.Booster(model_file=str(SERVED_DIR / f"gbdt_cgmacros_T{h}.txt"))
    cols = booster.feature_name()
    off = bool(meta_json["params"]["offset"])
    for name, X in (("served_full", X_full), ("served_full_noprofile", X_nop)):
        p = booster.predict(X[cols])
        preds[name] = p + g0 if off else p
    meta["served_full"] = {"columns": cols, "params": meta_json["params"], "set": meta_json["selected_feature_set"]}

    sets_saved = dict(G.FEATURE_SETS)
    for name in ("g0_hour", "g0_hour_profile"):
        cols_s = sets_saved[name]
        G.FEATURE_SETS = {name: cols_s}
        sel_name, p, scores = G.inner_select(cg, h, None, seed=G.RANDOM_STATE + h)
        G.FEATURE_SETS = sets_saved
        rows = G.expand(cg, None, np.random.default_rng(G.RANDOM_STATE + h + 7000),
                        G.N_METER_DRAWS, G.N_METER_DRAWS)
        model = G.fit_rows(rows, h, cols_s, p)
        out = model.predict(X_full[cols_s])
        preds[f"retrained_{name}"] = out + g0 if p["offset"] else out
        meta[f"retrained_{name}"] = {"columns": cols_s, "params": p,
                                     "inner_mae": float(scores[f"{name}|{G.GRID.index(p)}"]),
                                     "training_rows": int(len(rows))}
        _log(f"T{h}: retrained {name}: params={p} inner_mae={meta[f'retrained_{name}']['inner_mae']:.2f}")

    res = _frames_over_folds(coh, preds, b0, exc, h)
    _save(f"transfer_res_T{h}.pkl", {"res": res, "meta": meta, "b0": b0, "exc": exc,
                                     "n_cgmacros_meals": len(cg), "n_cgmacros_participants": int(cg["subject"].nunique())})
    # null for the observed gain: within-participant shuffle of the rise, predictions fixed
    base = coh.reset_index(drop=True)
    yv = base[f"y{h}"].to_numpy(dtype=float)
    rise = yv - g0
    groups = [np.flatnonzero(base["subject"].to_numpy() == s) for s in np.sort(base["subject"].unique())]
    b2_one = g0 + exc
    gains = {a: np.empty(N_PERM) for a in preds}
    for i in range(N_PERM):
        rng = np.random.default_rng([G.RANDOM_STATE, 31, h, i])
        new = rise.copy()
        for idx in groups:
            new[idx] = rng.permutation(rise[idx])
        yp = g0 + new
        e_b2 = np.abs(yp - b2_one).mean()
        for a, p in preds.items():
            gains[a][i] = e_b2 - np.abs(yp - p).mean()
    _save(f"transfer_null_T{h}.pkl", gains)
    _log(f"T{h}: transfer done")


# ---------------------------------------------------------------------------
# Prep and report
# ---------------------------------------------------------------------------

def prep() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    sh, D, vocab, info = build_shanghai()
    prof = load_profile()
    cohorts = {h: cohort(sh, h) for h in HORIZONS}
    ctx = {
        "sh": sh, "D": D, "grams": sh["total_grams"].to_numpy(dtype=float),
        "flags": sh[["is_fried_cooking", "is_large_portion"]].to_numpy(dtype=float),
        "cohort": cohorts,
        "groups": {h: [np.flatnonzero(cohorts[h]["subject"].to_numpy() == s)
                       for s in np.sort(cohorts[h]["subject"].unique())] for h in HORIZONS},
    }
    _save("ctx.pkl", ctx)
    meals, cleaning, _ = G.build_dataset(CGMACROS_ROOT)
    _save("cgmacros.pkl", meals)
    info["cgmacros_cleaning"] = cleaning
    info["cohort_sizes"] = {f"T{h}": {"meals": int(len(c)), "participants": int(c["subject"].nunique())}
                            for h, c in cohorts.items()}
    info["profile"] = profile_report(sh, prof)
    OUT.mkdir(parents=True, exist_ok=True)
    G._write(OUT / "data_summary.json", info)
    _log(json.dumps(info["cohort_sizes"]))


def _clarke_light(res: pd.DataFrame) -> dict:
    cb = G.clarke_block(res)
    return {m: {"zones_pct": cb[m]["zones_pct"], "per_participant_zone_a": cb[m]["per_participant_zone_a"],
                "zone_a_by_reference_range": cb[m]["zone_a_by_reference_range"]} for m in G.MODELS}


def _sens(res: pd.DataFrame, base: str, rng) -> dict:
    sub = res[(res["insulin_near_meal"] == 0) & (res["overlaps"] == 0)]
    lo, hi = G.subject_bootstrap_improvement(sub, base, rng)
    return {"n_meals_per_repeat": int(len(sub) // G.N_REPEATS), "n_participants": int(sub["subject"].nunique()),
            "gain_mgdl": _mae(sub["y"], sub[base]) - _mae(sub["y"], sub["gbdt"]),
            "mae_gbdt": _mae(sub["y"], sub["gbdt"]), "mae_base": _mae(sub["y"], sub[base]),
            "bootstrap_ci_95": [lo, hi]}


def _p(null: np.ndarray, obs: float) -> float:
    return (int(np.sum(null >= obs)) + 1) / (len(null) + 1)


def arm_block(res: pd.DataFrame, h: int, perm_null: np.ndarray | None, bases=("b2", "b1", "b0"),
              obs_perm: float | None = None) -> dict:
    rng = np.random.default_rng(G.RANDOM_STATE + h)
    obs = _mae(res["y"], res["b2"]) - _mae(res["y"], res["gbdt"])
    p = None if perm_null is None else _p(np.asarray(perm_null), obs if obs_perm is None else obs_perm)
    gates = {b: G.gate_vs(res, b, rng, p if b == "b2" else None) for b in bases}
    first = res[res["repeat"] == 1]
    err = res["gbdt"] - res["y"]
    return {
        "n_meals": int(len(first)), "n_participants": int(res["subject"].nunique()),
        "metrics": G.metric_block(res),
        "mean_error_gbdt_minus_y": float(err.mean()),
        "mean_error_b2_minus_y": float((res["b2"] - res["y"]).mean()),
        "clarke": _clarke_light(res), "gate": gates,
        "permutation": None if perm_null is None else {
            "n": int(len(perm_null)), "observed_gain_mgdl": obs,
            "observed_gain_like_for_like_mgdl": obs if obs_perm is None else obs_perm, "p": p,
            "null_mean": float(np.mean(perm_null)), "null_p95": float(np.percentile(perm_null, 95))},
        "sensitivity_no_insulin_no_overlap_vs_b2": _sens(res, "b2", rng),
    }


def report() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for h in HORIZONS:
        t = _load(f"transfer_res_T{h}.pkl")
        nulls = _load(f"transfer_null_T{h}.pkl")
        res_all = t["res"]
        arms = {}
        for a in [c for c in res_all.columns if c.startswith(("served", "retrained"))]:
            res = res_all.assign(gbdt=res_all[a])
            blk = arm_block(res, h, nulls[a], bases=("b2", "b1", "b0", "b2_sh"))
            arms[a] = blk
        summary[f"transfer_T{h}"] = {"arms": arms, "meta": t["meta"], "b0_cgmacros": t["b0"],
                                     "b2_excursion_cgmacros": t["exc"],
                                     "n_cgmacros_meals": t["n_cgmacros_meals"],
                                     "n_cgmacros_participants": t["n_cgmacros_participants"]}
        G._write(OUT / f"transfer_T{h}.json", summary[f"transfer_T{h}"])

        res_w = _load(f"within_res_T{h}.pkl")
        sels = _load(f"within_sel_T{h}.pkl")
        merged: dict[int, tuple] = {}
        ident = None
        for pf in sorted(WORK.glob("within_perm_*.pkl")):
            with open(pf, "rb") as fh:
                d = pickle.load(fh)
            merged.update(d["null"][h])
            ident = d["identity"].get(h, ident)
        nulls_w = {"s1": None, "s2": None, "incr": None}
        if merged:
            ids = sorted(merged)
            arr = np.array([merged[i] for i in ids])
            nulls_w["s1"] = arr[:, 0]
            ok = np.isfinite(arr[:, 1])
            nulls_w["s2"], nulls_w["incr"] = arr[ok, 1], arr[ok, 2]
        wblk = {}
        for name, col, null, obs_perm in (
                ("S1_nested_no_text", "gbdt_s1", nulls_w["s1"], None),
                ("S2_nested_with_text", "gbdt_s2", nulls_w["s2"], None if ident is None else ident[1])):
            wblk[name] = arm_block(res_w.assign(gbdt=res_w[col]), h, null, obs_perm=obs_perm)
        for s in SET_NAMES:  # fixed feature set, config chosen in inner folds (no permutation run)
            wblk[f"fixed_{s}"] = arm_block(res_w.assign(gbdt=res_w[f"pred_{s}"]), h, None, bases=("b2",))
        # text increment: S2 vs S1, and fixed-set text vs no text
        rng = np.random.default_rng(G.RANDOM_STATE + h + 1)
        inc = res_w.assign(gbdt=res_w["gbdt_s2"], base=res_w["gbdt_s1"])
        obs_inc = _mae(inc["y"], inc["base"]) - _mae(inc["y"], inc["gbdt"])
        pinc = None if nulls_w["incr"] is None else _p(nulls_w["incr"], obs_inc if ident is None else ident[2])
        wblk["text_increment_S2_vs_S1"] = G.gate_vs(inc, "base", rng, pinc)
        for no, yes in (("g0_hour_profile", "g0_hour_profile_text"), ("g0_hour", "g0_hour_text")):
            f = res_w.assign(gbdt=res_w[f"pred_{yes}"], base=res_w[f"pred_{no}"])
            wblk[f"text_increment_fixed_{yes}_vs_{no}"] = G.gate_vs(f, "base", rng, None)
        sel_counts = {"s1": {}, "s2": {}}
        for s in sels:
            for key in ("s1", "s2"):
                sel_counts[key][s[key]] = sel_counts[key].get(s[key], 0) + 1
        wblk["selection_frequency"] = sel_counts
        wblk["identity_check_pooled_gains_s1_s2_incr"] = None if ident is None else list(ident)
        summary[f"within_T{h}"] = wblk
        G._write(OUT / f"within_T{h}.json", {**wblk, "selections": [
            {k: v for k, v in s.items() if k != "inner_set_mae"} | {"inner_set_mae": s["inner_set_mae"]}
            for s in sels]})
    G._write(OUT / "summary.json", summary)
    _log("report written")


def _row(name, blk, h):
    g = blk["gate"]["b2"]
    m = blk["metrics"]
    cz = blk["clarke"]
    pz, bz = cz["gbdt"]["per_participant_zone_a"], cz["b2"]["per_participant_zone_a"]
    pp = blk["permutation"]["p"] if blk["permutation"] else None
    return (f"| {name} | {blk['n_participants']} / {blk['n_meals']} | {m['gbdt']['mae']:.2f} | {m['b2']['mae']:.2f} | "
            f"{g['improvement_mgdl']:+.2f} [{g['bootstrap_ci_95'][0]:.2f}, {g['bootstrap_ci_95'][1]:.2f}] | "
            f"{g['fold_wins_total']}/{g['n_folds']} | {'n/a' if pp is None else f'{pp:.3f}'} | "
            f"{'PASS' if g['passes'] else ('FAIL' if pp is not None else 'FAIL/n.p.')} | "
            f"{pz['median_zone_a']:.1f} ({pz['iqr_zone_a']:.1f}) vs {bz['median_zone_a']:.1f} ({bz['iqr_zone_a']:.1f}) |")


def tables() -> None:
    s = json.loads((OUT / "summary.json").read_text())
    head = ("| Arm | n people / meals | MAE | B2 MAE | Gain vs B2 [95% CI] | Folds won | Perm p | Gate | "
            "Zone A median (IQR), arm vs B2 |\n|---|---|---|---|---|---|---|---|---|")
    lines = []
    for h in HORIZONS:
        t = s[f"transfer_T{h}"]
        lines += [f"\n### Transfer T+{h}\n", head]
        for a, blk in t["arms"].items():
            lines.append(_row(a, blk, h))
        b = next(iter(t["arms"].values()))
        m = b["metrics"]
        lines.append(f"\nBaselines (CGMacros constants): B0 {m['b0']['mae']:.2f}, B1 {m['b1']['mae']:.2f}, "
                     f"B2 {m['b2']['mae']:.2f} MAE; B0 = {t['b0_cgmacros']:.2f} mg/dL, B2 excursion = "
                     f"{t['b2_excursion_cgmacros']:+.2f} mg/dL.")
        lines.append("\nGain of each arm vs the other baselines:\n")
        lines.append("| Arm | vs B1 [CI] | vs B0 [CI] | vs B2-Shanghai-oracle [CI] | MAE RMSE | mean error (pred-y) |\n|---|---|---|---|---|---|")
        for a, blk in t["arms"].items():
            gg = blk["gate"]
            f = lambda k: f"{gg[k]['improvement_mgdl']:+.2f} [{gg[k]['bootstrap_ci_95'][0]:.2f}, {gg[k]['bootstrap_ci_95'][1]:.2f}]"
            lines.append(f"| {a} | {f('b1')} | {f('b0')} | {f('b2_sh')} | {blk['metrics']['gbdt']['mae']:.2f} / "
                         f"{blk['metrics']['gbdt']['rmse']:.2f} | {blk['mean_error_gbdt_minus_y']:+.1f} |")
    for h in HORIZONS:
        w = s[f"within_T{h}"]
        lines += [f"\n### Within Shanghai T+{h}\n", head]
        for a in ("S1_nested_no_text", "S2_nested_with_text") + tuple(f"fixed_{x}" for x in SET_NAMES):
            lines.append(_row(a, w[a], h))
        b = w["S1_nested_no_text"]["metrics"]
        lines.append(f"\nBaselines inside Shanghai folds: B0 {b['b0']['mae']:.2f}, B1 {b['b1']['mae']:.2f}, "
                     f"B2 {b['b2']['mae']:.2f} MAE.  Selection frequency: {json.dumps(w['selection_frequency'])}")
        for k, v in w.items():
            if k.startswith("text_increment"):
                lines.append(f"- {k}: gain {v['improvement_mgdl']:+.2f} CI [{v['bootstrap_ci_95'][0]:.2f}, "
                             f"{v['bootstrap_ci_95'][1]:.2f}], folds {v['fold_wins_total']}/{v['n_folds']}, "
                             f"perm p {v['permutation_p']}, passes {v['passes']}")
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")
    _log("tables written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "transfer", "within-cv", "within-perm", "report", "tables"])
    ap.add_argument("--h", type=int)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n", type=int, default=N_PERM)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--s1-only", action="store_true")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prep":
        prep()
    elif a.cmd == "report":
        report()
    elif a.cmd == "tables":
        tables()
    elif a.cmd == "transfer":
        transfer(a.h)
    elif a.cmd == "within-cv":
        within_cv(a.workers)
    elif a.cmd == "within-perm":
        within_perm(a.workers, a.n, a.first, a.s1_only)


if __name__ == "__main__":
    main()
