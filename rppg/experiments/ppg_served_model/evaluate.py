"""Held-out evaluation of the PPG -> glucose candidates (ADR 0004 gate).

Primary protocol: GroupKFold(5) by subject at native 2190 Hz. Every candidate is
compared against the training-fold mean on the same folds. Outputs:
  rppg/experiments/ppg_served_model/ppg_served_model.json   (all numbers)
  rppg/models/reports/ppg_served_residuals.csv              (served model, out-of-fold)
Run with /tmp/ppg-venv/bin/python (or any python with the same sklearn/scipy).
"""
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from common import CACHE_PATH, HERE, REPO_ROOT
from candidates import CANDIDATES, RANDOM_STATE, fit_mean, get_X
from utils.clarke_grid import (clarke_error_grid, per_participant_zone_a,
                               zone_a_by_reference_range)

N_SPLITS = 5
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 1000
# Random-forest candidates re-run nested tuning inside every permutation (~10 s per CV run);
# 1000 permutations took >1 h per forest candidate, so they use 200 (min attainable p = 1/201).
N_PERMUTATIONS_FOREST = 200
FOREST_CANDIDATES = {"rf_handcrafted_nested", "papagei_pca10_rf_nested"}
MIN_FOLD_WINS = 4
MAX_PERMUTATION_P = 0.05
JSON_PATH = os.path.join(HERE, "ppg_served_model.json")
RESIDUALS_PATH = os.path.join(REPO_ROOT, "rppg", "models", "reports", "ppg_served_residuals.csv")


def run_cv(fit, X, y, groups, splitter):
    """Out-of-fold predictions; the training-fold mean is the baseline on the same rows."""
    n = len(y)
    pred, base, fold = np.empty(n), np.empty(n), np.zeros(n, dtype=int)
    tuned = []
    for i, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
        assert not (set(groups[tr]) & set(groups[te])), "subject leak across folds"
        m = fit(X[tr], y[tr], groups[tr])
        pred[te] = m.predict(X[te])
        base[te] = y[tr].mean()
        fold[te] = i
        tuned.append(getattr(m, "tuned_", None))
    return pred, base, fold, tuned


def _improvement(y, pred, base):
    return float(np.mean(np.abs(y - base)) - np.mean(np.abs(y - pred)))


def permuted_labels(y, groups, rng):
    """Shuffle glucose labels BETWEEN subjects (same scheme as glucose_estimator.evaluate)."""
    keys = list(np.unique(groups))
    subj_labels = {s: y[groups == s] for s in keys}
    order = rng.permutation(len(keys))
    y_perm = np.empty_like(y)
    for src, dst in zip(order, range(len(keys))):
        idx = np.flatnonzero(groups == keys[dst])
        donor = subj_labels[keys[src]]
        y_perm[idx] = rng.choice(donor, size=len(idx), replace=True)
    return y_perm


def _perm_improvement(fit, X, y_perm, groups):
    pred, base, _, _ = run_cv(fit, X, y_perm, groups, GroupKFold(n_splits=N_SPLITS))
    return _improvement(y_perm, pred, base)


def bootstrap_ci(y, pred, base, groups):
    """95% CI of baseline MAE minus model MAE, resampling SUBJECTS."""
    rng = np.random.default_rng(RANDOM_STATE)
    subjects = np.unique(groups)
    m_ae = np.abs(y - pred)
    b_ae = np.abs(y - base)
    s_m = np.array([m_ae[groups == s].sum() for s in subjects])
    s_b = np.array([b_ae[groups == s].sum() for s in subjects])
    cnt = np.array([(groups == s).sum() for s in subjects], dtype=float)
    diffs = np.empty(N_BOOTSTRAP)
    for k in range(N_BOOTSTRAP):
        idx = rng.choice(len(subjects), size=len(subjects), replace=True)
        n = cnt[idx].sum()
        diffs[k] = s_b[idx].sum() / n - s_m[idx].sum() / n
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def permutation_p(fit, X, y, groups, observed, n_perm):
    rng = np.random.default_rng(RANDOM_STATE + 1)
    perms = [permuted_labels(y, groups, rng) for _ in range(n_perm)]
    gains = []
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=ctx) as ex:
        futures = [ex.submit(_perm_improvement, fit, X, yp, groups) for yp in perms]
        for i, fut in enumerate(futures, start=1):
            gains.append(fut.result())
            if i % 25 == 0 or i == n_perm:
                print(f"  permutations done: {i}/{n_perm}", flush=True)
    count = sum(g >= observed for g in gains)
    return (count + 1) / (n_perm + 1), gains


def subject_stats(y, pred, groups):
    per = []
    for s in np.unique(groups):
        m = groups == s
        per.append(float(np.mean(np.abs(y[m] - pred[m]))))
    per = np.asarray(per)
    return {
        "n_subjects": int(len(per)),
        "mae_median": round(float(np.median(per)), 2),
        "mae_q25": round(float(np.percentile(per, 25)), 2),
        "mae_q75": round(float(np.percentile(per, 75)), 2),
    }


def summarize(name, description, fit, X, y, groups, with_permutation=True):
    pred, base, fold, tuned = run_cv(fit, X, y, groups, GroupKFold(n_splits=N_SPLITS))
    model_mae = float(np.mean(np.abs(y - pred)))
    base_mae = float(np.mean(np.abs(y - base)))
    folds = []
    for f in range(1, N_SPLITS + 1):
        m = fold == f
        fm = float(np.mean(np.abs(y[m] - pred[m])))
        fb = float(np.mean(np.abs(y[m] - base[m])))
        folds.append({"fold": f, "n_recordings": int(m.sum()),
                      "n_subjects": int(len(np.unique(groups[m]))),
                      "baseline_mae": round(fb, 2), "model_mae": round(fm, 2),
                      "model_wins": bool(fm < fb)})
    wins = sum(f["model_wins"] for f in folds)
    lo, hi = bootstrap_ci(y, pred, base, groups)
    improvement = base_mae - model_mae

    out = {
        "description": description,
        "mae": round(model_mae, 2),
        "rmse": round(float(np.sqrt(np.mean((y - pred) ** 2))), 2),
        "mae_gain_vs_mean": round(improvement, 2),
        "fold_wins": int(wins),
        "folds": folds,
        "bootstrap_ci_gain": [round(lo, 2), round(hi, 2)],
        "tuned_per_fold": tuned,
        "per_subject_mae": subject_stats(y, pred, groups),
        "clarke_pooled_pct": {k: round(v, 2) for k, v in clarke_error_grid(y, pred)[1].items()},
        "clarke_per_subject_zone_a": per_participant_zone_a(y, pred, groups, min_meals=2),
        "clarke_zone_a_by_reference_range": zone_a_by_reference_range(y, pred),
        "abs_residual_q80_q90": [round(float(q), 2)
                                 for q in np.percentile(np.abs(y - pred), [80, 90])],
    }
    if with_permutation:
        n_perm = N_PERMUTATIONS_FOREST if name in FOREST_CANDIDATES else N_PERMUTATIONS
        p, gains = permutation_p(fit, X, y, groups, improvement, n_perm)
        out["permutations"] = n_perm
        out["permutation_p"] = round(p, 4)
        out["permutation_null_gain_q95"] = round(float(np.percentile(gains, 95)), 2)
        reasons = {
            "mae_gain_positive": improvement > 0,
            "fold_wins_ge_4_of_5": wins >= MIN_FOLD_WINS,
            "bootstrap_ci_excludes_zero": lo > 0,
            "permutation_p_le_0.05": p <= MAX_PERMUTATION_P,
        }
        out["gate"] = {"criteria": reasons, "passes": all(reasons.values())}
    return out, pred, base


def main():
    cache = np.load(CACHE_PATH)
    y = cache["glucose"]
    groups = cache["subject_id"]
    rec = cache["recording_id"]
    sqi = cache["sqi"]
    n_subj = len(np.unique(groups))
    assert len(y) == 67 and n_subj == 23, (len(y), n_subj)

    result = {
        "protocol": {
            "primary": "GroupKFold(5) by subject, native 2190 Hz",
            "n_recordings": int(len(y)), "n_subjects": int(n_subj),
            "bootstrap": N_BOOTSTRAP, "permutations": N_PERMUTATIONS,
            "permutations_forest_candidates": N_PERMUTATIONS_FOREST,
            "gate": "MAE gain>0; fold wins>=4/5; bootstrap CI over subjects excludes 0 (lower>0); "
                    "subject-level label permutation p<=0.05",
            "selection_rule": "lowest pooled held-out MAE among non-baseline candidates",
        },
    }

    mean_out, mean_pred, mean_base = summarize(
        "mean", "Training-fold mean (B-mean baseline)", fit_mean,
        get_X(cache, "hand"), y, groups, with_permutation=False)
    mean_out["mae_gain_vs_mean"] = 0.0
    result["mean_baseline"] = mean_out

    candidates = {}
    preds = {}
    for name, (feat, fit, desc) in CANDIDATES.items():
        out, pred, _ = summarize(name, desc, fit, get_X(cache, feat), y, groups)
        candidates[name] = out
        preds[name] = pred
        print(f"{name}: MAE {out['mae']} gain {out['mae_gain_vs_mean']} "
              f"wins {out['fold_wins']} CI {out['bootstrap_ci_gain']} "
              f"p {out.get('permutation_p')} gate {out['gate']['passes']}", flush=True)
    result["candidates"] = candidates

    # Leave-one-subject-out sensitivity (point estimates only).
    loso = {}
    for name, (feat, fit, _) in CANDIDATES.items():
        p, b, _, _ = run_cv(fit, get_X(cache, feat), y, groups, LeaveOneGroupOut())
        loso[name] = {"mae": round(float(np.mean(np.abs(y - p))), 2),
                      "baseline_mae": round(float(np.mean(np.abs(y - b))), 2)}
    result["sensitivity_loso"] = loso

    # 30 Hz deployment-rate sensitivity for PaPaGei candidates (point + bootstrap; no permutation).
    sens30 = {}
    for name, (feat, fit, desc) in CANDIDATES.items():
        if not feat.startswith("papagei"):
            continue
        out, _, _ = summarize(name + "@30Hz", desc, fit, get_X(cache, "papagei_30hz"),
                              y, groups, with_permutation=False)
        sens30[name] = {k: out[k] for k in ("mae", "mae_gain_vs_mean", "fold_wins",
                                            "bootstrap_ci_gain", "folds")}
    result["sensitivity_30hz"] = sens30

    # Served-model selection (ticket rule: lowest MAE among non-baseline candidates).
    served = min(candidates, key=lambda k: candidates[k]["mae"])
    result["selected_served"] = served
    result["selected_served_passes_gate"] = candidates[served]["gate"]["passes"]

    res = y - preds[served]
    ae = np.abs(res)
    sq_rho, sq_p = spearmanr(ae, sqi)
    edges = np.quantile(sqi, [1 / 3, 2 / 3])
    tert = np.digitize(sqi, edges)
    result["served_residual_analysis"] = {
        "signed_residual_mean": round(float(res.mean()), 2),
        "abs_residual_q80_q90": [round(float(q), 2) for q in np.percentile(ae, [80, 90])],
        "spearman_abs_residual_vs_sqi": [round(float(sq_rho), 3), round(float(sq_p), 4)],
        "sqi_tertile_edges": [round(float(e), 3) for e in edges],
        "mae_by_sqi_tertile": [round(float(ae[tert == t].mean()), 2) for t in range(3)],
        "n_by_sqi_tertile": [int((tert == t).sum()) for t in range(3)],
        "sqi_below_0_3_count": int((sqi < 0.3).sum()),
    }

    os.makedirs(os.path.dirname(RESIDUALS_PATH), exist_ok=True)
    with open(RESIDUALS_PATH, "w") as fh:
        fh.write("subject_id,recording_id,glucose_true,glucose_pred,residual,fold\n")
        _, _, fold_ids, _ = run_cv(CANDIDATES[served][1], get_X(cache, CANDIDATES[served][0]),
                                   y, groups, GroupKFold(n_splits=N_SPLITS))
        for s, r, t, p, f in zip(groups, rec, y, preds[served], fold_ids):
            fh.write(f"{s},{r},{t:.1f},{p:.3f},{t - p:.3f},{f}\n")

    with open(JSON_PATH, "w") as fh:
        json.dump(result, fh, indent=2, default=lambda o: o.tolist()
                  if hasattr(o, "tolist") else str(o))
    print(f"served={served} passes_gate={result['selected_served_passes_gate']}")
    print(f"wrote {JSON_PATH} and {RESIDUALS_PATH}")


if __name__ == "__main__":
    main()
