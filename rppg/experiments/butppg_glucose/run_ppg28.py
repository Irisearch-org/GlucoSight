"""Ticket #28: does smartphone PPG track WITHIN-person glucose change; is per-user calibration viable?

Subjects with two distinct glycaemia values (sessions A = earlier record numbers, B = later), still recordings.
  (a) delta features (B - A) vs delta glucose: Spearman per feature (Holm), leave-one-subject-out
      delta-ridge (no intercept, antisymmetric) vs the zero-change baseline, sign accuracy.
  (b) calibration simulation: pred_B = model(B) + (glucose_A - model(A)) with a cross-subject model
      trained without that subject, vs carry-over (pred_B = glucose_A) and vs the training mean.

    python run_ppg28.py [N_PERM_RIDGE] [N_PERM_RF] [N_PERM_PAPAGEI]
"""
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import binomtest, spearmanr

from common import RANDOM_STATE, RESULTS_DIR, load_cache
from eval_lib import FITS, N_BOOTSTRAP, N_JOBS, permuted_labels, run_cv

FEATS = {"hand": "h_bright_raw", "pap": "e_bright_inv"}
ALPHAS = np.logspace(-1, 4, 6)
N_PERM_DELTA = 2000


# ---------------------------------------------------------------- data
def session_frame(c, mask):
    """One row per (subject, session) over `mask` rows: subject, order, glucose, mean features."""
    sid = c["session_id"][mask]
    ids, inv = np.unique(sid, return_inverse=True)
    cnt = np.bincount(inv)
    out = {"session_id": ids, "n_rec": cnt}
    first = np.zeros(len(ids), int); first[inv] = np.arange(mask.sum())
    for k in ("subj", "sess_order", "glucose"):
        out[k] = c[k][mask][first]
    for name, key in FEATS.items():
        X = c[key][mask]
        out[name] = np.stack([np.bincount(inv, weights=X[:, j]) / cnt for j in range(X.shape[1])], 1)
    return out


def pair_table(sf):
    """Per two-session subject: A (order 0) and B (order 1) rows."""
    rows = []
    for s in np.unique(sf["subj"]):
        m = np.flatnonzero(sf["subj"] == s)
        if len(m) != 2:
            continue
        a, b = m[np.argsort(sf["sess_order"][m])]
        if sf["glucose"][a] == sf["glucose"][b]:
            continue
        rows.append((s, a, b))
    return rows


# ---------------------------------------------------------------- (a) delta ridge
def _fit_delta_ridge(dX, dy, alpha):
    """Antisymmetric: train on (dX, dy) and (-dX, -dy), no intercept, features scaled by their RMS.
    Closed-form ridge (dual form when there are more features than rows)."""
    sd = np.sqrt(np.mean(dX ** 2, axis=0)) + 1e-12
    Z = np.vstack([dX, -dX]) / sd
    t = np.concatenate([dy, -dy])
    if Z.shape[1] > Z.shape[0]:
        coef = Z.T @ np.linalg.solve(Z @ Z.T + alpha * np.eye(len(t)), t)
    else:
        coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ t)
    return lambda d: (d / sd) @ coef


def delta_loso(dX, dy):
    """Leave-one-subject-out predictions; alpha chosen by inner leave-one-subject-out MAE."""
    n = len(dy)
    pred = np.empty(n)
    for i in range(n):
        tr = np.flatnonzero(np.arange(n) != i)
        best, best_err = None, np.inf
        for a in ALPHAS:
            err = 0.0
            for j in tr:
                tr2 = tr[tr != j]
                err += abs(_fit_delta_ridge(dX[tr2], dy[tr2], a)(dX[j:j + 1])[0] - dy[j])
            if err < best_err:
                best, best_err = a, err
        pred[i] = _fit_delta_ridge(dX[tr], dy[tr], best)(dX[i:i + 1])[0]
    return pred


def _delta_perm(args):
    dX, dy_perm = args
    p = delta_loso(dX, dy_perm)
    return float(np.mean(np.abs(dy_perm)) - np.mean(np.abs(dy_perm - p))), float(np.mean(np.sign(p) == np.sign(dy_perm)))


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def analysis_a(sf, pairs, names, n_perm, n_jobs=N_JOBS):
    a = np.array([p[1] for p in pairs]); b = np.array([p[2] for p in pairs])
    dy = sf["glucose"][b] - sf["glucose"][a]
    res = {"n_subjects": len(pairs), "delta_glucose_mgdl": {
        "abs_median": round(float(np.median(np.abs(dy))), 1),
        "abs_q25_q75": [round(float(np.percentile(np.abs(dy), 25)), 1), round(float(np.percentile(np.abs(dy), 75)), 1)],
        "abs_min_max": [round(float(np.abs(dy).min()), 1), round(float(np.abs(dy).max()), 1)],
        "n_abs_lt_10": int((np.abs(dy) < 10).sum()), "n_abs_lt_18": int((np.abs(dy) < 18).sum()),
        "n_positive": int((dy > 0).sum()), "n_negative": int((dy < 0).sum()),
        "mean_abs": round(float(np.mean(np.abs(dy))), 2)}}
    # per-feature Spearman, Holm
    dX = sf["hand"][b] - sf["hand"][a]
    rho, praw = zip(*[spearmanr(dX[:, j], dy) for j in range(dX.shape[1])])
    padj = holm(np.array(praw))
    res["spearman_hand"] = [{"feature": n, "rho": round(float(r), 3), "p": round(float(p), 4), "p_holm": round(float(q), 4)}
                            for n, r, p, q in zip(names, rho, praw, padj)]
    out = {}
    for tag, key in (("hand", "hand"), ("pap", "pap")):
        D = sf[key][b] - sf[key][a]
        pred = delta_loso(D, dy)
        mae, base = float(np.mean(np.abs(dy - pred))), float(np.mean(np.abs(dy)))
        k = int((np.sign(pred) == np.sign(dy)).sum())
        maj = max((dy > 0).sum(), (dy < 0).sum())
        rng = np.random.default_rng(RANDOM_STATE + 7)
        perms = [(D, rng.permutation(dy)) for _ in range(n_perm)]
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
            null = np.array(list(ex.map(_delta_perm, perms, chunksize=20)))
        obs_gain = base - mae
        # bootstrap CI of the MAE gain over subjects
        brng = np.random.default_rng(RANDOM_STATE)
        ae_b, ae_m = np.abs(dy), np.abs(dy - pred)
        bs = np.array([(ae_b[i] - ae_m[i]).mean() for i in (brng.integers(0, len(dy), len(dy)) for _ in range(N_BOOTSTRAP))])
        out[tag] = {
            "mae_delta_model": round(mae, 2), "mae_delta_zero_baseline": round(base, 2),
            "gain": round(obs_gain, 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
            "subjects_model_better": int((ae_m < ae_b).sum()),
            "sign_correct": f"{k}/{len(dy)}", "sign_acc": round(k / len(dy), 3),
            "majority_sign_rate": round(float(maj / len(dy)), 3),
            "binom_p_two_sided": round(float(binomtest(k, len(dy), 0.5).pvalue), 4),
            "perm_p_sign_acc": round(float((1 + np.sum(null[:, 1] >= k / len(dy))) / (1 + n_perm)), 4),
            "perm_p_gain": round(float((1 + np.sum(null[:, 0] >= obs_gain)) / (1 + n_perm)), 4),
            "n_perm": n_perm,
            "pearson_pred_vs_true_delta": round(float(np.corrcoef(pred, dy)[0, 1]), 3),
            "pred_delta": [round(float(v), 1) for v in pred], "true_delta": [round(float(v), 1) for v in dy],
        }
    res["delta_ridge_loso"] = out
    return res


# ---------------------------------------------------------------- (b) calibration simulation
def loso_recording_preds(fit, X, y, g):
    pred, base, _ = run_cv(fit, X, y, g, loso=True)
    return pred[0], base[0]


def calib_pairs(pred, base, y, g, sess, sess_order, eval_subjects):
    """Per (subject, ordering): errors of calibrated / carry-over / mean / uncalibrated model."""
    rows = []
    ids, inv = np.unique(sess, return_inverse=True)
    cnt = np.bincount(inv)
    Pm = np.bincount(inv, weights=pred) / cnt
    Bm = np.bincount(inv, weights=base) / cnt
    gs = np.zeros(len(ids)); gs[inv] = y
    ss = np.zeros(len(ids), int); ss[inv] = g
    so = np.zeros(len(ids), int); so[inv] = sess_order
    for s in eval_subjects:
        m = np.flatnonzero(ss == s)
        a, b = m[np.argsort(so[m])]
        for src, dst, label in ((a, b, "A>B"), (b, a, "B>A")):
            cal = Pm[dst] + (gs[src] - Pm[src])
            rows.append({"subject": int(s), "order": label, "g_src": gs[src], "g_dst": gs[dst],
                         "model_dst": Pm[dst], "model_src": Pm[src], "calibrated": cal,
                         "carry": gs[src], "mean": Bm[dst]})
    return rows


def _sim_perm(args):
    fit, X, y_perm, g, sess, so, eval_subjects = args
    pred, base = loso_recording_preds(fit, X, y_perm, g)
    rows = calib_pairs(pred, base, y_perm, g, sess, so, eval_subjects)
    carry = np.array([abs(r["carry"] - r["g_dst"]) for r in rows])
    cal = np.array([abs(r["calibrated"] - r["g_dst"]) for r in rows])
    mean = np.array([abs(r["mean"] - r["g_dst"]) for r in rows])
    return float(carry.mean() - cal.mean()), float(mean.mean() - cal.mean())


def permute_eval_labels(y, g, sess, so, eval_subjects, rng):
    """Shuffle the (A, B) label profiles among eval subjects only; others keep their labels."""
    ids, inv = np.unique(sess, return_inverse=True)
    gs = np.zeros(len(ids)); gs[inv] = y
    ss = np.zeros(len(ids), int); ss[inv] = g
    sidx = np.zeros(len(ids), int); sidx[inv] = so
    keep = np.isin(ss, eval_subjects)
    new = gs.copy()
    new[keep] = permuted_labels(gs[keep], ss[keep], sidx[keep], rng)
    return new[inv]


def boot_ci(diff_by_subject, n_boot=N_BOOTSTRAP):
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(diff_by_subject)
    d = np.array([diff_by_subject[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def analysis_b(c, mask, feat, model, eval_subjects, n_perm, delta_pred_by_subject=None, n_jobs=N_JOBS):
    X = c[FEATS[feat]][mask]
    y, g, sess, so = c["glucose"][mask], c["subj"][mask], c["session_id"][mask], c["sess_order"][mask]
    fit = FITS[model]
    pred, base = loso_recording_preds(fit, X, y, g)
    rows = calib_pairs(pred, base, y, g, sess, so, eval_subjects)
    subj = np.array([r["subject"] for r in rows])
    err = {k: np.array([abs(r[k] - r["g_dst"]) for r in rows]) for k in ("calibrated", "carry", "mean", "model_dst")}
    pm_delta = np.array([r["model_dst"] - r["model_src"] for r in rows])
    true_delta = np.array([r["g_dst"] - r["g_src"] for r in rows])
    out = {"n_pairs": len(rows), "n_subjects": len(eval_subjects),
           "mae": {k: round(float(v.mean()), 2) for k, v in err.items()},
           "model_delta_vs_true_delta": {
               "pearson": round(float(np.corrcoef(pm_delta, true_delta)[0, 1]), 3),
               "spearman": round(float(spearmanr(pm_delta, true_delta)[0]), 3),
               "sign_correct": int((np.sign(pm_delta) == np.sign(true_delta)).sum()), "of": len(rows),
               "pred_delta_sd": round(float(pm_delta.std()), 2), "true_delta_sd": round(float(true_delta.std()), 2)}}
    subs = np.array(eval_subjects)

    def by_subject(v):
        return np.array([v[subj == s].mean() for s in subs])

    for baseline in ("carry", "mean"):
        diff = by_subject(err[baseline] - err["calibrated"])      # positive = calibration better
        d = {"gain": round(float((err[baseline] - err["calibrated"]).mean()), 3),
             "ci95": boot_ci(diff), "subjects_calibration_better": int((diff > 0).sum()), "of_subjects": len(subs)}
        out[f"calibrated_vs_{baseline}"] = d
    if n_perm:
        rng = np.random.default_rng(RANDOM_STATE + 11)
        jobs = [(fit, X, permute_eval_labels(y, g, sess, so, eval_subjects, rng), g, sess, so, eval_subjects)
                for _ in range(n_perm)]
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
            null = np.array(list(ex.map(_sim_perm, jobs, chunksize=1)))
        for i, baseline in enumerate(("carry", "mean")):
            obs = out[f"calibrated_vs_{baseline}"]["gain"]
            out[f"calibrated_vs_{baseline}"]["perm_p"] = round(float((1 + np.sum(null[:, i] >= obs)) / (1 + n_perm)), 4)
        out["permutations"] = n_perm
    d = out["calibrated_vs_carry"]
    if n_perm:
        crit = {"gain_positive": d["gain"] > 0,
                "subjects_better_ge_80pct": d["subjects_calibration_better"] / d["of_subjects"] >= 0.8,
                "ci_excludes_zero": d["ci95"][0] > 0, "perm_p_le_0.05": d["perm_p"] <= 0.05}
        out["gate_vs_carry"] = {"criteria": {k: bool(v) for k, v in crit.items()}, "passes": bool(all(crit.values()))}
    out["rows"] = [{k: (round(float(v), 1) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in rows]
    return out, rows


def delta_ridge_as_calibration(sf, pairs, key="hand"):
    """Within-subject delta-ridge calibration: pred_B = g_A + delta_hat (and the reverse ordering)."""
    a = np.array([p[1] for p in pairs]); b = np.array([p[2] for p in pairs])
    D = sf[key][b] - sf[key][a]
    dy = sf["glucose"][b] - sf["glucose"][a]
    dhat = delta_loso(D, dy)
    err_cal = np.abs(dy - dhat)           # A>B and B>A errors are identical (antisymmetric, no intercept)
    return {"mae": round(float(err_cal.mean()), 2), "carry_mae": round(float(np.abs(dy).mean()), 2),
            "gain": round(float((np.abs(dy) - err_cal).mean()), 3), "ci95": boot_ci(np.abs(dy) - err_cal),
            "subjects_better": int((err_cal < np.abs(dy)).sum()), "of_subjects": len(dy)}


def main():
    n_ridge = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    n_rf = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n_pap = int(sys.argv[3]) if len(sys.argv) > 3 else 300   # 512-d LOSO nested ridge is ~20 s per permutation
    os.makedirs(RESULTS_DIR, exist_ok=True)
    c = load_cache()
    names = [str(n) for n in c["feature_names"]]
    base_ok = np.isfinite(c["h_bright_raw"]).all(1) & np.isfinite(c["e_bright_inv"]).all(1)
    out = {}
    jpath = os.path.join(RESULTS_DIR, "ppg28.json")
    if os.path.exists(jpath):          # resume: keep finished parts
        out = json.load(open(jpath))
    cohorts = {
        "finger_still": (c["site"] == "finger") & c["still"] & base_ok,
        "ear_still": (c["site"] == "ear") & c["still"] & base_ok,
        "finger_and_ear_still": c["still"] & base_ok,
    }
    for coh, mask in cohorts.items():
        t0 = time.time()
        sf = session_frame(c, mask)
        pairs = pair_table(sf)
        eval_subjects = [p[0] for p in pairs]
        print(coh, "recordings", int(mask.sum()), "subjects", len(np.unique(c["subj"][mask])),
              "two-session subjects", len(pairs), flush=True)
        r = out.get(coh) or {"n_recordings": int(mask.sum()), "n_subjects_total": int(len(np.unique(c["subj"][mask]))),
                             "n_two_session_subjects": len(pairs),
                             "recordings_per_session": {"min": int(sf["n_rec"].min()), "median": float(np.median(sf["n_rec"])),
                                                        "max": int(sf["n_rec"].max())}}
        if "a_delta_features" not in r:
            r["a_delta_features"] = analysis_a(sf, pairs, names, N_PERM_DELTA)
            r["delta_ridge_as_calibration"] = delta_ridge_as_calibration(sf, pairs)
        r.setdefault("b_calibration", {})
        print(coh, "(a) done", round(time.time() - t0), "s", flush=True)
        models = [("hand", "ridge", n_ridge), ("pap", "ridge", n_pap), ("hand", "rf", n_rf)]
        if coh != "finger_still":
            models = [("hand", "ridge", n_ridge), ("pap", "ridge", 0)]
        for feat, model, npm in models:
            if f"{feat}+{model}" in r["b_calibration"]:
                continue
            res, _ = analysis_b(c, mask, feat, model, eval_subjects, npm)
            r["b_calibration"][f"{feat}+{model}"] = res
            print(coh, feat, model, res["mae"], res["calibrated_vs_carry"], res.get("gate_vs_carry", {}).get("passes"),
                  round(time.time() - t0), "s", flush=True)
            out[coh] = r
            json.dump(out, open(os.path.join(RESULTS_DIR, "ppg28.json"), "w"), indent=1)
        out[coh] = r
        json.dump(out, open(os.path.join(RESULTS_DIR, "ppg28.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
