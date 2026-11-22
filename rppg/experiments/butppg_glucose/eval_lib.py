"""Subject-grouped repeated CV, baselines on the same folds, ADR 0004 gate statistics.

Effective n = people. Everything resamples or permutes SUBJECTS (C4); the baseline is the
training-fold mean on identical folds (C3); hyperparameters are tuned only on training folds
by the nested fits imported from ppg_served_model/candidates.py.
"""
import importlib.util
import os
import sys

import numpy as np

from common import REPO_ROOT, RANDOM_STATE
from utils.clarke_grid import clarke_error_grid, per_participant_zone_a, zone_a_by_reference_range

_spec = importlib.util.spec_from_file_location(
    "served_candidates",
    os.path.join(REPO_ROOT, "rppg", "experiments", "ppg_served_model", "candidates.py"))
cand = importlib.util.module_from_spec(_spec)
sys.modules["served_candidates"] = cand      # so the fit functions pickle into worker processes
_spec.loader.exec_module(cand)

fit_mean, fit_ridge_nested, fit_rf_nested = cand.fit_mean, cand.fit_ridge_nested, cand.fit_rf_nested
class _StdRidge:
    """Standardise + ridge with intercept, fitted for a whole alpha grid from one Gram matrix."""

    def __init__(self, mu, sd, coef, intercept):
        self.mu, self.sd, self.coef, self.intercept = mu, sd, coef, intercept

    def predict(self, X):
        return ((np.asarray(X, float) - self.mu) / self.sd) @ self.coef + self.intercept


def _gram_path(X, y, alphas):
    """Fit StandardScaler + Ridge(alpha) for every alpha with one eigendecomposition."""
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd == 0, 1.0, sd)            # sklearn StandardScaler: zero variance -> scale 1
    Z = (X - mu) / sd
    ym = y.mean()
    w, V = np.linalg.eigh(Z.T @ Z)
    Vb = V.T @ (Z.T @ (y - ym))
    return [_StdRidge(mu, sd, V @ (Vb / (w + a)), ym) for a in alphas]


def fit_ridge_fast(X, y, groups):
    """Numerically the same as ppg_served_model.candidates.fit_ridge_nested (same alpha grid, same
    inner GroupKFold, same pooled-MAE rule) but ~6x cheaper: one eigendecomposition per training set."""
    k = min(cand.INNER_SPLITS, len(np.unique(groups)))
    alphas = list(cand.RIDGE_ALPHAS)
    errs = np.zeros(len(alphas))
    from sklearn.model_selection import GroupKFold
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        for j, m in enumerate(_gram_path(X[tr], y[tr], alphas)):
            errs[j] += np.abs(m.predict(X[te]) - y[te]).sum()
    best = int(np.argmin(errs))
    m = _gram_path(X, y, [alphas[best]])[0]
    m.tuned_ = {"alpha": float(alphas[best])}
    return m


FITS = {"ridge": fit_ridge_fast, "rf": fit_rf_nested}

N_JOBS = int(os.environ.get("BUTPPG_JOBS", 8))
N_SPLITS = 5
N_REPEATS = 5
N_BOOTSTRAP = 2000
MIN_FOLD_WIN_FRAC = 0.8     # ADR 0004 "most folds": 4 of 5 -> 20 of 25 folds
MAX_PERM_P = 0.05


def subject_folds(subjects, n_splits, seed):
    """Random subject->fold assignment (sizes differ by at most one subject)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(subjects))
    return [perm[i::n_splits] for i in range(n_splits)]


def run_cv(fit, X, y, groups, repeats=N_REPEATS, n_splits=N_SPLITS, seed0=RANDOM_STATE, loso=False):
    """Out-of-fold model and training-mean predictions, shape (R, n), plus fold ids (R, n)."""
    subjects = np.unique(groups)
    n = len(y)
    R = 1 if loso else repeats
    pred, base, fold = np.empty((R, n)), np.empty((R, n)), np.zeros((R, n), dtype=int)
    for r in range(R):
        folds = [np.array([s]) for s in subjects] if loso else subject_folds(subjects, n_splits, seed0 + r)
        for k, held in enumerate(folds):
            te = np.isin(groups, held)
            tr = ~te
            assert not (set(groups[tr]) & set(groups[te]))
            m = fit(X[tr], y[tr], groups[tr])
            pred[r, te] = m.predict(X[te])
            base[r, te] = y[tr].mean()
            fold[r, te] = k
    return pred, base, fold


def session_means(v, sessions):
    """(R, n) recording values -> (R, n_sessions) means, plus the ordered session ids."""
    ids, inv = np.unique(sessions, return_inverse=True)
    cnt = np.bincount(inv)
    out = np.stack([np.bincount(inv, weights=row) / cnt for row in np.atleast_2d(v)])
    return out, ids, inv


def session_table(y, groups, sessions):
    ids, inv = np.unique(sessions, return_inverse=True)
    ys = np.zeros(len(ids))
    gs = np.zeros(len(ids), dtype=groups.dtype)
    ys[inv] = y
    gs[inv] = groups
    return ids, inv, ys, gs


def _subject_sums(ae_row_avg, groups, subjects):
    return np.array([ae_row_avg[groups == s].sum() for s in subjects]), \
        np.array([(groups == s).sum() for s in subjects], dtype=float)


def bootstrap_gain(ae_base, ae_model, groups, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    """95% CI of mean(ae_base) - mean(ae_model), resampling SUBJECTS. ae_* are per-unit
    (recording or session) errors already averaged over CV repeats."""
    rng = np.random.default_rng(seed)
    subjects = np.unique(groups)
    sb, cnt = _subject_sums(ae_base, groups, subjects)
    sm, _ = _subject_sums(ae_model, groups, subjects)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, len(subjects), len(subjects))
        n = cnt[idx].sum()
        diffs[k] = (sb[idx].sum() - sm[idx].sum()) / n
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def permuted_labels(session_labels, subj_of_session, sess_index, rng):
    """Shuffle label profiles BETWEEN subjects (donor's j-th session label -> recipient's j-th).

    session_labels: per-session glucose; subj_of_session: subject per session;
    sess_index: per-session order within subject. Returns permuted per-session labels.
    """
    subs = np.unique(subj_of_session)
    prof = {s: session_labels[subj_of_session == s][np.argsort(sess_index[subj_of_session == s])]
            for s in subs}
    order = rng.permutation(len(subs))
    out = np.empty_like(session_labels)
    for dst, src in zip(subs, subs[order]):
        m = np.flatnonzero(subj_of_session == dst)
        m = m[np.argsort(sess_index[m])]
        donor = prof[src]
        out[m] = donor[np.arange(len(m)) % len(donor)]
    return out


def _gains(y, pred, base, groups, sessions):
    """Mean over repeats of recording-level and session-level MAE gain (baseline - model)."""
    rec = float(np.mean(np.abs(y - base)) - np.mean(np.abs(y - pred)))
    ids, inv, ys, gs = session_table(y, groups, sessions)
    ps, _, _ = session_means(pred, sessions)
    bs, _, _ = session_means(base, sessions)
    ses = float(np.mean(np.abs(ys - bs)) - np.mean(np.abs(ys - ps)))
    return rec, ses


def _perm_job(args):
    fit, X, y_sess_perm, inv, groups, sessions, seed = args
    y = y_sess_perm[inv]
    pred, base, _ = run_cv(fit, X, y, groups, repeats=1, seed0=seed)
    rec, ses = _gains(y, pred[0], base[0], groups, sessions)
    return rec, ses


def permutation_test(fit, X, y, groups, sessions, sess_index_rec, n_perm, observed, n_jobs=N_JOBS):
    """Null distribution of gain over the mean with labels shuffled between subjects.
    Re-runs the full nested fit per permutation (1 CV repeat each). Returns p for rec and session level."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    ids, inv, ys, gs = session_table(y, groups, sessions)
    sidx = np.zeros(len(ids), dtype=int)
    sidx[inv] = sess_index_rec
    rng = np.random.default_rng(RANDOM_STATE + 1)
    jobs = [(fit, X, permuted_labels(ys, gs, sidx, rng)[0:len(ids)], inv, groups, sessions,
             RANDOM_STATE + 1000 + i) for i in range(n_perm)]
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
        res = list(ex.map(_perm_job, jobs, chunksize=1))
    res = np.array(res)
    p_rec = (1 + np.sum(res[:, 0] >= observed[0])) / (1 + n_perm)
    p_ses = (1 + np.sum(res[:, 1] >= observed[1])) / (1 + n_perm)
    return float(p_rec), float(p_ses), res


def clarke_summary(y, pred, groups):
    """Mean over repeats: pooled zone A %, per-subject zone A median/IQR, A by reference range."""
    R = pred.shape[0]
    pooled, med, iqr = [], [], []
    ranges = {}
    for r in range(R):
        pooled.append(clarke_error_grid(y, pred[r])[1]["A"])
        ps = per_participant_zone_a(y, pred[r], groups, min_meals=2)
        med.append(ps["median_zone_a"])
        iqr.append(ps["iqr_zone_a"])
        for row in zone_a_by_reference_range(y, pred[r]):
            d = ranges.setdefault(row["range"], {"n": [], "a": []})
            d["n"].append(row["n"])
            if row["zone_a_pct"] is not None:
                d["a"].append(row["zone_a_pct"])
    return {
        "zone_a_pooled_pct": round(float(np.mean(pooled)), 1),
        "zone_a_per_subject_median": round(float(np.mean(med)), 1),
        "zone_a_per_subject_iqr": round(float(np.mean(iqr)), 1),
        "zone_a_by_range": {k: {"n_recordings": int(round(np.mean(v["n"]))),
                                "zone_a_pct": round(float(np.mean(v["a"])), 1) if v["a"] else None}
                            for k, v in ranges.items()},
    }


def pearson(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize(fit, X, y, groups, sessions, sess_index, n_perm=0, loso=False, n_jobs=N_JOBS):
    """Run the grouped CV for one model and compute every reported statistic."""
    pred, base, fold = run_cv(fit, X, y, groups, loso=loso)
    R = pred.shape[0]
    ids, inv, ys, gs = session_table(y, groups, sessions)
    ps, _, _ = session_means(pred, sessions)
    bs, _, _ = session_means(base, sessions)

    ae_m = np.abs(y - pred).mean(0)           # per-recording, mean over repeats
    ae_b = np.abs(y - base).mean(0)
    ae_ms = np.abs(ys - ps).mean(0)
    ae_bs = np.abs(ys - bs).mean(0)
    gains = _gains(y, pred, base, groups, sessions)

    # fold wins (every outer fold of every repeat), recording level
    wins, nfold = 0, 0
    for r in range(R):
        for k in np.unique(fold[r]):
            m = fold[r] == k
            nfold += 1
            wins += int(np.mean(np.abs(y[m] - pred[r][m])) < np.mean(np.abs(y[m] - base[r][m])))
    lo, hi = bootstrap_gain(ae_b, ae_m, groups)
    lo_s, hi_s = bootstrap_gain(ae_bs, ae_ms, gs)

    out = {
        "n_recordings": int(len(y)), "n_sessions": int(len(ids)), "n_subjects": int(len(np.unique(groups))),
        "cv": "LOSO" if loso else f"{N_SPLITS}-fold x {R} repeats, subject-grouped",
        "recording": {
            "mae": round(float(ae_m.mean()), 2), "baseline_mae": round(float(ae_b.mean()), 2),
            "rmse": round(float(np.mean([np.sqrt(np.mean((y - pred[r]) ** 2)) for r in range(R)])), 2),
            "baseline_rmse": round(float(np.mean([np.sqrt(np.mean((y - base[r]) ** 2)) for r in range(R)])), 2),
            "pearson_r": round(float(np.nanmean([pearson(pred[r], y) for r in range(R)])), 3),
            "gain": round(gains[0], 3), "ci95": [round(lo, 3), round(hi, 3)],
            "fold_wins": [int(wins), int(nfold)],
        },
        "session": {
            "mae": round(float(ae_ms.mean()), 2), "baseline_mae": round(float(ae_bs.mean()), 2),
            "rmse": round(float(np.mean([np.sqrt(np.mean((ys - ps[r]) ** 2)) for r in range(R)])), 2),
            "pearson_r": round(float(np.nanmean([pearson(ps[r], ys) for r in range(R)])), 3),
            "gain": round(gains[1], 3), "ci95": [round(lo_s, 3), round(hi_s, 3)],
        },
        "pred_sd_mgdl": round(float(np.mean([pred[r].std() for r in range(R)])), 2),
        "clarke": clarke_summary(y, pred, groups),
        "clarke_baseline": clarke_summary(y, base, groups),
    }
    if n_perm:
        p_rec, p_ses, null = permutation_test(fit, X, y, groups, sessions, sess_index, n_perm, gains, n_jobs)
        out["permutations"] = n_perm
        out["recording"]["perm_p"] = round(p_rec, 4)
        out["session"]["perm_p"] = round(p_ses, 4)
        crit = {
            "gain_positive": gains[0] > 0,
            "fold_wins_ge_80pct": wins / nfold >= MIN_FOLD_WIN_FRAC,
            "ci_excludes_zero": lo > 0,
            "perm_p_le_0.05": p_rec <= MAX_PERM_P,
        }
        out["gate_recording"] = {"criteria": {k: bool(v) for k, v in crit.items()},
                                 "passes": bool(all(crit.values()))}
        crit_s = {"gain_positive": gains[1] > 0, "ci_excludes_zero": lo_s > 0,
                  "perm_p_le_0.05": p_ses <= MAX_PERM_P}
        out["gate_session"] = {"criteria": {k: bool(v) for k, v in crit_s.items()},
                               "passes": bool(all(crit_s.values()) and wins / nfold >= MIN_FOLD_WIN_FRAC)}
    return out, pred, base
