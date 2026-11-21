"""Promotes nlp/eval_dish_identity.py's M3_dish result -- M3 MAE 30.500 vs
M2_shipped's 31.148 mg/dL on ShanghaiT2DM, a gain that survives the M3p
permutation control at 31.252 -- from an offline eval script into a
gated artifact. Contract v2 does not serve it (the NLP path is retired).

Gated the same way `rppg/models/glucose_estimator.py` gates the PPG
estimator (ADR 0004: reuse that gate, don't invent a new evaluation
standard): a positive improvement, winning most folds, a bootstrap CI
over SUBJECTS excluding zero, and a subject-grouped label-permutation
test. If the gate fails, nothing is persisted.

Usage
-----
    python -m nlp.models.dish_identity_ridge --train
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from nlp.data.splits import grouped_folds
from nlp.eval_dish_identity import (
    CONTINUOUS,
    DEFAULT_LEGACY,
    DEFAULT_RESPONSE,
    FLAGS,
    MIN_DISH_MEALS,
    TARGET,
    _fit_and_predict,
    _standardise,
    dish_matrix,
    load_meals,
)
from rppg.models.glucose_estimator import (
    Evaluation,
    FoldResult,
    N_BOOTSTRAP,
    N_PERMUTATIONS,
    RANDOM_STATE,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(_HERE, "dish_identity_ridge.json")
REPORT_PATH = os.path.join(_HERE, "reports", "dish_identity_ridge.md")


# ---------------------------------------------------------------------
# M2 (shipped) vs M3 (+ dish) held-out predictions, per meal
# ---------------------------------------------------------------------

def _fold_predictions(
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    continuous: np.ndarray,
    flags: np.ndarray,
    groups: np.ndarray,
    dishes: np.ndarray,
    min_dish_meals: int,
) -> Tuple[np.ndarray, np.ndarray, List[FoldResult]]:
    pred_m2 = np.full(len(y), np.nan)
    pred_m3 = np.full(len(y), np.nan)
    fold_results: List[FoldResult] = []
    for i, (train, test) in enumerate(folds, start=1):
        # C4: held-out subjects must be genuinely unseen.
        assert not (set(groups[train]) & set(groups[test])), "subject leak across folds"

        scaled_train, scaled_test = _standardise(continuous[train], continuous[test])
        base_train = np.hstack([scaled_train, flags[train]])
        base_test = np.hstack([scaled_test, flags[test]])

        pm2, _ = _fit_and_predict(base_train, y[train], groups[train], base_test)

        frequency = dishes[train].sum(axis=0)
        columns = np.flatnonzero(frequency >= min_dish_meals)
        pm3, _ = _fit_and_predict(
            np.hstack([base_train, dishes[np.ix_(train, columns)]]),
            y[train], groups[train],
            np.hstack([base_test, dishes[np.ix_(test, columns)]]),
        )

        pred_m2[test] = pm2
        pred_m3[test] = pm3
        fold_results.append(FoldResult(
            fold=i, n_test=len(test), n_test_subjects=len(set(groups[test])),
            baseline_mae=float(np.mean(np.abs(y[test] - pm2))),
            model_mae=float(np.mean(np.abs(y[test] - pm3))),
        ))
    return pred_m2, pred_m3, fold_results


def evaluate_dish_addition(
    frame: pd.DataFrame,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    min_dish_meals: int = MIN_DISH_MEALS,
    run_bootstrap: bool = True,
    run_permutation: bool = True,
    n_bootstrap: int = N_BOOTSTRAP,
    n_permutations: int = N_PERMUTATIONS,
    random_state: int = RANDOM_STATE,
) -> Evaluation:
    """M3 (M2 + dish indicators) vs M2 (the shipped pipeline: g0 + grams +
    cooking flags), scored on subject-grouped folds, gated by
    `Evaluation.passes_gate()` (imported from rppg's estimator, not
    redeclared)."""
    y = frame[TARGET].to_numpy(dtype=float)
    continuous = frame[list(CONTINUOUS)].to_numpy(dtype=float)
    flags = frame[list(FLAGS)].to_numpy(dtype=float)
    groups = frame["subject_id"].to_numpy()
    dishes, _vocabulary = dish_matrix(frame)

    pred_m2, pred_m3, fold_results = _fold_predictions(
        folds, y, continuous, flags, groups, dishes, min_dish_meals
    )
    ev = Evaluation(folds=fold_results, y_true=y, y_pred=pred_m3,
                     y_baseline=pred_m2, groups=groups)

    if run_bootstrap:
        rng = np.random.default_rng(random_state)
        subjects = np.unique(groups)
        by_subject = {s: np.flatnonzero(groups == s) for s in subjects}
        diffs = []
        for _ in range(n_bootstrap):
            picked = rng.choice(subjects, size=len(subjects), replace=True)
            mask = np.concatenate([by_subject[s] for s in picked])
            diffs.append(
                float(np.mean(np.abs(y[mask] - pred_m2[mask])))
                - float(np.mean(np.abs(y[mask] - pred_m3[mask])))
            )
        ev.bootstrap_ci = (float(np.percentile(diffs, 2.5)),
                           float(np.percentile(diffs, 97.5)))

    if run_permutation:
        observed = ev.improvement
        rng = np.random.default_rng(random_state + 1)
        subjects = np.unique(groups)
        subj_labels = {s: y[groups == s] for s in subjects}
        keys = list(subj_labels)
        count = 0
        for _ in range(n_permutations):
            order = rng.permutation(len(keys))
            y_perm = np.empty_like(y)
            for src, dst in zip(order, range(len(keys))):
                idx = np.flatnonzero(groups == keys[dst])
                donor = subj_labels[keys[src]]
                y_perm[idx] = rng.choice(donor, size=len(idx), replace=True)
            perm_m2, perm_m3, _ = _fold_predictions(
                folds, y_perm, continuous, flags, groups, dishes, min_dish_meals
            )
            perm_improvement = (
                float(np.mean(np.abs(y_perm - perm_m2)))
                - float(np.mean(np.abs(y_perm - perm_m3)))
            )
            if perm_improvement >= observed:
                count += 1
        ev.permutation_p = (count + 1) / (n_permutations + 1)

    return ev


# ---------------------------------------------------------------------
# full-data fit (only reached once the gate above has passed)
# ---------------------------------------------------------------------

def _fit_full(frame: pd.DataFrame, min_dish_meals: int) -> Dict[str, Any]:
    """Fit M3 on the full committed set -- no held-out split, since the
    gate already proved the held-out gain is real. Penalty chosen by the
    same inner subject-grouped CV `_fit_and_predict` uses, not a fixed
    value, by calling it with train==test: it both selects alpha and
    returns that alpha, and we refit a plain Ridge at that alpha to get
    coefficients to persist (`_fit_and_predict` returns predictions, not
    the fitted estimator)."""
    from sklearn.linear_model import Ridge

    y = frame[TARGET].to_numpy(dtype=float)
    continuous = frame[list(CONTINUOUS)].to_numpy(dtype=float)
    flags = frame[list(FLAGS)].to_numpy(dtype=float)
    groups = frame["subject_id"].to_numpy()
    dishes, vocabulary = dish_matrix(frame)

    mean = continuous.mean(axis=0)
    scale = continuous.std(axis=0)
    scale[scale == 0] = 1.0
    scaled = (continuous - mean) / scale
    base = np.hstack([scaled, flags])

    frequency = dishes.sum(axis=0)
    columns = np.flatnonzero(frequency >= min_dish_meals)
    chosen_vocabulary = [vocabulary[c] for c in columns]
    X = np.hstack([base, dishes[:, columns]])

    _, best_alpha = _fit_and_predict(X, y, groups, X)
    model = Ridge(alpha=best_alpha).fit(X, y)

    return {
        "continuous_names": list(CONTINUOUS),
        "flag_names": list(FLAGS),
        "dish_vocabulary": chosen_vocabulary,
        "min_dish_meals": int(min_dish_meals),
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "alpha": float(best_alpha),
    }


def _write_report(frame, ev: Evaluation, passed: bool, reasons: List[str], path: str) -> None:
    """No Clarke grid here (unlike rppg's report) -- the target is a
    glucose RISE (a delta), and Clarke's zones are defined on absolute
    reference/prediction pairs, not deltas."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# NLP Dish-Identity Ridge — held-out evaluation",
        "",
        f"**Gate: {'PASSED' if passed else 'FAILED'}**",
        "",
        f"M3 (M2 + per-dish indicators) vs M2 (g0 + total_grams + cooking "
        f"flags), {len(ev.folds)}-fold GroupKFold by subject_id, on the "
        f"{TARGET} target (ShanghaiT2DM).",
        f"n = {len(frame)} meals, {frame['subject_id'].nunique()} subjects.",
        "",
        "## Results",
        "",
        "| Predictor | MAE | RMSE |",
        "|---|---|---|",
        f"| M2 (shipped) | {ev.baseline_mae:.3f} | "
        f"{np.sqrt(np.mean((ev.y_true - ev.y_baseline) ** 2)):.3f} |",
        f"| M3 (+ dish) | {ev.model_mae:.3f} | "
        f"{np.sqrt(np.mean((ev.y_true - ev.y_pred) ** 2)):.3f} |",
        "",
        "## Per fold",
        "",
        "| Fold | n | subjects | M2 MAE | M3 MAE | Improvement |",
        "|---|---|---|---|---|---|",
    ]
    for f in ev.folds:
        lines.append(
            f"| {f.fold} | {f.n_test} | {f.n_test_subjects} | "
            f"{f.baseline_mae:.3f} | {f.model_mae:.3f} | {f.improvement:+.3f} |"
        )
    lines += ["", "## Gate criteria", ""]
    lines += [f"- {r}" for r in reasons]
    lines += [
        "", "## Bound on any claim", "",
        f"n = {frame['subject_id'].nunique()} subjects, ShanghaiT2DM. Per "
        "ADR 0004, this is a single-dataset validation, not a claim about "
        "the target Indonesian population.",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# train + gate + persist
# ---------------------------------------------------------------------

def train_and_persist_from_frame(
    frame: pd.DataFrame,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    min_dish_meals: int = MIN_DISH_MEALS,
    artifact_path: str = ARTIFACT_PATH,
    report_path: str = REPORT_PATH,
    n_bootstrap: int = N_BOOTSTRAP,
    n_permutations: int = N_PERMUTATIONS,
) -> Evaluation:
    ev = evaluate_dish_addition(
        frame, folds, min_dish_meals=min_dish_meals,
        n_bootstrap=n_bootstrap, n_permutations=n_permutations,
    )
    passed, reasons = ev.passes_gate()
    _write_report(frame, ev, passed, reasons, report_path)

    if not passed:
        if os.path.exists(artifact_path):
            os.remove(artifact_path)
        print("\n".join(reasons))
        print(f"\nGATE FAILED — no artifact written. The "
              f"nlp_dish_identity_forecast mode stays off. Report: {report_path}")
        return ev

    full = _fit_full(frame, min_dish_meals)
    blob = {
        "model_version": "nlp-dish-identity-ridge-v1",
        "target": TARGET,
        **full,
        "n_meals": int(len(frame)),
        "n_subjects": int(frame["subject_id"].nunique()),
        "heldout_mae_m3": round(ev.model_mae, 3),
        "heldout_mae_m2": round(ev.baseline_mae, 3),
        "bootstrap_ci": (
            [round(c, 3) for c in ev.bootstrap_ci] if ev.bootstrap_ci else None
        ),
        "permutation_p": ev.permutation_p,
    }
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    with open(artifact_path, "w") as fh:
        json.dump(blob, fh, indent=2)
    print("\n".join(reasons))
    print(f"\nGATE PASSED — artifact written to {artifact_path}")
    print(f"NOTE: any claim from this model is bounded by n=%d subjects, "
          f"ShanghaiT2DM only." % blob["n_subjects"])
    return ev


def train_and_persist(
    response_path=DEFAULT_RESPONSE,
    legacy_path=DEFAULT_LEGACY,
    min_dish_meals: int = MIN_DISH_MEALS,
    n_splits: int = 5,
    artifact_path: str = ARTIFACT_PATH,
    report_path: str = REPORT_PATH,
    n_bootstrap: int = N_BOOTSTRAP,
    n_permutations: int = N_PERMUTATIONS,
) -> Evaluation:
    frame = load_meals(response_path, legacy_path)
    folds = grouped_folds(frame, n_splits=n_splits, group_column="subject_id")
    return train_and_persist_from_frame(
        frame, folds, min_dish_meals=min_dish_meals,
        artifact_path=artifact_path, report_path=report_path,
        n_bootstrap=n_bootstrap, n_permutations=n_permutations,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true", help="train, gate, persist")
    ap.add_argument("--min-dish-meals", type=int, default=MIN_DISH_MEALS)
    args = ap.parse_args()
    if not args.train:
        ap.print_help()
        return
    train_and_persist(min_dish_meals=args.min_dish_meals)


if __name__ == "__main__":
    main()
