"""Does knowing which dish beat knowing only how much? — the Path A method test.

Finding N6 is closed: `nlp/features/shanghai_nlp_response.csv` carries a strictly
causal `g0` and a T+60 rise for 3,045 of 3,060 meals. That makes the one question
this track exists to answer checkable for the first time:

    on a weighed cohort, does dish identity carry information about the glucose
    rise that total weight does not?

The design is a ladder of ridge regressions on the T+60 rise, scored on the same
subject-grouped folds (N5). The arms differ only in their features:

===============  ==========================================================
arm              features
===============  ==========================================================
``M0_constant``  the training-set mean rise
``M1_g0``        the pre-meal level
``M2_shipped``   ``g0`` + everything the shipped pipeline produces —
                 ``total_grams``, ``is_fried_cooking``, ``is_large_portion``
``M3_dish``      ``M2`` + one shrunk coefficient per frequent dish
``M3p_permuted`` ``M2`` + the *same* dish matrix with its rows permuted
===============  ==========================================================

**``M3 - M2`` is the answer, and ``M3p`` is why you can believe it.** This
repository has already been burned once by a metric that improved while the model
learned nothing — finding F15, where Ridge trained on pure noise beat the
constant baseline on Zone A. Dish features add parameters, so a gain from ``M3``
means nothing unless the identical procedure, on the identical parameter count,
with the dish-to-meal pairing destroyed, fails to gain. ``M3p`` is that control.

Three things are deliberately *generous to the baseline*:

* ``M2`` carries the two rule-based flags as well as the grams, so "the shipped
  pipeline" competes at full strength.
* Dishes too rare to estimate are dropped rather than bucketed, so those meals
  fall back to ``M2`` instead of being helped.
* The dish vocabulary is restricted to dishes seen at least ``MIN_DISH_MEALS``
  times **in the training fold** (and folds are grouped by subject), so a dish a
  test subject ate but no training subject ate contributes nothing. That is the
  real generalisation cost, and the per-fold coverage is reported.

Run::

    python3 -m nlp.eval_dish_identity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from nlp.data.diet_items import meal_items
from nlp.data.splits import ensure_subject_id, grouped_folds

_HERE = Path(__file__).resolve().parent
DEFAULT_RESPONSE = _HERE / "features" / "shanghai_nlp_response.csv"
DEFAULT_LEGACY = _HERE / "features" / "shanghai_nlp_features.csv"

TARGET = "rise_t60"
CONTINUOUS = ("g0_mgdl", "total_grams")
FLAGS = ("is_fried_cooking", "is_large_portion")

ARMS = ("M0_constant", "M1_g0", "M2_shipped", "M3_dish", "M3p_permuted")

#: Ridge penalties considered by the inner subject-grouped CV.
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
#: A dish must appear this often *in the training fold* to get a coefficient.
MIN_DISH_MEALS = 10
#: The diagnostics' OLS R^2 uses a stricter floor, since it is unregularised.
DIAGNOSTIC_DISH_MEALS = 50
#: Inner folds used to pick the penalty. Grouped, so the tuning does not leak.
INNER_SPLITS = 3
SEED = 20261018


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_meals(
    response_path: Path = DEFAULT_RESPONSE,
    legacy_path: Path = DEFAULT_LEGACY,
) -> pd.DataFrame:
    """Join the rebuilt response to the shipped features, on ``meal_id``."""
    response = pd.read_csv(response_path)
    legacy = pd.read_csv(legacy_path)
    columns = ["meal_id", *FLAGS, "total_grams"]

    frame = response.merge(
        legacy[columns], on="meal_id", how="inner", validate="one_to_one"
    )
    frame = frame[frame["valid_t60"] == 1].copy()
    frame = ensure_subject_id(frame)
    frame["dish_set"] = [
        tuple(sorted({name for name, _ in meal_items(text)}))
        for text in frame["dietary_intake"]
    ]
    frame = frame[
        frame[TARGET].notna() & frame["g0_mgdl"].notna() & frame["total_grams"].notna()
    ]
    return frame.reset_index(drop=True)


def dish_matrix(frame: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Dense 0/1 matrix of meal x dish over the whole vocabulary."""
    vocabulary = sorted({name for dishes in frame["dish_set"] for name in dishes})
    index = {name: column for column, name in enumerate(vocabulary)}
    matrix = np.zeros((len(frame), len(vocabulary)), dtype=float)
    for row, dishes in enumerate(frame["dish_set"]):
        for name in dishes:
            matrix[row, index[name]] = 1.0
    return matrix, vocabulary


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def _standardise(
    train: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Centre and scale on the training fold only."""
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _fit_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Ridge whose penalty is chosen by inner *subject-grouped* CV."""
    inner = GroupKFold(n_splits=min(INNER_SPLITS, len(set(groups_train))))
    inner_folds = list(inner.split(X_train, groups=groups_train))

    best_alpha, best_score = ALPHAS[0], np.inf
    for alpha in ALPHAS:
        errors = []
        for tr, va in inner_folds:
            model = Ridge(alpha=alpha).fit(X_train[tr], y_train[tr])
            errors.append(np.abs(model.predict(X_train[va]) - y_train[va]).mean())
        score = float(np.mean(errors))
        if score < best_score:
            best_score, best_alpha = score, alpha

    model = Ridge(alpha=best_alpha).fit(X_train, y_train)
    return model.predict(X_test), best_alpha


def run_ladder(
    frame: pd.DataFrame,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    min_dish_meals: int = MIN_DISH_MEALS,
    seed: int = SEED,
) -> dict:
    """Score every arm on the same folds. Returns errors and diagnostics."""
    y = frame[TARGET].to_numpy(dtype=float)
    continuous = frame[list(CONTINUOUS)].to_numpy(dtype=float)
    flags = frame[list(FLAGS)].to_numpy(dtype=float)
    groups = frame["subject_id"].to_numpy()

    dishes, vocabulary = dish_matrix(frame)
    permuted = dishes[np.random.default_rng(seed).permutation(len(frame))]

    predictions = {arm: np.full(len(frame), np.nan) for arm in ARMS}
    fold_errors: Dict[str, List[float]] = {arm: [] for arm in ARMS}
    alphas: Dict[str, List[float]] = {arm: [] for arm in ARMS}
    modelled_dishes: List[int] = []
    test_coverage: List[float] = []

    for train, test in folds:
        scaled_train, scaled_test = _standardise(continuous[train], continuous[test])
        base_train = np.hstack([scaled_train, flags[train]])
        base_test = np.hstack([scaled_test, flags[test]])

        predictions["M0_constant"][test] = y[train].mean()

        predictions["M1_g0"][test], alpha = _fit_and_predict(
            scaled_train[:, [0]], y[train], groups[train], scaled_test[:, [0]]
        )
        alphas["M1_g0"].append(alpha)

        predictions["M2_shipped"][test], alpha = _fit_and_predict(
            base_train, y[train], groups[train], base_test
        )
        alphas["M2_shipped"].append(alpha)

        frequency = dishes[train].sum(axis=0)
        columns = np.flatnonzero(frequency >= min_dish_meals)
        modelled_dishes.append(int(columns.size))
        test_coverage.append(float((dishes[np.ix_(test, columns)].sum(axis=1) > 0).mean()))

        predictions["M3_dish"][test], alpha = _fit_and_predict(
            np.hstack([base_train, dishes[np.ix_(train, columns)]]),
            y[train],
            groups[train],
            np.hstack([base_test, dishes[np.ix_(test, columns)]]),
        )
        alphas["M3_dish"].append(alpha)

        predictions["M3p_permuted"][test], alpha = _fit_and_predict(
            np.hstack([base_train, permuted[np.ix_(train, columns)]]),
            y[train],
            groups[train],
            np.hstack([base_test, permuted[np.ix_(test, columns)]]),
        )
        alphas["M3p_permuted"].append(alpha)

        for arm in ARMS:
            errors = np.abs(predictions[arm][test] - y[test])
            fold_errors[arm].append(float(errors.mean()))

    scored = {
        arm: {
            "mae": float(np.abs(predictions[arm] - y).mean()),
            "rmse": float(np.sqrt(np.mean((predictions[arm] - y) ** 2))),
            "fold_mae": fold_errors[arm],
            "fold_mae_sd": float(np.std(fold_errors[arm], ddof=1)),
            "chosen_alphas": [float(a) for a in alphas[arm]],
        }
        for arm in ARMS
    }

    scored["_diagnostics"] = {
        "n_meals": int(len(frame)),
        "n_subjects": int(frame["subject_id"].nunique()),
        "vocabulary_size": int(len(vocabulary)),
        "modelled_dishes_per_fold": modelled_dishes,
        "share_test_meals_with_a_modelled_dish": float(np.mean(test_coverage)),
        "target_sd_mgdl": float(y.std(ddof=1)),
        "target_mean_mgdl": float(y.mean()),
    }
    return scored


def feature_diagnostics(
    frame: pd.DataFrame,
    min_meals_per_subject: int = 10,
    min_dish_meals: int = 50,
) -> dict:
    """The simpler measurements the write-up quotes, so they have a command.

    Deliberately *not* part of the ridge ladder and **not comparable to it**:
    these are plain correlations and unregularised OLS held-out R², used to
    check the ladder's shape under a cruder estimator. Without this they would
    live only in prose, which is the failure mode `docs/METRICS.md` exists to
    catch.
    """
    y = frame[TARGET].to_numpy(dtype=float)
    continuous = ("g0_mgdl", "total_grams")

    correlations = {
        column: float(frame[column].corr(frame[TARGET]))
        for column in (*continuous, *FLAGS)
    }

    per_subject: Dict[str, dict] = {}
    for column in continuous:
        values = []
        for _, group in frame.groupby("subject_id"):
            if len(group) < min_meals_per_subject:
                continue
            value = group[column].corr(group[TARGET])
            if np.isfinite(value):
                values.append(float(value))
        per_subject[column] = {
            "n_subjects": len(values),
            "median_r": float(np.median(values)) if values else float("nan"),
        }

    dishes, _ = dish_matrix(frame)
    frequent = np.flatnonzero(dishes.sum(axis=0) >= min_dish_meals)
    folds = grouped_folds(frame, n_splits=5, group_column="subject_id")
    shipped = ("g0_mgdl", "total_grams", *FLAGS)

    def held_out_r2(columns: Tuple[str, ...], dish_columns=None) -> float:
        X = frame[list(columns)].to_numpy(dtype=float)
        if dish_columns is not None:
            X = np.hstack([X, dishes[:, dish_columns]])
        X = np.hstack([X, np.ones((len(X), 1))])
        errors = []
        for train, test in folds:
            beta, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
            errors.append(float(np.mean((X[test] @ beta - y[test]) ** 2)))
        return float(1 - np.mean(errors) / y.var())

    return {
        "correlation_with_target": correlations,
        "per_subject_correlation": per_subject,
        "min_meals_per_subject": min_meals_per_subject,
        "ols_r2": {
            "constant": held_out_r2(()),
            "g0": held_out_r2(("g0_mgdl",)),
            "g0_plus_shipped": held_out_r2(shipped),
            "g0_plus_frequent_dishes": held_out_r2(("g0_mgdl",), frequent),
            "shipped_plus_frequent_dishes": held_out_r2(shipped, frequent),
        },
        "frequent_dish_count": int(frequent.size),
        "frequent_dish_min_meals": min_dish_meals,
        "estimator": "unregularised OLS on the same subject-grouped folds",
        "comparable_to_ridge_ladder": False,
    }


def shuffled_folds(
    frame: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 0,
    test_size: float = 0.2,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Repeated subject-grouped held-out splits, for an uncertainty estimate."""
    groups = frame["subject_id"].to_numpy()
    splits = GroupShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=seed
    )
    return list(splits.split(frame, groups=groups))


def repeated_ladder(frame: pd.DataFrame, seeds: int = 5, **kwargs) -> dict:
    """Mean and spread of each arm's MAE across independent grouped splits.

    Scored per split, not pooled: ``GroupShuffleSplit`` rounds overlap and
    leave some meals unheld-out, so a pooled-by-index MAE would be computed
    over a ragged subset.
    """
    collected: Dict[str, List[float]] = {arm: [] for arm in ARMS}
    for seed in range(seeds):
        scored = run_ladder(
            frame, shuffled_folds(frame, seed=seed), seed=SEED + seed, **kwargs
        )
        for arm in ARMS:
            collected[arm].append(float(np.mean(scored[arm]["fold_mae"])))

    summary = {}
    for arm in ARMS:
        values = np.array(collected[arm])
        summary[arm] = {
            "mae_mean": float(values.mean()),
            "mae_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n_seeds": int(len(values)),
        }
    deltas = {
        "M3_minus_M2": [
            a - b
            for a, b in zip(collected["M3_dish"], collected["M2_shipped"])
        ],
        "M3p_minus_M2": [
            a - b
            for a, b in zip(collected["M3p_permuted"], collected["M2_shipped"])
        ],
        "M2_minus_M1": [
            a - b
            for a, b in zip(collected["M2_shipped"], collected["M1_g0"])
        ],
        "M2_minus_M0": [
            a - b
            for a, b in zip(collected["M2_shipped"], collected["M0_constant"])
        ],
    }
    summary["_deltas"] = {
        name: {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "per_seed": [float(v) for v in values],
        }
        for name, values in deltas.items()
    }
    return summary


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _print_ladder(title: str, scored: dict) -> None:
    diagnostics = scored["_diagnostics"]
    print()
    print(title)
    print("-" * 78)
    print(f"  meals {diagnostics['n_meals']}  subjects {diagnostics['n_subjects']}  "
          f"target SD {diagnostics['target_sd_mgdl']:.1f} mg/dL  "
          f"mean {diagnostics['target_mean_mgdl']:.1f}")
    print(f"  dish vocabulary {diagnostics['vocabulary_size']}  "
          f"modelled per fold {diagnostics['modelled_dishes_per_fold']}  "
          f"test meals with a modelled dish "
          f"{diagnostics['share_test_meals_with_a_modelled_dish']:.1%}")
    print()
    print(f"  {'arm':<14} {'MAE':>8} {'RMSE':>8} {'fold SD':>8} {'vs M0':>9} {'vs M2':>9}")
    reference = scored["M0_constant"]["mae"]
    baseline = scored["M2_shipped"]["mae"]
    for arm in ARMS:
        entry = scored[arm]
        delta0 = "        —" if arm == "M0_constant" else f"{entry['mae'] - reference:+9.3f}"
        delta2 = "" if arm == "M2_shipped" else f"{entry['mae'] - baseline:+9.3f}"
        print(f"  {arm:<14} {entry['mae']:8.3f} {entry['rmse']:8.3f} "
              f"{entry['fold_mae_sd']:8.3f} {delta0} {delta2}")
    print()


def _print_spread(title: str, spread: dict, seeds: int) -> None:
    print(f"{title} — {seeds} independent subject-grouped 80/20 splits")
    print("-" * 78)
    for name, stat in spread["_deltas"].items():
        print(f"  {name:<14} {stat['mean']:+7.3f} ± {stat['sd']:.3f} mg/dL MAE")
    print()


def _print_diagnostics(diagnostics: dict) -> None:
    print("Diagnostics — separate estimator, NOT comparable to the ladder")
    print("-" * 78)
    for column, value in diagnostics["correlation_with_target"].items():
        print(f"  r({column}, {TARGET}) = {value:+.3f}")
    for column, stat in diagnostics["per_subject_correlation"].items():
        print(f"  median per-subject r({column}) = {stat['median_r']:+.3f} "
              f"over {stat['n_subjects']} subjects")
    print()
    print(f"  held-out OLS R^2 ({diagnostics['frequent_dish_count']} dishes with "
          f">= {diagnostics['frequent_dish_min_meals']} meals)")
    for name, value in diagnostics["ols_r2"].items():
        print(f"    {name:<32} {value:.4f}")
    print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Does dish identity beat total grams on the T+60 rise?"
    )
    parser.add_argument("--response", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--min-dish-meals", type=int, default=MIN_DISH_MEALS)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    frame = load_meals(args.response, args.legacy)
    folds = grouped_folds(frame, n_splits=args.splits, group_column="subject_id")

    print("=" * 78)
    print("NLP PATH A — dish identity vs total grams, on the T+60 rise")
    print("=" * 78)
    print(f"Folds: {args.splits} GroupKFold by subject_id. "
          f"Dishes need >= {args.min_dish_meals} training meals for a coefficient.")
    print("M3 - M2 is the answer. M3p is the control: same matrix, pairing destroyed.")

    primary = run_ladder(frame, folds, min_dish_meals=args.min_dish_meals)
    _print_ladder("All meals with a T+60 rise", primary)

    unmedicated = frame[frame["insulin_near_meal"] == 0].reset_index(drop=True)
    unmedicated_folds = grouped_folds(
        unmedicated, n_splits=args.splits, group_column="subject_id"
    )
    secondary = run_ladder(
        unmedicated, unmedicated_folds, min_dish_meals=args.min_dish_meals
    )
    _print_ladder(
        "Robustness — no insulin dose within 30 min of the meal", secondary
    )

    spread = repeated_ladder(
        frame, seeds=args.seeds, min_dish_meals=args.min_dish_meals
    )
    _print_spread("Uncertainty — all meals", spread, args.seeds)

    unmedicated_spread = repeated_ladder(
        unmedicated, seeds=args.seeds, min_dish_meals=args.min_dish_meals
    )
    _print_spread(
        "Uncertainty — no insulin dose within 30 min", unmedicated_spread, args.seeds
    )

    diagnostics = feature_diagnostics(
        frame, min_dish_meals=DIAGNOSTIC_DISH_MEALS
    )
    _print_diagnostics(diagnostics)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "config": {
                        "splits": args.splits,
                        "seeds": args.seeds,
                        "min_dish_meals": args.min_dish_meals,
                        "diagnostic_dish_meals": DIAGNOSTIC_DISH_MEALS,
                        "group_column": "subject_id",
                        "target": TARGET,
                        "alphas": list(ALPHAS),
                    },
                    "all_meals": primary,
                    "unmedicated": secondary,
                    "repeated": spread,
                    "repeated_unmedicated": unmedicated_spread,
                    "diagnostics": diagnostics,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
