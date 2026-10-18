"""BIG IDEAs — does the meal carry information the glucose history does not?

BIG IDEAs pairs a food log with a Dexcom G6 trace for 16 pre-diabetic
participants over 8-10 days. That is a *real event-level pairing* of meal
composition with a glucose outcome, which is the condition
``docs/CGMACROS_PIPELINE_SPEC.md`` sets for treating a component study as
more than an isolated track.

This module answers three questions, in order:

1. **Meal response.** Does glucose actually rise after a logged meal, how far
   and how fast, and what predicts the size of the rise?
2. **History baselines.** Against B0 / B1 / B2 / a mean-reversion-corrected
   history model, what does the meal add? Leave-one-out throughout, and the
   mean-reversion term is separated out so the meal is not credited for it.
3. **The CV ceiling.** The macros in the food log are ORACLE values. A
   deployed system gets them from a photo, and ``cv/MODEL_CARD.md`` says there
   is no portion estimation. So what portion accuracy does CV actually need?

Question 3 is the one that decides whether CV is load-bearing in the fusion,
and it is answered two ways: by degrading the macros directly, and by
re-measuring the model with coarse portion buckets learned leave-one-
participant-out (the best case for a classifier that cannot size a plate).

Run::

    python -m forecasting.evaluate.bigideas_response --data-dir ~/Downloads/Bigideas-dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.data.bigideas import (
    HORIZONS_MINUTES,
    default_data_dir,
    load_cohort,
    load_food_log,
)

PRIMARY_HORIZON = 60
MIN_MEALS_PER_PARTICIPANT = 12
MIN_GROUP_OBSERVATIONS = 3
MACRO_FEATURES = ("carbs_g", "kcal", "protein_g", "fat_g", "fiber_g")


# ----------------------------------------------------------------------
# Estimation primitives
# ----------------------------------------------------------------------

def loo_predict_linear(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out OLS predictions, computed with the hat matrix.

    Algebraically identical to refitting n times with each point held out, but
    O(1) per point. Verified against the explicit loop to 6e-15. Every model
    comparison in this module uses this, so no score is inflated by fitting
    and scoring on the same rows.
    """
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    residual = y - design @ beta
    leverage = np.clip(
        np.einsum("ij,jk,ik->i", design, xtx_inv, design), 0.0, 1.0 - 1e-9
    )
    return y - residual / (1.0 - leverage)


def _design(*columns: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(columns[0]), 1))] + [c[:, None] for c in columns])


def safe_correlation(a, b) -> float:
    """Pearson r, NaN when there is too little variation to mean anything."""
    pair = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(pair) < 4 or pair.a.nunique() < 2 or pair.b.nunique() < 2:
        return float("nan")
    return float(pair.a.corr(pair.b))


def crossing_point(xs, ys) -> float | None:
    """First x where y crosses from <= 0 to > 0, linearly interpolated."""
    for i in range(1, len(ys)):
        if ys[i - 1] <= 0 < ys[i]:
            return float(xs[i - 1] + (xs[i] - xs[i - 1])
                         * (0 - ys[i - 1]) / (ys[i] - ys[i - 1]))
    return None


def _finite_participants(cohort: pd.DataFrame, horizon: int):
    """Per-participant frames with enough meals to fit anything."""
    target = f"glucose_{horizon}_mgdl"
    for participant_id, group in cohort.groupby("participant_id"):
        usable = group.dropna(subset=[target, "g0_mgdl", "carbs_g"]).copy()
        if len(usable) < MIN_MEALS_PER_PARTICIPANT:
            continue
        usable["delta"] = usable[target] - usable["g0_mgdl"]
        yield participant_id, usable


# ----------------------------------------------------------------------
# 1. Meal response
# ----------------------------------------------------------------------

def meal_response_summary(cohort: pd.DataFrame, horizons=HORIZONS_MINUTES) -> pd.DataFrame:
    """The average glucose rise after a meal, pooled across participants."""
    rows = []
    for horizon in horizons:
        rises = cohort[f"delta_{horizon}_mgdl"].dropna()
        rows.append({
            "horizon_minutes": horizon,
            "n_meals": int(len(rises)),
            "n_participants": int(
                cohort.loc[rises.index, "participant_id"].nunique()
            ),
            "mean_mgdl": float(rises.mean()),
            "median_mgdl": float(rises.median()),
            "sd_mgdl": float(rises.std()),
            "p25_mgdl": float(rises.quantile(0.25)),
            "p75_mgdl": float(rises.quantile(0.75)),
        })
    return pd.DataFrame(rows)


def predictor_correlations(cohort: pd.DataFrame, horizon: int = PRIMARY_HORIZON) -> pd.DataFrame:
    """What correlates with the size of the rise?

    The pooled column treats every meal as independent, which it is not —
    meals within one participant are correlated. The median-per-participant
    column is the honest one; the pooled number is shown only so the
    difference is visible.
    """
    target = f"delta_{horizon}_mgdl"
    features = list(MACRO_FEATURES) + ["g0_mgdl", "hour_of_day", "n_items"]
    rows = []
    for feature in features:
        per_participant = [
            safe_correlation(group[feature], group[target])
            for _, group in cohort.groupby("participant_id")
        ]
        per_participant = [r for r in per_participant if np.isfinite(r)]
        rows.append({
            "feature": feature,
            "pooled_r": safe_correlation(cohort[feature], cohort[target]),
            "median_per_participant_r": float(np.median(per_participant))
            if per_participant else float("nan"),
            "participants": len(per_participant),
        })
    return pd.DataFrame(rows)


def carbohydrate_terciles(cohort: pd.DataFrame, horizon: int = PRIMARY_HORIZON) -> pd.DataFrame:
    """Rise by carbohydrate band, pooled.

    ``g0_mgdl`` is reported alongside because it is the confound: participants
    eat larger meals at lower glucose, so an unstratified comparison credits
    the meal with the regression to the mean.
    """
    target = f"delta_{horizon}_mgdl"
    usable = cohort.dropna(subset=["carbs_g", target]).copy()
    if len(usable) < 30:
        return pd.DataFrame()
    usable["band"] = pd.qcut(usable.carbs_g, 3, labels=["low", "mid", "high"],
                             duplicates="drop")
    columns = {"g0_mgdl", "carbs_g"} | {
        f"delta_{h}_mgdl" for h in HORIZONS_MINUTES
    }
    return (
        usable.groupby("band", observed=True)
        .agg(**{c: (c, "median") for c in sorted(columns)},
             n_meals=("carbs_g", "size"))
        .reset_index()
    )


# ----------------------------------------------------------------------
# 2. History baselines
# ----------------------------------------------------------------------

def history_baselines(cohort: pd.DataFrame, horizon: int = PRIMARY_HORIZON) -> tuple:
    """Leave-one-out MAE for the history-only and meal-aware models.

        B0   constant        the participant's mean glucose
        B1   persistence     the reading at the meal
        B2   g0 + mean rise  the project's baseline
        B2g  g0 + fitted slope, no meal term — mean reversion pulled out
        B3   g0 + fitted slope + carbs

    ``B2g`` exists to stop the meal being credited for mean reversion. Compare
    B3 against B2g, not against B2.
    """
    target = f"glucose_{horizon}_mgdl"
    rows = []
    for participant_id, usable in _finite_participants(cohort, horizon):
        actual = usable[target].to_numpy(float)
        g0 = usable.g0_mgdl.to_numpy(float)
        carbs = usable.carbs_g.to_numpy(float)
        delta = usable.delta.to_numpy(float)

        predictions = {
            "B0 constant": np.full(len(usable), float(usable.g0_mgdl.mean())),
            "B1 persistence": g0,
            "B2 g0 + mean rise": g0 + loo_predict_linear(_design(g0 * 0 + 1), delta),
            "B2g g0 + fitted slope": g0 + loo_predict_linear(_design(g0), delta),
            "B3 g0 + slope + carbs": g0 + loo_predict_linear(_design(g0, carbs), delta),
        }
        for model, predicted in predictions.items():
            rows.append({
                "participant_id": participant_id,
                "model": model,
                "n_meals": len(usable),
                "mae_mgdl": float(np.abs(actual - predicted).mean()),
            })

    per_participant = pd.DataFrame(rows)
    summary = (
        per_participant.groupby("model", sort=False)
        .agg(n_meals=("n_meals", "sum"),
             participants=("participant_id", "nunique"),
             mae_mgdl=("mae_mgdl", "mean"))
        .reset_index()
    )
    return summary, per_participant


# ----------------------------------------------------------------------
# 3. The CV ceiling
# ----------------------------------------------------------------------

def portion_noise_sweep(cohort: pd.DataFrame, horizon: int = PRIMARY_HORIZON,
                        sigmas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0),
                        repeats: int = 12, seed: int = 0) -> pd.DataFrame:
    """Multiplicative portion error on the oracle macros.

    Answers "how wrong can the gram estimate be", and is deliberately shown
    next to the reliability sweep because on its own it is misleading: scaling
    carbs by noise leaves the correlation with the truth high (SD 1.0 still
    leaves r ~ 0.6), so the meal survives. Correlation, not precision, is the
    binding constraint.
    """
    rng = np.random.default_rng(seed)
    target = f"glucose_{horizon}_mgdl"
    rows = []

    for sigma in sigmas:
        base_scores, meal_scores, carb_errors, correlations = [], [], [], []
        for _, usable in _finite_participants(cohort, horizon):
            actual = usable[target].to_numpy(float)
            g0 = usable.g0_mgdl.to_numpy(float)
            carbs = usable.carbs_g.to_numpy(float)
            delta = usable.delta.to_numpy(float)

            base = g0 + loo_predict_linear(_design(g0), delta)
            base_error = float(np.abs(actual - base).mean())

            scores = []
            for _ in range(repeats):
                noisy = (np.clip(carbs * (1 + rng.normal(0, sigma, len(carbs))), 0, None)
                         if sigma > 0 else carbs)
                predicted = g0 + loo_predict_linear(_design(g0, noisy), delta)
                scores.append(float(np.abs(actual - predicted).mean()))
                correlations.append(float(np.corrcoef(noisy, carbs)[0, 1]))
            base_scores.append(base_error)
            meal_scores.append(float(np.mean(scores)))
            carb_errors.append(float(np.abs(noisy - carbs).mean()))

        base_mae, meal_mae = float(np.mean(base_scores)), float(np.mean(meal_scores))
        rows.append({
            "portion_sd": sigma,
            "carb_error_g": float(np.mean(carb_errors)),
            "carb_correlation": float(np.mean(correlations)),
            "mae_history_only_mgdl": base_mae,
            "mae_with_meal_mgdl": meal_mae,
            "meal_gain_mgdl": meal_mae - base_mae,
        })
    return pd.DataFrame(rows)


def carb_reliability_sweep(cohort: pd.DataFrame, horizon: int = PRIMARY_HORIZON,
                           correlations=(1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0),
                           repeats: int = 12, seed: int = 1) -> pd.DataFrame:
    """Corrupt the carb estimate's RANKING, not its precision.

    Misidentification and "no portion output" do not add symmetric noise to
    the gram count; they destroy the correlation between estimated and true
    carbohydrates. At rho = 0 the dish is known but every meal gets one
    textbook value, and the carb column is collinear with the intercept.

    Returns one row per target correlation, with ``meal_gain_mgdl`` measured
    against the mean-reversion-corrected history model (B2g).
    """
    rng = np.random.default_rng(seed)
    target = f"glucose_{horizon}_mgdl"
    rows = []

    for rho in correlations:
        base_scores, meal_scores, achieved, improved, used = [], [], [], 0, 0
        for _, usable in _finite_participants(cohort, horizon):
            actual = usable[target].to_numpy(float)
            g0 = usable.g0_mgdl.to_numpy(float)
            carbs = usable.carbs_g.to_numpy(float)
            delta = usable.delta.to_numpy(float)
            mean, sd = float(carbs.mean()), float(carbs.std())
            if sd <= 0:
                continue

            base = g0 + loo_predict_linear(_design(g0), delta)
            base_error = float(np.abs(actual - base).mean())

            scores, rhos = [], []
            for _ in range(repeats):
                standardised = (carbs - mean) / sd
                corrupted = (rho * standardised
                             + np.sqrt(max(0.0, 1 - rho ** 2))
                             * rng.normal(0, 1, len(carbs)))
                noisy = np.clip(mean + sd * corrupted, 0, None)
                predicted = g0 + loo_predict_linear(_design(g0, noisy), delta)
                scores.append(float(np.abs(actual - predicted).mean()))
                rhos.append(float(np.corrcoef(noisy, carbs)[0, 1]))

            base_scores.append(base_error)
            meal_scores.append(float(np.mean(scores)))
            achieved.append(float(np.mean(rhos)))
            improved += int(np.mean(scores) < base_error)
            used += 1

        base_mae, meal_mae = float(np.mean(base_scores)), float(np.mean(meal_scores))
        rows.append({
            "target_correlation": rho,
            "achieved_correlation": float(np.mean(achieved)),
            "mae_history_only_mgdl": base_mae,
            "mae_with_meal_mgdl": meal_mae,
            "meal_gain_mgdl": meal_mae - base_mae,
            "participants_improved": improved,
            "participants": used,
        })
    return pd.DataFrame(rows)


def _textbook_lookup(others: pd.DataFrame, buckets: int) -> tuple[dict, dict, float]:
    """Learn one carb value per (dish, unit, size bucket) from other participants."""
    lookup: dict[tuple, tuple[np.ndarray, dict[int, float]]] = {}
    for (food_key, unit), group in others.groupby(["food_key", "unit"], dropna=True):
        usable = group.dropna(subset=["amount", "carbs_g"])
        if len(usable) < MIN_GROUP_OBSERVATIONS:
            continue
        amounts = usable.amount.to_numpy(float)
        carbs = usable.carbs_g.to_numpy(float)
        cuts = (np.unique(np.quantile(
            amounts, np.linspace(0, 1, buckets + 1)[1:-1]))
            if buckets > 1 else np.array([]))
        assignment = np.digitize(amounts, cuts)
        medians = {int(b): float(np.median(carbs[assignment == b]))
                   for b in np.unique(assignment)}
        lookup[(food_key, unit)] = (cuts, medians)

    per_food = others.dropna(subset=["carbs_g"]).groupby("food_key").carbs_g.median()
    return lookup, per_food.to_dict(), float(others.carbs_g.median())


def _cohort_estimates(mine: pd.DataFrame, others: pd.DataFrame, buckets: int) -> np.ndarray:
    """Cold start: the dish must be one other participants have logged."""
    lookup, per_food, global_median = _textbook_lookup(others, buckets)
    estimates = []
    for item in mine.itertuples(index=False):
        entry = lookup.get((item.food_key, item.unit))
        if entry is not None and np.isfinite(item.amount):
            cuts, medians = entry
            bucket = int(np.digitize([item.amount], cuts)[0])
            if bucket in medians:
                estimates.append(medians[bucket])
                continue
        estimates.append(float(per_food.get(item.food_key, global_median)))
    return np.asarray(estimates, dtype=float)


def _loo_bucket_estimates(items: pd.DataFrame, buckets: int) -> np.ndarray:
    """Personalised: the dish must be one THIS participant has logged before.

    Leave-one-item-out, so an item never informs its own estimate. This is the
    deployment case for a calibrated system — the app learns your usual
    portions — and is the only version of the experiment with enough repeated
    observations per dish to actually measure what portion information buys.
    """
    items = items.reset_index(drop=True)
    estimates = np.full(len(items), np.nan)

    for _, group in items.groupby(["food_key", "unit"], dropna=True):
        positions = group.index.to_numpy()
        amounts = group.amount.to_numpy(float)
        carbs = group.carbs_g.to_numpy(float)
        for i in range(len(group)):
            keep = np.ones(len(group), bool)
            keep[i] = False
            rest_amounts, rest_carbs = amounts[keep], carbs[keep]
            usable = np.isfinite(rest_amounts) & np.isfinite(rest_carbs)
            rest_amounts, rest_carbs = rest_amounts[usable], rest_carbs[usable]
            if len(rest_carbs) == 0:
                continue
            if len(rest_carbs) >= 2 and buckets > 1 and np.isfinite(amounts[i]):
                cuts = np.unique(np.quantile(
                    rest_amounts, np.linspace(0, 1, buckets + 1)[1:-1]))
                bucket = int(np.digitize([amounts[i]], cuts)[0])
                same = rest_carbs[np.digitize(rest_amounts, cuts) == bucket]
                if len(same):
                    estimates[positions[i]] = float(np.median(same))
                    continue
            estimates[positions[i]] = float(np.median(rest_carbs))
    return estimates


def portion_bucket_sweep(data_dir: Path, cohort: pd.DataFrame,
                         horizon: int = PRIMARY_HORIZON,
                         bucket_counts=(1, 2, 3, 4, 5, 8),
                         sources=("self", "cohort")) -> pd.DataFrame:
    """How much does KNOWING THE PORTION add, on top of knowing the dish?

    ``cv/MODEL_CARD.md`` says CV cannot estimate portion, so the best a vision
    model can do today is name the dish and hand over one textbook carb value.
    This asks what a crude size estimate would add — the cheapest available
    fix to that track.

    Buckets are cut on the dish's amount distribution and each bucket takes
    that bucket's median carbohydrates. ``bucket_counts=1`` is identification
    only; larger values mean the system can also tell a small plate from a
    large one.

    Two textbook sources, because the cold-start version is starved:

    ``cohort``  dish values learned from OTHER participants (cold start).
                Most dishes are logged by one person, so there is nothing to
                bucket on and this mostly measures identification alone.
    ``self``    dish values learned from the participant's own earlier meals,
                leave-one-item-out (deployment, personalised). Repeated dishes
                per person are what make coarse portion measurable at all.

    The threshold to clear is the break-even carb correlation from
    :func:`carb_reliability_sweep`, not zero.
    """
    items = _load_items(data_dir, cohort)
    target = f"glucose_{horizon}_mgdl"

    rows = []
    for source in sources:
        for buckets in bucket_counts:
            base_scores, meal_scores, carb_corrs, improved, used = [], [], [], 0, 0
            for participant_id, usable in _finite_participants(cohort, horizon):
                mine = items[items.participant_id == participant_id]
                others = items[items.participant_id != participant_id]
                if mine.empty or others.empty:
                    continue

                if source == "self":
                    estimates = _loo_bucket_estimates(mine, buckets)
                    # A dish the participant has never logged before is a cold
                    # start; fall back to what other participants logged.
                    _, per_food, global_median = _textbook_lookup(others, buckets)
                    cold = [float(per_food.get(item.food_key, global_median))
                            for item in mine.itertuples(index=False)]
                    estimates = np.where(np.isfinite(estimates), estimates, cold)
                else:
                    estimates = _cohort_estimates(mine, others, buckets)

                mine = mine.assign(estimated_carbs_g=estimates)
                meals = (
                    mine.groupby("t0")
                    .agg(true_carbs_g=("carbs_g", "sum"),
                         estimated_carbs_g=("estimated_carbs_g", "sum"))
                    .reset_index()
                )
                joined = usable.merge(meals, on="t0", how="inner").dropna(
                    subset=["true_carbs_g", "estimated_carbs_g"])
                if (len(joined) < MIN_MEALS_PER_PARTICIPANT
                        or joined.estimated_carbs_g.std() == 0):
                    continue

                actual = joined[target].to_numpy(float)
                g0 = joined.g0_mgdl.to_numpy(float)
                estimated = joined.estimated_carbs_g.to_numpy(float)
                delta = actual - g0

                base = g0 + loo_predict_linear(_design(g0), delta)
                base_error = float(np.abs(actual - base).mean())
                meal_error = float(np.abs(
                    actual - (g0 + loo_predict_linear(_design(g0, estimated), delta))
                ).mean())

                base_scores.append(base_error)
                meal_scores.append(meal_error)
                carb_corrs.append(float(np.corrcoef(
                    estimated, joined.true_carbs_g)[0, 1]))
                improved += int(meal_error < base_error)
                used += 1

            if not used:
                continue
            base_mae, meal_mae = float(np.mean(base_scores)), float(np.mean(meal_scores))
            rows.append({
                "textbook": source,
                "portion_buckets": buckets,
                "mean_carb_correlation": float(np.mean(carb_corrs)),
                "median_carb_correlation": float(np.median(carb_corrs)),
                "mae_history_only_mgdl": base_mae,
                "mae_with_meal_mgdl": meal_mae,
                "meal_gain_mgdl": meal_mae - base_mae,
                "participants_improved": improved,
                "participants": used,
            })
    return pd.DataFrame(rows)


def _load_items(data_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """Item-level food logs for every participant in the cohort."""
    frames = []
    for participant_id in cohort.participant_id.unique():
        food_log = load_food_log(data_dir / f"Food_Log_{participant_id}.csv")
        food_log = food_log.dropna(subset=["carbs_g"])
        food_log["participant_id"] = participant_id
        frames.append(food_log)
    return pd.concat(frames, ignore_index=True)


def carbohydrate_sensitivity(cohort: pd.DataFrame,
                             horizon: int = PRIMARY_HORIZON) -> dict:
    """Milligrams per decilitre of glucose per gram of carbohydrate.

    The number to quote whenever someone asks what a carbohydrate error costs.
    Fitted per participant as the ``carbs_g`` coefficient in
    ``delta ~ 1 + g0_mgdl + carbs_g``, so the starting glucose is controlled.

    ``agents/cv/CLAUDE.md`` states that a 30 g carbohydrate error "can shift
    predicted glucose by 40-60 mg/dL". That figure is not reproducible here;
    see ``docs/BIGIDEAS_MEAL_RESPONSE_FINDINGS.md``.
    """
    slopes = []
    for _, usable in _finite_participants(cohort, horizon):
        design = _design(usable.g0_mgdl.to_numpy(float),
                         usable.carbs_g.to_numpy(float))
        beta, *_ = np.linalg.lstsq(design, usable.delta.to_numpy(float),
                                   rcond=None)
        slopes.append(float(beta[2]))
    slopes = np.asarray(slopes)
    median = float(np.median(slopes))
    return {
        "horizon_minutes": horizon,
        "participants": int(len(slopes)),
        "median_mgdl_per_g": median,
        "mean_mgdl_per_g": float(slopes.mean()),
        "min_mgdl_per_g": float(slopes.min()),
        "max_mgdl_per_g": float(slopes.max()),
        "glucose_shift_for_30g_median_mgdl": 30.0 * median,
    }


def _within_dish_variance(items: pd.DataFrame,
                          require_varying_amount: bool) -> tuple[float, int]:
    """Pooled within-dish variance of ``log(carbs)`` for repeated dishes."""
    total, count = 0.0, 0
    for _, group in items.groupby(["participant_id", "food_key"]):
        if len(group) < 2:
            continue
        if require_varying_amount and group.amount.nunique() <= 1:
            continue
        values = group.log_carbs.to_numpy(float)
        total += float(((values - values.mean()) ** 2).sum())
        count += len(values)
    if count == 0:
        return float("nan"), 0
    return total / count, count


def carbohydrate_variance_decomposition(data_dir: Path,
                                        cohort: pd.DataFrame) -> dict:
    """Split carbohydrate variance into "which dish" and "how much of it".

    This is what decides where a vision track should spend its effort. If the
    variance lives in *which* dish was eaten, the bottleneck is class breadth
    and identification. If it lives in portion size, the bottleneck is
    estimation of amount.

    Within-dish variance is computed on dishes a participant logged at least
    twice, which is the only place portion variation is observable. Both are on
    the log scale, because carbohydrate scales multiplicatively with amount.

    Two versions are reported. About half of repeated dishes are logged at an
    identical amount, which forces their within-dish variance to zero and
    biases the share downward. ``portion_share_variable_amount_only`` restricts
    to dishes where the amount actually moved and is the softer bound. Quote
    the range between them.
    """
    items = _load_items(data_dir, cohort)
    items = items[(items.carbs_g > 0) & (items.amount > 0)].copy()
    if items.empty:
        return {}
    items["log_carbs"] = np.log(items.carbs_g)

    overall = items.log_carbs.to_numpy(float)
    total_variance = float(((overall - overall.mean()) ** 2).mean())
    all_variance, all_n = _within_dish_variance(items, False)
    varying_variance, varying_n = _within_dish_variance(items, True)

    per_participant = items.groupby("participant_id").food_key.nunique()
    return {
        "items": int(len(items)),
        "items_in_repeated_dishes": int(all_n),
        "items_in_varying_amount_dishes": int(varying_n),
        "distinct_dishes": int(items.food_key.nunique()),
        "distinct_dishes_per_participant_median": float(per_participant.median()),
        "distinct_dishes_per_participant_max": int(per_participant.max()),
        "total_variance": total_variance,
        "within_dish_variance": float(all_variance),
        "between_dish_variance": float(total_variance - all_variance),
        "portion_share": float(all_variance / total_variance),
        "dish_share": float(1 - all_variance / total_variance),
        "portion_share_variable_amount_only": float(
            varying_variance / total_variance),
        "repeated_dishes_logged_at_one_amount": float(1 - varying_n / all_n)
        if all_n else float("nan"),
    }


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def _print_frame(frame: pd.DataFrame, **kwargs) -> None:
    if len(frame):
        with pd.option_context("display.width", 220, "display.max_columns", 60):
            print(frame.round(kwargs.pop("round", 3)).to_string(index=False, **kwargs))


def run(data_dir: Path, horizons=HORIZONS_MINUTES,
        bucket_counts=(1, 2, 3, 4, 5, 8)) -> dict:
    cohort, problems = load_cohort(data_dir, horizons)

    print("=" * 74)
    print(f"BIG IDEAs {data_dir}  —  meal response")
    print("=" * 74)
    print(f"participants : {cohort.participant_id.nunique()}  "
          f"(unusable: {len(problems)})")
    print(f"meals        : {len(cohort)}")
    for problem in problems:
        print(f"  skipped {problem}")
    print()

    response = meal_response_summary(cohort, horizons)
    print("--- glucose rise after a logged meal " + "-" * 37)
    _print_frame(response, round=1)
    print()

    print(f"--- what predicts the T+{PRIMARY_HORIZON} rise? " + "-" * 34)
    _print_frame(predictor_correlations(cohort, PRIMARY_HORIZON), round=3)
    print("  pooled_r treats every meal as independent, which it is not;")
    print("  median_per_participant_r is the honest column.")
    print()

    terciles = carbohydrate_terciles(cohort, PRIMARY_HORIZON)
    print("--- rise by carbohydrate band, pooled " + "-" * 36)
    _print_frame(terciles, round=1)
    print("  g0_mgdl is the confound: larger meals are eaten at lower glucose.")
    print()

    summary, per_participant = history_baselines(cohort, PRIMARY_HORIZON)
    b2g = summary.loc[summary.model == "B2g g0 + fitted slope", "mae_mgdl"]
    b2g = float(b2g.iloc[0]) if len(b2g) else float("nan")
    summary = summary.assign(vs_B2g=(summary.mae_mgdl - b2g).round(2))
    print(f"--- history baselines, T+{PRIMARY_HORIZON} (leave-one-out MAE) " + "-" * 22)
    _print_frame(summary, round=2)
    print("  B2g removes mean reversion. Compare B3 against B2g, not B2: the")
    print("  difference between the two is the meal's actual contribution.")
    print()

    noise = portion_noise_sweep(cohort, PRIMARY_HORIZON)
    print("--- portion PRECISION: degrade the gram estimate " + "-" * 26)
    _print_frame(noise, round=3)
    print()

    reliability = carb_reliability_sweep(cohort, PRIMARY_HORIZON)
    print("--- portion RANKING: destroy the correlation " + "-" * 30)
    _print_frame(reliability, round=3)
    break_even = crossing_point(
        reliability.target_correlation.to_numpy(float),
        reliability.meal_gain_mgdl.to_numpy(float),
    )
    if break_even is not None:
        print(f"  BREAK-EVEN carb correlation: {break_even:.2f}")
        print("  Below this the meal term stops paying for itself.")
    print()

    buckets = portion_bucket_sweep(data_dir, cohort, PRIMARY_HORIZON, bucket_counts)
    print("--- what does knowing the PORTION add? " + "-" * 35)
    print("  1 bucket  = dish identified, one fixed serving (CV today)")
    print("  k buckets = dish identified AND sized into k portion bands")
    print("  self  = dish values from this participant's own earlier meals")
    print("  cohort= dish values from other participants (cold start)")
    _print_frame(buckets, round=3)
    if break_even is not None and len(buckets):
        print(f"  break-even is r = {break_even:.2f}")
        for source in buckets.textbook.unique():
            subset = buckets[buckets.textbook == source]
            identity = subset.loc[subset.portion_buckets == 1,
                                  "mean_carb_correlation"]
            best = subset.loc[subset.mean_carb_correlation.idxmax()]
            identity = float(identity.iloc[0]) if len(identity) else float("nan")
            print(f"  {source:6s}: identity-only r = {identity:.3f} -> "
                  f"best k = {int(best.portion_buckets)} at r = "
                  f"{best.mean_carb_correlation:.3f}  "
                  f"(+{best.mean_carb_correlation - identity:.3f})")
    print()

    sensitivity = carbohydrate_sensitivity(cohort, PRIMARY_HORIZON)
    print("--- how much does a carbohydrate error cost? " + "-" * 30)
    print(f"  fitted per participant: delta ~ 1 + g0_mgdl + carbs_g")
    print(f"  median slope  : {sensitivity['median_mgdl_per_g']:+.3f} mg/dL per g")
    print(f"  range         : {sensitivity['min_mgdl_per_g']:+.3f} "
          f".. {sensitivity['max_mgdl_per_g']:+.3f}")
    print(f"  a 30 g error  : {sensitivity['glucose_shift_for_30g_median_mgdl']:+.1f} mg/dL "
          f"at T+{PRIMARY_HORIZON}")
    print("  agents/cv/CLAUDE.md claims 40-60 mg/dL for the same 30 g error.")
    print()

    decomposition = carbohydrate_variance_decomposition(data_dir, cohort)
    print("--- where does carbohydrate variance actually live? " + "-" * 24)
    if decomposition:
        print(f"  dishes in the logs          : "
              f"{decomposition['distinct_dishes']}")
        print(f"  dishes per participant      : median "
              f"{decomposition['distinct_dishes_per_participant_median']:.0f}, "
              f"max {decomposition['distinct_dishes_per_participant_max']}")
        print(f"  variance from PORTION SIZE  : "
              f"{100 * decomposition['portion_share']:.1f}% "
              f"({100 * decomposition['portion_share_variable_amount_only']:.1f}% "
              f"on dishes where the amount moved)")
        print(f"  variance from WHICH DISH    : "
              f"{100 * decomposition['dish_share']:.1f}%")
        print(f"  {100 * decomposition['repeated_dishes_logged_at_one_amount']:.0f}% "
              f"of repeated dishes are logged at one amount, which forces their")
        print("  within-dish variance to zero; the two figures bracket it.")
    print()

    return {
        "data_dir": str(data_dir),
        "participants": int(cohort.participant_id.nunique()),
        "meals": int(len(cohort)),
        "problems": problems,
        "meal_response": response.to_dict(orient="records"),
        "predictor_correlations": predictor_correlations(
            cohort, PRIMARY_HORIZON).to_dict(orient="records"),
        "carbohydrate_terciles": terciles.to_dict(orient="records"),
        "history_baselines": summary.round(4).to_dict(orient="records"),
        "portion_noise_sweep": noise.round(4).to_dict(orient="records"),
        "carb_reliability_sweep": reliability.round(4).to_dict(orient="records"),
        "break_even_carb_correlation": break_even,
        "portion_bucket_sweep": buckets.round(4).to_dict(orient="records"),
        "carbohydrate_sensitivity": sensitivity,
        "carbohydrate_variance_decomposition": decomposition,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--bucket-counts", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5, 8])
    arguments = parser.parse_args(argv)

    result = run(arguments.data_dir, HORIZONS_MINUTES, tuple(arguments.bucket_counts))
    if arguments.json:
        arguments.json.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(f"written: {arguments.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
