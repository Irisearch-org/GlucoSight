from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forecasting.data.cgmacros import TARGET_HORIZONS_MINUTES

MAX_G0_AGE_MINUTES = 360.0


FEATURE_COLUMNS = (
    "g0_mgdl",
    "g0_age_minutes",
    "history_n",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "carbs_g",
    "protein_g",
    "fat_g",
    "fiber_g",
    "amount_consumed_pct",
)


@dataclass(frozen=True)
class MetricRow:
    horizon: str
    model: str
    n_events: int
    n_participants: int
    mae_mgdl: float
    rmse_mgdl: float
    participant_median_mae_mgdl: float
    participant_iqr_mae_mgdl: float


def build_event_features(
    event: pd.Series,
    sparse_readings: pd.DataFrame,
    participant_frame: pd.DataFrame,
) -> dict[str, Any]:
    timestamp = pd.Timestamp(event["event_timestamp"])
    g0 = float(sparse_readings["Dexcom GL"].iloc[-1]) if not sparse_readings.empty else np.nan
    g0_ts = sparse_readings["Timestamp"].iloc[-1] if not sparse_readings.empty else pd.NaT
    age = (
        float((timestamp - g0_ts).total_seconds() / 60.0)
        if pd.notna(g0_ts) else np.nan
    )
    source_rows = participant_frame.loc[
        participant_frame["source_row_id"].eq(int(event["source_row_id"]))
    ]
    if len(source_rows) != 1:
        raise ValueError(f"event source row is not unique: {event['event_id']}")
    source = source_rows.iloc[0]
    hour = timestamp.hour + timestamp.minute / 60.0
    return {
        "event_id": event["event_id"],
        "participant_id": event["participant_id"],
        "event_timestamp": timestamp,
        "g0_mgdl": g0,
        "g0_age_minutes": age,
        "history_n": float(len(sparse_readings)),
        "history_schedule_id": "fixed_0700_1200_2000_tolerance15_v1",
        "hour_of_day_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_of_day_cos": float(np.cos(2 * np.pi * hour / 24)),
        "carbs_g": pd.to_numeric(source.get("Carbs"), errors="coerce"),
        "protein_g": pd.to_numeric(source.get("Protein"), errors="coerce"),
        "fat_g": pd.to_numeric(source.get("Fat"), errors="coerce"),
        "fiber_g": pd.to_numeric(source.get("Fiber"), errors="coerce"),
        "amount_consumed_pct": pd.to_numeric(source.get("Amount Consumed"), errors="coerce"),
        "target_T+60": event.get("T+60"),
        "target_T+120": event.get("T+120"),
        "isolated_T+60": bool(event.get("isolated_T+60", False)),
        "isolated_T+120": bool(event.get("isolated_T+120", False)),
    }


def _target(event: pd.Series, horizon: int) -> float:
    value = event.get(f"target_T+{horizon}")
    return float(value["target_mgdl"]) if isinstance(value, dict) else np.nan


def _metric_row(horizon: str, model: str, y_true, y_pred, groups) -> MetricRow:
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    per_participant = (
        pd.DataFrame({"group": groups, "abs_error": np.abs(errors)})
        .groupby("group")["abs_error"]
        .mean()
    )
    q25, q75 = np.percentile(per_participant.to_numpy(), [25, 75])
    return MetricRow(
        horizon=horizon,
        model=model,
        n_events=len(errors),
        n_participants=int(pd.Series(groups).nunique()),
        mae_mgdl=float(np.mean(np.abs(errors))),
        rmse_mgdl=float(np.sqrt(np.mean(errors**2))),
        participant_median_mae_mgdl=float(np.median(per_participant)),
        participant_iqr_mae_mgdl=float(q75 - q25),
    )


def evaluate_baselines(
    events: pd.DataFrame,
    *,
    n_splits: int = 5,
    isolated_only: bool = False,
    feature_columns=FEATURE_COLUMNS,
    include_meal_features: bool = True,
) -> dict[str, Any]:
    """Evaluate participant-held-out baselines in explicitly matched cohorts."""
    if events.empty:
        raise ValueError("no event rows supplied")
    groups = events["participant_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two participants are required")
    n_splits = min(int(n_splits), len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    reports = {}

    for horizon in TARGET_HORIZONS_MINUTES:
        label = f"T+{horizon}"
        cohort_rows = {
            "all_target_events": [],
            "history_available_events": [],
        }
        for fold, (train_idx, test_idx) in enumerate(splitter.split(events, groups=groups)):
            train = events.iloc[train_idx].reset_index(drop=True)
            test = events.iloc[test_idx].reset_index(drop=True)
            if set(train["participant_id"]) & set(test["participant_id"]):
                raise AssertionError("participant leakage across evaluation split")

            y_train = np.asarray([_target(row, horizon) for _, row in train.iterrows()])
            y_test = np.asarray([_target(row, horizon) for _, row in test.iterrows()])
            train_valid = np.isfinite(y_train)
            test_valid = np.isfinite(y_test)
            if isolated_only:
                train_valid &= train[f"isolated_{label}"].to_numpy(dtype=bool)
                test_valid &= test[f"isolated_{label}"].to_numpy(dtype=bool)
            if not train_valid.any() or not test_valid.any():
                continue

            train_g0 = train["g0_mgdl"].to_numpy(dtype=float)
            test_g0 = test["g0_mgdl"].to_numpy(dtype=float)
            train_age = train["g0_age_minutes"].to_numpy(dtype=float)
            test_age = test["g0_age_minutes"].to_numpy(dtype=float)
            train_history_valid = (
                train_valid & np.isfinite(train_g0) & np.isfinite(train_age)
                & (train_age <= MAX_G0_AGE_MINUTES)
            )
            test_history_valid = (
                test_valid & np.isfinite(test_g0) & np.isfinite(test_age)
                & (test_age <= MAX_G0_AGE_MINUTES)
            )
            columns = tuple(feature_columns) if include_meal_features else tuple(
                column for column in feature_columns
                if column in {"g0_mgdl", "g0_age_minutes", "history_n", "hour_of_day_sin", "hour_of_day_cos"}
            )
            X_train = train.loc[:, list(columns)].replace([np.inf, -np.inf], np.nan)
            X_test = test.loc[:, list(columns)].replace([np.inf, -np.inf], np.nan)
            history_columns = tuple(
                column for column in feature_columns
                if column in {"g0_mgdl", "g0_age_minutes", "history_n", "hour_of_day_sin", "hour_of_day_cos"}
            )
            if not history_columns:
                raise ValueError("history-only evaluation requires causal history features")
            X_history_train = train.loc[:, list(history_columns)].replace([np.inf, -np.inf], np.nan)
            X_history_test = test.loc[:, list(history_columns)].replace([np.inf, -np.inf], np.nan)

            all_mean = float(np.mean(y_train[train_valid]))
            all_test = test.loc[test_valid].reset_index(drop=True)
            all_truth = y_test[test_valid]
            cohort_rows["all_target_events"].append(
                {
                    "fold": fold,
                    "event_ids": all_test["event_id"].astype(str).tolist(),
                    "participant_ids": all_test["participant_id"].astype(str).tolist(),
                    "y_true": all_truth.tolist(),
                    "predictions": {"B0": np.full(len(all_test), all_mean).tolist()},
                }
            )

            if not train_history_valid.any() or not test_history_valid.any():
                continue
            history_mean = float(np.mean(y_train[train_history_valid]))
            train_excursion = y_train[train_history_valid] - train_g0[train_history_valid]
            mean_excursion = float(np.mean(train_excursion))
            history_model = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                Ridge(alpha=10.0),
            )
            history_model.fit(
                X_history_train.loc[train_history_valid], y_train[train_history_valid]
            )
            model = None
            if include_meal_features:
                model = make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    StandardScaler(),
                    Ridge(alpha=10.0),
                )
                model.fit(X_train.loc[train_history_valid], y_train[train_history_valid])
            history_test = test.loc[test_history_valid].reset_index(drop=True)
            history_truth = y_test[test_history_valid]
            history_g0 = test_g0[test_history_valid]
            candidate_name = "Ridge_history_plus_meal" if include_meal_features else "Ridge_history_only"
            candidate_prediction = (
                model.predict(X_test.loc[test_history_valid])
                if model is not None
                else history_model.predict(X_history_test.loc[test_history_valid])
            )
            cohort_rows["history_available_events"].append(
                {
                    "fold": fold,
                    "event_ids": history_test["event_id"].astype(str).tolist(),
                    "participant_ids": history_test["participant_id"].astype(str).tolist(),
                    "y_true": history_truth.tolist(),
                    "predictions": {
                        "B0": np.full(len(history_test), history_mean).tolist(),
                        "B1": history_g0.tolist(),
                        "B2": (history_g0 + mean_excursion).tolist(),
                        candidate_name: candidate_prediction.tolist(),
                    },
                }
            )

        horizon_report = {}
        for cohort_name, folds in cohort_rows.items():
            if not folds:
                horizon_report[cohort_name] = {
                    "n_events": 0,
                    "n_participants": 0,
                    "rows": {},
                    "fold_predictions": [],
                }
                continue
            names = sorted(set().union(*(fold["predictions"].keys() for fold in folds)))
            tables = {}
            for name in names:
                selected = [fold for fold in folds if name in fold["predictions"]]
                for fold in selected:
                    if len(fold["event_ids"]) != len(fold["predictions"][name]) or len(fold["event_ids"]) != len(fold["y_true"]):
                        raise AssertionError(f"event denominator changed within fold for {cohort_name}/{name}")
                truth = [value for fold in selected for value in fold["y_true"]]
                pred = [value for fold in selected for value in fold["predictions"][name]]
                pids = [value for fold in selected for value in fold["participant_ids"]]
                tables[name] = asdict(_metric_row(label, name, truth, pred, pids))
            all_events = set().union(*(set(fold["event_ids"]) for fold in folds))
            all_participants = set().union(*(set(fold["participant_ids"]) for fold in folds))
            horizon_report[cohort_name] = {
                "n_events": len(all_events),
                "n_participants": len(all_participants),
                "rows": tables,
                "fold_predictions": folds,
            }
        reports[label] = horizon_report

    schedule_ids = sorted(set(events.get("history_schedule_id", pd.Series(dtype=str)).dropna().astype(str)))
    return {
        "evaluation": "participant_grouped_cold_start",
        "input_history": "deterministic waking/pre-meal/bedtime sparse schedule",
        "history_schedule_ids": schedule_ids,
        "input_history_limitation": "simulated missing-at-random; not observed fingerstick behavior",
        "n_participants": len(unique_groups),
        "n_splits": n_splits,
        "isolated_only": isolated_only,
        "horizons": reports,
    }
