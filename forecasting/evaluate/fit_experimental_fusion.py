"""Fit the `experimental_fusion` serving artifact (Q6/Q8 of the 2026-10-20
grill, recorded here rather than as an ADR because this is explicitly a
temporary, ungated demo path, not a settled architecture decision).

What this is NOT: a single model jointly trained on CV+NLP+PPG+glucose.
No dataset pairs all three with glucose (ADR 0004), so that would require
the simulated join DATA_STRATEGY.md §3 already rejected. Instead this fits
TWO separate small ridge models, each on the one real dataset that actually
supports it, and `integration.predictor.ExperimentalFusionPredictor` blends
their outputs at serve time, weighted by each modality's own confidence
(contract.modality_weight) -- never a single fabricated feature vector.

    CV adjustment   -- CGMacros ground-truth weighed macros -> T+60 glucose
                       (45 participants, the ADR 0004 approach)
    NLP/PPG adjustment -- BIG IDEAs food-log text (+ wrist BVP once
                       downloaded) -> T+60 excursion (16 participants)

Deliberately bypasses the 4-criterion evidence gate `rppg.models.glucose_estimator`
and `nlp.models.dish_identity_ridge` use: every artifact here is written
regardless of whether it beats baseline, labelled "experimental_ungated".
It must never become the default predictor (see integration/predictor.py
and integration/api.py: default_predictor() is untouched).

The CGMacros range-violation gate (docs/CGMACROS_DATA_QUALITY_REPORT.md) is
NOT bypassed the same way -- that one is a data-integrity check, not an
accuracy check. Violating rows are filtered out instead, so the ~80% of
rows that agree with the dataset's own documented ranges still train it.

Usage::

    python -m forecasting.evaluate.fit_experimental_fusion \\
        --cgmacros-dir data/forecasting/cgmacros_extracted \\
        --bigideas-dir ~/Downloads/Bigideas-dataset \\
        --output integration/models/experimental_fusion.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from forecasting.data.cgmacros import (
    SOURCE_RANGES, extract_events, load_participant, build_targets,
)
from forecasting.data.sparsify import sparse_history
from forecasting.evaluate.baselines import (
    FEATURE_COLUMNS, build_event_features, evaluate_baselines,
)
from forecasting.data.bigideas import default_data_dir, load_cohort, HORIZONS_MINUTES
from utils.cooking_method import is_fried

RIDGE_ALPHA = 10.0  # matches forecasting/evaluate/baselines.py; F9 flagged alpha=20 as pathological


# ---------------------------------------------------------------------
# small, dependency-free ridge: fit once, persist as plain numbers (JSON),
# same shape as nlp/models/dish_identity_ridge.py and
# rppg/models/glucose_estimator.py's artifacts -- no pickled sklearn object.
# ---------------------------------------------------------------------

def _fit_ridge(X: pd.DataFrame, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> Dict[str, Any]:
    median = X.median(numeric_only=True)
    filled = X.fillna(median)
    mean = filled.mean()
    scale = filled.std(ddof=0)
    scale = scale.where(scale > 1e-9, 1.0)
    Xs = ((filled - mean) / scale).to_numpy(dtype=float)
    # Ridge closed form with an explicit intercept column, bias term unpenalised.
    design = np.hstack([np.ones((len(Xs), 1)), Xs])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {
        "feature_names": list(X.columns),
        "median": median.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "intercept": float(beta[0]),
        "coef": beta[1:].tolist(),
        "n_rows": int(len(X)),
    }


# ---------------------------------------------------------------------
# CV adjustment: CGMacros ground-truth macros -> T+60 glucose
# ---------------------------------------------------------------------

def _valid_mask(events: pd.DataFrame) -> pd.Series:
    """Rows within every SOURCE_RANGES field's documented min/max.

    docs/CGMACROS_DATA_QUALITY_REPORT.md: 136/564 sampled rows violate the
    dataset's own data dictionary. Filtered out, not bypassed -- a data
    integrity problem is not the same kind of gate as an accuracy gate.
    """
    ok = pd.Series(True, index=events.index)
    for column, (lo, hi) in SOURCE_RANGES.items():
        if column not in events.columns:
            continue
        values = pd.to_numeric(events[column], errors="coerce")
        ok &= values.isna() | ((values >= lo) & (values <= hi))
    return ok


def build_filtered_cgmacros_table(data_root: str | Path) -> tuple[pd.DataFrame, int]:
    root = Path(data_root)
    records, excluded = [], 0
    for path in sorted(root.glob("CGMacros-*.csv")):
        frame = load_participant(root, path.name)
        events = extract_events(frame)
        valid = _valid_mask(events).set_axis(events["event_id"])
        targets, _ = build_targets(frame, events)
        for _, event in targets.iterrows():
            if not bool(valid.get(event["event_id"], False)):
                excluded += 1
                continue
            history = sparse_history(frame, event["event_timestamp"])
            records.append(build_event_features(event, history, frame))
    if not records:
        raise ValueError(f"no valid meal events survived filtering under {root}")
    return pd.DataFrame.from_records(records), excluded


def fit_cgmacros_meal_model(data_root: str | Path) -> Dict[str, Any]:
    events, excluded = build_filtered_cgmacros_table(data_root)
    target = events["target_T+60"].apply(
        lambda v: float(v["target_mgdl"]) if isinstance(v, dict) else np.nan
    )
    usable = events.loc[target.notna() & events["g0_mgdl"].notna()].copy()
    y = target.loc[usable.index].to_numpy(dtype=float)
    X = usable[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")

    fit = _fit_ridge(X, y)

    # The honest, held-out number -- participant-GroupKFold, already-tested
    # code (forecasting/evaluate/baselines.py). This is NOT the artifact
    # above (that one is fit on all filtered rows, for serving); it is the
    # number that answers "does this actually beat baseline".
    held_out = None
    if usable["participant_id"].nunique() >= 2:
        report = evaluate_baselines(usable, include_meal_features=True)
        rows = report["horizons"]["T+60"]["history_available_events"]["rows"]
        held_out = {name: row["mae_mgdl"] for name, row in rows.items()}

    fit.update({
        "target": "glucose_t60_mgdl_absolute",
        "n_participants": int(usable["participant_id"].nunique()),
        "rows_excluded_range_violation": int(excluded),
        "held_out_mae_mgdl_grouped_by_participant": held_out,
        "provenance": "CGMacros v1.0.0, ground-truth weighed macros (ADR 0004)",
    })
    return fit


# ---------------------------------------------------------------------
# NLP/PPG adjustment: BIG IDEAs food-log text (+ wrist BVP) -> T+60 excursion
# ---------------------------------------------------------------------

def _ppg_feature_for_meals(data_dir: Path, cohort: pd.DataFrame,
                           window_minutes: float = 5.0) -> pd.Series:
    """pulse_rate_bpm in the window just before each meal, from real BVP.

    Only participants whose BVP_<id>.csv has actually been downloaded
    contribute a real value; everyone else is NaN (median-imputed by
    _fit_ridge, same as any other missing feature). Reuses the same
    feature extractor the live phone capture uses
    (rppg/features/extractor.py) so the coefficient learned here applies
    to the same field name (`ppg_pulse_rate_bpm`) the live API populates
    from a real finger-camera capture -- a different acquisition site
    (wrist wearable vs finger+flash), flagged, not hidden.
    """
    from rppg.features.extractor import extract_features

    values = pd.Series(np.nan, index=cohort.index)
    for participant_id in cohort["participant_id"].unique():
        bvp_path = data_dir / f"BVP_{participant_id}.csv"
        if not bvp_path.is_file():
            continue
        bvp = pd.read_csv(bvp_path, names=["datetime", "bvp"], header=0)
        bvp["datetime"] = pd.to_datetime(bvp["datetime"], errors="coerce")
        bvp = bvp.dropna(subset=["datetime", "bvp"]).sort_values("datetime")
        if len(bvp) < 64 * 10:
            continue
        fs = 1.0 / bvp["datetime"].diff().dt.total_seconds().median()
        rows = cohort.index[cohort["participant_id"] == participant_id]
        for idx in rows:
            t0 = cohort.loc[idx, "t0"]
            window = bvp.loc[
                (bvp["datetime"] >= t0 - pd.Timedelta(minutes=window_minutes))
                & (bvp["datetime"] < t0)
            ]
            if len(window) < int(fs * 10):
                continue
            feats = extract_features(window["bvp"].to_numpy(dtype=float), fs)
            if np.isfinite(feats["pulse_rate_bpm"]):
                values.loc[idx] = feats["pulse_rate_bpm"]
    return values


def fit_ppg_absolute_model() -> Dict[str, Any]:
    """PPG adjustment, take 2: BIG IDEAs' wrist BVP needs a 4.7 GB PhysioNet
    download that turned out too slow to get (2026-10-20). Pivoting to the
    dataset already on disk -- rppg/PPG_Dataset, the project's own contact
    PPG (finger+flash, the real acquisition device, not a wrist wearable)
    paired with a glucometer reading. No download needed.

    Known going in: docs/adr/0003 already closed PPG->glucose estimation on
    this exact dataset -- a constant predictor wins, independently
    replicated in a 2026 ACM ISWC paper. This fit will very likely
    reproduce that same near-zero/negative result. That's not a bug; it's
    the honest answer this dataset has always given. Reusing
    rppg.models.glucose_estimator's own loader and evaluate() (already
    tested, this is not new signal-processing code) rather than re-deriving
    either.
    """
    from rppg.models.glucose_estimator import (
        FEATURE_NAMES, FS_DEPLOY, DatasetUnavailable, build_feature_table, evaluate,
    )
    try:
        X, y, groups, dropped = build_feature_table(fs_target=FS_DEPLOY)
    except DatasetUnavailable as exc:
        return {"available": False, "reason": str(exc)}

    frame = pd.DataFrame(X, columns=list(FEATURE_NAMES))
    fit = _fit_ridge(frame, y)

    held_out = None
    if len(np.unique(groups)) >= 2:
        ev = evaluate(X, y, groups, dropped=dropped)
        held_out = {
            "model_mae": ev.model_mae,
            "baseline_mae_constant_predictor": ev.baseline_mae,
            "fold_wins": ev.fold_wins,
            "n_folds": len(ev.folds),
            "bootstrap_ci": ev.bootstrap_ci,
            "permutation_p": ev.permutation_p,
            "passes_evidence_gate": ev.passes_gate()[0],
        }

    fit.update({
        "available": True,
        "target": "glucose_absolute_mgdl_from_ppg_features_alone",
        "n_recordings": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
        "dropped": dropped,
        "held_out_gate_evaluation": held_out,
        "known_result": (
            "ADR 0003: this exact 4-criterion gate already failed on this "
            "dataset (constant predictor wins). Reproduced here, ungated, "
            "for serving wiring only -- not a re-litigation of that finding."
        ),
        "provenance": "rppg/PPG_Dataset, contact PPG (finger+flash) + glucometer, 23 subjects, 67 recordings",
    })
    return fit


def fit_bigideas_meal_model(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    cohort, problems = load_cohort(data_dir, HORIZONS_MINUTES)
    target = "glucose_60_mgdl"
    usable = cohort.dropna(subset=[target, "g0_mgdl", "carbs_g"]).copy()
    usable["delta_t60"] = usable[target] - usable["g0_mgdl"]
    usable["is_fried_cooking"] = [
        is_fried(str(name)) for name in usable.get("item_names", "")
    ]

    feature_cols = ["g0_mgdl", "carbs_g", "is_fried_cooking"]
    ppg_subjects: List[str] = []
    ppg = _ppg_feature_for_meals(data_dir, usable)
    if ppg.notna().any():
        usable["ppg_pulse_rate_bpm"] = ppg
        feature_cols.append("ppg_pulse_rate_bpm")
        ppg_subjects = sorted(
            usable.loc[ppg.notna(), "participant_id"].unique().tolist()
        )

    X = usable[feature_cols]
    y = usable["delta_t60"].to_numpy(dtype=float)
    fit = _fit_ridge(X, y)
    fit.update({
        "target": "glucose_t60_mgdl_delta_from_g0",
        "n_participants": int(usable["participant_id"].nunique()),
        "n_meals": int(len(usable)),
        "problems": problems,
        "ppg_subjects_with_real_bvp": ppg_subjects,
        "ppg_evaluation_status": (
            "UNEVALUATED -- directional only, n={} subjects, no held-out "
            "split possible at this n (C4 needs several held-out people)"
            .format(len(ppg_subjects))
            if ppg_subjects else
            "no BVP data available yet; ppg_pulse_rate_bpm has no coefficient"
        ),
        "nlp_evaluation_note": (
            "is_fried_cooking's held-out contribution is already measured "
            "and published: docs/BIGIDEAS_MEAL_RESPONSE_FINDINGS.md finding "
            "B12 -- B4 (+is_fried) is numerically identical to B3 at this "
            "cohort's 1.7% fried-item prevalence. This fit is the serving "
            "artifact, not a re-measurement of that finding."
        ),
        "provenance": "BIG IDEAs PhysioNet 1.1.2, real food-log + Dexcom pairing (16 participants)",
    })
    return fit


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cgmacros-dir", type=Path,
                        default=Path("data/forecasting/cgmacros_extracted"))
    parser.add_argument("--bigideas-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path,
                        default=Path("integration/models/experimental_fusion.json"))
    args = parser.parse_args(argv)

    blob = {
        "model_version": "experimental-fusion-v0",
        "label": "experimental_ungated",
        "warning": (
            "Bypasses the project's 4-criterion evidence gate by explicit "
            "product decision (2026-10-20). Not validated. Never the "
            "default predictor -- see integration/predictor.default_predictor()."
        ),
        "cgmacros_meal": fit_cgmacros_meal_model(args.cgmacros_dir),
        "bigideas_meal": fit_bigideas_meal_model(args.bigideas_dir),
        "ppg_absolute": fit_ppg_absolute_model(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(blob, indent=2, default=str) + "\n")
    print(f"wrote {args.output}")
    print("cgmacros_meal:", {k: v for k, v in blob["cgmacros_meal"].items()
                              if k in ("n_rows", "n_participants", "rows_excluded_range_violation",
                                       "held_out_mae_mgdl_grouped_by_participant")})
    print("bigideas_meal:", {k: v for k, v in blob["bigideas_meal"].items()
                              if k in ("n_rows", "n_meals", "n_participants",
                                       "ppg_subjects_with_real_bvp", "ppg_evaluation_status")})
    print("ppg_absolute:", {k: v for k, v in blob["ppg_absolute"].items()
                             if k in ("available", "n_recordings", "n_subjects",
                                      "held_out_gate_evaluation")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
