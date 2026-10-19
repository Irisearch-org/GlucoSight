from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from forecasting.data.cgmacros import (
    DATASET_VERSION,
    SOURCE_RANGES,
    audit_dataset,
    build_targets,
    extract_events,
    load_participant,
)
from forecasting.data.sparsify import sparse_history
from forecasting.evaluate.baselines import build_event_features, evaluate_baselines


HISTORY_FEATURES = (
    "g0_mgdl",
    "g0_age_minutes",
    "history_n",
    "hour_of_day_sin",
    "hour_of_day_cos",
)


def build_event_table(data_root: str | Path) -> pd.DataFrame:
    root = Path(data_root)
    records = []
    participant_files = sorted(root.glob("CGMacros-*.csv"))
    for path in participant_files:
        frame = load_participant(root, path)
        events = extract_events(frame)
        targets, _ = build_targets(frame, events)
        for _, event in targets.iterrows():
            history = sparse_history(frame, event["event_timestamp"])
            record = build_event_features(event, history, frame)
            records.append(record)
    if not records:
        raise ValueError(f"no meal events found in {root}")
    return pd.DataFrame.from_records(records)


def _range_gate(audit: dict) -> dict:
    violations = audit["range_violations"]
    invalid = {
        field: counts
        for field, counts in violations.items()
        if counts["below_minimum"] or counts["above_maximum"]
    }
    return {
        "passed": not invalid,
        "source_dictionary_ranges": SOURCE_RANGES,
        "violations": invalid,
        "reason": None if not invalid else "meal features violate published source ranges; meal-feature modeling is blocked pending provenance/unit reconciliation",
    }


def run(data_root: str | Path, *, isolated_only: bool = False) -> dict:
    # require_metadata=False: bio.csv/gut_health_test.csv feed only the
    # oracle-tier fields a phone can never serve (age/BMI/labs/gut panel).
    # The range-violation gate below never reads them, so their absence
    # (this environment's copy of CGMacros ships without them) must not
    # block the audit itself -- audit_dataset reports what it actually
    # found either way (see its own `metadata_available` field).
    audit = asdict(audit_dataset(data_root, require_metadata=False))
    gate = _range_gate(audit)
    events = build_event_table(data_root)
    if events["participant_id"].nunique() < 2:
        history_evaluation = {
            "status": "blocked",
            "reason": "participant-held-out evaluation requires at least two participants",
            "n_participants": int(events["participant_id"].nunique()),
            "horizons": {},
        }
    else:
        history_evaluation = evaluate_baselines(
            events,
            isolated_only=isolated_only,
            include_meal_features=False,
            feature_columns=HISTORY_FEATURES,
        )
        history_evaluation["status"] = "completed"
    meal_feature_evaluation = None
    if gate["passed"]:
        meal_feature_evaluation = evaluate_baselines(
            events,
            isolated_only=isolated_only,
            include_meal_features=True,
        )
    return {
        "dataset": "CGMacros",
        "dataset_version": DATASET_VERSION,
        "task": "event-indexed Dexcom glucose estimation",
        "time_semantics": "dataset-reported meal/photo event time; not established as first bite",
        "target_rule": "exact observed Dexcom timestamp at event time + horizon; no interpolation",
        "event_policy": "all meal markers retained; isolated-only sensitivity also available",
        "input_history_schedule": "fixed waking/pre-meal/bedtime sparse schedule; simulated data, not actual fingerstick behavior",
        "target_selection": {
            "event_count": len(events),
            "events_with_any_exact_history_g0": int(events["g0_mgdl"].notna().sum()),
            "events_without_sparse_g0": int(events["g0_mgdl"].isna().sum()),
            "stale_g0_over_360_minutes": int((events["g0_age_minutes"] > 360).sum()),
        },
        "audit": audit,
        "meal_feature_gate": gate,
        "history_only_evaluation": history_evaluation,
        "meal_feature_evaluation": meal_feature_evaluation,
        "claim_limit": "No trained glucose predictor or modality improvement is established by this baseline pipeline.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run participant-held-out CGMacros baselines")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--isolated-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.data_root, isolated_only=args.isolated_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("Meal-feature gate:", "PASS" if report["meal_feature_gate"]["passed"] else "BLOCKED")
    if not report["meal_feature_gate"]["passed"]:
        print(report["meal_feature_gate"]["reason"])
    for horizon, cohorts in report["history_only_evaluation"]["horizons"].items():
        print(horizon)
        for cohort_name, cohort in cohorts.items():
            print(f"  {cohort_name}: events={cohort['n_events']} participants={cohort['n_participants']}")
            for name, row in cohort["rows"].items():
                print(
                    f"    {name}: MAE={row['mae_mgdl']:.2f} RMSE={row['rmse_mgdl']:.2f} "
                    f"participant-median-MAE={row['participant_median_mae_mgdl']:.2f} "
                    f"participant-IQR={row['participant_iqr_mae_mgdl']:.2f}"
                )
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
