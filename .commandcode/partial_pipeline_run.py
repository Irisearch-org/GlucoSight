"""Run forecasting.evaluate.run_cgmacros's pipeline against the 15/45-participant
partial CGMacros sample, substituting the already-computed range-violation
result for audit_dataset() (which needs bio.csv/gut_health_test.csv -- not
recovered from the truncated download, see docs/CGMACROS_DATA_QUALITY_REPORT.md).

Mirrors run_cgmacros.run(): meal-feature evaluation only runs if the gate
passes. It doesn't here (same violations found in the real audit), so this
reports B0/B1/B2 history-only baselines, same as the full pipeline would.

Run: python -I .commandcode/partial_pipeline_run.py <abs path to extracted csvs>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecasting.data.cgmacros import DATASET_VERSION, SOURCE_RANGES
from forecasting.evaluate.baselines import evaluate_baselines
from forecasting.evaluate.run_cgmacros import HISTORY_FEATURES, build_event_table

# From docs/CGMACROS_DATA_QUALITY_REPORT.md -- reproduced against this same
# 15-participant subset; the full 45-participant gate is extrapolated to also
# fail, not separately verified.
KNOWN_RANGE_VIOLATIONS = {
    "Calories": {"below_minimum": 13, "above_maximum": 9},
    "Carbs": {"below_minimum": 0, "above_maximum": 8},
    "Protein": {"below_minimum": 86, "above_maximum": 0},
    "Fat": {"below_minimum": 0, "above_maximum": 0},
    "Fiber": {"below_minimum": 0, "above_maximum": 13},
    "Amount Consumed": {"below_minimum": 0, "above_maximum": 37},
}


def main(data_root: str) -> None:
    root = Path(data_root)
    events = build_event_table(root)
    gate = {
        "passed": False,
        "source_dictionary_ranges": SOURCE_RANGES,
        "violations": KNOWN_RANGE_VIOLATIONS,
        "reason": "meal features violate published source ranges; meal-feature modeling is blocked pending provenance/unit reconciliation",
    }
    n_participants = events["participant_id"].nunique()
    history_evaluation = evaluate_baselines(
        events,
        isolated_only=False,
        include_meal_features=False,
        feature_columns=HISTORY_FEATURES,
    )
    history_evaluation["status"] = "completed"
    report = {
        "dataset": "CGMacros",
        "dataset_version": DATASET_VERSION,
        "sample": "PARTIAL -- 15/45 participants (truncated download, see docs/CGMACROS_DATA_QUALITY_REPORT.md)",
        "n_participants": int(n_participants),
        "n_events": len(events),
        "meal_feature_gate": gate,
        "history_only_evaluation": history_evaluation,
        "meal_feature_evaluation": None,
        "claim_limit": "No trained glucose predictor or modality improvement is established by this baseline pipeline. Partial-sample numbers are not the full-cohort result.",
    }
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("partial_pipeline_result.json")
    out_path.write_text(json.dumps(report, indent=2, default=str))

    print("Meal-feature gate: BLOCKED (known from real audit)")
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
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
