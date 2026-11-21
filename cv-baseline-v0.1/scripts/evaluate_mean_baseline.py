#!/usr/bin/env python3
"""Build the CGMacros meal manifest and evaluate the CV B0 baseline."""

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from cv_baseline.b0_evaluation import (  # noqa: E402
    evaluate_population_mean,
    render_markdown_report,
    write_json,
    write_jsonl,
)
from cv_baseline.cgmacros_manifest import build_meal_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the image-free CV B0 population-mean baseline on CGMacros"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="directory containing CGMacros-001 style participant folders",
    )
    parser.add_argument(
        "--data-output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "cgmacros" / "results" / "b0_population_mean",
        help="ignored directory for manifest and per-meal predictions",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PACKAGE_ROOT / "reports" / "b0_population_mean",
        help="directory for the tracked summary report and metrics",
    )
    args = parser.parse_args()

    manifest = build_meal_manifest(args.dataset_root)
    predictions, metrics = evaluate_population_mean(manifest.records)
    summary = manifest.summary()
    metrics["manifest_build"] = summary

    write_jsonl(args.data_output_dir / "meal_manifest.jsonl", manifest.records)
    write_jsonl(args.data_output_dir / "predictions.jsonl", predictions)
    write_json(args.data_output_dir / "exclusions.json", manifest.exclusions)
    write_json(args.data_output_dir / "warnings.json", manifest.warnings)
    write_json(args.report_dir / "metrics.json", metrics)
    (args.report_dir / "report.md").parent.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "report.md").write_text(
        render_markdown_report(metrics, summary), encoding="utf-8"
    )

    print(f"Participants discovered : {summary['participants_discovered']}")
    print(f"Meal rows discovered    : {summary['meal_rows_discovered']}")
    print(f"Eligible meals          : {summary['eligible_meals']}")
    print(f"Excluded meals          : {summary['excluded_meals']}")
    print(f"Warnings                : {summary['warnings']}")
    print(f"Manifest                : {args.data_output_dir / 'meal_manifest.jsonl'}")
    print(f"Predictions             : {args.data_output_dir / 'predictions.jsonl'}")
    print(f"Metrics                 : {args.report_dir / 'metrics.json'}")
    print(f"Report                  : {args.report_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
