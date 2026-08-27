"""Evaluation and reporting utilities for the CV B0 baseline."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .mean_baseline import B0_MODEL_VERSION, predict_population_mean


MACROS = ("carbs_g", "protein_g", "fat_g", "fiber_g")


def _metric_block(true_values: Sequence[float], predicted_values: Sequence[float]) -> dict:
    true_array = np.asarray(true_values, dtype=np.float64)
    predicted_array = np.asarray(predicted_values, dtype=np.float64)
    error = predicted_array - true_array
    return {
        "mae_g": float(np.mean(np.abs(error))),
        "rmse_g": float(math.sqrt(float(np.mean(np.square(error))))),
        "n_meals": int(true_array.size),
    }


def _manifest_hash(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def evaluate_population_mean(records: Iterable[Mapping[str, object]]) -> Tuple[List[dict], dict]:
    ordered = sorted(
        (dict(record) for record in records),
        key=lambda item: (str(item["participant_id"]), str(item["timestamp"])),
    )
    if not ordered:
        raise ValueError("No eligible CGMacros meals were provided for B0 evaluation")

    fixed_prediction = predict_population_mean(str(ordered[0]["meal_id"]))

    predictions: List[dict] = []
    for record in ordered:
        baseline = predict_population_mean(str(record["meal_id"]))
        true_values = {macro: float(record[f"true_{macro}"]) for macro in MACROS}
        predicted_values = {macro: float(baseline[macro]) for macro in MACROS}
        predictions.append(
            {
                "meal_id": record["meal_id"],
                "participant_id": record["participant_id"],
                "timestamp": record["timestamp"],
                "meal_type": record["meal_type"],
                "before_image_path": record["before_image_path"],
                "ground_truth_provenance": record["ground_truth_provenance"],
                "true": true_values,
                "predicted": predicted_values,
                "absolute_error": {
                    macro: abs(predicted_values[macro] - true_values[macro])
                    for macro in MACROS
                },
                "cv_features": baseline,
            }
        )

    pooled = {
        macro: _metric_block(
            [float(record[f"true_{macro}"]) for record in ordered],
            [float(fixed_prediction[macro]) for _ in ordered],
        )
        for macro in MACROS
    }

    by_participant: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for record in ordered:
        by_participant[str(record["participant_id"])].append(record)

    participant_metrics: Dict[str, dict] = {}
    for participant_id, participant_records in sorted(by_participant.items()):
        participant_metrics[participant_id] = {
            macro: _metric_block(
                [float(record[f"true_{macro}"]) for record in participant_records],
                [float(fixed_prediction[macro]) for _ in participant_records],
            )
            for macro in MACROS
        }

    participant_summary = {}
    for macro in MACROS:
        mae_values = np.asarray(
            [metrics[macro]["mae_g"] for metrics in participant_metrics.values()],
            dtype=np.float64,
        )
        rmse_values = np.asarray(
            [metrics[macro]["rmse_g"] for metrics in participant_metrics.values()],
            dtype=np.float64,
        )
        participant_summary[macro] = {
            "mae_median_g": float(np.median(mae_values)),
            "mae_q1_g": float(np.percentile(mae_values, 25, method="linear")),
            "mae_q3_g": float(np.percentile(mae_values, 75, method="linear")),
            "rmse_median_g": float(np.median(rmse_values)),
            "rmse_q1_g": float(np.percentile(rmse_values, 25, method="linear")),
            "rmse_q3_g": float(np.percentile(rmse_values, 75, method="linear")),
            "n_participants": len(participant_metrics),
        }

    by_meal_type_records: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for record in ordered:
        by_meal_type_records[str(record["meal_type"])].append(record)
    by_meal_type = {
        meal_type: {
            macro: _metric_block(
                [float(record[f"true_{macro}"]) for record in meal_records],
                [float(fixed_prediction[macro]) for _ in meal_records],
            )
            for macro in MACROS
        }
        for meal_type, meal_records in sorted(by_meal_type_records.items())
    }

    first = ordered[0]
    metrics = {
        "baseline": {
            "name": "B0 population mean",
            "model_version": B0_MODEL_VERSION,
            "reads_images": False,
            "fixed_predictions": fixed_prediction,
        },
        "dataset": {
            "source_dataset": first["source_dataset"],
            "dataset_version": first["dataset_version"],
            "dataset_license": first["dataset_license"],
            "ground_truth_provenance": first["ground_truth_provenance"],
            "manifest_sha256": _manifest_hash(ordered),
        },
        "selection": {
            "meals_evaluated": len(ordered),
            "participants_evaluated": len(by_participant),
            "meal_type_counts": dict(sorted(Counter(str(r["meal_type"]) for r in ordered).items())),
        },
        "pooled": pooled,
        "participant_summary": participant_summary,
        "by_meal_type": by_meal_type,
    }
    return predictions, metrics


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_markdown_report(metrics: Mapping[str, object], manifest_summary: Mapping[str, object]) -> str:
    pooled = metrics["pooled"]
    participant = metrics["participant_summary"]
    selection = metrics["selection"]
    dataset = metrics["dataset"]
    meal_types = metrics["by_meal_type"]

    lines = [
        "# CV B0 Population-Mean Baseline — CGMacros",
        "",
        "## Summary",
        "",
        f"- Dataset: CGMacros {dataset['dataset_version']}",
        f"- License: {dataset['dataset_license']}",
        f"- Meals evaluated: {selection['meals_evaluated']}",
        f"- Participants evaluated: {selection['participants_evaluated']}",
        f"- Ground-truth provenance: `{dataset['ground_truth_provenance']}`",
        f"- Manifest SHA-256: `{dataset['manifest_sha256']}`",
        "- Images read by predictor: **no**",
        "",
        "CGMacros' local data dictionary describes meal macros as reported estimates. "
        "This report therefore does not label them as weighed ground truth.",
        "",
        "## Fixed Prediction",
        "",
        "`carbs_g=45.0`, `protein_g=18.0`, `fat_g=12.0`, `fiber_g=4.0`, "
        "`gi_category=1`.",
        "",
        "## Pooled Metrics",
        "",
        "| Macro | MAE (g) | RMSE (g) | Meals |",
        "|---|---:|---:|---:|",
    ]
    for macro in MACROS:
        block = pooled[macro]
        lines.append(
            f"| {macro} | {block['mae_g']:.2f} | {block['rmse_g']:.2f} | {block['n_meals']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Participant Distribution",
            "",
            "| Macro | MAE median [IQR] (g) | RMSE median [IQR] (g) | Participants |",
            "|---|---:|---:|---:|",
        ]
    )
    for macro in MACROS:
        block = participant[macro]
        lines.append(
            f"| {macro} | {block['mae_median_g']:.2f} "
            f"[{block['mae_q1_g']:.2f}, {block['mae_q3_g']:.2f}] | "
            f"{block['rmse_median_g']:.2f} "
            f"[{block['rmse_q1_g']:.2f}, {block['rmse_q3_g']:.2f}] | "
            f"{block['n_participants']} |"
        )

    lines.extend(["", "## Diagnostics by Meal Type", ""])
    for meal_type, blocks in meal_types.items():
        lines.extend(
            [
                f"### {meal_type}",
                "",
                "| Macro | MAE (g) | RMSE (g) | Meals |",
                "|---|---:|---:|---:|",
            ]
        )
        for macro in MACROS:
            block = blocks[macro]
            lines.append(
                f"| {macro} | {block['mae_g']:.2f} | {block['rmse_g']:.2f} | {block['n_meals']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Manifest Audit",
            "",
            f"- Participant folders discovered: {manifest_summary['participants_discovered']}",
            f"- Meal-start rows discovered: {manifest_summary['meal_rows_discovered']}",
            f"- Eligible meals: {manifest_summary['eligible_meals']}",
            f"- Excluded meals: {manifest_summary['excluded_meals']}",
            f"- Manifest warnings: {manifest_summary['warnings']}",
            f"- Exclusion reasons: `{json.dumps(manifest_summary['exclusion_reasons'], sort_keys=True)}`",
            f"- Warning codes: `{json.dumps(manifest_summary['warning_codes'], sort_keys=True)}`",
            "",
            "Meal rows require complete macros within the ranges documented by CGMacros "
            "(0-176 g per macro) and a resolvable start image. "
            "Image-only rows after a meal start are retained as after-meal metadata and never "
            "become independent B0 inputs.",
            "",
            "## Interpretation",
            "",
            "B0 is a deliberately non-informative reference, not a model-quality claim. "
            "Every later CV result must be evaluated on the same manifest and printed beside "
            "these values.",
            "",
        ]
    )
    return "\n".join(lines)
