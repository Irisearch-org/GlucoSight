#!/usr/bin/env python3
"""Tests for CGMacros manifest construction and the image-free CV B0."""

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_baseline.b0_evaluation import evaluate_population_mean  # noqa: E402
from cv_baseline.cgmacros_manifest import build_meal_manifest  # noqa: E402
from cv_baseline.mean_baseline import (  # noqa: E402
    B0_MODEL_VERSION,
    predict_population_mean,
)


CSV_FIELDS = [
    "Unnamed: 0",
    "Timestamp",
    "Libre GL",
    "Dexcom GL",
    "HR",
    "Calories (Activity)",
    "METs",
    "Meal Type",
    "Calories",
    "Carbs",
    "Protein",
    "Fat",
    "Fiber",
    "Amount Consumed ",
    "Image path",
]


def _row(index, timestamp, meal_type="", image_path="", **macros):
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "Unnamed: 0": str(index),
            "Timestamp": timestamp,
            "Meal Type": meal_type,
            "Image path": image_path,
            "Amount Consumed ": str(macros.pop("amount_consumed", "")),
        }
    )
    for source, value in macros.items():
        row[source.capitalize()] = str(value)
    return row


class PopulationMeanPredictionTests(unittest.TestCase):
    def test_exact_contract_and_no_image_access(self):
        with patch("builtins.open", side_effect=AssertionError("image was opened")):
            result = predict_population_mean("CGMacros-001_20200501T142300")

        self.assertEqual(
            result,
            {
                "carbs_g": 45.0,
                "protein_g": 18.0,
                "fat_g": 12.0,
                "fiber_g": 4.0,
                "gi_category": 1,
                "food_class": "unknown",
                "cv_confidence": 0.0,
                "cv_present": 1,
                "carbs_source": "population_mean",
                "portion_reported": 1,
                "cv_model_version": B0_MODEL_VERSION,
            },
        )

    def test_empty_meal_id_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_population_mean("  ")


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.participant = self.root / "CGMacros-001"
        self.photos = self.participant / "photos"
        self.photos.mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_csv(self, rows):
        csv_path = self.participant / "CGMacros-001.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_pairs_after_images_and_repairs_timestamp_name_mismatch(self):
        before = "00000001-PHOTO-2020-5-1-10-0-0.jpg"
        declared_after = "00000002-PHOTO-2020-5-1-10-30-0.jpg"
        actual_after = "00000002-PHOTO-2020-5-1-10-35-0.jpg"
        (self.photos / before).write_bytes(b"before")
        (self.photos / actual_after).write_bytes(b"after")

        self._write_csv(
            [
                _row(
                    1,
                    "2020-05-01 10:00:00",
                    meal_type="Breakfast",
                    image_path=f"photos/{before}",
                    carbs=50,
                    protein=20,
                    fat=10,
                    fiber=5,
                    amount_consumed=80,
                ),
                _row(2, "2020-05-01 10:30:00", image_path=f"photos/{declared_after}"),
            ]
        )

        result = build_meal_manifest(self.root)

        self.assertEqual(len(result.records), 1)
        meal = result.records[0]
        self.assertEqual(meal["meal_id"], "CGMacros-001_20200501T100000")
        self.assertTrue(meal["before_image_path"].endswith(before))
        self.assertEqual(len(meal["after_image_paths"]), 1)
        self.assertTrue(meal["after_image_paths"][0].endswith(actual_after))
        self.assertEqual(meal["amount_consumed_pct"], 80.0)
        self.assertEqual(meal["ground_truth_provenance"], "cgmacros_reported_estimate")
        self.assertEqual(result.summary()["warning_codes"], {"after_image_path_repaired": 1})

    def test_excludes_missing_image_and_missing_macro(self):
        valid_photo = "00000003-PHOTO-2020-5-1-12-0-0.jpg"
        (self.photos / valid_photo).write_bytes(b"meal")
        self._write_csv(
            [
                _row(
                    1,
                    "2020-05-01 10:00:00",
                    meal_type="Breakfast",
                    image_path="photos/does-not-exist.jpg",
                    carbs=50,
                    protein=20,
                    fat=10,
                    fiber=5,
                ),
                _row(
                    2,
                    "2020-05-01 12:00:00",
                    meal_type="Lunch",
                    image_path=f"photos/{valid_photo}",
                    carbs=60,
                    protein=20,
                    fat=10,
                ),
            ]
        )

        result = build_meal_manifest(self.root)

        self.assertEqual(result.records, [])
        self.assertEqual(
            result.summary()["exclusion_reasons"],
            {"missing_or_invalid_macros": 1, "missing_or_unresolved_before_image": 1},
        )

    def test_normalises_meal_type_and_excludes_documented_range_outlier(self):
        first_photo = "00000004-PHOTO-2020-5-1-8-0-0.jpg"
        second_photo = "00000005-PHOTO-2020-5-1-12-0-0.jpg"
        (self.photos / first_photo).write_bytes(b"meal")
        (self.photos / second_photo).write_bytes(b"meal")
        self._write_csv(
            [
                _row(
                    1,
                    "2020-05-01 08:00:00",
                    meal_type="breakfast",
                    image_path=f"photos/{first_photo}",
                    carbs=50,
                    protein=20,
                    fat=10,
                    fiber=5,
                ),
                _row(
                    2,
                    "2020-05-01 12:00:00",
                    meal_type="snack 1",
                    image_path=f"photos/{second_photo}",
                    carbs=50,
                    protein=20,
                    fat=10,
                    fiber=2300,
                ),
            ]
        )

        result = build_meal_manifest(self.root)

        self.assertEqual([record["meal_type"] for record in result.records], ["Breakfast"])
        self.assertEqual(
            result.summary()["exclusion_reasons"],
            {"macro_out_of_documented_range": 1},
        )


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def _record(meal_id, participant_id, timestamp, meal_type, carbs):
        return {
            "meal_id": meal_id,
            "participant_id": participant_id,
            "timestamp": timestamp,
            "meal_type": meal_type,
            "before_image_path": f"{participant_id}/photos/{meal_id}.jpg",
            "true_carbs_g": float(carbs),
            "true_protein_g": 18.0,
            "true_fat_g": 12.0,
            "true_fiber_g": 4.0,
            "source_dataset": "cgmacros",
            "dataset_version": "1.0.0",
            "dataset_license": "CC BY-NC-SA 4.0",
            "ground_truth_provenance": "cgmacros_reported_estimate",
        }

    def test_metrics_match_manual_calculation(self):
        records = [
            self._record("m1", "P1", "2020-01-01 08:00:00", "Breakfast", 50),
            self._record("m2", "P1", "2020-01-02 08:00:00", "Breakfast", 70),
            self._record("m3", "P2", "2020-01-01 12:00:00", "Lunch", 45),
        ]

        predictions, metrics = evaluate_population_mean(records)

        self.assertEqual(len(predictions), 3)
        self.assertAlmostEqual(metrics["pooled"]["carbs_g"]["mae_g"], 10.0)
        self.assertAlmostEqual(
            metrics["pooled"]["carbs_g"]["rmse_g"], math.sqrt(650.0 / 3.0)
        )
        summary = metrics["participant_summary"]["carbs_g"]
        self.assertAlmostEqual(summary["mae_median_g"], 7.5)
        self.assertAlmostEqual(summary["mae_q1_g"], 3.75)
        self.assertAlmostEqual(summary["mae_q3_g"], 11.25)
        self.assertEqual(metrics["selection"]["participants_evaluated"], 2)
        self.assertEqual(metrics["by_meal_type"]["Lunch"]["carbs_g"]["mae_g"], 0.0)

    def test_empty_manifest_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_population_mean([])


if __name__ == "__main__":
    unittest.main()
