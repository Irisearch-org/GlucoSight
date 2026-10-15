from __future__ import annotations

import pandas as pd

from forecasting.data.cgmacros import audit_dataset
from forecasting.evaluate.run_cgmacros import _range_gate, run


def test_meal_feature_range_gate_blocks_training_when_source_values_fail(tmp_path):
    frame = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2025-01-01 06:00", periods=121, freq="min"),
            "Dexcom GL": [100.0] * 121,
            "Meal Type": [None] * 121,
            "Calories": [None] * 121,
            "Carbs": [None] * 121,
            "Protein": [None] * 121,
            "Fat": [None] * 121,
            "Fiber": [None] * 121,
            "Amount Consumed": [None] * 121,
            "Image path": [None] * 121,
        }
    )
    frame.loc[0, ["Meal Type", "Calories", "Carbs", "Protein", "Fat", "Fiber", "Amount Consumed"]] = [
        "lunch", 300, 30, 10, 5, 900, 100
    ]
    frame.to_csv(tmp_path / "CGMacros-001.csv", index=False)
    pd.DataFrame({"subject": [1]}).to_csv(tmp_path / "bio.csv", index=False)
    pd.DataFrame({"subject": [1]}).to_csv(tmp_path / "gut_health_test.csv", index=False)

    report = run(tmp_path)
    assert report["meal_feature_gate"]["passed"] is False
    assert report["meal_feature_evaluation"] is None
    assert report["meal_feature_gate"]["violations"]["Fiber"]["above_maximum"] == 1
    assert report["history_only_evaluation"]["status"] == "blocked"
    assert "at least two participants" in report["history_only_evaluation"]["reason"]
    assert report["history_only_evaluation"]["horizons"] == {}
