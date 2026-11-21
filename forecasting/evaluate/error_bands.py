"""90th percentile of absolute held-out error per horizon and g0 source -> served error bands.

Input: the `_predictions_T{h}.csv.gz` files `gbdt_forecaster` writes next to its results (one row
per held-out test meal x g0 draw, columns `y`, `gbdt`, `g0_source`). The served bands come from two
no-permutation nested runs (same outer folds, inner selection and seeds as the committed trial):

    python -m forecasting.evaluate.gbdt_forecaster --data-root <CGMacros> --permutations 0 \
        --out /tmp/gbdt_bands_main
    python -m forecasting.evaluate.gbdt_forecaster --data-root <CGMacros> --permutations 0 \
        --ppg-residuals rppg/models/reports/ppg_served_residuals.csv --out /tmp/gbdt_bands_ppg
    python -m forecasting.evaluate.error_bands --cgm-meter /tmp/gbdt_bands_main --ppg /tmp/gbdt_bands_ppg

CGM and meter bands come from the first run (models trained on CGM + meter g0, exactly like the
served models); the PPG band comes from the second (trial's simulated PPG g0, models also trained
with PPG rows). Prints JSON to paste into integration/models/served.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILE = 90.0


def band(df: pd.DataFrame, source: str) -> float:
    sub = df[df["g0_source"] == source]
    if sub.empty:
        raise ValueError(f"no rows for g0_source={source!r}")
    return float(np.percentile(np.abs(sub["gbdt"] - sub["y"]), QUANTILE))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cgm-meter", type=Path, required=True)
    ap.add_argument("--ppg", type=Path, required=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=[60, 120])
    args = ap.parse_args(argv)

    out = {}
    for h in args.horizons:
        main_df = pd.read_csv(args.cgm_meter / f"_predictions_T{h}.csv.gz")
        ppg_df = pd.read_csv(args.ppg / f"_predictions_T{h}.csv.gz")
        out[f"t{h}"] = {
            "meter": round(band(main_df, "meter"), 1),
            "cgm": round(band(main_df, "CGM"), 1),
            "ppg": round(band(ppg_df, "ppg"), 1),
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
