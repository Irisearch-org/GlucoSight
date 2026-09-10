"""Result reporting that cannot omit the baselines.

Finding C3: "Clarke Zone A is +/-20% of reference; a constant predictor
plausibly reaches 50-70% Zone A on a T2D cohort. No fusion result is
meaningful without B0/B1/B2 in the same table."

DATA_STRATEGY §7 adds: "no track reports a model number that is not printed
next to its baseline in the same table. That is finding C3, and in a
compressed week it is the first thing that gets skipped."

So `results_table` refuses to build a table that has model predictions and
no baselines. It is easier to comply than to work around, which is the only
kind of discipline that survives a deadline.

This module deliberately takes plain arrays rather than a dataset loader, so
it can be used directly from the Kaggle notebook where CGMacros lives:

    from integration.report import results_table, print_table
    tbl = results_table(
        y_true=y_true, participant_ids=pids, horizon="T+60",
        baselines={"B0": b0, "B1": b1, "B2": b2},
        models={"GBDT (deployable tier)": pred_gbdt},
    )
    print_table(tbl)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from utils.clarke_grid import (
    clarke_error_grid,
    per_participant_zone_a,
    zone_a_by_reference_range,
)

REQUIRED_BASELINES = ("B0", "B1", "B2")

BASELINE_MEANING = {
    "B0": "population mean of the training split (predicts one number for everyone)",
    "B1": "persistence — the last causal pre-meal reading, unchanged",
    "B2": "persistence + the training-split mean excursion",
}


class MissingBaseline(ValueError):
    """Raised when a table would report a model without its baselines."""


def _row(name: str, kind: str, y_true, y_pred, participant_ids) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    _, pct = clarke_error_grid(y_true, y_pred)
    row = {
        "name": name,
        "kind": kind,
        "n": int(len(y_true)),
        "rmse": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 2),
        "mae": round(float(np.mean(np.abs(y_true - y_pred))), 2),
        "zone_a": pct["A"],
        "zone_ab": round(pct["A"] + pct["B"], 2),
        "zone_c": pct["C"],
        "zone_d": pct["D"],
        "zone_e": pct["E"],
    }
    if participant_ids is not None:
        row.update(per_participant_zone_a(y_true, y_pred, participant_ids))
    return row


def results_table(
    *,
    y_true: Sequence[float],
    horizon: str,
    baselines: Mapping[str, Sequence[float]],
    models: Mapping[str, Sequence[float]],
    participant_ids: Optional[Sequence] = None,
    n_held_out_participants: Optional[int] = None,
    require_baselines: bool = True,
) -> Dict[str, Any]:
    """Build one results table. Baselines are not optional.

    Every prediction array must cover the SAME samples as `y_true`. The
    Sprint 2 notebook compared a calibrated model on 1534 meals against an
    uncalibrated one on 1669 and reported both in one table; that comparison
    is not like-for-like and the length check below is what catches it.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()

    if require_baselines:
        missing = [b for b in REQUIRED_BASELINES if b not in baselines]
        if missing and models:
            raise MissingBaseline(
                f"refusing to report {sorted(models)} without baselines "
                f"{missing}. Finding C3: a Clarke Zone A number without "
                f"B0/B1/B2 beside it is not a result. Pass them, or set "
                f"require_baselines=False and state in the caption that the "
                f"table is incomplete."
            )

    rows: List[Dict[str, Any]] = []
    for group, kind in ((baselines, "baseline"), (models, "model")):
        for name, preds in group.items():
            preds = np.asarray(preds, dtype=float).ravel()
            if len(preds) != len(y_true):
                raise ValueError(
                    f"{name!r} has {len(preds)} predictions for {len(y_true)} "
                    f"references. Rows in one table must cover the same "
                    f"samples — comparing a model evaluated on a subset "
                    f"against a baseline evaluated on the full set overstates "
                    f"the model by however the subset differs."
                )
            rows.append(_row(name, kind, y_true, preds, participant_ids))

    best_baseline = max(
        (r for r in rows if r["kind"] == "baseline"),
        key=lambda r: r["zone_a"], default=None,
    )
    for r in rows:
        if r["kind"] == "model" and best_baseline is not None:
            r["zone_a_over_best_baseline"] = round(
                r["zone_a"] - best_baseline["zone_a"], 2)
            r["rmse_over_best_baseline"] = round(
                r["rmse"] - best_baseline["rmse"], 2)

    return {
        "horizon": horizon,
        "n_samples": int(len(y_true)),
        "n_held_out_participants": n_held_out_participants,
        "best_baseline": None if best_baseline is None else best_baseline["name"],
        "rows": rows,
        "reference_histogram": _histogram(y_true),
        "stratified_by_reference": None,
    }


def add_stratification(table: Dict[str, Any], y_true, y_pred) -> Dict[str, Any]:
    """Attach Zone A stratified by reference range, per finding C3."""
    table["stratified_by_reference"] = zone_a_by_reference_range(y_true, y_pred)
    return table


def _histogram(y_true: np.ndarray, bins=(0, 70, 100, 140, 180, 250, 400)) -> List[Dict]:
    """The reference-glucose histogram C3 requires with every Clarke result.

    Without it, "Zone E: 0.0%" reads as "safe" when it means "no data there"
    (DATA_STRATEGY §4.4)."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = int(((y_true >= lo) & (y_true < hi)).sum())
        out.append({"range": f"[{lo}, {hi})", "n": n,
                    "pct": round(100 * n / len(y_true), 2)})
    return out


def print_table(table: Dict[str, Any]) -> None:
    rows = table["rows"]
    has_pp = any("median_zone_a" in r for r in rows)

    header = f"{'Setting':<34}{'kind':<10}{'n':>6}{'RMSE':>9}{'MAE':>8}{'ZoneA%':>9}{'A+B%':>8}{'C%':>6}{'D%':>6}{'E%':>6}"
    if has_pp:
        header += f"{'pp-med':>9}{'pp-IQR':>9}"
    line = "=" * len(header)

    print(line)
    print(f"  {table['horizon']}   n={table['n_samples']}"
          + (f"   held-out participants={table['n_held_out_participants']}"
             if table["n_held_out_participants"] is not None else ""))
    print(line)
    print(header)
    print("-" * len(header))
    for r in rows:
        s = (f"{r['name']:<34}{r['kind']:<10}{r['n']:>6}{r['rmse']:>9.2f}"
             f"{r['mae']:>8.2f}{r['zone_a']:>9.1f}{r['zone_ab']:>8.1f}"
             f"{r['zone_c']:>6.1f}{r['zone_d']:>6.1f}{r['zone_e']:>6.1f}")
        if has_pp:
            s += f"{r.get('median_zone_a', float('nan')):>9.1f}{r.get('iqr_zone_a', float('nan')):>9.1f}"
        print(s)
    print("-" * len(header))

    for r in rows:
        if "zone_a_over_best_baseline" in r:
            delta = r["zone_a_over_best_baseline"]
            verdict = "above" if delta > 0 else "at or below"
            print(f"  {r['name']}: {delta:+.1f} pts Zone A {verdict} the best "
                  f"baseline ({table['best_baseline']}), "
                  f"RMSE {r['rmse_over_best_baseline']:+.2f} mg/dL")

    print(f"\n  Reference glucose distribution (n={table['n_samples']}):")
    for b in table["reference_histogram"]:
        print(f"    {b['range']:<14} n={b['n']:<6} {b['pct']:>5.1f}%")

    if table.get("stratified_by_reference"):
        print("\n  Zone A stratified by reference range:")
        for s in table["stratified_by_reference"]:
            if s["n"] == 0:
                print(f"    {s['range']:<14} n=0      NO DATA — not 'safe here'")
            else:
                print(f"    {s['range']:<14} n={s['n']:<6} "
                      f"Zone A {s['zone_a_pct']:>5.1f}%  A+B {s['zone_ab_pct']:>5.1f}%")

    print("\n  Baselines in this table:")
    for name in REQUIRED_BASELINES:
        if any(r["name"] == name for r in rows):
            print(f"    {name} = {BASELINE_MEANING[name]}")
    print(line)
