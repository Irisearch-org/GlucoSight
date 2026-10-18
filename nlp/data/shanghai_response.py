"""Rebuild the ShanghaiT2DM meal response variable from the raw workbooks.

Finding N6 in ``docs/NLP_PATH_A_GATE.md`` established that the committed
Path A feature table carries glucose *at* the meal and no outcome, because
the generator kept only the meal rows and discarded the 15-minute CGM grid.
The grid was never missing -- it is 112,379 readings in the v3 release --
only unused.

This module turns each meal row into a supervised example:

``g0``
    The last CGM reading at or before ``t0``.  Never a future reading, and
    never one older than ``max_g0_staleness_minutes``.
``glucose_t{h}``
    The reading nearest ``t0 + h``, accepted only within
    ``outcome_tolerance_minutes``.
``rise_t{h}``
    ``glucose_t{h} - g0``, the quantity the BIG IDEAs analysis showed is the
    only place a meal's value is visible (B3/B4).

Exclusion is explicit, never silent: every rejected horizon carries a
``reason_t{h}``, and :func:`build_response_table` returns a summary counting
them.  Nothing is imputed and nothing is clipped.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .shanghai_schema import (
    SCHEMA_VERSION,
    SOURCE_DATASET,
    read_sheet,
    workbook_paths,
)

#: Meal cells that record the *absence* of a meal rather than a meal.
PLACEHOLDER_TEXTS = frozenset(
    {
        "",
        "/",
        "-",
        "—",
        "–",
        "not available",
        "data not available",
        "data  not available",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "未记录",
        "未进食",
        "无",
    }
)

#: Every value ``reason_t{h}`` can take, in priority order.
REASONS = (
    "no_g0",
    "g0_stale",
    "g0_out_of_range",
    "no_outcome",
    "outcome_out_of_range",
    "series_gap",
)


@dataclass(frozen=True)
class ResponseConfig:
    """Rules for anchoring a meal to the CGM grid."""

    horizons_minutes: Tuple[int, ...] = (60, 120)
    #: How old the pre-meal reading may be. One grid step (15 min) is exact.
    max_g0_staleness_minutes: float = 15.0
    #: How far the outcome reading may sit from ``t0 + h``. One grid step.
    outcome_tolerance_minutes: float = 15.0
    #: Largest tolerated hole in the series between ``g0`` and the outcome.
    max_series_gap_minutes: float = 45.0
    min_glucose_mgdl: float = 40.0
    max_glucose_mgdl: float = 400.0
    #: Length of an identical-value run that is worth flagging as a clamp.
    clamp_run_min: int = 4
    #: A bolus this close to the meal confounds the rise.
    insulin_window_minutes: float = 30.0


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def repeat_run_lengths(values: np.ndarray) -> np.ndarray:
    """Length of the identical-value run each position belongs to."""
    values = np.asarray(values)
    length = values.size
    runs = np.ones(length, dtype=int)
    if length == 0:
        return runs
    start = 0
    for index in range(1, length + 1):
        if index == length or values[index] != values[start]:
            runs[start:index] = index - start
            start = index
    return runs


def meal_mask(sheet: pd.DataFrame) -> pd.Series:
    """Rows that record an actual meal."""
    text = sheet["dietary_intake"].fillna("").astype(str).str.strip()
    folded = text.str.lower()
    return (text != "") & ~folded.isin(PLACEHOLDER_TEXTS)


def meal_id(participant_id: str, t0, source_row: int) -> str:
    """Stable id in the legacy scheme, so old and new tables join."""
    stamp = pd.Timestamp(t0).strftime("%Y%m%d%H%M")
    return f"SH-{participant_id}-{stamp}-{int(source_row):04d}"


def _minutes(delta: np.timedelta64) -> float:
    return float(delta / np.timedelta64(1, "m"))


def _cgm_series(sheet: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    ok = sheet["timestamp"].notna() & sheet["cgm_mgdl"].notna()
    subset = sheet.loc[ok, ["timestamp", "cgm_mgdl"]]
    if subset.empty:
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype=float)
    order = np.argsort(subset["timestamp"].to_numpy(), kind="stable")
    timestamps = subset["timestamp"].to_numpy()[order].astype("datetime64[ns]")
    glucose = subset["cgm_mgdl"].to_numpy(dtype=float)[order]
    return timestamps, glucose


def _insulin_times(sheet: pd.DataFrame) -> np.ndarray:
    """Timestamps of meal-timed insulin: bolus or a subcutaneous dose."""
    doses = sheet[["insulin_bolus_iu", "insulin_sc_iu"]].fillna(0.0)
    given = (doses > 0).any(axis=1) & sheet["timestamp"].notna()
    if not given.any():
        return np.array([], dtype="datetime64[ns]")
    return (
        sheet.loc[given, "timestamp"].to_numpy().astype("datetime64[ns]")
    )


# ---------------------------------------------------------------------
# core
# ---------------------------------------------------------------------


def compute_responses(
    sheet: pd.DataFrame,
    config: Optional[ResponseConfig] = None,
) -> pd.DataFrame:
    """Attach ``g0`` and per-horizon outcomes to every meal row."""
    cfg = config or ResponseConfig()
    timestamps, glucose = _cgm_series(sheet)
    runs = repeat_run_lengths(glucose)
    insulin = _insulin_times(sheet)

    participant = str(sheet["participant_id"].iloc[0])
    subject = str(sheet["subject_id"].iloc[0])
    period = int(sheet["period_index"].iloc[0])

    records: List[dict] = []
    for position in np.flatnonzero(meal_mask(sheet).to_numpy()):
        row = sheet.iloc[position]
        t0 = row["timestamp"]
        record: Dict[str, object] = {
            "meal_id": "",
            "participant_id": participant,
            "subject_id": subject,
            "period_index": period,
            "source_row": int(row["source_row"]),
            "t0_timestamp_iso": pd.NaT if pd.isna(t0) else pd.Timestamp(t0).isoformat(),
            "dietary_intake": row["dietary_intake"],
            "dietary_intake_cn": row["dietary_intake_cn"],
            "g0_mgdl": math.nan,
            "g0_staleness_min": math.nan,
            "g0_repeat_run": 0,
            "insulin_near_meal": 0,
            "cgm_clamped_at_g0": 0,
        }
        for horizon in cfg.horizons_minutes:
            record[f"glucose_t{horizon}"] = math.nan
            record[f"rise_t{horizon}"] = math.nan
            record[f"valid_t{horizon}"] = 0
            record[f"reason_t{horizon}"] = ""

        if pd.isna(t0):
            for horizon in cfg.horizons_minutes:
                record[f"reason_t{horizon}"] = "no_g0"
            record["meal_id"] = meal_id(participant, pd.Timestamp("1970-01-01"), record["source_row"])
            records.append(record)
            continue

        t0_np = np.datetime64(pd.Timestamp(t0).to_datetime64(), "ns")
        record["meal_id"] = meal_id(participant, t0, record["source_row"])

        if insulin.size:
            near = np.abs((insulin - t0_np) / np.timedelta64(1, "m")) <= cfg.insulin_window_minutes
            record["insulin_near_meal"] = int(bool(near.any()))

        index = int(np.searchsorted(timestamps, t0_np, side="right")) - 1
        if timestamps.size == 0 or index < 0:
            for horizon in cfg.horizons_minutes:
                record[f"reason_t{horizon}"] = "no_g0"
            records.append(record)
            continue

        g0 = float(glucose[index])
        staleness = _minutes(t0_np - timestamps[index])
        record["g0_mgdl"] = g0
        record["g0_staleness_min"] = staleness
        record["g0_repeat_run"] = int(runs[index])
        record["cgm_clamped_at_g0"] = int(
            runs[index] >= cfg.clamp_run_min and g0 <= cfg.min_glucose_mgdl
        )

        for horizon in cfg.horizons_minutes:
            reason = ""
            target = t0_np + np.timedelta64(int(horizon), "m")

            if staleness > cfg.max_g0_staleness_minutes:
                reason = "g0_stale"
            elif not (cfg.min_glucose_mgdl <= g0 <= cfg.max_glucose_mgdl):
                reason = "g0_out_of_range"
            else:
                candidates = []
                after = int(np.searchsorted(timestamps, target))
                for candidate in (after, after - 1):
                    if 0 <= candidate < timestamps.size:
                        candidates.append(candidate)
                if not candidates:
                    reason = "no_outcome"
                else:
                    best = min(
                        candidates,
                        key=lambda k: abs(_minutes(timestamps[k] - target)),
                    )
                    offset = abs(_minutes(timestamps[best] - target))
                    outcome = float(glucose[best])
                    if offset > cfg.outcome_tolerance_minutes:
                        reason = "no_outcome"
                    elif best <= index:
                        reason = "no_outcome"
                    else:
                        span = np.diff(timestamps[index : best + 1])
                        if span.size and max(_minutes(s) for s in span) > cfg.max_series_gap_minutes:
                            reason = "series_gap"
                        elif not (cfg.min_glucose_mgdl <= outcome <= cfg.max_glucose_mgdl):
                            reason = "outcome_out_of_range"
                        else:
                            record[f"glucose_t{horizon}"] = outcome
                            record[f"rise_t{horizon}"] = outcome - g0
                            record[f"valid_t{horizon}"] = 1

            record[f"reason_t{horizon}"] = reason

        records.append(record)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame

    # How close the next meal is: a T+60 rise after a meal 20 minutes later
    # is not a single-meal response.
    ordered = frame.sort_values("t0_timestamp_iso", kind="stable")
    stamps = pd.to_datetime(ordered["t0_timestamp_iso"], errors="coerce")
    gaps = stamps.shift(-1) - stamps
    frame.loc[ordered.index, "next_meal_min"] = gaps.dt.total_seconds() / 60.0
    frame["next_meal_min"] = frame["next_meal_min"].astype(float)
    for horizon in cfg.horizons_minutes:
        frame[f"overlaps_next_meal_t{horizon}"] = (
            frame["next_meal_min"].fillna(math.inf).lt(horizon)
        ).astype(int)

    return frame


def summarise(frame: pd.DataFrame, config: Optional[ResponseConfig] = None) -> Dict[str, object]:
    """Counts that make the denominators auditable."""
    cfg = config or ResponseConfig()
    summary: Dict[str, object] = {
        "meals": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()) if len(frame) else 0,
        "periods": int(frame["participant_id"].nunique()) if len(frame) else 0,
        "reason_vocabulary": list(REASONS),
    }
    for horizon in cfg.horizons_minutes:
        valid = frame[f"valid_t{horizon}"].eq(1) if len(frame) else pd.Series(dtype=int)
        reasons = (
            frame.loc[~valid, f"reason_t{horizon}"].value_counts().to_dict()
            if len(frame)
            else {}
        )
        rises = frame.loc[valid, f"rise_t{horizon}"] if len(frame) else pd.Series(dtype=float)
        summary[f"t{horizon}"] = {
            "valid_meals": int(valid.sum()),
            "coverage_pct": round(100.0 * valid.sum() / len(frame), 2) if len(frame) else 0.0,
            "excluded_by_reason": {str(k): int(v) for k, v in reasons.items()},
            "median_g0_mgdl": round(float(frame["g0_mgdl"].median()), 2) if len(frame) else None,
            "median_rise_mgdl": round(float(rises.median()), 2) if rises.notna().any() else None,
            "median_abs_rise_mgdl": round(float(rises.abs().median()), 2) if rises.notna().any() else None,
            "overlapping_next_meal": int(frame[f"overlaps_next_meal_t{horizon}"].sum()) if len(frame) else 0,
        }
    summary["insulin_near_meal"] = int(frame["insulin_near_meal"].sum()) if len(frame) else 0
    summary["cgm_clamped_at_g0"] = int(frame["cgm_clamped_at_g0"].sum()) if len(frame) else 0
    return summary


def build_response_table(
    data_dir: Path,
    config: Optional[ResponseConfig] = None,
    *,
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Read every workbook in ``data_dir`` and return the response table."""
    cfg = config or ResponseConfig()
    frames: List[pd.DataFrame] = []
    paths = workbook_paths(data_dir)
    if limit is not None:
        paths = paths[:limit]
    for path in paths:
        sheet = read_sheet(path)
        frames.append(compute_responses(sheet, cfg))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(frame):
        frame = frame.sort_values(
            ["subject_id", "t0_timestamp_iso"], kind="stable"
        ).reset_index(drop=True)
    return frame, summarise(frame, cfg)


OUTPUT_COLUMNS = (
    "meal_id",
    "participant_id",
    "subject_id",
    "period_index",
    "source_row",
    "t0_timestamp_iso",
    "dietary_intake",
    "dietary_intake_cn",
    "g0_mgdl",
    "g0_staleness_min",
    "g0_repeat_run",
    "cgm_clamped_at_g0",
    "insulin_near_meal",
    "glucose_t60",
    "rise_t60",
    "valid_t60",
    "reason_t60",
    "overlaps_next_meal_t60",
    "glucose_t120",
    "rise_t120",
    "valid_t120",
    "reason_t120",
    "overlaps_next_meal_t120",
    "next_meal_min",
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the ShanghaiT2DM meal response (g0 and rises)."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--horizons",
        type=str,
        default="60,120",
        help="comma-separated horizons in minutes (default 60,120)",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    horizons = tuple(int(part) for part in args.horizons.split(",") if part.strip())
    cfg = ResponseConfig(horizons_minutes=horizons)
    frame, summary = build_response_table(args.data_dir, cfg, limit=args.limit)

    columns = [c for c in OUTPUT_COLUMNS if c in frame.columns]
    extra = [c for c in frame.columns if c not in columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame[columns + extra].to_csv(args.output, index=False)

    payload = {
        "generated_from": Path(args.data_dir).name,
        "source_dataset": SOURCE_DATASET,
        "schema_version": SCHEMA_VERSION,
        "config": {
            "horizons_minutes": list(cfg.horizons_minutes),
            "max_g0_staleness_minutes": cfg.max_g0_staleness_minutes,
            "outcome_tolerance_minutes": cfg.outcome_tolerance_minutes,
            "max_series_gap_minutes": cfg.max_series_gap_minutes,
            "min_glucose_mgdl": cfg.min_glucose_mgdl,
            "max_glucose_mgdl": cfg.max_glucose_mgdl,
            "insulin_window_minutes": cfg.insulin_window_minutes,
        },
        "summary": summary,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2))
    print(f"\nwrote {args.output} ({len(frame)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
