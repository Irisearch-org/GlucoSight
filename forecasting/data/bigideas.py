"""BIG IDEAs — food log + Dexcom CGM loader and meal alignment.

BIG IDEAs (PhysioNet 1.1.2, ODC-BY 1.0) is 16 pre-diabetic participants with
8-10 days of wristband PPG/BVP and a Dexcom G6 trace each, plus a food log.

This module reads the two files that carry a meal-level pairing:

    Food_Log_<id>.csv    meals, free text, macros, meal timestamps
    Dexcom_<id>.csv      raw Dexcom Clarity export (preamble + EGV rows)

The other BIG IDEAs streams (ACC / EDA / TEMP / BVP) are deliberately not read
here. ACC / EDA / TEMP are not inputs to any GlucoSight track; BVP is a 64 Hz
wristband signal and is fetched separately by the PPG component study.

Naming follows the interface contract where the concepts correspond:
``g0_mgdl``, ``carbs_g``, ``protein_g``, ``fat_g``, ``fiber_g``.

Raw source data stays outside the repository. Point ``--data-dir`` (or
``BIGIDEAS_DIR``) at a local download.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.data.cgmacros import TARGET_HORIZONS_MINUTES

DATASET_VERSION = "1.1.2"
LICENCE = "ODC-BY 1.0"
SOURCE_URL = "https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/"

FOOD_LOG_RE = re.compile(r"^Food_Log_(\d{3})\.csv$")
DEXCOM_RE = re.compile(r"^Dexcom_(\d{3})\.csv$")

GLUCOSE_COL = "Glucose Value (mg/dL)"
TIMESTAMP_COL = "Timestamp (YYYY-MM-DDThh:mm:ss)"

# The project's primary horizons, plus T+30 because the BIG IDEAs response
# peaks early enough that T+30 has to be checked rather than assumed.
HORIZONS_MINUTES = (30,) + tuple(TARGET_HORIZONS_MINUTES)

# Contract-style output names for the two target horizons.
TARGET_COLUMNS = {60: "glucose_60_mgdl", 120: "glucose_120_mgdl"}

# Food-log nutrient columns. ``total_carb`` is the Oracle value: the number the
# participant's food-search app looked up for the logged amount. There is no
# measurement error in it, which is exactly the ceiling a CV estimate is
# measured against.
NUTRIENT_COLUMNS = {
    "carbs_g": "total_carb",
    "kcal": "calorie",
    "protein_g": "protein",
    "fat_g": "total_fat",
    "fiber_g": "dietary_fiber",
}

# Some participant food logs ship WITHOUT a header row and with fewer columns
# (003 has 11, not 14). Position is then all we have, so the layout is rebuilt
# from the canonical order and verified against the content before use.
LEADING_COLUMNS = [
    "date", "time", "time_begin", "logged_food", "amount", "unit",
    "searched_food",
]
TRAILING_COLUMNS = [
    "calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat",
]

DEFAULT_G0_MAX_AGE_MINUTES = 15.0
DEFAULT_HORIZON_TOLERANCE_MINUTES = 7.5


def _sum_keeping_missing(values: pd.Series) -> float:
    """Sum, returning NaN when every value is missing.

    ``Series.sum()`` returns 0.0 for an all-NaN column, which would silently
    turn "this participant's export has no protein column" into "this meal
    contained no protein".
    """
    return values.sum(min_count=1)


class DatasetFormatError(ValueError):
    """Raised when a file is not the BIG IDEAs export we expect."""


def default_data_dir() -> Path:
    env = os.environ.get("BIGIDEAS_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Downloads" / "Bigideas-dataset"


def _read_food_log(path: Path) -> pd.DataFrame:
    """Read a food log, repairing the headerless exports."""
    frame = pd.read_csv(path, low_memory=False)
    if "logged_food" in frame.columns:
        return frame
    raw = pd.read_csv(path, header=None, low_memory=False)
    raw.columns = _headerless_names(raw, path)
    return raw


def _headerless_names(frame: pd.DataFrame, path: Path) -> list[str]:
    n_columns = frame.shape[1]
    if n_columns < len(LEADING_COLUMNS):
        raise DatasetFormatError(
            f"{path.name}: {n_columns} columns and no header row; cannot infer "
            f"the layout"
        )
    names = LEADING_COLUMNS + TRAILING_COLUMNS[: n_columns - len(LEADING_COLUMNS)]

    first_column = frame.iloc[:, 0].astype(str)
    if first_column.str.match(r"^\d{4}-\d{2}-\d{2}$").mean() <= 0.9:
        raise DatasetFormatError(
            f"{path.name}: no header row and column 0 is not a date "
            f"(first value {first_column.iloc[0]!r}); refusing to guess"
        )
    if frame.iloc[:, 3].astype(str).str.strip().eq("").mean() > 0.5:
        raise DatasetFormatError(
            f"{path.name}: column 3 is empty but should hold 'logged_food'"
        )
    return names


def load_food_log(path: Path) -> pd.DataFrame:
    """Load one participant's food log at item level.

    Returns one row per logged item. Items sharing a ``t0`` are one meal; group
    with :func:`build_meal_table`. Nutrient values stay NaN when the export
    omits them — nothing is filled with zero.
    """
    frame = _read_food_log(path)

    t0 = pd.to_datetime(frame.get("time_begin"), errors="coerce")
    if "date" in frame.columns:
        # Most exports use 'time'; 007 / 013 / 015 / 016 use 'time_of_day'.
        for clock in ("time", "time_of_day"):
            if clock in frame.columns:
                fallback = pd.to_datetime(
                    frame["date"].astype(str) + " " + frame[clock].astype(str),
                    errors="coerce",
                )
                t0 = t0.fillna(fallback)
    if t0.isna().all():
        raise DatasetFormatError(f"{path.name}: no usable meal timestamps")

    food = frame.get("logged_food")
    # Some exports omit 'searched_food' entirely, and some rows leave it blank.
    food_key = frame.get("searched_food")
    food_key = food if food_key is None else food_key.fillna(food)

    out = pd.DataFrame({
        "t0": t0,
        "food": food,
        "food_key": food_key,
        "amount": pd.to_numeric(frame.get("amount"), errors="coerce"),
        "unit": frame.get("unit"),
    })
    for name, column in NUTRIENT_COLUMNS.items():
        # Absent columns must be NaN, never a filled zero. Assigning None
        # would create an object column that sums to 0.0 downstream.
        if column in frame.columns:
            out[name] = pd.to_numeric(frame[column], errors="coerce")
        else:
            out[name] = np.nan

    return out.dropna(subset=["t0"]).sort_values("t0").reset_index(drop=True)


def load_dexcom(path: Path) -> pd.DataFrame:
    """Load a raw Dexcom Clarity export, keeping EGV rows with a value."""
    frame = pd.read_csv(path, low_memory=False)
    if TIMESTAMP_COL not in frame.columns:
        raise DatasetFormatError(
            f"{path.name}: no {TIMESTAMP_COL!r} column; is this a Clarity export?"
        )

    events = frame[frame.get("Event Type", "").eq("EGV")].copy()
    events["ts"] = pd.to_datetime(events[TIMESTAMP_COL], errors="coerce")
    events["g0_mgdl"] = pd.to_numeric(events[GLUCOSE_COL], errors="coerce")
    events = events.dropna(subset=["ts", "g0_mgdl"])[["ts", "g0_mgdl"]]
    events = (events.sort_values("ts")
                    .drop_duplicates("ts")
                    .reset_index(drop=True))
    if events.empty:
        raise DatasetFormatError(f"{path.name}: no usable EGV rows")
    return events


def glucose_before(
    dexcom: pd.DataFrame,
    t0: pd.Timestamp,
    max_age_minutes: float = DEFAULT_G0_MAX_AGE_MINUTES,
) -> float:
    """The last reading at or before ``t0``, within ``max_age_minutes``.

    Strictly causal by construction: a reading taken after the meal can never
    be used as the pre-meal level.
    """
    prior = dexcom[dexcom.ts <= t0]
    if prior.empty:
        return float("nan")
    last = prior.iloc[-1]
    age = (t0 - last.ts).total_seconds() / 60.0
    return float(last.g0_mgdl) if age <= max_age_minutes else float("nan")


def glucose_near(
    dexcom: pd.DataFrame,
    target: pd.Timestamp,
    tolerance_minutes: float = DEFAULT_HORIZON_TOLERANCE_MINUTES,
) -> float:
    """The reading nearest ``target``, within ``tolerance_minutes``."""
    index = (dexcom.ts - target).abs().idxmin()
    offset = abs((dexcom.ts[index] - target).total_seconds()) / 60.0
    if offset > tolerance_minutes:
        return float("nan")
    return float(dexcom.g0_mgdl[index])


def build_meal_table(
    dexcom: pd.DataFrame,
    food_log: pd.DataFrame,
    horizons=HORIZONS_MINUTES,
    tolerance_minutes: float = DEFAULT_HORIZON_TOLERANCE_MINUTES,
    max_age_minutes: float = DEFAULT_G0_MAX_AGE_MINUTES,
) -> pd.DataFrame:
    """Group logged items into meals and attach the glucose response.

    One row per meal: summed nutrients, ``g0_mgdl``, the glucose at each
    horizon, and the rise since the meal. Missing stays missing.
    """
    totals = {
        name: pd.NamedAgg(column=name, aggfunc=_sum_keeping_missing)
        for name in NUTRIENT_COLUMNS
    }
    meals = (
        food_log.groupby("t0")
        .agg(
            n_items=pd.NamedAgg(column="food", aggfunc="size"),
            item_names=pd.NamedAgg(
                column="food",
                aggfunc=lambda values: " + ".join(values.astype(str)),
            ),
            **totals,
        )
        .reset_index()
        .sort_values("t0")
        .reset_index(drop=True)
    )

    rows = []
    for meal in meals.itertuples(index=False):
        record = {
            "t0": meal.t0,
            "item_names": meal.item_names,
            "n_items": int(meal.n_items),
            "hour_of_day": meal.t0.hour + meal.t0.minute / 60.0,
            "g0_mgdl": glucose_before(dexcom, meal.t0, max_age_minutes),
        }
        for name in NUTRIENT_COLUMNS:
            record[name] = getattr(meal, name)
        for horizon in horizons:
            glucose = glucose_near(
                dexcom, meal.t0 + pd.Timedelta(minutes=horizon), tolerance_minutes
            )
            record[f"glucose_{horizon}_mgdl"] = glucose
            g0 = record["g0_mgdl"]
            record[f"delta_{horizon}_mgdl"] = (
                glucose - g0 if np.isfinite(glucose) and np.isfinite(g0) else np.nan
            )
        rows.append(record)

    return pd.DataFrame(rows)


def discover_participants(data_dir: Path) -> list[str]:
    """Participant ids that have both a food log and a Dexcom export."""
    food_logs = {
        m.group(1) for name in os.listdir(data_dir)
        if (m := FOOD_LOG_RE.match(name))
    }
    dexcoms = {
        m.group(1) for name in os.listdir(data_dir)
        if (m := DEXCOM_RE.match(name))
    }
    return sorted(food_logs & dexcoms)


def load_participant(
    data_dir: Path,
    participant_id: str,
    horizons=HORIZONS_MINUTES,
) -> pd.DataFrame:
    """Load and align one participant's meals.

    The Dexcom window is attached as ``dexcom_start`` / ``dexcom_end`` so a
    caller can tell "no glucose response" apart from "these two files are not
    even from the same fortnight".
    """
    food_log = load_food_log(data_dir / f"Food_Log_{participant_id}.csv")
    dexcom = load_dexcom(data_dir / f"Dexcom_{participant_id}.csv")
    meals = build_meal_table(dexcom, food_log, horizons)
    meals["participant_id"] = participant_id
    meals["dexcom_start"] = dexcom.ts.min()
    meals["dexcom_end"] = dexcom.ts.max()
    meals["dexcom_n"] = len(dexcom)
    return meals


def load_cohort(
    data_dir: Path,
    horizons=HORIZONS_MINUTES,
) -> tuple[pd.DataFrame, list[str]]:
    """Load every pairable participant. Returns (meals, problems).

    A participant whose food-log window does not overlap its Dexcom window
    contributes nothing and is returned in ``problems`` with the reason rather
    than silently adding empty rows.
    """
    frames, problems = [], []
    for participant_id in discover_participants(data_dir):
        try:
            meals = load_participant(data_dir, participant_id, horizons)
        except (DatasetFormatError, OSError) as error:
            problems.append(f"{participant_id}: {error}")
            continue

        horizon = f"delta_{max(horizons)}_mgdl"
        if meals.empty or meals[horizon].notna().sum() == 0:
            if meals.empty:
                reason = "food log produced no meals"
            else:
                reason = (
                    f"no meal falls inside the Dexcom window "
                    f"(food log {meals.t0.min():%Y-%m-%d}..{meals.t0.max():%Y-%m-%d}, "
                    f"Dexcom {meals.dexcom_start.iloc[0]:%Y-%m-%d}"
                    f"..{meals.dexcom_end.iloc[0]:%Y-%m-%d})"
                )
            problems.append(f"{participant_id}: {reason}")
            continue
        frames.append(meals)

    if not frames:
        raise DatasetFormatError(f"no usable participants found in {data_dir}")
    return pd.concat(frames, ignore_index=True), problems
