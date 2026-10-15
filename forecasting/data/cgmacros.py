from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DATASET_VERSION = "1.0.0"
PARTICIPANT_RE = re.compile(r"^CGMacros-(\d{3})$")
MEAL_TYPES = {
    "breakfast": "breakfast",
    "lunch": "lunch",
    "dinner": "dinner",
    "snack": "snack",
    "snacks": "snack",
    "snack 1": "snack",
}
NUTRIENT_COLUMNS = ("Calories", "Carbs", "Protein", "Fat", "Fiber")
TARGET_HORIZONS_MINUTES = (60, 120)
SOURCE_RANGES = {
    "Calories": (30.0, 1180.0),
    "Carbs": (0.0, 176.0),
    "Protein": (3.0, 176.0),
    "Fat": (0.0, 176.0),
    "Fiber": (0.0, 176.0),
    "Amount Consumed": (0.0, 100.0),
}


class DatasetFormatError(ValueError):
    pass


@dataclass(frozen=True)
class AuditReport:
    dataset_version: str
    participant_files: int
    participant_ids: tuple[int, ...]
    bio_rows: int
    bio_subjects: tuple[int, ...]
    gut_rows: int
    gut_subjects: tuple[int, ...]
    unmatched_bio_subjects: tuple[int, ...]
    extra_gut_subjects: tuple[int, ...]
    source_events: int
    events_by_participant: dict[str, int]
    missingness: dict[str, int]
    range_violations: dict[str, dict[str, int]]
    meal_type_counts: dict[str, int]
    unknown_meal_types: dict[str, int]
    dexcom_observations: int
    dexcom_interval_minutes: dict[str, float]
    candidate_targets: dict[str, int]
    paired_candidate_events: int
    isolated_candidate_targets: dict[str, int]
    isolated_paired_events: int
    close_event_pairs_under_120m: int
    csv_checksums: dict[str, str]


def _numeric_ids(values: Iterable[Any], label: str) -> list[int]:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    if numeric.isna().any():
        raise DatasetFormatError(f"{label} contains missing or non-numeric subject IDs")
    ids = numeric.astype(int).tolist()
    if len(ids) != len(set(ids)):
        raise DatasetFormatError(f"{label} contains duplicate subject IDs")
    return ids


def _participant_id(path: Path) -> int:
    match = PARTICIPANT_RE.fullmatch(path.stem)
    if not match:
        raise DatasetFormatError(f"invalid participant filename: {path.name}")
    return int(match.group(1))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DatasetFormatError(f"missing required file: {path}")
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    return frame


def _require_columns(frame: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DatasetFormatError(f"{path.name} missing columns: {missing}")


def load_participant(root: str | Path, participant_file: str | Path) -> pd.DataFrame:
    root_path = Path(root)
    path = Path(participant_file)
    if not path.is_absolute():
        path = root_path / path
    pid = _participant_id(path)
    frame = _read_csv(path)
    _require_columns(frame, ("Timestamp", "Dexcom GL", "Meal Type", "Calories", "Carbs", "Protein", "Fat"), path)
    for optional_column in ("Fiber", "Amount Consumed", "Image path"):
        if optional_column not in frame:
            frame[optional_column] = pd.NA
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    if frame["Timestamp"].isna().any():
        raise DatasetFormatError(f"{path.name} contains invalid timestamps")
    frame["Dexcom GL"] = pd.to_numeric(frame["Dexcom GL"], errors="coerce")
    frame["participant_id"] = f"cgmacros-{pid:03d}"
    frame["source_row_id"] = range(len(frame))
    frame["source_event_label"] = frame["Meal Type"]
    normalized = frame["Meal Type"].astype("string").str.strip().str.casefold()
    frame["meal_type"] = normalized.map(MEAL_TYPES)
    frame["event_candidate"] = normalized.notna() & normalized.ne("")
    frame["unknown_meal_type"] = frame["event_candidate"] & frame["meal_type"].isna()
    return frame.sort_values(["Timestamp", "source_row_id"], kind="stable").reset_index(drop=True)


def load_metadata(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root_path = Path(root)
    bio_path = root_path / "bio.csv"
    gut_path = root_path / "gut_health_test.csv"
    bio = _read_csv(bio_path)
    gut = _read_csv(gut_path)
    _require_columns(bio, ("subject",), bio_path)
    _require_columns(gut, ("subject",), gut_path)
    bio_ids = _numeric_ids(bio["subject"], "bio.csv")
    gut_ids = _numeric_ids(gut["subject"], "gut_health_test.csv")
    bio["subject"] = bio_ids
    gut["subject"] = gut_ids
    details = {
        "bio_ids": tuple(sorted(bio_ids)),
        "gut_ids": tuple(sorted(gut_ids)),
        "extra_gut_ids": tuple(sorted(set(gut_ids) - set(bio_ids))),
        "missing_gut_ids": tuple(sorted(set(bio_ids) - set(gut_ids))),
    }
    return bio, gut, details


def join_participant_metadata(
    participant_ids: Iterable[int], bio: pd.DataFrame, gut: pd.DataFrame
) -> pd.DataFrame:
    participant_ids = sorted(set(int(pid) for pid in participant_ids))
    bio_ids = _numeric_ids(bio["subject"], "bio metadata")
    gut_ids = _numeric_ids(gut["subject"], "gut metadata")
    expected = set(participant_ids)
    missing_bio = expected - set(bio_ids)
    if missing_bio:
        raise DatasetFormatError(f"bio.csv missing cohort subjects: {sorted(missing_bio)}")
    bio_subset = bio[bio["subject"].isin(expected)].copy()
    gut_subset = gut[gut["subject"].isin(expected)].copy()
    if len(bio_subset) != len(expected):
        raise DatasetFormatError("bio metadata join is not one-to-one for participant cohort")
    if len(gut_subset) != len(expected):
        missing = expected - set(gut_ids)
        raise DatasetFormatError(
            f"gut metadata missing cohort subjects: {sorted(missing)}; do not join positionally"
        )
    return bio_subset.merge(
        gut_subset,
        on="subject",
        how="inner",
        validate="one_to_one",
        suffixes=("_bio", "_gut"),
    )


def extract_events(frame: pd.DataFrame) -> pd.DataFrame:
    events = frame.loc[frame["event_candidate"]].copy()
    if events["unknown_meal_type"].any():
        labels = sorted(events.loc[events["unknown_meal_type"], "source_event_label"].astype(str).unique())
        raise DatasetFormatError(f"unknown non-empty meal type labels: {labels}")
    event_columns = [
        "Timestamp",
        "participant_id",
        "source_row_id",
        "source_event_label",
        "meal_type",
        "unknown_meal_type",
        *NUTRIENT_COLUMNS,
        "Amount Consumed",
        "Image path",
    ]
    events = events.loc[:, event_columns].copy()
    events["dataset_version"] = DATASET_VERSION
    events["event_id"] = events.apply(
        lambda row: f"{DATASET_VERSION}:{row['participant_id']}:{int(row['source_row_id'])}",
        axis=1,
    )
    return events.reset_index(drop=True)


def _nearest_exact_observation(
    cgm: pd.DataFrame, event_time: pd.Timestamp, horizon_minutes: int
) -> dict[str, Any] | None:
    target_time = event_time + pd.Timedelta(minutes=horizon_minutes)
    matches = cgm.loc[
        (cgm["Timestamp"] == target_time)
        & cgm["Dexcom GL"].notna()
        & (cgm["Timestamp"] > event_time)
    ]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return {
        "target_mgdl": float(row["Dexcom GL"]),
        "target_timestamp": row["Timestamp"],
        "actual_delta_t_minutes": float(
            (row["Timestamp"] - event_time).total_seconds() / 60.0
        ),
        "target_offset_minutes": 0.0,
    }


def build_targets(
    participant_frame: pd.DataFrame, events: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    events = extract_events(participant_frame) if events is None else events.copy()
    cgm = participant_frame.loc[:, ["Timestamp", "Dexcom GL"]].dropna(subset=["Dexcom GL"])
    cgm = cgm.sort_values("Timestamp", kind="stable")
    records = []
    counts = {
        "source_events": len(events),
        "T+60": 0,
        "T+120": 0,
        "paired": 0,
        "isolated_T+60": 0,
        "isolated_T+120": 0,
        "isolated_paired": 0,
    }
    ordered_events = events.sort_values(["Timestamp", "source_row_id"], kind="stable")
    next_timestamps = ordered_events["Timestamp"].shift(-1).to_dict()
    for event_index, event in events.iterrows():
        next_event = next_timestamps.get(event_index)
        row = {
            "event_id": event["event_id"],
            "participant_id": event["participant_id"],
            "event_timestamp": event["Timestamp"],
            "source_row_id": int(event["source_row_id"]),
            "meal_type": event["meal_type"],
            "source_event_label": event["source_event_label"],
        }
        complete = True
        isolated_complete = True
        for horizon in TARGET_HORIZONS_MINUTES:
            label = f"T+{horizon}"
            target = _nearest_exact_observation(cgm, event["Timestamp"], horizon)
            isolated = pd.isna(next_event) or next_event >= event["Timestamp"] + pd.Timedelta(minutes=horizon)
            row[label] = target
            row[f"isolated_{label}"] = bool(isolated)
            counts[label] += int(target is not None)
            counts[f"isolated_{label}"] += int(target is not None and isolated)
            complete &= target is not None
            isolated_complete &= target is not None and isolated
        counts["paired"] += int(complete)
        counts["isolated_paired"] += int(isolated_complete)
        records.append(row)
    return pd.DataFrame.from_records(records), counts


def build_history(
    readings: pd.DataFrame, event_time: Any
) -> pd.DataFrame:
    event_ts = pd.Timestamp(event_time)
    frame = readings.copy()
    if "Timestamp" not in frame or "Dexcom GL" not in frame:
        raise DatasetFormatError("readings require Timestamp and Dexcom GL columns")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    frame["Dexcom GL"] = pd.to_numeric(frame["Dexcom GL"], errors="coerce")
    history = frame.loc[
        frame["Timestamp"].notna()
        & frame["Dexcom GL"].notna()
        & (frame["Timestamp"] < event_ts)
    ].sort_values("Timestamp", kind="stable").reset_index(drop=True)
    if not history.empty and not history["Timestamp"].max() < event_ts:
        raise AssertionError("causal history includes an event-time or future reading")
    return history


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(root: str | Path) -> AuditReport:
    root_path = Path(root)
    files = sorted(
        p for p in root_path.glob("CGMacros-*.csv")
        if PARTICIPANT_RE.fullmatch(p.stem)
    )
    if not files:
        raise DatasetFormatError(f"no participant CSVs found under {root_path}")
    pids = [_participant_id(path) for path in files]
    if len(pids) != len(set(pids)):
        raise DatasetFormatError("duplicate participant IDs from filenames")

    bio, gut, metadata = load_metadata(root_path)
    frames = []
    for path in files:
        frame = load_participant(root_path, path)
        _require_columns(frame, ("Calories", "Carbs", "Protein", "Fat"), path)
        for optional_column in ("Fiber", "Amount Consumed", "Image path"):
            if optional_column not in frame:
                frame[optional_column] = pd.NA
        frames.append(frame)
    join_participant_metadata(pids, bio, gut)

    event_frames = [extract_events(frame) for frame in frames]
    all_events = pd.DataFrame.from_records(
        [record for event_frame in event_frames for record in event_frame.to_dict("records")]
    )
    missingness = {}
    for column in (*NUTRIENT_COLUMNS, "Amount Consumed", "Image path"):
        if column not in all_events:
            all_events[column] = pd.NA
        missingness[column] = int(all_events[column].isna().sum())
    range_violations = {}
    for column, (minimum, maximum) in SOURCE_RANGES.items():
        values = pd.to_numeric(all_events[column], errors="coerce")
        range_violations[column] = {
            "below_minimum": int((values < minimum).sum()),
            "above_maximum": int((values > maximum).sum()),
        }
    normalized_type_counts = all_events["meal_type"].value_counts().sort_index().to_dict()
    raw_type_counts = all_events["source_event_label"].astype(str).value_counts().sort_index().to_dict()

    candidate_targets = {f"T+{h}": 0 for h in TARGET_HORIZONS_MINUTES}
    isolated_targets = {f"T+{h}": 0 for h in TARGET_HORIZONS_MINUTES}
    paired = 0
    isolated_paired = 0
    close_pairs = 0
    event_counts = {}
    dexcom_deltas = []
    dexcom_count = 0
    for frame in frames:
        participant = str(frame["participant_id"].iloc[0])
        events = extract_events(frame)
        event_counts[participant] = len(events)
        targets, counts = build_targets(frame, events)
        candidate_targets["T+60"] += counts["T+60"]
        candidate_targets["T+120"] += counts["T+120"]
        isolated_targets["T+60"] += counts["isolated_T+60"]
        isolated_targets["T+120"] += counts["isolated_T+120"]
        paired += counts["paired"]
        isolated_paired += counts["isolated_paired"]
        times = frame.loc[frame["Dexcom GL"].notna(), "Timestamp"].sort_values()
        dexcom_count += len(times)
        dexcom_deltas.extend(times.diff().dt.total_seconds().dropna() / 60.0)
        event_times = events["Timestamp"].sort_values().tolist()
        close_pairs += sum(
            (right - left).total_seconds() < 120 * 60
            for left, right in zip(event_times, event_times[1:])
        )

    intervals = pd.Series(dexcom_deltas, dtype=float)
    interval_summary = {
        "p10": float(intervals.quantile(0.10)),
        "median": float(intervals.median()),
        "p90": float(intervals.quantile(0.90)),
        "max": float(intervals.max()),
    }
    checksums = {path.name: _sha256(path) for path in [*files, root_path / "bio.csv", root_path / "gut_health_test.csv"]}
    return AuditReport(
        dataset_version=DATASET_VERSION,
        participant_files=len(files),
        participant_ids=tuple(sorted(pids)),
        bio_rows=len(bio),
        bio_subjects=metadata["bio_ids"],
        gut_rows=len(gut),
        gut_subjects=metadata["gut_ids"],
        unmatched_bio_subjects=metadata["missing_gut_ids"],
        extra_gut_subjects=metadata["extra_gut_ids"],
        source_events=len(all_events),
        events_by_participant=event_counts,
        missingness=missingness,
        range_violations=range_violations,
        meal_type_counts={str(k): int(v) for k, v in normalized_type_counts.items()},
        unknown_meal_types={str(k): int(v) for k, v in raw_type_counts.items()},
        dexcom_observations=dexcom_count,
        dexcom_interval_minutes=interval_summary,
        candidate_targets=candidate_targets,
        paired_candidate_events=paired,
        isolated_candidate_targets=isolated_targets,
        isolated_paired_events=isolated_paired,
        close_event_pairs_under_120m=int(close_pairs),
        csv_checksums=checksums,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a local CGMacros v1.0.0 copy")
    parser.add_argument("data_root", type=Path, help="Directory with participant CSVs, bio.csv and gut_health_test.csv")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    result = asdict(audit_dataset(args.data_root))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
