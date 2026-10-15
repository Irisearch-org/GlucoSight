"""Standalone range-violation check against partially-recovered CGMacros CSVs.

Reuses forecasting.data.cgmacros's load_participant()/extract_events()/SOURCE_RANGES
directly, skipping audit_dataset()'s bio.csv/gut_health_test.csv join (those files
weren't recovered from the truncated download). Only 15/45 participants available
-- this is a sanity check on the violation pattern, not a full reproduction.

Run: python -I .commandcode/partial_range_audit.py <extracted_csv_dir>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from forecasting.data.cgmacros import (
    PARTICIPANT_RE,
    SOURCE_RANGES,
    extract_events,
    load_participant,
)


def main(data_root: str) -> None:
    root = Path(data_root)
    files = sorted(p for p in root.glob("CGMacros-*.csv") if PARTICIPANT_RE.fullmatch(p.stem))
    if not files:
        raise SystemExit(f"no participant CSVs found under {root}")

    frames = []
    for path in files:
        frame = load_participant(root, path.name)
        for optional_column in ("Fiber", "Amount Consumed", "Image path"):
            if optional_column not in frame:
                frame[optional_column] = pd.NA
        frames.append(frame)

    event_frames = [extract_events(frame) for frame in frames]
    all_events = pd.DataFrame.from_records(
        [record for ef in event_frames for record in ef.to_dict("records")]
    )

    violating_rows = set()
    per_field = {}
    for column, (minimum, maximum) in SOURCE_RANGES.items():
        values = pd.to_numeric(all_events[column], errors="coerce")
        below = values < minimum
        above = values > maximum
        bad = all_events[below | above]
        violating_rows.update(bad.index)
        per_field[column] = {
            "range": [minimum, maximum],
            "below_minimum": int(below.sum()),
            "above_maximum": int(above.sum()),
            "examples": [
                {
                    "value": None if pd.isna(row[column]) else float(row[column]),
                    "participant_id": row["participant_id"],
                    "timestamp": str(row["Timestamp"]),
                }
                for _, row in bad.head(5).iterrows()
            ],
        }

    result = {
        "participants": [p.stem for p in files],
        "n_participants": len(files),
        "n_events": len(all_events),
        "range_violations": per_field,
        "total_distinct_violating_rows": len(violating_rows),
    }
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("partial_audit_result.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
