"""Build a meal-level manifest from the raw CGMacros participant CSV files."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DATASET_VERSION = "1.0.0"
DATASET_LICENSE = "CC BY-NC-SA 4.0"
GROUND_TRUTH_PROVENANCE = "cgmacros_reported_estimate"
MACRO_COLUMNS = {
    "carbs_g": "carbs",
    "protein_g": "protein",
    "fat_g": "fat",
    "fiber_g": "fiber",
}
DOCUMENTED_MACRO_RANGES_G = {
    "carbs_g": (0.0, 176.0),
    "protein_g": (0.0, 176.0),
    "fat_g": (0.0, 176.0),
    "fiber_g": (0.0, 176.0),
}
PHOTO_ID_RE = re.compile(r"^(\d+)-")


@dataclass
class ManifestBuildResult:
    records: List[dict]
    exclusions: List[dict]
    warnings: List[dict]
    participants_discovered: int
    meal_rows_discovered: int

    def summary(self) -> dict:
        return {
            "participants_discovered": self.participants_discovered,
            "meal_rows_discovered": self.meal_rows_discovered,
            "eligible_meals": len(self.records),
            "excluded_meals": len(self.exclusions),
            "exclusion_reasons": dict(
                sorted(Counter(item["reason"] for item in self.exclusions).items())
            ),
            "warnings": len(self.warnings),
            "warning_codes": dict(
                sorted(Counter(item["code"] for item in self.warnings).items())
            ),
        }


def _normalise_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        (key or "").strip().lower(): (value or "").strip()
        for key, value in row.items()
    }


def _parse_optional_float(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _meal_id(participant_id: str, timestamp: str) -> str:
    try:
        parsed = datetime.fromisoformat(timestamp)
        stamp = parsed.strftime("%Y%m%dT%H%M%S")
    except ValueError:
        stamp = re.sub(r"[^0-9A-Za-z]+", "", timestamp)
    if not stamp:
        raise ValueError("meal timestamp is empty or invalid")
    return f"{participant_id}_{stamp}"


def _normalise_meal_type(value: str) -> str:
    normalised = value.strip().lower()
    if normalised.startswith("snack"):
        return "Snack"
    canonical = {
        "breakfast": "Breakfast",
        "lunch": "Lunch",
        "dinner": "Dinner",
    }
    return canonical.get(normalised, value.strip())


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _resolve_image(
    participant_dir: Path,
    dataset_root: Path,
    raw_path: str,
) -> Tuple[Optional[str], str]:
    """Resolve an image path, repairing known timestamp/name mismatches by ID."""

    if not raw_path:
        return None, "missing"

    normalised = raw_path.replace("\\", "/")
    candidate = participant_dir.joinpath(*Path(normalised).parts)
    if candidate.is_file():
        return _relative_posix(candidate, dataset_root), "exact"

    filename = Path(normalised).name
    match = PHOTO_ID_RE.match(filename)
    if not match:
        return None, "unresolved"

    photo_dir = participant_dir.joinpath(*Path(normalised).parent.parts)
    matches = sorted(photo_dir.glob(f"{match.group(1)}-*.jpg"))
    if len(matches) == 1:
        return _relative_posix(matches[0], dataset_root), "photo_id_fallback"
    return None, "ambiguous_photo_id" if matches else "unresolved"


def _participant_csv(participant_dir: Path) -> Optional[Path]:
    expected = participant_dir / f"{participant_dir.name}.csv"
    if expected.is_file():
        return expected
    candidates = sorted(participant_dir.glob("*.csv"))
    return candidates[0] if len(candidates) == 1 else None


def build_meal_manifest(dataset_root: Path) -> ManifestBuildResult:
    """Parse CGMacros into eligible meal records.

    ``dataset_root`` must be the directory containing ``CGMacros-001`` style
    participant directories.  Only meal-start rows with complete non-negative
    macros and a resolvable before image are eligible.  Later image-only rows
    are retained as ``after_image_paths`` for audit but never become samples.
    """

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CGMacros dataset root not found: {root}")

    participant_dirs = sorted(
        path for path in root.glob("CGMacros-*") if path.is_dir()
    )
    if not participant_dirs:
        raise ValueError(f"No CGMacros participant directories found under: {root}")

    records: List[dict] = []
    exclusions: List[dict] = []
    warnings: List[dict] = []
    meal_rows_discovered = 0

    def finalise(current: Optional[dict]) -> None:
        if current is None:
            return

        exclusion_base = {
            "participant_id": current["participant_id"],
            "timestamp": current["timestamp"],
            "meal_type": current["meal_type"],
            "row_number": current["row_number"],
        }

        # Audit all after-meal paths even when the meal itself is later
        # excluded (for example because its before image is missing).
        after_paths: List[str] = []
        for after in current["after_images"]:
            resolved, mode = _resolve_image(
                current["participant_dir"], root, after["raw_path"]
            )
            if resolved is None:
                warnings.append(
                    {
                        **exclusion_base,
                        "code": "after_image_unresolved",
                        "declared_path": after["raw_path"],
                        "row_number": after["row_number"],
                        "resolution": mode,
                    }
                )
                continue
            if mode != "exact":
                warnings.append(
                    {
                        **exclusion_base,
                        "code": "after_image_path_repaired",
                        "declared_path": after["raw_path"],
                        "resolved_path": resolved,
                        "row_number": after["row_number"],
                        "resolution": mode,
                    }
                )
            if resolved not in after_paths:
                after_paths.append(resolved)

        missing_macros = [
            output_name
            for output_name, source_name in MACRO_COLUMNS.items()
            if _parse_optional_float(current["row"].get(source_name, "")) is None
        ]
        if missing_macros:
            exclusions.append(
                {**exclusion_base, "reason": "missing_or_invalid_macros", "fields": missing_macros}
            )
            return

        macros = {
            output_name: float(current["row"][source_name])
            for output_name, source_name in MACRO_COLUMNS.items()
        }
        if any(value < 0 for value in macros.values()):
            exclusions.append({**exclusion_base, "reason": "negative_macro_value"})
            return
        out_of_range = {
            macro: value
            for macro, value in macros.items()
            if not (
                DOCUMENTED_MACRO_RANGES_G[macro][0]
                <= value
                <= DOCUMENTED_MACRO_RANGES_G[macro][1]
            )
        }
        if out_of_range:
            exclusions.append(
                {
                    **exclusion_base,
                    "reason": "macro_out_of_documented_range",
                    "values": out_of_range,
                    "documented_ranges_g": {
                        macro: list(DOCUMENTED_MACRO_RANGES_G[macro])
                        for macro in out_of_range
                    },
                }
            )
            return

        before_path, before_mode = _resolve_image(
            current["participant_dir"], root, current["row"].get("image path", "")
        )
        if before_path is None:
            exclusions.append(
                {
                    **exclusion_base,
                    "reason": "missing_or_unresolved_before_image",
                    "image_path": current["row"].get("image path", ""),
                    "resolution": before_mode,
                }
            )
            return
        if before_mode != "exact":
            warnings.append(
                {
                    **exclusion_base,
                    "code": "before_image_path_repaired",
                    "declared_path": current["row"].get("image path", ""),
                    "resolved_path": before_path,
                    "resolution": before_mode,
                }
            )

        try:
            generated_meal_id = _meal_id(current["participant_id"], current["timestamp"])
        except ValueError:
            exclusions.append({**exclusion_base, "reason": "invalid_timestamp"})
            return

        records.append(
            {
                "meal_id": generated_meal_id,
                "participant_id": current["participant_id"],
                "timestamp": current["timestamp"],
                "meal_type": current["meal_type"],
                "before_image_path": before_path,
                "before_image_resolution": before_mode,
                "after_image_paths": after_paths,
                **{f"true_{name}": value for name, value in macros.items()},
                "amount_consumed_pct": _parse_optional_float(
                    current["row"].get("amount consumed", "")
                ),
                "source_dataset": "cgmacros",
                "dataset_version": DATASET_VERSION,
                "dataset_license": DATASET_LICENSE,
                "ground_truth_provenance": GROUND_TRUTH_PROVENANCE,
            }
        )

    for participant_dir in participant_dirs:
        csv_path = _participant_csv(participant_dir)
        if csv_path is None:
            warnings.append(
                {
                    "participant_id": participant_dir.name,
                    "code": "participant_csv_missing_or_ambiguous",
                }
            )
            continue

        current: Optional[dict] = None
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row_number, raw_row in enumerate(reader, start=2):
                row = _normalise_row(raw_row)
                meal_type = row.get("meal type", "")
                image_path = row.get("image path", "")

                if meal_type:
                    finalise(current)
                    meal_rows_discovered += 1
                    current = {
                        "participant_id": participant_dir.name,
                        "participant_dir": participant_dir,
                        "timestamp": row.get("timestamp", ""),
                        "meal_type": _normalise_meal_type(meal_type),
                        "row": row,
                        "row_number": row_number,
                        "after_images": [],
                    }
                elif image_path and current is not None:
                    current["after_images"].append(
                        {"raw_path": image_path, "row_number": row_number}
                    )

        finalise(current)

    records.sort(key=lambda item: (item["participant_id"], item["timestamp"]))
    return ManifestBuildResult(
        records=records,
        exclusions=exclusions,
        warnings=warnings,
        participants_discovered=len(participant_dirs),
        meal_rows_discovered=meal_rows_discovered,
    )
