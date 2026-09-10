"""Single source of truth for Interface Contract v1.2.

Before this module the repo had three schemas and no shared validator:

  * `cv/cv_baseline/schema.py`  — `sample_id`, `confidence`, `feature_status`,
    `calories_kcal`. Not a contract v1.2 record. Its own docstring says it
    should be replaced by the shared validator when one exists, "bukan
    dipertahankan berdampingan" — not kept alongside. This is that validator.
  * `nlp/nlp_baseline/predict.py` — close to v1.2 already.
  * `rppg/features/extractor.py` — has `to_contract_dict()` and
    `validate_contract_output()`; the most contract-mature of the three.

`integration/adapters/` translates each of those into the schema defined
here. Tracks keep their native output; nothing downstream reads it directly.

Rules this module enforces, all from CLAUDE.md and the contract:
  * C2 — no glucose_history entry at or after t0.
  * C1 — delta_t_minutes is carried, never assumed to be 60 or 120.
  * H1 — every modality has a `*_present` mask separate from its confidence.
  * H1 — low confidence DOWN-WEIGHTS; it never excludes, because excluding
    changes input dimensionality at inference time.
  * H7 — gi_category and portion_reported are one-hot, never integers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "1.2"

VALID_SOURCE_DATASETS = {
    "cgmacros", "shanghai_t2dm", "big_ideas", "ppg_dataset", "live_capture",
}
VALID_CARBS_SOURCES = {"weighed", "cv_regressed", "class_lookup", "population_mean"}
VALID_NLP_SOURCES = {"extracted", "lookup", "rule_based", "unavailable"}
VALID_GLUCOSE_RANGES = {"normal", "borderline", "high"}

# Confidence below this down-weights a modality. It never drops it (H1).
CONFIDENCE_FLOOR = 0.3
# A modality is never zeroed out entirely, however bad its confidence:
# a hard zero is indistinguishable from "the value really is zero".
MIN_MODALITY_WEIGHT = 0.1

# Maximum age of the last causal glucose reading before it stops counting as
# a pre-meal value. 6 hours.
#
# NOT IN CONTRACT v1.2 — proposed for v2.0, and found by running the demo:
# with a glucose history from January and a first bite in September, the
# causality filter passed happily (every reading IS before t0) and handed
# the predictor a 237-day-old reading as `g0`. B1 persistence and B2
# persistence-plus-excursion are both meaningless on a reading that old, but
# nothing errored and the prediction looked entirely reasonable.
#
# This is the C2 failure mode's mirror image: C2 stops readings that are too
# NEW from entering the history. Nothing stopped readings that are too OLD.
MAX_G0_AGE_MINUTES = 360.0


@dataclass(frozen=True)
class Field:
    name: str
    kind: type
    required: bool
    fallback: Any = None
    lo: Optional[float] = None
    hi: Optional[float] = None
    allowed: Optional[frozenset] = None
    note: str = ""

    def check(self, value: Any) -> Optional[str]:
        if value is None:
            return None if not self.required else f"{self.name}: required, got None"
        if self.kind is float and isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
        elif self.kind is int and isinstance(value, bool):
            return f"{self.name}: bool is not an int here"
        elif not isinstance(value, self.kind):
            return f"{self.name}: expected {self.kind.__name__}, got {type(value).__name__}"
        if self.allowed is not None and value not in self.allowed:
            return f"{self.name}: {value!r} not in {sorted(self.allowed)}"
        if self.lo is not None and value < self.lo:
            return f"{self.name}: {value} below contract minimum {self.lo}"
        if self.hi is not None and value > self.hi:
            return f"{self.name}: {value} above contract maximum {self.hi}"
        return None


# ---------------------------------------------------------------------
# §0 Meal envelope
# ---------------------------------------------------------------------
ENVELOPE_FIELDS: Tuple[Field, ...] = (
    Field("schema_version", str, True, SCHEMA_VERSION,
          allowed=frozenset({SCHEMA_VERSION})),
    Field("meal_id", str, True),
    Field("participant_id", str, True, note="GroupKFold key — C4"),
    Field("source_dataset", str, True, allowed=frozenset(VALID_SOURCE_DATASETS)),
    Field("t0_timestamp", str, True, note="FIRST BITE, not the photo — C1"),
    Field("delta_t_minutes", float, True, lo=0.0, hi=240.0,
          note="actual horizon to the label — never assume 60/120 (C1)"),
)

# ---------------------------------------------------------------------
# CV track
# ---------------------------------------------------------------------
CV_FIELDS: Tuple[Field, ...] = (
    Field("carbs_g", float, True, 45.0, lo=0.0, hi=300.0),
    Field("protein_g", float, True, 18.0, lo=0.0, hi=150.0),
    Field("fat_g", float, True, 12.0, lo=0.0, hi=150.0),
    Field("fiber_g", float, True, 4.0, lo=0.0, hi=50.0),
    Field("gi_category", int, True, 1, allowed=frozenset({0, 1, 2})),
    Field("food_class", str, False, "unknown"),
    Field("cv_confidence", float, True, 0.0, lo=0.0, hi=1.0),
    Field("cv_present", int, True, 0, allowed=frozenset({0, 1})),
    Field("carbs_source", str, True, "population_mean",
          allowed=frozenset(VALID_CARBS_SOURCES)),
    Field("portion_reported", int, True, 1, allowed=frozenset({0, 1, 2})),
    Field("cv_model_version", str, True, "unknown"),
)

# ---------------------------------------------------------------------
# NLP track (reduced in v1.2)
# ---------------------------------------------------------------------
NLP_FIELDS: Tuple[Field, ...] = (
    Field("is_fried_cooking", int, True, 0, allowed=frozenset({0, 1})),
    Field("is_large_portion", int, True, 0, allowed=frozenset({0, 1})),
    Field("nlp_confidence", float, True, 0.0, lo=0.0, hi=1.0),
    Field("nlp_present", int, True, 0, allowed=frozenset({0, 1})),
    Field("nlp_feature_source", str, True, "unavailable",
          allowed=frozenset(VALID_NLP_SOURCES)),
    Field("nlp_model_version", str, True, "unknown"),
)

# ---------------------------------------------------------------------
# Contact PPG track
# ---------------------------------------------------------------------
PPG_FIELDS: Tuple[Field, ...] = (
    Field("ppg_pulse_rate_bpm", float, True, 75.0, lo=40.0, hi=200.0),
    Field("ppg_hrv_rmssd", float, False, 30.0, lo=0.0, hi=200.0),
    Field("ppg_signal_quality", float, True, 0.0, lo=0.0, hi=1.0),
    Field("ppg_perfusion_index", float, False, 2.0, lo=0.0, hi=20.0),
    Field("ppg_glucose_estimate", float, False, None, lo=70.0, hi=400.0,
          note="RING-FENCED — see is_ringfenced() below (E1)"),
    Field("ppg_present", int, True, 0, allowed=frozenset({0, 1})),
    Field("ppg_clip_fraction", float, True, 1.0, lo=0.0, hi=1.0),
    Field("ppg_n_windows", int, True, 0, lo=0, hi=3),
    Field("ppg_model_version", str, True, "unknown"),
)

# ---------------------------------------------------------------------
# Context produced outside the three tracks
# ---------------------------------------------------------------------
CONTEXT_FIELDS: Tuple[Field, ...] = (
    Field("time_since_last_meal_hours", float, False, None, lo=0.0, hi=72.0),
)

# ---------------------------------------------------------------------
# Forecasting output
# ---------------------------------------------------------------------
FORECAST_FIELDS: Tuple[Field, ...] = (
    Field("glucose_t60", float, True, lo=70.0, hi=400.0),
    Field("glucose_t120", float, True, lo=70.0, hi=400.0),
    Field("glucose_range", str, True, allowed=frozenset(VALID_GLUCOSE_RANGES)),
    Field("prediction_confidence", float, True, lo=0.0, hi=1.0),
    Field("delta_t_minutes", float, True, lo=0.0, hi=240.0),
    Field("model_version", str, True),
)

MEAL_INPUT_FIELDS: Tuple[Field, ...] = (
    ENVELOPE_FIELDS + CV_FIELDS + NLP_FIELDS + PPG_FIELDS + CONTEXT_FIELDS
)
_BY_NAME = {f.name: f for f in MEAL_INPUT_FIELDS}

MODALITY_FIELDS = {
    "cv": CV_FIELDS,
    "nlp": NLP_FIELDS,
    "ppg": PPG_FIELDS,
}


class ContractViolation(ValueError):
    """Raised when a record cannot be made contract-compliant."""


# ---------------------------------------------------------------------
# Causality — finding C2
# ---------------------------------------------------------------------

def _parse_ts(value) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ContractViolation(f"unparseable timestamp {value!r}") from exc


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def filter_causal_history(
    glucose_history: Sequence[Dict[str, Any]],
    t0_timestamp,
) -> List[Dict[str, Any]]:
    """Keep only readings strictly before t0. Filter FIRST, interpolate after.

    This is finding C2, the first of the four silent-failure rules. A reading
    at or after t0 is a prediction target; letting one into the input history
    produces a clean run, a good loss curve and a worthless model. Nothing
    errors on its own, so it is asserted here rather than hoped for.
    """
    t0 = _naive(_parse_ts(t0_timestamp))
    kept = []
    for entry in glucose_history or []:
        if "timestamp" not in entry or "value" not in entry:
            raise ContractViolation(
                f"glucose_history entry needs 'timestamp' and 'value': {entry!r}"
            )
        if _naive(_parse_ts(entry["timestamp"])) < t0:
            kept.append(entry)
    kept.sort(key=lambda e: _naive(_parse_ts(e["timestamp"])))
    return kept


def assert_causal_history(glucose_history, t0_timestamp) -> None:
    """Hard assertion for loaders. Raises rather than silently dropping."""
    t0 = _naive(_parse_ts(t0_timestamp))
    for entry in glucose_history or []:
        ts = _naive(_parse_ts(entry["timestamp"]))
        if ts >= t0:
            raise ContractViolation(
                f"C2 LABEL LEAKAGE: glucose_history contains {ts} which is at or "
                f"after t0={t0}. The T+60/T+120 readings are the prediction "
                f"targets. Filter before interpolating, never after."
            )


# ---------------------------------------------------------------------
# Ring-fence — finding E1
# ---------------------------------------------------------------------

def is_ringfenced(field_name: str) -> bool:
    """`ppg_glucose_estimate` may never enter the headline fusion feature set.

    Contract §Ring-fence rules: never a sole prediction, never surfaced to a
    user, reported in its own section with its own n, and fusion ablations
    run with and without it. `integration/features.py` honours this by
    excluding it from every default tier; an ablation must opt in explicitly.
    """
    return field_name == "ppg_glucose_estimate"


# ---------------------------------------------------------------------
# Validation and fallbacks
# ---------------------------------------------------------------------

def validate_meal_input(record: Dict[str, Any], strict: bool = True) -> List[str]:
    """Return a list of contract violations. Empty list means compliant."""
    errors: List[str] = []

    for f in MEAL_INPUT_FIELDS:
        if f.name not in record:
            if f.required:
                errors.append(f"{f.name}: required field missing")
            continue
        err = f.check(record[f.name])
        if err:
            errors.append(err)

    if strict:
        unknown = set(record) - set(_BY_NAME) - {"glucose_history"}
        if unknown:
            errors.append(
                f"unknown fields not in contract v{SCHEMA_VERSION}: "
                f"{sorted(unknown)}. Add them to the contract or drop them — "
                f"a field nobody agreed on is a field nobody validates."
            )

    # Missing != negative (CLAUDE.md rule 8). A present mask of 0 alongside a
    # confidence above zero is contradictory and must not pass silently.
    for modality in MODALITY_FIELDS:
        present = record.get(f"{modality}_present")
        conf = record.get(f"{modality}_confidence")
        if modality == "ppg":
            conf = record.get("ppg_signal_quality")
        if present == 0 and conf is not None and conf > 0.0:
            errors.append(
                f"{modality}_present=0 but {modality} confidence={conf} > 0 — "
                f"'no capture' and 'a bad capture' are different states (H1)"
            )

    if "glucose_history" in record and record.get("t0_timestamp"):
        try:
            assert_causal_history(record["glucose_history"], record["t0_timestamp"])
        except ContractViolation as exc:
            errors.append(str(exc))

    return errors


def apply_fallbacks(record: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing optional fields with contract fallbacks.

    Fallbacks are applied to VALUES only. The `*_present` masks are never
    inferred here: a caller that does not set them is telling us nothing, and
    guessing would collapse the very distinction H1 exists to preserve.
    """
    out = dict(record)
    for f in MEAL_INPUT_FIELDS:
        if out.get(f.name) is None and f.fallback is not None:
            if f.name.endswith("_present"):
                continue
            out[f.name] = f.fallback
    out.setdefault("schema_version", SCHEMA_VERSION)
    for modality in MODALITY_FIELDS:
        out.setdefault(f"{modality}_present", 0)
    return out


def modality_weight(record: Dict[str, Any], modality: str) -> float:
    """Down-weight, never exclude (H1 / CLAUDE.md rule 9).

    Returns a weight in [MIN_MODALITY_WEIGHT, 1.0]:
      * absent modality      -> MIN_MODALITY_WEIGHT (not 0 — see below)
      * confidence >= floor  -> 1.0
      * below the floor      -> scaled linearly down to the minimum

    The weight never reaches 0. Multiplying a feature by zero is
    indistinguishable from that feature genuinely being zero, which is the
    same error as encoding missingness as a value.
    """
    if modality not in MODALITY_FIELDS:
        raise KeyError(f"unknown modality {modality!r}")
    if not record.get(f"{modality}_present", 0):
        return MIN_MODALITY_WEIGHT
    key = "ppg_signal_quality" if modality == "ppg" else f"{modality}_confidence"
    conf = float(record.get(key) or 0.0)
    if conf >= CONFIDENCE_FLOOR:
        return 1.0
    scale = conf / CONFIDENCE_FLOOR
    return MIN_MODALITY_WEIGHT + (1.0 - MIN_MODALITY_WEIGHT) * scale


def glucose_range(value_mgdl: float) -> str:
    """Contract thresholds for the UI traffic light."""
    if value_mgdl < 140.0:
        return "normal"
    if value_mgdl < 200.0:
        return "borderline"
    return "high"


def validate_forecast_output(record: Dict[str, Any]) -> List[str]:
    errors = []
    for f in FORECAST_FIELDS:
        if f.name not in record:
            errors.append(f"{f.name}: required field missing")
            continue
        err = f.check(record[f.name])
        if err:
            errors.append(err)
    if not errors:
        expected = glucose_range(record["glucose_t60"])
        if record["glucose_range"] != expected:
            errors.append(
                f"glucose_range={record['glucose_range']!r} does not match "
                f"glucose_t60={record['glucose_t60']} (expected {expected!r})"
            )
    return errors
