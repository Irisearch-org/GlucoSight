"""Assemble one contract-compliant meal record from the three tracks.

This is the fusion layer's input side. It takes whatever each track
produced — including nothing at all — and produces one validated meal
record, plus the causally-derived glucose-history features.

The four silent-failure rules from CLAUDE.md are enforced here rather than
trusted:

  C1 timing    — `delta_t_minutes` is computed from real timestamps and
                 carried as a feature. Nothing assumes 60 or 120.
  C2 causality — `glucose_history` is filtered strictly before t0, and the
                 derived features are computed only from what survives.
  C4 splits    — not this module's job, but `participant_id` is required so
                 the grouping key always exists downstream.
  C3 baselines — not this module's job; see `integration/report.py`.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from integration import contract
from integration.adapters import cv_to_contract, nlp_to_contract, ppg_to_contract

# Provenance fields the adapters attach for the audit trail. They are not
# contract fields, so they are separated out before validation rather than
# being allowed to fail the strict unknown-field check.
_PROVENANCE = ("gi_source", "cv_feature_status", "nlp_calibrated")


def _to_dt(value) -> datetime:
    dt = contract._parse_ts(value)
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def derive_history_features(
    glucose_history: Sequence[Dict[str, Any]],
    t0_timestamp,
) -> Dict[str, Optional[float]]:
    """Causal features from the glucose history. Filter first, derive second.

    Returns None for every feature when no causal reading exists, rather than
    a zero or a population mean. `g0` absent is a real state — the first meal
    of a record has no prior reading — and it must not be encoded as a
    glucose value of 0 or of 100.
    """
    t0 = _to_dt(t0_timestamp)
    causal = contract.filter_causal_history(glucose_history, t0_timestamp)

    empty = {
        "g0": None,
        "g0_age_minutes": None,
        "glucose_slope_30min": None,
        "glucose_mean_6h": None,
        "g0_rejected_as_stale": False,
    }
    if not causal:
        return empty

    times = [_to_dt(e["timestamp"]) for e in causal]
    values = [float(e["value"]) for e in causal]

    g0 = values[-1]
    g0_age = (t0 - times[-1]).total_seconds() / 60.0

    # Causal but stale. A reading from last week is before t0 and therefore
    # passes C2, but it is not a pre-meal glucose value and persistence from
    # it is not a baseline. Report the age, refuse the value.
    if g0_age > contract.MAX_G0_AGE_MINUTES:
        return {
            **empty,
            "g0_age_minutes": g0_age,
            "g0_rejected_as_stale": True,
        }

    # Slope over the 30 minutes before the last reading. Two readings at the
    # same timestamp would divide by zero; guard rather than emit inf.
    slope = None
    cutoff = times[-1].timestamp() - 30 * 60
    window = [(t, v) for t, v in zip(times, values) if t.timestamp() >= cutoff]
    if len(window) >= 2:
        dt_min = (window[-1][0] - window[0][0]).total_seconds() / 60.0
        if dt_min > 0:
            slope = (window[-1][1] - window[0][1]) / dt_min

    cutoff_6h = t0.timestamp() - 6 * 3600
    recent = [v for t, v in zip(times, values) if t.timestamp() >= cutoff_6h]
    mean_6h = sum(recent) / len(recent) if recent else None

    return {
        "g0": g0,
        "g0_age_minutes": g0_age,
        "glucose_slope_30min": slope,
        "glucose_mean_6h": mean_6h,
    }


def assemble(
    *,
    meal_id: str,
    participant_id: str,
    t0_timestamp,
    source_dataset: str,
    delta_t_minutes: float,
    cv_record: Optional[Dict[str, Any]] = None,
    nlp_record: Optional[Dict[str, Any]] = None,
    ppg_record: Optional[Dict[str, Any]] = None,
    glucose_history: Optional[Sequence[Dict[str, Any]]] = None,
    portion_reported: int = 1,
    carbs_source_override: Optional[str] = None,
    time_since_last_meal_hours: Optional[float] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """Build one validated contract v1.2 meal record.

    Any of the three track records may be None. A missing modality yields
    contract fallbacks with `*_present=0`; it never yields a missing column,
    because a missing column would change input dimensionality at inference
    time — which is exactly what the down-weight-never-exclude rule forbids.
    """
    if source_dataset not in contract.VALID_SOURCE_DATASETS:
        raise contract.ContractViolation(
            f"source_dataset {source_dataset!r} not in "
            f"{sorted(contract.VALID_SOURCE_DATASETS)}"
        )

    t0 = _to_dt(t0_timestamp)

    record: Dict[str, Any] = {
        "schema_version": contract.SCHEMA_VERSION,
        "meal_id": meal_id,
        "participant_id": participant_id,
        "source_dataset": source_dataset,
        "t0_timestamp": t0.isoformat(),
        "delta_t_minutes": float(delta_t_minutes),
    }

    cv_block = cv_to_contract(
        cv_record, meal_id,
        portion_reported=portion_reported,
        carbs_source_override=carbs_source_override,
    )
    nlp_block = nlp_to_contract(nlp_record, meal_id)
    ppg_block = ppg_to_contract(ppg_record, meal_id)

    provenance: Dict[str, Any] = {}
    for block in (cv_block, nlp_block, ppg_block):
        for key, value in block.items():
            if key == "meal_id":
                continue
            if key in _PROVENANCE:
                provenance[key] = value
            else:
                record[key] = value

    if time_since_last_meal_hours is not None:
        record["time_since_last_meal_hours"] = float(time_since_last_meal_hours)

    causal_history = contract.filter_causal_history(glucose_history or [], t0)
    n_dropped = len(glucose_history or []) - len(causal_history)
    record["glucose_history"] = causal_history

    if validate:
        errors = contract.validate_meal_input(record, strict=True)
        if errors:
            raise contract.ContractViolation(
                "meal record is not contract v1.2 compliant:\n  - "
                + "\n  - ".join(errors)
            )

    derived = derive_history_features(causal_history, t0)

    return {
        **record,
        **derived,
        "hour_of_day_sin": math.sin(2 * math.pi * t0.hour / 24.0),
        "hour_of_day_cos": math.cos(2 * math.pi * t0.hour / 24.0),
        "_provenance": {
            **provenance,
            "n_history_readings_dropped_as_noncausal": n_dropped,
            "modality_weights": {
                m: contract.modality_weight(record, m)
                for m in contract.MODALITY_FIELDS
            },
        },
    }


def strip_derived(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return just the contract fields, for validation or serialisation."""
    derived = {"g0", "g0_age_minutes", "glucose_slope_30min", "glucose_mean_6h",
               "g0_rejected_as_stale", "hour_of_day_sin", "hour_of_day_cos",
               "_provenance"}
    return {k: v for k, v in record.items() if k not in derived}
