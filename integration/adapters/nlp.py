"""NLP track (nlp-baseline-v0.1) -> contract v1.2.

The NLP track is already close: `nlp/nlp_baseline/predict.py` emits
`is_fried_cooking`, `is_large_portion`, `nlp_confidence`, `nlp_present`,
`nlp_feature_source` and `nlp_model_version` directly. The adapter's job is
to strip the track-local extras (`sample_id`, `modality`, `feature_status`,
`created_at`) and to enforce two things the contract is explicit about and
the track cannot enforce alone.

1. **`nlp_present=0` means "no dietary record for this meal".** All-zero
   labels with `nlp_present=1` means "a record exists and reports none of
   these attributes". These are different states and must never share an
   encoding (H1). The track gets this right; the adapter asserts it so a
   future change cannot quietly break it.

2. **`nlp_confidence` must be calibrated** (isotonic or Platt), not a raw
   score. Forecasting uses it as a gating weight, and an uncalibrated score
   makes that gate meaningless. The current baseline emits a rule-based
   score from `nlp/data/derive_labels.py::_confidence`, which is NOT
   calibrated. `nlp_calibrated` records that fact rather than letting an
   uncalibrated number be used as if it were a probability.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from integration import contract

# Feature sources whose confidence is a rule output rather than a fitted,
# calibrated probability. ClickUp task [NLP] "Calibrate nlp_confidence
# (isotonic/Platt)" is open; until it lands, everything here is uncalibrated.
_UNCALIBRATED_SOURCES = {"rule_based", "lookup", "unavailable"}


def nlp_to_contract(
    record: Optional[Dict[str, Any]],
    meal_id: str,
) -> Dict[str, Any]:
    """Translate one nlp-baseline record into the contract's NLP block."""
    if record is None:
        block = {f.name: f.fallback for f in contract.NLP_FIELDS}
        block.update({
            "meal_id": meal_id,
            "nlp_present": 0,
            "nlp_confidence": 0.0,
            "nlp_feature_source": "unavailable",
            "nlp_model_version": "absent",
            "nlp_calibrated": False,
        })
        return block

    present = int(record.get("nlp_present", 0))
    source = record.get("nlp_feature_source", "unavailable")
    if source not in contract.VALID_NLP_SOURCES:
        raise contract.ContractViolation(
            f"nlp_feature_source {source!r} not in "
            f"{sorted(contract.VALID_NLP_SOURCES)}"
        )

    confidence = float(record.get("nlp_confidence", 0.0) or 0.0)

    if present == 0:
        # "No record" carries no information about the labels. Confidence
        # must be zero so it cannot be read as a weak positive.
        if confidence > 0.0:
            raise contract.ContractViolation(
                f"nlp_present=0 with nlp_confidence={confidence} — 'no dietary "
                f"record' and 'a record reporting nothing' must not share an "
                f"encoding (contract, NLP notes)"
            )
        if source != "unavailable":
            raise contract.ContractViolation(
                f"nlp_present=0 requires nlp_feature_source='unavailable', "
                f"got {source!r}"
            )

    if source == "unavailable" and present == 1:
        raise contract.ContractViolation(
            "nlp_feature_source='unavailable' with nlp_present=1 is incoherent"
        )

    return {
        "meal_id": meal_id,
        "is_fried_cooking": int(record.get("is_fried_cooking", 0)),
        "is_large_portion": int(record.get("is_large_portion", 0)),
        "nlp_confidence": confidence,
        "nlp_present": present,
        "nlp_feature_source": source,
        "nlp_model_version": record.get("nlp_model_version", "unknown"),
        # Provenance, stripped before validation.
        "nlp_calibrated": source not in _UNCALIBRATED_SOURCES,
    }
