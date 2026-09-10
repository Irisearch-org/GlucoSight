"""Contact PPG track -> contract v1.2.

`rppg/features/extractor.py` already has `to_contract_dict()` and
`validate_contract_output()`, so this adapter is thin. It exists for two
reasons the track cannot handle on its own.

1. **The ring-fence (E1).** `ppg_glucose_estimate` is the track's novelty
   claim and stays in the contract, but it may never be a sole prediction,
   never be surfaced to a user, and must be isolable in ablations. The
   adapter carries it, and `integration/features.py` keeps it out of every
   default feature tier. `strip_ringfenced()` here is what the API calls
   before anything reaches a response body.

2. **Signal quality down-weights, it does not exclude (H1).** The v1.0
   contradiction between the contract and `agents/forecasting/CLAUDE.md` was
   resolved in favour of down-weighting. The weight is computed centrally in
   `contract.modality_weight`; the adapter only guarantees that a bad
   capture arrives with an honest `ppg_signal_quality` rather than being
   dropped upstream.

Note on `ppg_n_windows`: the contract bounds it at 0-3 (three 10-second
windows from a 30-second capture) and specifies features as the median
across windows. A capture yielding zero valid windows is present=0.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from integration import contract

_RINGFENCED = ("ppg_glucose_estimate",)


def ppg_to_contract(
    record: Optional[Dict[str, Any]],
    meal_id: str,
) -> Dict[str, Any]:
    """Translate one rppg contract dict into the contract's PPG block."""
    if record is None:
        block = {f.name: f.fallback for f in contract.PPG_FIELDS}
        block.update({
            "meal_id": meal_id,
            "ppg_present": 0,
            "ppg_signal_quality": 0.0,
            "ppg_clip_fraction": 1.0,
            "ppg_n_windows": 0,
            "ppg_model_version": "absent",
            "ppg_glucose_estimate": None,
        })
        return block

    out: Dict[str, Any] = {"meal_id": meal_id}
    for f in contract.PPG_FIELDS:
        value = record.get(f.name)
        if value is None:
            value = f.fallback
        if value is not None and f.kind is float:
            value = float(value)
        elif value is not None and f.kind is int:
            value = int(value)
        out[f.name] = value

    n_windows = int(out.get("ppg_n_windows") or 0)
    if n_windows == 0:
        # No valid window means no measurement, whatever else the record says.
        out["ppg_present"] = 0
        out["ppg_signal_quality"] = 0.0

    if not out.get("ppg_present"):
        out["ppg_signal_quality"] = 0.0
        # A glucose estimate from a capture we are calling absent is not a
        # measurement of anything.
        out["ppg_glucose_estimate"] = None

    return out


def strip_ringfenced(record: Dict[str, Any]) -> Dict[str, Any]:
    """Remove ring-fenced fields before a record leaves the system.

    Contract §Ring-fence rule 2: `ppg_glucose_estimate` is never surfaced to
    a user. The API calls this on every response body.
    """
    return {k: v for k, v in record.items() if k not in _RINGFENCED}
