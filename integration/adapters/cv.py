"""CV track (cv-baseline-v0.1) -> contract v1.2.

The CV track emits its own v0.1 record:

    {"sample_id", "modality", "food_top1", "food_top3", "carbs_g",
     "protein_g", "fat_g", "fiber_g", "calories_kcal", "confidence",
     "feature_status", "model_version", "created_at", ...}

The contract wants `meal_id`, `cv_confidence`, `cv_present`, `carbs_source`,
`portion_reported`, `gi_category`, `cv_model_version`. Four fields the
contract requires have no producer in the CV track at all:

  gi_category       — derived here from the food class, see GI_BY_CLASS.
  carbs_source      — derived from `feature_status` + the macro provenance
                      fields the track already emits.
  portion_reported  — NOT a CV output. It comes from the in-app question
                      ("porsi: kecil / sedang / besar"). The contract says CV
                      passes it through so Forecasting receives one dict; the
                      adapter takes it as an argument for the same reason.
  cv_present        — from feature_status, not from confidence.

`feature_status == "mock"` maps to cv_present=0. A MockClassifier record is
not a measurement, and the CV track's own schema calls it "TIDAK valid untuk
downstream". Letting it through with present=1 would put fabricated macros
into a fusion result.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from integration import contract

# Glycemic index band by food class. 0=low (<55), 1=medium (55-70), 2=high (>70).
#
# PROVENANCE: these are assignments made by the integration layer from the
# Indonesian food classes the CV baseline recognises, using published GI
# values for the dominant carbohydrate component (Atkinson, Foster-Powell &
# Brand-Miller, "International Tables of Glycemic Index and Glycemic Load
# Values", Diabetes Care 2008/2021 revisions) — white rice GI ~73, glutinous
# rice ~86, wheat noodles ~49, cassava ~46.
#
# They are NOT measured on Indonesian preparations and are NOT validated.
# `gi_source` travels with the record so a downstream consumer can tell an
# assignment from a measurement. The CV track should own this table once it
# has evidence; until then the default is medium, which is what the contract
# specifies as the fallback.
GI_BY_CLASS: Dict[str, int] = {
    "nasi_goreng": 2,       # fried white rice
    "nasi_uduk": 2,         # coconut-rice, still white rice dominant
    "lontong": 2,           # compressed rice cake
    "ketupat": 2,
    "bubur_ayam": 2,        # rice porridge, gelatinised -> higher GI
    "gudeg": 1,             # jackfruit stew, served with rice
    "rendang": 0,           # beef, minimal carbohydrate
    "bebek_betutu": 0,      # duck, minimal carbohydrate
    "ayam_goreng": 0,
    "sate": 0,
    "soto": 1,
    "bakso": 1,             # wheat-flour meatball + noodle
    "mie_goreng": 1,        # wheat noodle, lower GI than white rice
    "pempek": 1,            # sago/tapioca fishcake
    "gado_gado": 0,         # vegetable + peanut sauce
    "tempe_goreng": 0,
    "tahu_goreng": 0,
}

DEFAULT_GI = 1  # contract fallback: medium

# feature_status -> (cv_present, carbs_source)
_STATUS_MAP = {
    "ok": (1, "class_lookup"),
    "low_confidence": (1, "class_lookup"),
    "mock": (0, "population_mean"),
    "error": (0, "population_mean"),
}


def cv_to_contract(
    record: Optional[Dict[str, Any]],
    meal_id: str,
    portion_reported: int = 1,
    carbs_source_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Translate one cv-baseline record into the contract's CV block.

    Args:
        record: a cv-baseline-v0.1 dict, or None when no photo was taken.
        meal_id: the envelope key this CV record belongs to.
        portion_reported: 0=kecil, 1=sedang, 2=besar, from the in-app
            question. Defaults to the contract fallback (1, sedang).
        carbs_source_override: set to "weighed" on CGMacros, where breakfast
            and lunch macros are weighed rather than looked up
            (DATA_STRATEGY §5.3 — finding C5 does not bite there).

    Returns:
        A contract v1.2 CV block. Absent CV yields every field at its
        contract fallback with cv_present=0 and cv_confidence=0.0.
    """
    if portion_reported not in (0, 1, 2):
        raise contract.ContractViolation(
            f"portion_reported must be 0/1/2, got {portion_reported!r}"
        )
    # Validated here, before the absent-photo branch: an invalid override
    # must fail whether or not a photo happened to be taken.
    if (carbs_source_override is not None
            and carbs_source_override not in contract.VALID_CARBS_SOURCES):
        raise contract.ContractViolation(
            f"carbs_source_override {carbs_source_override!r} not in "
            f"{sorted(contract.VALID_CARBS_SOURCES)}"
        )

    if record is None:
        block = {f.name: f.fallback for f in contract.CV_FIELDS}
        block.update({
            "cv_present": 0,
            "cv_confidence": 0.0,
            "carbs_source": "population_mean",
            "portion_reported": portion_reported,
            "cv_model_version": "absent",
            "food_class": "unknown",
            "gi_category": DEFAULT_GI,
            "meal_id": meal_id,
            "gi_source": "contract_fallback",
        })
        return block

    status = record.get("feature_status", "error")
    present, carbs_source = _STATUS_MAP.get(status, (0, "population_mean"))

    # An error record has null macros by design — the CV track's own comment
    # says a 0 would read downstream as "food with no calories". Fall back to
    # the contract's population means and mark the modality absent.
    macros = {}
    for name, fallback in (("carbs_g", 45.0), ("protein_g", 18.0),
                           ("fat_g", 12.0), ("fiber_g", 4.0)):
        value = record.get(name)
        if value is None:
            macros[name] = fallback
            present = 0
            carbs_source = "population_mean"
        else:
            macros[name] = float(value)

    food_class = record.get("food_top1") or "unknown"
    if food_class in GI_BY_CLASS:
        gi_category = GI_BY_CLASS[food_class]
        gi_source = "class_table"
    else:
        gi_category = DEFAULT_GI
        gi_source = "contract_fallback"

    confidence = record.get("confidence")
    confidence = 0.0 if confidence is None else float(confidence)
    if not present:
        # H1: present=0 and a positive confidence are contradictory states.
        confidence = 0.0

    if carbs_source_override is not None and present:
        carbs_source = carbs_source_override

    return {
        "meal_id": meal_id,
        **macros,
        "gi_category": gi_category,
        "food_class": food_class,
        "cv_confidence": confidence,
        "cv_present": present,
        "carbs_source": carbs_source,
        "portion_reported": portion_reported,
        "cv_model_version": record.get("model_version", "unknown"),
        # Not contract fields — provenance carried for the audit trail and
        # stripped by fusion.assemble() before validation.
        "gi_source": gi_source,
        "cv_feature_status": status,
    }
