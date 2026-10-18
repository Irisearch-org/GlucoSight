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

from integration import contract, dish_table

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


def _validate_portion_and_override(
    portion_reported: int, carbs_source_override: Optional[str]
) -> None:
    """Shared by `cv_to_contract` and `cv_items_to_contract`.

    Validated before either function's absent-photo branch: an invalid
    portion or override must fail whether or not a photo happened to be
    taken, and whether it was one dish or a list.
    """
    if portion_reported not in (0, 1, 2):
        raise contract.ContractViolation(
            f"portion_reported must be 0/1/2, got {portion_reported!r}"
        )
    if (carbs_source_override is not None
            and carbs_source_override not in contract.VALID_CARBS_SOURCES):
        raise contract.ContractViolation(
            f"carbs_source_override {carbs_source_override!r} not in "
            f"{sorted(contract.VALID_CARBS_SOURCES)}"
        )


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
    _validate_portion_and_override(portion_reported, carbs_source_override)

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

    # Mock/error/unknown records are not measured inputs. Never pass their
    # arbitrary values through while marking the modality absent.
    if not present:
        return {
            "meal_id": meal_id,
            "carbs_g": 45.0,
            "protein_g": 18.0,
            "fat_g": 12.0,
            "fiber_g": 4.0,
            "gi_category": DEFAULT_GI,
            "food_class": "unknown",
            "cv_confidence": 0.0,
            "cv_present": 0,
            "carbs_source": "population_mean",
            "portion_reported": portion_reported,
            "cv_model_version": record.get("model_version", "unknown"),
            "gi_source": "contract_fallback",
            "cv_feature_status": status,
        }

    macro_fields = (("carbs_g", 45.0), ("protein_g", 18.0),
                    ("fat_g", 12.0), ("fiber_g", 4.0))
    if any(record.get(name) is None for name, _ in macro_fields):
        return {
            "meal_id": meal_id,
            **{name: fallback for name, fallback in macro_fields},
            "gi_category": DEFAULT_GI,
            "food_class": "unknown",
            "cv_confidence": 0.0,
            "cv_present": 0,
            "carbs_source": "population_mean",
            "portion_reported": portion_reported,
            "cv_model_version": record.get("model_version", "unknown"),
            "gi_source": "contract_fallback",
            "cv_feature_status": status,
        }
    macros = {name: float(record[name]) for name, _ in macro_fields}

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


def cv_items_to_contract(
    items: Optional[list],
    meal_id: str,
    portion_reported: int = 1,
    carbs_source_override: Optional[str] = None,
    model_version: str = "vlm-dish-id-v0",
) -> Dict[str, Any]:
    """Translate a VLM's list of detected dishes into the contract's CV block.

    Args:
        items: stands in for the VLM's output — [{"food_name": str,
            "confidence": float}, ...], one entry per dish it lists on the
            plate (GLOSSARY.md "mixed plate"), or None/empty when no photo
            was taken.
        meal_id: the envelope key this CV record belongs to.
        portion_reported: 0=kecil, 1=sedang, 2=besar, from the in-app
            question. Defaults to the contract fallback (1, sedang).
        carbs_source_override: set to "weighed" on CGMacros, same as
            `cv_to_contract` — see its docstring.
        model_version: identifies which VLM/prompt produced `items`.

    Returns:
        A contract v1.2 CV block. Each name is resolved through the shared
        matcher (`integration.dish_table`) against the dish table; macros
        sum across every matched item, `cv_confidence` is the minimum
        confidence across matched items (one shaky item pulls the whole
        meal down), and `carbs_source` becomes `class_lookup_multi` once
        more than one item matched, distinguishing a mixed-plate result
        from a single-dish lookup.

        An item the matcher can't resolve is dropped from the sum, never
        defaulted to a population mean silently — see
        docs/SPEC_DISH_IDENTIFICATION.md, Implementation Decisions. This
        holds even when *every* item fails to match: the result still
        falls back to population_mean (there's nothing else to predict
        from), but `cv_item_count`/`cv_items_unmatched` still carry the
        count, the same as the partial-match case.
    """
    _validate_portion_and_override(portion_reported, carbs_source_override)

    def absent_block(item_count: int, unmatched_count: int) -> Dict[str, Any]:
        return {
            "meal_id": meal_id,
            "carbs_g": 45.0, "protein_g": 18.0, "fat_g": 12.0, "fiber_g": 4.0,
            "gi_category": DEFAULT_GI,
            "food_class": "unknown",
            "cv_confidence": 0.0,
            "cv_present": 0,
            "carbs_source": "population_mean",
            "portion_reported": portion_reported,
            "cv_model_version": model_version,
            "gi_source": "contract_fallback",
            "cv_item_count": item_count,
            "cv_items_unmatched": unmatched_count,
        }

    if not items:
        return absent_block(item_count=0, unmatched_count=0)

    rows = dish_table.load_dish_table()
    matched = []
    for item in items:
        m = dish_table.match_dish(item.get("food_name", ""), rows)
        if m is None:
            continue
        confidence = float(item.get("confidence", 0.0) or 0.0)
        matched.append((m.row, confidence))

    if not matched:
        return absent_block(item_count=0, unmatched_count=len(items))

    totals = {"carbs_g": 0.0, "protein_g": 0.0, "fat_g": 0.0, "fiber_g": 0.0}
    # Tracks the matched item with the most carbs, to pick that item's GI
    # category for the whole meal (docs/SPEC_DISH_IDENTIFICATION.md:
    # "Default to the GI category of the item contributing the most carbs").
    dominant_carbs_so_far = -1.0
    gi_category, gi_source = DEFAULT_GI, "contract_fallback"
    for row, _ in matched:
        macros = dish_table.serving_macros(row)
        for key in totals:
            totals[key] += macros[key]
        if macros["carbs_g"] > dominant_carbs_so_far:
            dominant_carbs_so_far = macros["carbs_g"]
            if row.food_class in GI_BY_CLASS:
                gi_category, gi_source = GI_BY_CLASS[row.food_class], "class_table"
            else:
                gi_category, gi_source = DEFAULT_GI, "contract_fallback"

    confidence = min(c for _, c in matched)
    carbs_source = carbs_source_override or (
        "class_lookup" if len(matched) == 1 else "class_lookup_multi"
    )

    return {
        "meal_id": meal_id,
        **{k: round(v, 1) for k, v in totals.items()},
        "gi_category": gi_category,
        # Joined, not a list: food_class is a plain str contract field
        # (CV_FIELDS), and nothing downstream parses a mixed-plate value
        # back apart today. Readable for logging/audit, not a structured
        # multi-value field — revisit if a consumer ever needs the list.
        "food_class": "+".join(row.food_class for row, _ in matched),
        "cv_confidence": confidence,
        "cv_present": 1,
        "carbs_source": carbs_source,
        "portion_reported": portion_reported,
        "cv_model_version": model_version,
        "gi_source": gi_source,
        "cv_item_count": len(matched),
        "cv_items_unmatched": len(items) - len(matched),
    }
