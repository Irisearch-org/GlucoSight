"""Adapter tests — the three track schemas reconciled to contract v1.2.

These are the tests that would have caught the integration gap: CV emits a
v0.1 record that no contract validator has ever seen, and until now nothing
checked that the translation preserved meaning.
"""

import json
import pathlib

import pytest

from integration import contract
from integration.adapters import (
    cv_items_to_contract, cv_to_contract, nlp_to_contract, ppg_to_contract,
)

REPO = pathlib.Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------

def test_cv_real_committed_features_translate_cleanly():
    """Runs against the CV track's actual committed output, not a mock."""
    path = REPO / "cv/features/cv/cv-baseline-v0.1/features.jsonl"
    if not path.exists():
        pytest.skip("CV feature dump not present")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert records, "CV feature dump is empty"

    for rec in records:
        block = cv_to_contract(rec, meal_id=rec["sample_id"])
        for f in contract.CV_FIELDS:
            assert f.name in block, f"{f.name} missing for {rec['sample_id']}"
            err = f.check(block[f.name])
            assert err is None, f"{rec['sample_id']}: {err}"


def test_cv_mock_records_are_marked_absent():
    """A MockClassifier output is not a measurement.

    The CV track's own schema calls feature_status='mock' "TIDAK valid untuk
    downstream". Letting it through with cv_present=1 would put fabricated
    macros into a fusion result.
    """
    block = cv_to_contract(
        {"food_top1": "nasi_goreng", "carbs_g": 60.0, "protein_g": 10.0,
         "fat_g": 5.0, "fiber_g": 2.0, "confidence": 0.99,
         "feature_status": "mock", "model_version": "mock-v0"},
        meal_id="m1",
    )
    assert block["cv_present"] == 0
    assert block["cv_confidence"] == 0.0
    assert block["carbs_g"] == 45.0
    assert block["protein_g"] == 18.0
    assert block["fat_g"] == 12.0
    assert block["fiber_g"] == 4.0
    assert block["food_class"] == "unknown"


def test_cv_partial_macro_record_falls_back_as_a_whole_modality():
    block = cv_to_contract(
        {"food_top1": "nasi_goreng", "carbs_g": 60.0, "protein_g": 10.0,
         "fat_g": None, "fiber_g": 2.0, "confidence": 0.99,
         "feature_status": "ok", "model_version": "v0"}, "m1")
    assert block["cv_present"] == 0
    assert block["carbs_g"] == 45.0
    assert block["protein_g"] == 18.0
    assert block["fat_g"] == 12.0
    assert block["fiber_g"] == 4.0
    assert block["food_class"] == "unknown"


def test_cv_error_record_nulls_become_fallbacks_not_zeros():
    """0 g carbs reads downstream as 'food with no calories'."""
    block = cv_to_contract(
        {"food_top1": None, "carbs_g": None, "protein_g": None, "fat_g": None,
         "fiber_g": None, "confidence": None, "feature_status": "error",
         "model_version": "cv-baseline-v0.1"},
        meal_id="m1",
    )
    assert block["carbs_g"] == 45.0
    assert block["cv_present"] == 0
    assert block["carbs_source"] == "population_mean"


def test_cv_absent_photo_yields_full_fallback_block():
    block = cv_to_contract(None, meal_id="m1", portion_reported=0)
    assert block["cv_present"] == 0
    assert block["carbs_g"] == 45.0
    assert block["portion_reported"] == 0
    assert block["carbs_source"] == "population_mean"


def test_cv_portion_is_passed_through_not_predicted():
    """portion_reported is the in-app question, not a CV output (C5)."""
    rec = {"food_top1": "gudeg", "carbs_g": 34.0, "protein_g": 7.0,
           "fat_g": 16.0, "fiber_g": 5.4, "confidence": 0.9,
           "feature_status": "ok", "model_version": "cv-baseline-v0.1"}
    assert cv_to_contract(rec, "m1", portion_reported=0)["portion_reported"] == 0
    assert cv_to_contract(rec, "m2", portion_reported=2)["portion_reported"] == 2


def test_cv_rejects_invalid_portion():
    with pytest.raises(contract.ContractViolation):
        cv_to_contract(None, "m1", portion_reported=5)


def test_cv_gi_provenance_distinguishes_table_from_fallback():
    known = cv_to_contract(
        {"food_top1": "nasi_goreng", "carbs_g": 60.0, "protein_g": 10.0,
         "fat_g": 5.0, "fiber_g": 2.0, "confidence": 0.9,
         "feature_status": "ok", "model_version": "v0"}, "m1")
    assert known["gi_category"] == 2
    assert known["gi_source"] == "class_table"

    unknown = cv_to_contract(
        {"food_top1": "makanan_tidak_dikenal", "carbs_g": 60.0, "protein_g": 10.0,
         "fat_g": 5.0, "fiber_g": 2.0, "confidence": 0.9,
         "feature_status": "ok", "model_version": "v0"}, "m2")
    assert unknown["gi_category"] == 1
    assert unknown["gi_source"] == "contract_fallback"


def test_cv_weighed_override_only_applies_when_present():
    """On CGMacros the macros are weighed — but not for an absent photo."""
    rec = {"food_top1": "gudeg", "carbs_g": 34.0, "protein_g": 7.0,
           "fat_g": 16.0, "fiber_g": 5.4, "confidence": 0.9,
           "feature_status": "ok", "model_version": "v0"}
    assert cv_to_contract(rec, "m1", carbs_source_override="weighed")["carbs_source"] == "weighed"
    absent = cv_to_contract(None, "m2", carbs_source_override="weighed")
    assert absent["carbs_source"] == "population_mean"


def test_cv_rejects_unknown_carbs_source():
    with pytest.raises(contract.ContractViolation):
        cv_to_contract(None, "m1", carbs_source_override="vibes")


# ---------------------------------------------------------------------
# CV — mixed plates (docs/SPEC_DISH_IDENTIFICATION.md)
#
# `items` stands in for a VLM's listed dishes: [{"food_name", "confidence"}].
# The VLM call itself is not exercised here (per the spec's own Testing
# Decisions) — these fixtures are what the adapter receives regardless of
# which VLM produced them.
# ---------------------------------------------------------------------

def test_cv_single_item_behaves_like_the_single_dish_case():
    block = cv_items_to_contract(
        [{"food_name": "rendang", "confidence": 0.9}], meal_id="m1",
    )
    assert block["cv_present"] == 1
    assert block["carbs_g"] == 3.5
    assert block["cv_confidence"] == 0.9
    assert block["carbs_source"] == "class_lookup"
    assert block["food_class"] == "rendang"


def test_cv_mixed_plate_sums_macros_and_takes_minimum_confidence():
    """Nasi padang: rice plus a side dish, not one label (Task 5)."""
    block = cv_items_to_contract(
        [{"food_name": "nasi goreng", "confidence": 0.95},
         {"food_name": "rendang daging", "confidence": 0.7}],
        meal_id="m1",
    )
    assert block["cv_present"] == 1
    assert block["carbs_g"] == 67.5 + 3.5  # nasi_goreng (250g) + rendang (100g)
    assert block["cv_confidence"] == 0.7  # the weaker item, not the average
    assert block["carbs_source"] == "class_lookup_multi"


def test_cv_mixed_plate_drops_unmatched_items_not_defaults_them():
    block = cv_items_to_contract(
        [{"food_name": "rendang", "confidence": 0.9},
         {"food_name": "makanan antah berantah", "confidence": 0.5}],
        meal_id="m1",
    )
    # One real item matched: behaves like the single-dish case, not a
    # population-mean blend with the unmatched item.
    assert block["carbs_g"] == 3.5
    assert block["carbs_source"] == "class_lookup"
    assert block["cv_items_unmatched"] == 1


def test_cv_mixed_plate_output_satisfies_every_contract_field():
    block = cv_items_to_contract(
        [{"food_name": "nasi goreng", "confidence": 0.95},
         {"food_name": "rendang", "confidence": 0.7}],
        meal_id="m1",
    )
    for f in contract.CV_FIELDS:
        assert f.name in block, f"{f.name} missing"
        err = f.check(block[f.name])
        assert err is None, err


def test_cv_no_items_behaves_like_absent_photo():
    assert cv_items_to_contract(None, "m1") == cv_items_to_contract([], "m1")
    assert cv_items_to_contract(None, "m1")["cv_present"] == 0


def test_cv_all_items_unmatched_behaves_like_absent_photo():
    block = cv_items_to_contract(
        [{"food_name": "makanan antah berantah", "confidence": 0.5}], "m1",
    )
    assert block["cv_present"] == 0
    assert block["carbs_source"] == "population_mean"
    # code-review finding: this must stay visible even when the whole list
    # fails to match, not just the partial-match case — "not defaulted to a
    # population mean silently" applies at the whole-list level too.
    assert block["cv_items_unmatched"] == 1
    assert block["cv_item_count"] == 0


def test_cv_mixed_plate_sets_carbs_source_override_like_the_single_dish_case():
    """CGMacros weighs macros; this path needs the same override cv_to_contract has."""
    block = cv_items_to_contract(
        [{"food_name": "rendang", "confidence": 0.9}],
        meal_id="m1", carbs_source_override="weighed",
    )
    assert block["carbs_source"] == "weighed"


# ---------------------------------------------------------------------
# NLP
# ---------------------------------------------------------------------

def test_nlp_absent_record_is_unavailable_not_all_zero_labels():
    block = nlp_to_contract(None, "m1")
    assert block["nlp_present"] == 0
    assert block["nlp_feature_source"] == "unavailable"
    assert block["nlp_confidence"] == 0.0


def test_nlp_distinguishes_no_record_from_a_record_reporting_nothing():
    no_record = nlp_to_contract(None, "m1")
    empty_record = nlp_to_contract(
        {"is_fried_cooking": 0, "is_large_portion": 0, "nlp_confidence": 0.8,
         "nlp_present": 1, "nlp_feature_source": "rule_based",
         "nlp_model_version": "v0.1"}, "m2")
    assert no_record["nlp_present"] != empty_record["nlp_present"]
    assert no_record["is_fried_cooking"] == empty_record["is_fried_cooking"] == 0


def test_nlp_incoherent_present_and_confidence_is_rejected():
    with pytest.raises(contract.ContractViolation, match="must not share an encoding"):
        nlp_to_contract(
            {"nlp_present": 0, "nlp_confidence": 0.7,
             "nlp_feature_source": "unavailable", "nlp_model_version": "v0"}, "m1")


def test_nlp_unavailable_with_present_is_rejected():
    with pytest.raises(contract.ContractViolation, match="incoherent"):
        nlp_to_contract(
            {"nlp_present": 1, "nlp_confidence": 0.5,
             "nlp_feature_source": "unavailable", "nlp_model_version": "v0"}, "m1")


def test_nlp_reports_calibration_unavailable_and_zero_confidence():
    block = nlp_to_contract(
        {"is_fried_cooking": 1, "is_large_portion": 0, "nlp_confidence": 0.0,
         "nlp_calibration_status": "unavailable", "nlp_present": 1,
         "nlp_feature_source": "rule_based", "nlp_model_version": "v0.1"}, "m1")
    assert block["nlp_calibrated"] is False
    assert block["nlp_calibration_status"] == "unavailable"
    assert block["nlp_confidence"] == 0.0


def test_nlp_rejects_unknown_feature_source():
    with pytest.raises(contract.ContractViolation):
        nlp_to_contract(
            {"nlp_present": 1, "nlp_confidence": 0.5,
             "nlp_feature_source": "guessing", "nlp_model_version": "v0"}, "m1")


def test_nlp_real_predict_output_translates():
    """Round-trip through the NLP track's actual predict()."""
    pytest.importorskip("pandas")
    try:
        from nlp.nlp_baseline.predict import predict
    except Exception as exc:
        pytest.skip(f"nlp predict unavailable: {exc}")
    out = predict({"meal_id": "m1", "participant_id": "P1",
                   "dietary_intake": "fried rice, large portion"})
    block = nlp_to_contract(out, "m1")
    for f in contract.NLP_FIELDS:
        assert f.check(block[f.name]) is None


# ---------------------------------------------------------------------
# NLP — cooking-method correction (docs/SPEC_DISH_IDENTIFICATION.md, H2)
#
# `cooking_method_text` stands in for whatever live text exists. Checked
# for redundancy against `cv_food_class` (the dish CV already identified)
# before being treated as independent information.
# ---------------------------------------------------------------------

_SOTO_RECORD = {"is_fried_cooking": 0, "is_large_portion": 0,
                "nlp_confidence": 0.8, "nlp_present": 1,
                "nlp_feature_source": "rule_based", "nlp_model_version": "v0.1"}


def test_nlp_cooking_method_text_corrects_when_cv_doesnt_already_imply_it():
    """soto is not fried by name; the text is genuinely new information."""
    block = nlp_to_contract(
        _SOTO_RECORD, "m1",
        cv_food_class="soto", cooking_method_text="soto ayam digoreng dulu",
    )
    assert block["is_fried_cooking"] == 1
    assert block["nlp_feature_source"] == "rule_based"
    assert block.get("nlp_cooking_method_redundant_with_cv") is False


def test_nlp_cooking_method_text_flagged_redundant_when_cv_already_implies_it():
    """nasi_goreng is fried by name — text saying so adds no new information."""
    block = nlp_to_contract(
        {**_SOTO_RECORD, "is_fried_cooking": 1}, "m1",
        cv_food_class="nasi_goreng", cooking_method_text="nasi goreng pete",
    )
    assert block["nlp_cooking_method_redundant_with_cv"] is True


def test_nlp_cooking_method_correction_is_noop_without_a_mention():
    block = nlp_to_contract(_SOTO_RECORD, "m1", cv_food_class="soto")
    assert block["is_fried_cooking"] == 0
    assert block["nlp_cooking_method_redundant_with_cv"] is None


def test_nlp_cooking_method_correction_does_not_fire_without_a_record():
    """Scoped: corrects an existing record, doesn't manufacture one from text alone."""
    block = nlp_to_contract(None, "m1", cooking_method_text="semuanya digoreng")
    assert block["nlp_present"] == 0
    assert block["is_fried_cooking"] == 0


# ---------------------------------------------------------------------
# PPG
# ---------------------------------------------------------------------

def test_ppg_absent_capture_is_full_fallback():
    block = ppg_to_contract(None, "m1")
    assert block["ppg_present"] == 0
    assert block["ppg_signal_quality"] == 0.0
    assert block["ppg_clip_fraction"] == 1.0
    assert block["ppg_pulse_rate_bpm"] == 75.0


def test_ppg_zero_windows_means_absent():
    block = ppg_to_contract(
        {"ppg_pulse_rate_bpm": 82.0, "ppg_signal_quality": 0.9,
         "ppg_present": 1, "ppg_clip_fraction": 0.1, "ppg_n_windows": 0,
         "ppg_model_version": "v0"}, "m1")
    assert block["ppg_present"] == 0
    assert block["ppg_signal_quality"] == 0.0


def test_ppg_bad_capture_is_kept_and_downweighted_not_dropped():
    block = ppg_to_contract(
        {"ppg_pulse_rate_bpm": 82.0, "ppg_signal_quality": 0.05,
         "ppg_present": 1, "ppg_clip_fraction": 0.9, "ppg_n_windows": 1,
         "ppg_model_version": "v0"}, "m1")
    assert block["ppg_present"] == 1
    assert block["ppg_pulse_rate_bpm"] == 82.0
    w = contract.modality_weight(block, "ppg")
    assert 0 < w < 1.0


def test_ppg_strip_ringfenced_removes_the_glucose_estimate():
    from integration.adapters.ppg import strip_ringfenced
    block = ppg_to_contract(
        {"ppg_pulse_rate_bpm": 82.0, "ppg_signal_quality": 0.9,
         "ppg_present": 1, "ppg_clip_fraction": 0.1, "ppg_n_windows": 3,
         "ppg_glucose_estimate": 145.0, "ppg_model_version": "v0"}, "m1")
    assert block["ppg_glucose_estimate"] == 145.0
    assert "ppg_glucose_estimate" not in strip_ringfenced(block)


def test_ppg_real_extractor_contract_dict_translates():
    """Round-trip through the PPG track's own to_contract_dict()."""
    pytest.importorskip("scipy")
    try:
        import numpy as np
        from rppg.features.extractor import extract_features, to_contract_dict
    except Exception as exc:
        pytest.skip(f"rppg extractor unavailable: {exc}")
    fs = 30.0
    t = np.arange(0, 10, 1 / fs)
    signal = 1.0 + 0.05 * np.sin(2 * np.pi * 1.2 * t)
    feats = extract_features(signal, fs)
    native = to_contract_dict(feats, glucose_estimate=None, present=1, n_windows=1)
    block = ppg_to_contract(native, "m1")
    for f in contract.PPG_FIELDS:
        assert f.check(block[f.name]) is None, f.name
