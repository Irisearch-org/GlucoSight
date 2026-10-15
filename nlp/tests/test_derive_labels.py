from nlp.data.derive_labels import _is_fried, _is_large
from nlp.nlp_baseline.classifier import classify_context
from nlp.nlp_baseline.predict import predict


def test_bahasa_fried_lexicon_and_negation():
    assert _is_fried("nasi digoreng", "") == 1
    assert _is_fried("gorengan", "") == 1
    assert _is_fried("ayam tidak digoreng", "") == 0


def test_bahasa_portion_words():
    assert _is_large("nasi porsi besar", "") == 1
    assert _is_large("nasi jumbo", "") == 1
    assert _is_large("porsi kecil", "") == 0
    assert _is_large("rice 150 g", "") == 0


def test_alternate_classifier_uses_same_bahasa_rules_without_fake_confidence():
    record = classify_context("nasi tidak digoreng, porsi besar")
    assert record["is_fried_cooking"] == 0
    assert record["is_large_portion"] == 1
    assert record["nlp_confidence"] == 0.0
    assert record["nlp_calibration_status"] == "unavailable"


def test_predict_does_not_claim_uncalibrated_confidence():
    record = predict({"text": "nasi goreng porsi besar, digoreng"})
    assert record["is_fried_cooking"] == 1
    assert record["is_large_portion"] == 1
    assert record["nlp_confidence"] == 0.0
    assert record["nlp_calibration_status"] == "unavailable"
