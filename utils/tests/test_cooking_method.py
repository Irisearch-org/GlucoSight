"""Pins the exact behavior moved from nlp/data/derive_labels.py (code
review, 2026-10-19) — same test cases nlp/tests already covers indirectly
through the NLP baseline, pinned here at the actual seam."""

from utils.cooking_method import is_fried, is_large, total_grams


def test_is_fried_matches_indonesian_and_english_keywords():
    assert is_fried("ayam goreng") == 1
    assert is_fried("fried chicken") == 1
    assert is_fried("grilled chicken") == 0


def test_is_fried_negation_overrides_the_keyword():
    assert is_fried("tidak digoreng, dikukus saja") == 0


def test_is_large_from_explicit_wording():
    assert is_large("porsi besar") == 1
    assert is_large("small portion") == 0


def test_is_large_falls_back_to_total_grams():
    assert is_large("500g nasi") == 1
    assert is_large("200g nasi") == 0


def test_total_grams_sums_every_match():
    assert total_grams("250g rice + 150g chicken") == 400.0
