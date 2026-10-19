"""`nlp.nlp_baseline.predict` — the live NLP path that `integration/api.py`
calls. It already computes `total_grams` internally (via
`nlp.data.derive_labels._total_grams`) but never returned it, and never
derived dish identity via `nlp.data.diet_items.meal_items`. Both are needed
by the gated dish-identity model (`nlp/models/dish_identity_ridge.py`) once
it is wired in — see plan `drifting-inventing-bentley.md`, change 2.
"""

from __future__ import annotations

from nlp.nlp_baseline.predict import predict


def test_predict_returns_total_grams_and_dish_names_for_structured_text():
    out = predict({"text": "Rice 150 g | Cabbage 20 g"})
    assert out["total_grams"] == 170.0
    assert set(out["dish_names"]) == {"rice", "cabbage"}


def test_predict_returns_zero_and_empty_for_unstructured_free_text():
    """Documents the known live-input limitation (plan change 2): Bahasa
    free prose does not match diet_items.ITEM_RE's structured grammar, so
    total_grams/dish_names come back as the existing "nothing parsed"
    sentinel (0.0 — the value `_total_grams` already returns when no gram
    quantity is found, per nlp/data/derive_labels.py's own convention) on
    real demo input. Not a bug."""
    out = predict({"text": "nasi goreng porsi besar, digoreng"})
    assert out["total_grams"] == 0.0
    assert out["dish_names"] == []


def test_predict_returns_zero_and_empty_for_missing_input():
    out = predict({"text": ""})
    assert out["total_grams"] == 0.0
    assert out["dish_names"] == []
