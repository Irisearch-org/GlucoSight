"""Guards the measurements in `docs/NLP_PATH_A_GATE.md`.

These are property tests, not constants: they assert the invariants that
made each finding a finding. If someone fixes the pipeline (adds a
composition table, regroups the folds, restores the raw data), the
corresponding test here should start failing, and the document should be
revised alongside it.
"""
from __future__ import annotations

import pytest

from nlp.eval_path_a_gate import (
    DEFAULT_FEATURES,
    base_sentence,
    dish_names,
    feature_collapse,
    load_features,
    parse_items,
    path_b_leakage,
    portion_decomposition,
    richness,
    split_unit,
    subject_of,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_FEATURES.exists(), reason="Path A feature file not present"
)


@pytest.fixture(scope="module")
def rows():
    return load_features()


# --------------------------------------------------------------------------
# parsing, which every measurement below depends on
# --------------------------------------------------------------------------

def test_parses_grams_and_millilitres():
    items = parse_items([{"participant_id": "p", "dietary_intake": "Rice 150 g | Milk 250 ml"}])
    assert items == [("p", "rice", 150.0), ("p", "milk", 250.0)]


def test_weight_is_anchored_at_the_end_of_the_segment():
    # A digit inside the item name must not be mistaken for the weight.
    items = parse_items([{"participant_id": "p", "dietary_intake": "Vitamin B12 50 g"}])
    assert items == [("p", "vitamin b12", 50.0)]


def test_segment_without_a_weight_is_dropped_not_guessed():
    assert parse_items([{"participant_id": "p", "dietary_intake": "Soup | Rice 100 g"}]) == [
        ("p", "rice", 100.0)
    ]


def test_zero_weight_is_dropped_because_it_breaks_the_log_transform():
    assert parse_items([{"participant_id": "p", "dietary_intake": "Rice 0 g"}]) == []


def test_unparseable_segments_are_rare(rows):
    after = len(parse_items(rows))
    before = sum(len(r["dietary_intake"].split("|")) for r in rows)
    assert after <= before
    assert after / before > 0.99


# --------------------------------------------------------------------------
# N1 — the gate's verdict
# --------------------------------------------------------------------------

def test_n1_records_are_rich_enough_that_the_lookup_branch_does_not_apply(rows):
    r = richness(rows)
    # The gate's "uniformly 米饭 100g" case would give a handful of names.
    assert r["distinct_dish_names"] > 100
    assert r["multi_item_meals"] > 0
    assert r["distinct_meal_texts"] / r["meals"] > 0.5


# --------------------------------------------------------------------------
# N2 — the dish name is discarded
# --------------------------------------------------------------------------

def test_n2_feature_vector_carries_less_than_a_tenth_of_the_meal_identity(rows):
    c = feature_collapse(rows)
    assert c["distinct_dish_names"] > 1000
    assert c["distinct_carb_densities"] == 1
    assert c["distinct_feature_states"] / c["distinct_meal_texts"] < 0.5


# --------------------------------------------------------------------------
# N5 — the split unit is the period, not the patient
# --------------------------------------------------------------------------

def test_n5_period_ids_outnumber_subjects(rows):
    s = split_unit(rows)
    assert s["periods"] > s["subjects"]
    assert s["subjects_with_multiple_periods"] > 0


def test_n5_grouping_on_period_id_leaks_every_repeat_visit(rows):
    s = split_unit(rows)
    # The invariant: every period belonging to a multi-period subject ends up
    # in a test fold while that subject is in train. That is the whole leak.
    multi = {}
    for r in rows:
        multi.setdefault(subject_of(r["participant_id"]), set()).add(r["participant_id"])
    expected = sum(len(v) for v in multi.values() if len(v) > 1)
    assert s["leaking_test_groups"] == expected
    assert s["leaking_meal_share"] > 0.15


def test_subject_of_strips_the_period_and_date():
    assert subject_of("2069_1_20210705") == "2069"


# --------------------------------------------------------------------------
# N7 — Path B leakage
# --------------------------------------------------------------------------

def test_base_sentence_strips_one_prefix(rows=None):
    assert base_sentence("Tadi Aku makan ayam goreng.") == "aku makan ayam goreng."
    assert base_sentence("Aku tadi makan ayam goreng.") == "makan ayam goreng."
    assert base_sentence("Aku makan ayam goreng.") == "aku makan ayam goreng."


def test_n7_most_test_sentences_have_a_near_duplicate_in_train():
    pb = path_b_leakage()
    if pb is None:
        pytest.skip("Path B splits not present")
    n, d = (int(x) for x in pb["test_bases_in_train"].split("/"))
    assert n / d > 0.5
    # 400 "unique" strings are actually a small template set under prefixes.
    assert pb["distinct_base_sentences"] < pb["distinct_sentences"] / 2


# --------------------------------------------------------------------------
# N8 — weighed portion variation
# --------------------------------------------------------------------------

def test_n8_portion_share_is_much_larger_than_the_self_reported_floor(rows):
    d = portion_decomposition(rows)
    within = d["within_participant_x_dish"]["portion_share"]
    pooled = d["pooled_by_dish_name"]["portion_share"]
    # BIG IDEAs B7 measured 0.025 on self-reported amounts.
    assert within > 0.10
    assert d["within_participant_x_dish"]["which_dish_share"] > 0.5


def test_n8_pooling_across_people_inflates_the_portion_term(rows):
    d = portion_decomposition(rows)
    # Between-person differences are real variation, but they are not the
    # within-person portion term B7 measured.
    assert (
        d["pooled_by_dish_name"]["portion_share"]
        > d["within_participant_x_dish"]["portion_share"]
    )


def test_n8_single_weight_share_is_below_the_self_reported_figure(rows):
    d = portion_decomposition(rows)
    assert d["within_participant_x_dish"]["single_weight_share"] < 0.51


def test_dish_names_are_lowercased_and_whitespace_normalised(rows):
    names = dish_names(rows)
    assert all(n == " ".join(n.split()) and n == n.lower() for n in names)
