"""The matcher — GLOSSARY.md "matcher": food-name string -> nearest dish-table
row, or no-match. Shared by CV (VLM-detected item names) and NLP
(cooking-method/dish mentions in text). See docs/SPEC_DISH_IDENTIFICATION.md.
"""

from integration.dish_table import DishRow, load_dish_table, match_dish, serving_macros

_ROWS = (
    DishRow(
        food_class="nasi_goreng", tkpi_item_name="Nasi goreng", serving_g=250,
        carbs_g_100g=27.0, protein_g_100g=4.5, fat_g_100g=6.5,
        fiber_ratio_of_carbs=0.03, mapping_confidence="high",
    ),
    DishRow(
        food_class="rendang", tkpi_item_name="Rendang daging sapi", serving_g=100,
        carbs_g_100g=3.5, protein_g_100g=22.6, fat_g_100g=10.5,
        fiber_ratio_of_carbs=0.14, mapping_confidence="high",
    ),
)


def test_exact_food_class_matches_itself():
    match = match_dish("nasi_goreng", _ROWS)
    assert match is not None
    assert match.row.food_class == "nasi_goreng"


def test_unrelated_name_returns_no_match():
    """An unmatched name must be visible as no-match, never a nearest-row guess."""
    assert match_dish("makanan antah berantah", _ROWS) is None


def test_near_miss_name_still_resolves():
    """A VLM/free-text name rarely spells the food_class exactly."""
    match = match_dish("nasi goreng spesial", _ROWS)
    assert match is not None
    assert match.row.food_class == "nasi_goreng"
    assert match.row.food_class != "rendang"


def test_real_dish_table_loads_and_matches():
    """The adapters call this against the actual committed table, not a fixture."""
    rows = load_dish_table()
    classes = {row.food_class for row in rows}
    assert "rendang" in classes
    assert len(rows) == 10  # cv/data/tkpi_macro_lookup.csv, as of this ticket

    match = match_dish("rendang daging", rows)
    assert match is not None
    assert match.row.food_class == "rendang"


def test_serving_macros_matches_the_committed_cv_baseline_value():
    """rendang: 100 g serving, 3.5 g carbs/100g -> cv/cv_baseline/macro_lookup.py
    has already published 3.5 g carbs for this exact row; pin to that, not a
    hand-derived number, so the two don't silently diverge."""
    row = next(r for r in load_dish_table() if r.food_class == "rendang")
    macros = serving_macros(row)
    assert macros["carbs_g"] == 3.5
    assert macros["fiber_g"] == round(3.5 * 0.14, 1)
