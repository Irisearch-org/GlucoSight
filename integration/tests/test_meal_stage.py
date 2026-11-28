"""Meal stage: confidence rule at the matcher seam, mixed plates, portion -> grams, macros."""

import csv

import pytest

from integration.dishes import DishRow, DishTable, Match, TABLE_FIELDS
from integration.stages import meal
from integration.stages.meal import (
    MARGIN_MIN, SIM_MIN, GeminiKeyError, MealError, VisionUnavailable, identify_photo,
    load_gemini_key, resolve_meal,
)


def row(id, name, *, fibre=1.0, serving=100.0, carbs=10.0, protein=5.0, fat=2.0, kcal=100.0):
    return DishRow(id, name, "dish", kcal, carbs, protein, fat, fibre, serving,
                   "assumed" if serving else None, "test", id, "exact", ())


A, B, C = row("A", "Aa"), row("B", "Bb"), row("C", "Cc")


class StubTable:
    """DishTable stand-in that returns scripted matches (exact similarity control)."""
    build_hash = "stub"

    def __init__(self, scripted: dict[str, Match | None], rows=(A, B, C)):
        self.scripted, self.rows = scripted, {r.id: r for r in rows}

    def match(self, name):
        return self.scripted[name]

    def get(self, row_id):
        return self.rows[row_id]


class FakeNamer:
    model_version = "fake-vlm"

    def __init__(self, names):
        self.names, self.calls = names, 0

    def name_dishes(self, image, mime):
        self.calls += 1
        return list(self.names)


def m(top, sim, margin, *others):
    return Match(top, sim, margin, (top, *others))


def run(scripted, names=None):
    names = names or list(scripted)
    return identify_photo(b"img", "image/jpeg", FakeNamer(names), StubTable(scripted))


def test_rule_constants_are_the_adr_0006_values():
    assert (SIM_MIN, MARGIN_MIN) == (0.85, 0.0)


@pytest.mark.parametrize("sim,margin,accepted", [
    (0.85, 0.0, True),         # both bounds inclusive
    (0.8499, 0.20, False),     # similarity just under
    (1.0, 0.0, True),          # a tie at the top is accepted: dev chose margin 0
    (0.60, 0.40, False),
])
def test_accept_abstain_boundaries(sim, margin, accepted):
    [item] = run({"x": m(A, sim, margin, B, C)})
    assert item.accepted is accepted
    assert (item.row is A) is accepted
    assert item.similarity == sim and item.margin == margin
    assert item.suggestions == (A, B, C)       # suggestions are kept either way


def test_mixed_plate_one_uncertain_item():
    items = run({
        "nasi": m(A, 0.93, 0.30, B, C),
        "lauk aneh": m(B, 0.71, 0.02, C, A),
        "sambal": m(C, 0.88, 0.12, A, B),
    })
    assert [i.accepted for i in items] == [True, False, True]
    assert [i.vlm_name for i in items] == ["nasi", "lauk aneh", "sambal"]
    assert items[1].row is None and items[1].suggestions[0] is B


def test_duplicate_names_and_duplicate_accepted_rows_collapse():
    items = identify_photo(
        b"i", "image/jpeg", FakeNamer(["Nasi", "nasi ", "nasi putih", "", "sup"]),
        StubTable({
            "Nasi": m(A, 0.9, 0.3, B, C), "nasi putih": m(A, 0.85, 0.2, B, C),
            "sup": m(B, 0.6, 0.01, A, C),
        }))
    assert [i.vlm_name for i in items] == ["Nasi", "sup"]    # same row accepted twice -> once


def test_no_names_gives_no_items():
    assert run({}, names=[]) == []


def test_name_with_nothing_comparable_needs_manual_entry():
    class Blank(StubTable):
        def match(self, name):
            raise ValueError("empty dish name")

    [item] = identify_photo(b"i", "image/jpeg", FakeNamer(["???"]), Blank({}))
    assert not item.accepted and item.suggestions == ()


def test_namer_failure_propagates_as_vision_unavailable():
    class Down:
        model_version = "v"

        def name_dishes(self, image, mime):
            raise VisionUnavailable("down")

    with pytest.raises(VisionUnavailable):
        identify_photo(b"i", "image/jpeg", Down(), StubTable({}))


def test_identify_photo_with_real_table_and_real_matcher(tmp_path):
    path = tmp_path / "t.csv"
    rows = [["R1", "Rendang", "dish", "190", "8", "22", "8", "", "100", "assumed", "t", "R1", "exact", "rendang padang"],
            ["R2", "Sate", "dish", "280", "12", "17", "20", "", "120", "assumed", "t", "R2", "exact", ""],
            ["R3", "Soto", "dish", "60", "1", "3", "4", "", "250", "assumed", "t", "R3", "exact", ""]]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TABLE_FIELDS)
        w.writerows(rows)

    items = identify_photo(b"i", "image/jpeg", FakeNamer(["Rendang Padang", "bakso"]),
                           DishTable.load(path))
    assert items[0].accepted and items[0].row.id == "R1"
    assert not items[1].accepted and len(items[1].suggestions) == 3


# --- resolve_meal ----------------------------------------------------------------------

T = StubTable({}, rows=(
    row("A", "Aa", carbs=20.0, protein=4.0, fat=1.0, kcal=110.0, fibre=2.0, serving=100.0),
    row("B", "Bb", carbs=5.0, protein=10.0, fat=8.0, kcal=140.0, fibre=1.0, serving=200.0),
    row("N", "NoFibre", carbs=10.0, protein=1.0, fat=1.0, kcal=50.0, fibre=None, serving=150.0),
    row("S", "NoServing", carbs=10.0, protein=1.0, fat=1.0, kcal=50.0, serving=None),
))


def resolved(items, **kw):
    return resolve_meal(items, T, vlm_model_version=kw.get("v"))


@pytest.mark.parametrize("portion,factor", [(0, 0.5), (1, 1.0), (2, 1.5)])
def test_portion_scales_the_serving(portion, factor):
    meal_ = resolved([{"row_id": "B", "portion": portion, "source": "manual"}])
    assert meal_.items[0].grams == pytest.approx(200.0 * factor)


def test_explicit_grams_win_over_portion():
    meal_ = resolved([{"row_id": "B", "grams": 50, "portion": 2, "source": "manual"}])
    assert meal_.items[0].grams == 50.0


def test_totals_are_sums_over_items_and_carry_provenance():
    meal_ = resolved([
        {"row_id": "A", "grams": 150, "source": "photo", "similarity": 0.91, "margin": 0.12},
        {"row_id": "B", "portion": 1, "source": "manual"},
    ], v="fake-vlm")
    assert meal_.carbs_g == pytest.approx(20 * 1.5 + 5 * 2.0)
    assert meal_.protein_g == pytest.approx(4 * 1.5 + 10 * 2.0)
    assert meal_.fat_g == pytest.approx(1 * 1.5 + 8 * 2.0)
    assert meal_.kcal == pytest.approx(110 * 1.5 + 140 * 2.0)
    assert meal_.fibre_g == pytest.approx(2 * 1.5 + 1 * 2.0)
    assert (meal_.dish_table_build, meal_.vlm_model_version) == ("stub", "fake-vlm")
    first = meal_.items[0]
    assert (first.source, first.similarity, first.margin) == ("photo", 0.91, 0.12)
    assert first.macros()["carbs_g"] == pytest.approx(30.0)


def test_fibre_is_none_when_any_item_fibre_is_unknown_never_zero():
    meal_ = resolved([{"row_id": "A", "grams": 100, "source": "manual"},
                      {"row_id": "N", "grams": 100, "source": "manual"}])
    assert meal_.fibre_g is None
    assert meal_.carbs_g == pytest.approx(30.0)          # the other macros still sum
    assert meal_.items[1].macros()["fibre_g"] is None


@pytest.mark.parametrize("bad,msg", [
    ([], "minimal satu"),
    ([{"row_id": "ZZZ", "grams": 10, "source": "manual"}], "tidak ada"),
    ([{"row_id": 7, "grams": 10, "source": "manual"}], "tidak ada"),
    ([{"grams": 10, "source": "manual"}], "tidak ada"),
    ([{"row_id": "A", "source": "manual"}], "berat"),
    ([{"row_id": "A", "grams": 0, "source": "manual"}], "gram"),
    ([{"row_id": "A", "grams": -5, "source": "manual"}], "gram"),
    ([{"row_id": "A", "grams": float("nan"), "source": "manual"}], "gram"),
    ([{"row_id": "A", "grams": 99999, "source": "manual"}], "gram"),
    ([{"row_id": "A", "grams": True, "source": "manual"}], "gram"),
    ([{"row_id": "A", "portion": 3, "source": "manual"}], "porsi"),
    ([{"row_id": "S", "portion": 1, "source": "manual"}], "takaran"),
    ([{"row_id": "A", "grams": 10, "source": "web"}], "sumber"),
])
def test_invalid_input_raises_meal_error(bad, msg):
    with pytest.raises(MealError, match=f"(?i){msg}"):
        resolved(bad)


def test_serving_less_row_is_fine_with_explicit_grams():
    assert resolved([{"row_id": "S", "grams": 30, "source": "manual"}]).items[0].grams == 30.0


# --- key file and the vision namer (HTTP seam) ------------------------------------------

def test_load_gemini_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    f = tmp_path / "k.env"
    f.write_text("OTHER=1\nGEMINI_API_KEY='abc123'\n")
    assert load_gemini_key(f) == "abc123"
    with pytest.raises(GeminiKeyError):
        load_gemini_key(tmp_path / "absent.env")
    f.write_text("OTHER=1\n")
    with pytest.raises(GeminiKeyError):
        load_gemini_key(f)
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")  # Heroku config var: no file needed
    assert load_gemini_key(tmp_path / "absent.env") == "from-env"


def _jpeg() -> bytes:
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


def test_gemini_vision_namer_parses_names_and_ignores_confidence(monkeypatch):
    sent = {}

    class Resp:
        ok, status_code, text = True, 200, ""

        def json(self):
            import json
            payload = {"items": [{"food_name": " Rendang ", "confidence": 0.99},
                                 {"food_name": "", "confidence": 0.5},
                                 {"food_name": "Nasi putih", "confidence": 0.1}]}
            return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}

    def fake_post(url, params, json, timeout):
        sent.update(url=url, body=json)
        return Resp()

    monkeypatch.setattr(meal.requests, "post", fake_post)
    namer = meal.GeminiVisionNamer("k", "gemini-3.5-flash-lite")
    assert namer.model_version == "gemini-3.5-flash-lite"
    assert namer.name_dishes(_jpeg(), "image/png") == ["Rendang", "Nasi putih"]
    assert sent["url"].endswith("gemini-3.5-flash-lite:generateContent")
    assert sent["body"]["contents"][0]["parts"][0]["text"] == meal.VLM_PROMPT
    assert sent["body"]["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "image/jpeg"


def test_gemini_vision_namer_errors(monkeypatch):
    class Bad:
        ok, status_code, text = False, 403, "no"

    monkeypatch.setattr(meal.requests, "post", lambda *a, **k: Bad())
    namer = meal.GeminiVisionNamer("k", "m")
    with pytest.raises(VisionUnavailable):
        namer.name_dishes(_jpeg(), "image/png")
    with pytest.raises(MealError):
        namer.name_dishes(b"not an image", "image/jpeg")

    class Garbage:
        ok, status_code, text = True, 200, ""

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "{not json"}]}}]}

    monkeypatch.setattr(meal.requests, "post", lambda *a, **k: Garbage())
    with pytest.raises(VisionUnavailable):
        namer.name_dishes(_jpeg(), "image/png")
