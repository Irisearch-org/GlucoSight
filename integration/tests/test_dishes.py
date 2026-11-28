"""DishTable, the character n-gram matcher and the TKPI layout parser.

Every composition value below is invented: TKPI values are never committed.
"""

import csv
import hashlib

import pytest

from integration import dishes
from integration.dishes import DishTable, tkpi

FIELDS = dishes.TABLE_FIELDS
ROWS = [
    # id, name, kind, kcal, carbs, protein, fat, fibre, serving_g, serving_source, source, ref, conf, aliases
    ["T001", "Nasi contoh", "ingredient", "100", "20", "2", "0.5", "0.4", "100", "contoh", "test", "T001", "exact", "nasi|rice"],
    ["T002", "Rendang contoh", "dish", "200", "5", "20", "10", "", "100", "assumed", "test", "T002", "approximate", "rendang"],
    ["T003", "Sate contoh", "dish", "250", "8", "18", "15", "", "", "", "computed", "recipe_x", "computed", "sate ayam|sate_madura"],
    ["T004", "Soto contoh", "dish", "60", "3", "4", "2", "0.2", "250", "assumed", "test", "T004", "exact", ""],
]


def write_table(path, rows=ROWS):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    return path


@pytest.fixture
def table_path(tmp_path):
    return write_table(tmp_path / "dish_table.csv")


@pytest.fixture
def table(table_path):
    return DishTable.load(table_path)


def test_load_parses_nullable_fields(table):
    nasi = table.get("T001")
    assert (nasi.kind, nasi.fibre_g, nasi.serving_g) == ("ingredient", 0.4, 100.0)
    assert nasi.aliases == ("nasi", "rice")
    rendang = table.get("T002")
    assert rendang.fibre_g is None and rendang.serving_g == 100.0
    sate = table.get("T003")
    assert sate.serving_g is None and sate.serving_source is None
    assert table.get("T004").aliases == ()


def test_get_unknown_id_raises_key_error(table):
    with pytest.raises(KeyError):
        table.get("NOPE")


def test_build_hash_is_sha256_prefix_of_bytes(table_path, tmp_path):
    t = DishTable.load(table_path)
    assert t.build_hash == hashlib.sha256(table_path.read_bytes()).hexdigest()[:12]
    other = write_table(tmp_path / "other.csv", ROWS[:2])
    assert DishTable.load(other).build_hash != t.build_hash


def test_load_rejects_missing_columns_and_duplicate_ids(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,name\nA,B\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        DishTable.load(bad)
    dup = write_table(tmp_path / "dup.csv", [ROWS[0], ROWS[0]])
    with pytest.raises(ValueError, match="duplicate"):
        DishTable.load(dup)


def test_search_prefix_alias_case_insensitive(table):
    assert [r.id for r in table.search("NAS")] == ["T001"]
    assert [r.id for r in table.search("rice")] == ["T001"]            # alias
    assert [r.id for r in table.search("sate_madura")] == ["T003"]      # underscore normalised
    assert [r.id for r in table.search("rendang")] == ["T002"]


def test_search_ranks_exact_then_prefix_then_word_then_substring(table):
    ids = [r.id for r in table.search("contoh")]
    assert set(ids) == {"T001", "T002", "T003", "T004"}                 # word match in every name
    assert [r.id for r in table.search("ndang")] == ["T002"]            # substring only


def test_search_limit_and_empty_query(table):
    assert len(table.search("contoh", limit=2)) == 2
    assert table.search("   ") == []
    assert table.search("zzz") == []


def test_match_returns_best_row_margin_and_suggestions(table):
    m = table.match("nasi putih")
    assert m.row.id == "T001"
    assert 0.5 < m.similarity < 1.0                       # shares "nasi", not "putih"
    assert m.margin == pytest.approx(m.similarity - _second(table, "nasi putih"))
    assert m.suggestions[0].id == "T001" and len(m.suggestions) == 3
    assert len({r.id for r in m.suggestions}) == 3


def _second(table, name):
    """Independent top-2 similarity: max cosine over candidates, per row."""
    scores = [max(_cos(name, c) for c in cands) for cands in table._candidates]
    return sorted(scores, reverse=True)[1]


def _cos(a, b):
    va = dishes._VECTORIZER.transform([dishes._match_text(a)])
    vb = dishes._VECTORIZER.transform([dishes._match_text(b)])
    return float((va @ vb.T).toarray()[0, 0])


def test_exact_name_or_alias_scores_one(table):
    assert table.match("Rendang Contoh").similarity == pytest.approx(1.0)
    m = table.match("rice")                                # alias of T001
    assert m.row.id == "T001" and m.similarity == pytest.approx(1.0)


def test_match_uses_best_alias_not_just_the_name(table):
    # "sate madura" equals an alias of T003 (stored as "sate_madura"), not its canonical name.
    m = table.match("Sate Madura")
    assert m.row.id == "T003" and m.similarity == pytest.approx(1.0)


def test_match_folds_accents_case_and_punctuation(table):
    assert table.match("RENDANG!!").similarity == pytest.approx(table.match("rendang").similarity)
    assert table.match("rèndang").row.id == "T002"


def test_score_depends_only_on_the_two_strings(tmp_path):
    # Hashed n-grams: adding rows never moves an existing name's score (thresholds stay valid).
    small = DishTable.load(write_table(tmp_path / "s.csv", ROWS[:2]))
    big = DishTable.load(write_table(tmp_path / "b.csv", ROWS))
    assert small.match("rendang sapi").similarity == pytest.approx(big.match("rendang sapi").similarity)


def test_match_on_empty_table_is_none_and_blank_name_is_an_error(tmp_path):
    empty = write_table(tmp_path / "empty.csv", [])
    assert DishTable.load(empty).match("nasi") is None
    with pytest.raises(ValueError):
        DishTable.load(write_table(tmp_path / "t.csv")).match("  ")


# --- TKPI layout parser (synthetic layout, invented values) -----------------------------

def _layout(*body: str) -> str:
    return "\n".join([""] * (tkpi.TABLE_START - 1) + list(body) + [""])


def test_tkpi_parser_wrapped_names_fibre_and_empty_fibre_cell():
    text = _layout(
        "AX001   Contoh satu              KZGMI-2001    10.0   100      5.0       2.0     15.0     1.2     0.6   10",
        "        Contoh",
        "AX002                            KZGMI-2001    10.0   100      5.0       2.0     15.0             0.6   10",
        "        dua, masakan",
        "AX003   Tanpa angka               KZGMI-2001    10.0   100      5.0       2.0     15.0      -      0.6   10",
        "AX004   Serat besar              KZGMI-2001    10.0   100      5.0       2.0     15.0    20.0     0.6   10",
    )
    got = {e.code: e for e in tkpi.parse_layout(text)}
    assert got["AX001"].name == "Contoh satu"
    assert got["AX001"].fibre_g == 1.2
    assert got["AX002"].name == "Contoh dua, masakan"
    assert got["AX002"].fibre_g is None          # empty cell: the Abu value is not read as fibre
    assert got["AX003"].fibre_g is None          # "-"
    assert got["AX004"].fibre_g is None          # fibre larger than carbs
    assert got["AX001"].carbs_g == 15.0 and got["AX001"].energy_kcal == 100.0


def test_tkpi_atwater_plausibility():
    ok = tkpi.TkpiEntry("AX1", "a", 100.0, 5.0, 2.0, 15.0, None)       # 4*5+4*15+9*2 = 98
    bad = tkpi.TkpiEntry("AX2", "b", 340.0, 10.0, 1.7, 6.3, None)      # carbohydrate off by 10x
    assert ok.atwater_ok() and not bad.atwater_ok()
