"""DishTable, the embedding matcher and the TKPI layout parser.

Every composition value below is invented: TKPI values are never committed.
"""

import csv
import hashlib

import numpy as np
import pytest

from integration import dishes
from integration.dishes import DishTable, GeminiEmbedder, GeminiUnavailable, tkpi

FIELDS = dishes.TABLE_FIELDS
ROWS = [
    # id, name, kind, kcal, carbs, protein, fat, fibre, serving_g, serving_source, source, ref, conf, aliases
    ["T001", "Nasi contoh", "ingredient", "100", "20", "2", "0.5", "0.4", "100", "contoh", "test", "T001", "exact", "nasi|rice"],
    ["T002", "Rendang contoh", "dish", "200", "5", "20", "10", "", "100", "assumed", "test", "T002", "approximate", "rendang"],
    ["T003", "Sate contoh", "dish", "250", "8", "18", "15", "", "", "", "computed", "recipe_x", "computed", "sate ayam|sate_madura"],
    ["T004", "Soto contoh", "dish", "60", "3", "4", "2", "0.2", "250", "assumed", "test", "T004", "exact", ""],
]

# Orthogonal-ish 4-d vectors per candidate text.
VECS = {
    "nasi contoh": [1, 0, 0, 0], "nasi": [1, 0.1, 0, 0], "rice": [1, 0, 0.1, 0],
    "rendang contoh": [0, 1, 0, 0], "rendang": [0, 1, 0.05, 0],
    "sate contoh": [0, 0, 1, 0], "sate ayam": [0, 0, 1, 0.1], "sate madura": [0, 0, 1, 0.2],
    "soto contoh": [0, 0, 0, 1],
    # queries
    "nasi putih": [1, 0.2, 0, 0], "rendang sapi": [0, 1, 0.3, 0], "sate": [0, 0.3, 1, 0.2],
}


class FakeEmbedder:
    model = "fake-embed"

    def __init__(self):
        self.calls: list[tuple[int, bool]] = []

    def embed(self, texts, *, query=False):
        self.calls.append((len(texts), query))
        v = np.asarray([VECS[t] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


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
    m = table.match("nasi putih", FakeEmbedder())
    assert m.row.id == "T001"
    assert 0.9 < m.similarity <= 1.0
    assert m.margin == pytest.approx(m.similarity - _second(table, "nasi putih"))
    assert m.suggestions[0].id == "T001" and len(m.suggestions) == 3
    assert len({r.id for r in m.suggestions}) == 3


def _second(table, name):
    """Independent top-2 similarity: max cosine over candidates, per row."""
    fe = FakeEmbedder()
    q = fe.embed([name], query=True)[0]
    scores = []
    for cands in table._candidates:
        scores.append(max(float(fe.embed([c])[0] @ q) for c in cands))
    return sorted(scores, reverse=True)[1]


def test_match_uses_best_alias_not_just_the_name(table):
    # "sate" is closest to the alias "sate ayam"/"sate madura", not the canonical name.
    m = table.match("sate", FakeEmbedder())
    assert m.row.id == "T003"


def test_query_and_documents_use_the_asymmetric_sides(table):
    fe = FakeEmbedder()
    table.match("rendang sapi", fe)
    table.match("nasi putih", fe)
    doc_calls = [c for c in fe.calls if not c[1]]
    query_calls = [c for c in fe.calls if c[1]]
    assert len(doc_calls) == 1 and doc_calls[0][0] == 9   # every candidate string, once
    assert query_calls == [(1, True), (1, True)]


def test_match_on_empty_table_is_none_and_blank_name_is_an_error(tmp_path):
    empty = write_table(tmp_path / "empty.csv", [])
    assert DishTable.load(empty).match("nasi", FakeEmbedder()) is None
    with pytest.raises(ValueError):
        DishTable.load(write_table(tmp_path / "t.csv")).match("  ", FakeEmbedder())


def test_embedding_cache_is_reused_across_loads_and_keyed_by_build(table_path):
    DishTable.load(table_path).match("nasi putih", (first := FakeEmbedder()))
    assert any(not q for _, q in first.calls)
    assert table_path.with_suffix(".embeddings.npz").exists()

    second = FakeEmbedder()
    DishTable.load(table_path).match("nasi putih", second)
    assert second.calls == [(1, True)]                      # no document re-embedding

    write_table(table_path, ROWS[:3])                       # new build -> new hash -> re-embed
    third = FakeEmbedder()
    DishTable.load(table_path).match("nasi putih", third)
    assert any(not q for _, q in third.calls)


def test_gemini_embedder_batches_normalises_and_maps_failures(monkeypatch):
    posted = []

    class Resp:
        def __init__(self, ok, status, payload):
            self.ok, self.status_code, self._p, self.text = ok, status, payload, "x"

        def json(self):
            return self._p

    def fake_post(url, params, json, timeout):
        posted.append((url, json["requests"][0]["taskType"], len(json["requests"])))
        return Resp(True, 200, {"embeddings": [{"values": [3.0, 4.0]} for _ in json["requests"]]})

    monkeypatch.setattr(dishes.requests, "post", fake_post)
    e = GeminiEmbedder("k", "gemini-embedding-001")
    out = e.embed([f"t{i}" for i in range(dishes.EMBED_BATCH + 5)], query=True)
    assert out.shape == (dishes.EMBED_BATCH + 5, 2)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert [p[2] for p in posted] == [dishes.EMBED_BATCH, 5]
    assert {p[1] for p in posted} == {"RETRIEVAL_QUERY"}
    assert posted[0][0].endswith("gemini-embedding-001:batchEmbedContents")

    monkeypatch.setattr(dishes.requests, "post", lambda *a, **k: Resp(False, 403, {}))
    with pytest.raises(GeminiUnavailable):
        e.embed(["x"])


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


def test_strings_shared_by_rows_are_embedded_once(tmp_path):
    rows = [ROWS[0], [*ROWS[1][:-1], "nasi"]]          # T002 also carries the alias "nasi"
    t = DishTable.load(write_table(tmp_path / "t.csv", rows))
    fe = FakeEmbedder()
    t.warm(fe)
    # nasi contoh, nasi, rice (T001) + rendang contoh (T002); the shared "nasi" is not repeated
    assert fe.calls == [(4, False)]


def test_gemini_embedder_fails_fast_on_daily_quota(monkeypatch):
    slept = []

    class Resp:
        ok, status_code = False, 429
        text = "Embed content requests per day limit exceeded on Free Tier"

    monkeypatch.setattr(dishes.requests, "post", lambda *a, **k: Resp())
    monkeypatch.setattr(dishes.time, "sleep", slept.append)
    with pytest.raises(GeminiUnavailable, match="daily quota"):
        GeminiEmbedder("k", "m").embed(["x"])
    assert slept == []
