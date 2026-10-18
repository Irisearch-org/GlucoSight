"""The dish table and the matcher — GLOSSARY.md terms.

The dish table is the single place a dish's macro composition lives
(currently `cv/data/tkpi_macro_lookup.csv`, sourced from TKPI). The matcher
maps a free-text or VLM-generated food-name string to its nearest dish-table
row, or to no-match. Both CV (on each item a VLM lists for a mixed plate)
and NLP (on cooking-method/dish mentions in text) call the same function —
see docs/SPEC_DISH_IDENTIFICATION.md.
"""

from __future__ import annotations

import csv
import difflib
import math
import pathlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_DISH_TABLE_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "cv" / "data" / "tkpi_macro_lookup.csv"
)


@dataclass(frozen=True)
class DishRow:
    food_class: str
    tkpi_item_name: str
    serving_g: float
    carbs_g_100g: float
    protein_g_100g: float
    fat_g_100g: float
    fiber_ratio_of_carbs: float
    mapping_confidence: str


@dataclass(frozen=True)
class DishMatch:
    row: DishRow
    similarity: float


def serving_macros(row: DishRow) -> Dict[str, float]:
    """One matched row's macros for its standard serving.

    Mirrors `cv/cv_baseline/macro_lookup.py::MacroLookup.get_macros` exactly
    (same scale and fiber-from-carbs formula, same rounding) so a
    single-item mixed-plate result and the pre-existing single-dish path
    don't silently diverge on the same row.
    """
    scale = row.serving_g / 100.0
    carbs = row.carbs_g_100g * scale
    fiber = carbs * row.fiber_ratio_of_carbs
    return {
        "carbs_g": round(carbs, 1),
        "protein_g": round(row.protein_g_100g * scale, 1),
        "fat_g": round(row.fat_g_100g * scale, 1),
        "fiber_g": round(fiber, 1),
    }


def load_dish_table(path: pathlib.Path = DEFAULT_DISH_TABLE_PATH) -> List[DishRow]:
    """Read the dish table (currently `cv/data/tkpi_macro_lookup.csv`)."""
    with open(path, newline="", encoding="utf-8") as f:
        return [
            DishRow(
                food_class=r["food_class"],
                tkpi_item_name=r["tkpi_item_name"],
                serving_g=float(r["serving_g"]),
                carbs_g_100g=float(r["carbs_g_100g"]),
                protein_g_100g=float(r["protein_g_100g"]),
                fat_g_100g=float(r["fat_g_100g"]),
                fiber_ratio_of_carbs=float(r["fiber_ratio_of_carbs"]),
                mapping_confidence=r["mapping_confidence"],
            )
            for r in csv.DictReader(f)
        ]


# Below this similarity, treat it as no match rather than a guess dressed
# up as a match. An unmatched name must be visible in the data, not
# silently defaulted (docs/SPEC_DISH_IDENTIFICATION.md, Implementation
# Decisions).
MATCH_THRESHOLD = 0.6


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def match_dish(
    name: str, rows: Sequence[DishRow], threshold: float = MATCH_THRESHOLD
) -> Optional[DishMatch]:
    """Match `name` to the dish-table row it most resembles, or None.

    Similarity-based (string-similarity ratio against each row's
    `food_class` and `tkpi_item_name`), not an exact key and not a trained
    classifier — matches the shared "matcher" defined in GLOSSARY.md.

    **Correction, measured live, not a future concern:** an earlier version
    of this docstring claimed a learned embedding "buys nothing a string
    ratio doesn't already give" at this table's size. That's false, caught
    by `cv/experiments/vlm_vs_mobilenet/run_dish_list_gas_vlm.py` against a
    real photo: `difflib.SequenceMatcher("nasi gudeg", "nasi goreng")`
    scores 0.762 — higher than `SequenceMatcher("nasi gudeg", "gudeg")` at
    0.667 — because the shared "nasi " prefix dominates a whole-string
    ratio over the word that actually distinguishes the dish. No threshold
    separates every true match from every false one; checked, they overlap.
    Use `match_dish_by_embedding` (below) instead when an embedder is
    available — this function stays as the offline, dependency-free
    fallback and keeps the existing test suite fast and network-free.
    """
    query = _normalize(name)
    best: Optional[DishMatch] = None
    for row in rows:
        score = max(
            difflib.SequenceMatcher(a=query, b=_normalize(candidate)).ratio()
            for candidate in (row.food_class, row.tkpi_item_name)
        )
        if best is None or score > best.similarity:
            best = DishMatch(row=row, similarity=score)
    if best is None or best.similarity < threshold:
        return None
    return best


# A (text, is_query) -> embedding-vector function. `is_query` distinguishes
# the live item being matched (True) from a dish-table candidate being
# indexed (False) — Gemini's embedding API scores noticeably better with
# asymmetric query/document embeddings than a single symmetric call (see
# EMBEDDING_MATCH_THRESHOLD's measurement note). The real implementation (a
# Gemini API call) lives in cv/experiments/vlm_vs_mobilenet/, not here —
# this module stays free of network/credential code, same discipline as
# the rest of integration/. Dependency-injected so tests can use a fake one.
Embedder = Callable[[str, bool], Sequence[float]]

# Measured against a real Gemini embedding call (gemini-embedding-001,
# task-type-aware query/document split), 8 real items from one live VLM
# run (cv/experiments/vlm_vs_mobilenet/run_dish_list_gas_vlm.py) against
# the 10-row dish table:
#
#   true matches (want to keep):  0.669, 0.784, 0.816  (+ near-1.0 for
#                                 exact-name items, not all retested)
#   false positives (garnishes/
#   condiments, want to reject):  0.612, 0.622, 0.649, 0.679, 0.735
#
# No threshold separates these perfectly — 0.669 (a true match, "ayam
# betutu" -> bebek_betutu) sits below 0.735 (a false positive, "kacang
# goreng" -> nasi_goreng). 0.75 rejects every measured false positive at
# the cost of that one hardest true match. Chosen deliberately: a wrongly
# matched item silently pollutes the macro sum with the wrong dish's
# values; an unmatched item is just dropped (cv_items_to_contract already
# treats that as safe, not a fabricated number) — so this trades recall
# for precision on purpose. This is a measurement from one small smoke
# test (8 queries), not a validated, tuned threshold — expect to revisit
# with more data before trusting it beyond a smoke test.
EMBEDDING_MATCH_THRESHOLD = 0.75


@dataclass(frozen=True)
class DishTableIndex:
    """Precomputed embeddings for every row's candidate strings.

    Build once per process (each row-candidate costs one embedding call);
    matching against the index then costs one new call per query, not one
    per row.
    """
    rows: Sequence[DishRow]
    vectors: Dict[Tuple[int, str], Sequence[float]]


def build_dish_table_index(rows: Sequence[DishRow], embedder: Embedder) -> DishTableIndex:
    vectors: Dict[Tuple[int, str], Sequence[float]] = {}
    for i, row in enumerate(rows):
        for candidate in (row.food_class, row.tkpi_item_name):
            key = (i, _normalize(candidate))
            if key not in vectors:
                vectors[key] = embedder(_normalize(candidate), False)
    return DishTableIndex(rows=tuple(rows), vectors=vectors)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def match_dish_by_embedding(
    name: str,
    index: DishTableIndex,
    embedder: Embedder,
    threshold: float = EMBEDDING_MATCH_THRESHOLD,
) -> Optional[DishMatch]:
    """Like `match_dish`, but cosine similarity over embeddings, not
    character overlap — fixes the word-identity failure mode `match_dish`
    has (see its docstring). Still an open problem in its own right — see
    EMBEDDING_MATCH_THRESHOLD's measurement note."""
    query_vec = embedder(_normalize(name), True)
    best: Optional[DishMatch] = None
    for i, row in enumerate(index.rows):
        score = max(
            _cosine_similarity(query_vec, index.vectors[(i, _normalize(candidate))])
            for candidate in (row.food_class, row.tkpi_item_name)
        )
        if best is None or score > best.similarity:
            best = DishMatch(row=row, similarity=score)
    if best is None or best.similarity < threshold:
        return None
    return best
