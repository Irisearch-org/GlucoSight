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
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

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

    ponytail: plain difflib ratio, no embedding model. At this table's
    current size (10 rows, short Indonesian names) a learned embedding
    buys nothing a string ratio doesn't already give. Upgrade to the
    retrieval approach nlp/docs/LITERATURE_REVIEW.md names once the table
    grows enough that real near-misses stop sharing substrings with their
    row (synonyms, not typos/suffixes).
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
