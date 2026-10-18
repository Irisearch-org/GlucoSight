"""Parsing the dietary record into (dish, grams) items.

A ShanghaiT2DM meal is a multi-line string:

    "Rice 150 g | Vegetable 100 g | Sliced lotus root 50 g"

Every consumer that wants dish names or weights goes through here, so the item
grammar lives in one place instead of being copied per script.

One deliberate limitation: **an item is only parsed when it carries a weight.**
``"Soup"`` with no grams is skipped rather than guessed at. The Day 1 gate
reports its counts under that rule (`docs/NLP_PATH_A_GATE.md`, N1) and the
dish-identity test reports its counts under the same rule, so the two numbers
are comparable. That is why the gate counts 1,825 distinct dish names while a
looser parser counts 1,838.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

#: ``"Rice 150 g"`` / ``"Milk 250 ml"`` -> ``("Rice", 150.0)``. Anchored at the
#: end so an item name containing a digit does not split the wrong way.
ITEM_RE = re.compile(r"^(?P<name>.*?)\s*(?P<grams>\d+(?:\.\d+)?)\s*(?:g|ml)\s*$", re.I)

#: The record's item separator.
SEPARATOR = "|"


def normalise_name(name: str) -> str:
    """Collapse whitespace and case so ``"Rice "`` and ``"rice"`` are one dish."""
    return re.sub(r"\s+", " ", str(name).strip().lower())


def meal_items(text: str) -> List[Tuple[str, float]]:
    """``(dish name, grams)`` for every weighted item in one meal's record."""
    items: List[Tuple[str, float]] = []
    for segment in str(text or "").split(SEPARATOR):
        match = ITEM_RE.match(segment.strip())
        if not match:
            continue
        grams = float(match.group("grams"))
        name = normalise_name(match.group("name"))
        if name and grams > 0:
            items.append((name, grams))
    return items


def parse_items(rows: Iterable[dict]) -> List[Tuple[str, str, float]]:
    """``(participant_id, dish name, grams)`` for every parsed item."""
    return [
        (row["participant_id"], name, grams)
        for row in rows
        for name, grams in meal_items(row.get("dietary_intake", ""))
    ]


def dish_names(rows: Iterable[dict]) -> List[str]:
    """Every dish name, with repeats."""
    return [name for _, name, _ in parse_items(rows)]
