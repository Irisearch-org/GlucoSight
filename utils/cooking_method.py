"""Cooking-method and portion-size text matching — rule-based, no training.

Moved from `nlp/data/derive_labels.py` (code review, 2026-10-19): it was
`forecasting/`'s only cross-track import, and it reached into another
track's underscore-prefixed (private) helper. The matcher itself isn't
NLP-specific business logic — it's a keyword/pattern utility, the same
shape as `utils/clarke_grid.py` — so it lives here, public, and any track
can call it directly instead of reaching into another track's internals.

`nlp/data/derive_labels.py` re-exports these names unchanged so existing
NLP-track imports don't need to change.
"""

from __future__ import annotations

import re

FRIED_EN = ("fried", "crispy", "deep-fried")
FRIED_ID = ("digoreng", "goreng", "menggoreng", "gorengan")
FRIED_CN = ("炒", "煎", "炸")
FRIED_NEGATIONS = ("tidak digoreng", "bukan digoreng", "tidak goreng", "bukan goreng", "not fried", "not deep-fried")
LARGE_PORTION_PATTERNS = (
    re.compile(r"\b(?:jumbo|large|big)\b", re.I),
    re.compile(r"\b(?:porsi|portion)\s+(?:besar|large|jumbo|extra besar)\b", re.I),
    re.compile(r"\b(?:besar|large|jumbo)\s+(?:porsi|portion)\b", re.I),
)
SMALL_PORTION_PATTERNS = (
    re.compile(r"\b(?:porsi|portion)\s+(?:kecil|small)\b", re.I),
    re.compile(r"\b(?:kecil|small)\s+(?:porsi|portion)\b", re.I),
)
GRAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:g|ml)\b", re.I)
LARGE_THRESHOLD_G = 400


def total_grams(text: str) -> float:
    return sum(float(m.group(1)) for m in GRAM_RE.finditer(text or ""))


def is_fried(en: str, cn: str = "") -> int:
    """Rule-based: does the text mention a frying verb? EN/ID/CN keywords,
    with negation handling ("tidak digoreng" -> 0, not 1)."""
    combined = f"{en or ''} {cn or ''}".casefold()
    if any(negation in combined for negation in FRIED_NEGATIONS):
        return 0
    return int(
        any(term in combined for term in (*FRIED_EN, *FRIED_ID, *FRIED_CN))
    )


def is_large(en: str, cn: str = "") -> int:
    """Rule-based: explicit large/small portion wording first, then a
    400g+ total as a fallback signal."""
    combined = f"{en or ''} {cn or ''}"
    if any(pattern.search(combined) for pattern in LARGE_PORTION_PATTERNS):
        return 1
    if any(pattern.search(combined) for pattern in SMALL_PORTION_PATTERNS):
        return 0
    total = total_grams(en) or total_grams(cn)
    return int(total > LARGE_THRESHOLD_G)
