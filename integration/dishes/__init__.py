"""The dish table and the character n-gram matcher (GLOSSARY.md: dish table, matcher).

``DishTable`` is the one place a food's composition lives. It is built locally from
TKPI 2017 by ``python -m integration.dishes.build`` into a gitignored CSV (the repo is
public and TKPI has no open licence), then loaded here.

Matching is local and needs no API (ADR 0006): every row's name and aliases, and the VLM's
dish name, become hashed character 2-4-gram count vectors (``char_wb``, accents and
punctuation folded, L2-normalised); a row's score is the maximum cosine over its candidate
strings; the top-1/top-2 gap over distinct rows is the margin. Hashing keeps a score a
function of the two strings alone, so it does not shift when rows are added. The accept rule
(similarity and margin thresholds) is NOT here: it lives once, in ``integration.stages.meal``.
"""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

SUGGESTION_COUNT = 3
MATCHER_VERSION = "charwb-2-4-hashed-cosine-v1"
_VECTORIZER = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2**20,
                                alternate_sign=False, norm="l2")
TABLE_FIELDS = (
    "id", "name", "kind", "energy_kcal", "carbs_g", "protein_g", "fat_g", "fibre_g",
    "serving_g", "serving_source", "source", "source_ref", "confidence", "aliases",
)


@dataclass(frozen=True)
class DishRow:
    id: str
    name: str
    kind: Literal["dish", "ingredient"]
    energy_kcal: float
    carbs_g: float
    protein_g: float
    fat_g: float
    fibre_g: float | None  # per 100 g edible portion; None = unknown (missing is not zero)
    serving_g: float | None
    serving_source: str | None
    source: str
    source_ref: str | None
    confidence: Literal["exact", "approximate", "computed"]
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Match:
    row: DishRow
    similarity: float
    margin: float  # top-1 similarity minus top-2 (distinct rows)
    suggestions: tuple[DishRow, ...]  # top-3 rows, row first


def normalise_text(s: str) -> str:
    return s.strip().lower().replace("_", " ")


def _match_text(s: str) -> str:
    """What the matcher compares: ASCII-folded, lower case, punctuation and underscores as spaces."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def _opt_float(s: str) -> float | None:
    return float(s) if s.strip() else None


def _read_rows(path: Path) -> tuple[DishRow, ...]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(TABLE_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = [DishRow(
            id=r["id"], name=r["name"], kind=r["kind"],
            energy_kcal=float(r["energy_kcal"]), carbs_g=float(r["carbs_g"]),
            protein_g=float(r["protein_g"]), fat_g=float(r["fat_g"]),
            fibre_g=_opt_float(r["fibre_g"]), serving_g=_opt_float(r["serving_g"]),
            serving_source=r["serving_source"] or None, source=r["source"],
            source_ref=r["source_ref"] or None, confidence=r["confidence"],
            aliases=tuple(a for a in r["aliases"].split("|") if a),
        ) for r in reader]
    ids = [r.id for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate row ids")
    return tuple(rows)


class DishTable:
    """Immutable view of one built dish-table CSV plus its matcher index."""

    def __init__(self, rows: tuple[DishRow, ...], build_hash: str):
        self.rows = rows
        self.build_hash = build_hash
        self._by_id = {r.id: r for r in rows}
        self._candidates = [list(dict.fromkeys(normalise_text(s) for s in (r.name, *r.aliases)))
                            for r in rows]
        texts = [_match_text(t) for cands in self._candidates for t in cands]
        self._doc_vectors = _VECTORIZER.transform(texts) if texts else None  # sparse, one per candidate
        lengths = [len(c) for c in self._candidates]
        self._row_starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.intp)

    @classmethod
    def load(cls, path: Path) -> "DishTable":
        path = Path(path)
        data = path.read_bytes()
        build_hash = hashlib.sha256(data).hexdigest()[:12]
        return cls(_read_rows(path), build_hash)

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, row_id: str) -> DishRow:
        return self._by_id[row_id]

    def search(self, q: str, limit: int = 10) -> list[DishRow]:
        """Autocomplete: case-insensitive; exact, then prefix, then word-prefix, then substring."""
        needle = normalise_text(q)
        if not needle or limit <= 0:
            return []
        scored: list[tuple[int, int, DishRow]] = []
        for i, (row, cands) in enumerate(zip(self.rows, self._candidates)):
            best = None
            for rank_base, cand in enumerate(cands):
                # the canonical name (first candidate) outranks aliases at equal strength
                tie = 0 if rank_base == 0 else 1
                if cand == needle:
                    rank = 0
                elif cand.startswith(needle):
                    rank = 1
                elif any(w.startswith(needle) for w in cand.replace(",", " ").split()):
                    rank = 2
                elif needle in cand:
                    rank = 3
                else:
                    continue
                key = rank * 2 + tie
                best = key if best is None else min(best, key)
            if best is not None:
                scored.append((best, i, row))
        scored.sort(key=lambda t: (t[0], len(t[2].name), t[1]))
        return [row for _, _, row in scored[:limit]]

    def match(self, name: str) -> Match | None:
        if not self.rows:
            return None
        query = _match_text(name)
        if not query:
            raise ValueError("empty dish name")
        q = _VECTORIZER.transform([query])
        scores = (self._doc_vectors @ q.T).toarray().ravel()
        row_scores = np.maximum.reduceat(scores, self._row_starts)
        order = np.argsort(-row_scores, kind="stable")
        top = order[:SUGGESTION_COUNT]
        top1 = float(row_scores[order[0]])
        top2 = float(row_scores[order[1]]) if len(order) > 1 else 0.0
        return Match(
            row=self.rows[order[0]], similarity=top1, margin=top1 - top2,
            suggestions=tuple(self.rows[i] for i in top),
        )
