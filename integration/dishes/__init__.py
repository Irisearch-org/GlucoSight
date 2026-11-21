"""The dish table and the embedding matcher (GLOSSARY.md: dish table, matcher).

``DishTable`` is the one place a food's composition lives. It is built locally from
TKPI 2017 by ``python -m integration.dishes.build`` into a gitignored CSV (the repo is
public and TKPI has no open licence), then loaded here.

Matching reproduces the dish-ID trial (``cv/experiments/dish_id_trial``): the VLM's
dish name is embedded as a RETRIEVAL_QUERY, every row's name and aliases as
RETRIEVAL_DOCUMENT, both L2-normalised; a row's score is the maximum cosine over its
candidate strings; the top-1/top-2 gap over distinct rows is the margin. The accept
rule (similarity and margin thresholds) is NOT here: it lives once, in
``integration.stages.meal``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import requests

SUGGESTION_COUNT = 3
EMBED_BATCH = 50
EMBED_TIMEOUT_S = 60
EMBED_MAX_RETRIES = 8
EMBED_QUOTA_WAIT_S = 61  # rejected requests still count toward the per-minute quota: wait out the window
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
TABLE_FIELDS = (
    "id", "name", "kind", "energy_kcal", "carbs_g", "protein_g", "fat_g", "fibre_g",
    "serving_g", "serving_source", "source", "source_ref", "confidence", "aliases",
)


class GeminiUnavailable(RuntimeError):
    """The Gemini API failed (network, quota, rejected request)."""


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


class Embedder(Protocol):
    """Seam: GeminiEmbedder (real) and a deterministic fake in tests."""

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        """L2-normalised rows. ``query`` selects the query side of the asymmetric model."""
        ...


def normalise_text(s: str) -> str:
    return s.strip().lower().replace("_", " ")


def _l2(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.where(norms == 0.0, 1.0, norms)


class GeminiEmbedder:
    """``gemini-embedding-001`` over the REST batchEmbedContents endpoint."""

    def __init__(self, api_key: str, model: str):
        self.model = model
        self._api_key = api_key

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        task = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            body = {"requests": [
                {"model": f"models/{self.model}", "taskType": task,
                 "content": {"parts": [{"text": t}]}} for t in batch]}
            out.extend(e["values"] for e in self._post(body)["embeddings"])
        return _l2(np.asarray(out, dtype=np.float32))

    def _post(self, body: dict) -> dict:
        url = f"{GEMINI_BASE_URL}{self.model}:batchEmbedContents"
        for attempt in range(EMBED_MAX_RETRIES + 1):
            try:
                resp = requests.post(url, params={"key": self._api_key}, json=body,
                                     timeout=EMBED_TIMEOUT_S)
            except requests.RequestException as exc:
                raise GeminiUnavailable(f"embedding request failed: {type(exc).__name__}") from exc
            if resp.ok:
                return resp.json()
            if resp.status_code not in (429, 500, 503) or attempt == EMBED_MAX_RETRIES:
                raise GeminiUnavailable(f"embedding HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code == 429 and "per day" in resp.text:
                raise GeminiUnavailable(f"embedding daily quota exhausted: {resp.text[:300]}")
            time.sleep(EMBED_QUOTA_WAIT_S if resp.status_code == 429 else 2 ** attempt * 2)
        raise AssertionError("unreachable")


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
    """Immutable view of one built dish-table CSV plus its cached embeddings."""

    def __init__(self, rows: tuple[DishRow, ...], build_hash: str, cache_path: Path | None):
        self.rows = rows
        self.build_hash = build_hash
        self._cache_path = cache_path
        self._by_id = {r.id: r for r in rows}
        self._candidates = [list(dict.fromkeys(normalise_text(s) for s in (r.name, *r.aliases)))
                            for r in rows]
        self._lock = threading.Lock()
        self._doc_vectors: np.ndarray | None = None
        self._row_starts: np.ndarray | None = None

    @classmethod
    def load(cls, path: Path) -> "DishTable":
        path = Path(path)
        data = path.read_bytes()
        build_hash = hashlib.sha256(data).hexdigest()[:12]
        return cls(_read_rows(path), build_hash, path.with_suffix(".embeddings.npz"))

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

    def warm(self, embedder: Embedder) -> None:
        """Embed (or load from cache) every row's candidate strings now, not on first match."""
        if self.rows:
            self._document_matrix(embedder)

    def match(self, name: str, embedder: Embedder) -> Match | None:
        if not self.rows:
            return None
        query = normalise_text(name)
        if not query:
            raise ValueError("empty dish name")
        docs, starts = self._document_matrix(embedder)
        q = _l2(np.asarray(embedder.embed([query], query=True), dtype=np.float32))[0]
        row_scores = np.maximum.reduceat(docs @ q, starts)
        order = np.argsort(-row_scores, kind="stable")
        top = order[:SUGGESTION_COUNT]
        top1 = float(row_scores[order[0]])
        top2 = float(row_scores[order[1]]) if len(order) > 1 else 0.0
        return Match(
            row=self.rows[order[0]], similarity=top1, margin=top1 - top2,
            suggestions=tuple(self.rows[i] for i in top),
        )

    def _document_matrix(self, embedder: Embedder) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if self._doc_vectors is None:
                texts = [t for cands in self._candidates for t in cands]
                unique = list(dict.fromkeys(texts))  # a string shared by rows is embedded once
                model = getattr(embedder, "model", None)
                vectors = self._read_cache(unique, model)
                if vectors is None:
                    vectors = _l2(np.asarray(embedder.embed(unique, query=False), dtype=np.float32))
                    if vectors.shape[0] != len(unique):
                        raise ValueError("embedder returned a wrong number of vectors")
                    self._write_cache(unique, model, vectors)
                position = {t: i for i, t in enumerate(unique)}
                vectors = vectors[[position[t] for t in texts]]
                lengths = [len(c) for c in self._candidates]
                self._row_starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.intp)
                self._doc_vectors = vectors
            return self._doc_vectors, self._row_starts

    def _meta(self, texts: list[str], model: str | None) -> str:
        return json.dumps({"build_hash": self.build_hash, "model": model, "n": len(texts)})

    def _read_cache(self, texts: list[str], model: str | None) -> np.ndarray | None:
        if self._cache_path is None or not self._cache_path.exists():
            return None
        with np.load(self._cache_path, allow_pickle=False) as z:
            if str(z["meta"]) != self._meta(texts, model):
                return None
            return z["vectors"]

    def _write_cache(self, texts: list[str], model: str | None, vectors: np.ndarray) -> None:
        if self._cache_path is None:
            return
        tmp = self._cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, meta=np.array(self._meta(texts, model)), vectors=vectors)
        tmp.replace(self._cache_path)
