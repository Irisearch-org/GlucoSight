"""Shared fixtures for the v2 integration tests.

Fakes live only at the two real seams (`Embedder`, `VisionNamer`). The dish table is a tiny
synthetic CSV (invented values, nothing from TKPI); the forecaster and PPG ridge are the real
served artifacts when present, else the tests that need them are skipped.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from integration import pipeline
from integration.dishes import TABLE_FIELDS, DishTable
from integration.errors import ModelUnavailable
from integration.pipeline import REPO_ROOT, Deps
from integration.stages.forecast import Forecaster
from integration.stages.glucose import PPGModel

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)  # 19:00 in Jakarta

TABLE_ROWS = [
    # id, name, kind, kcal, carbs, protein, fat, fibre, serving_g
    ("TST01", "Nasi putih", "ingredient", 180, 40.0, 3.0, 0.5, 0.5, 100),
    ("TST02", "Ayam goreng", "dish", 250, 5.0, 25.0, 15.0, "", 80),
    ("TST03", "Tempe goreng", "dish", 300, 12.0, 18.0, 20.0, 2.0, ""),
]


def write_table(path: Path) -> Path:
    lines = [",".join(TABLE_FIELDS)]
    for rid, name, kind, kcal, c, p, f, fibre, serving in TABLE_ROWS:
        lines.append(",".join(str(x) for x in (
            rid, name, kind, kcal, c, p, f, fibre, serving,
            "", "test", rid, "approximate", "")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class FakeEmbedder:
    """Deterministic 8-d embedder. Row names are basis vectors e0..e2; query strings below are
    chosen to land confidently on a row, ambiguously between two rows, or nowhere."""

    model = "fake-embedding"

    _KNOWN = {
        "nasi putih": np.eye(8)[0],
        "ayam goreng": np.eye(8)[1],
        "tempe goreng": np.eye(8)[2],
        "gado-gado": 0.7 * np.eye(8)[0] + 0.714 * np.eye(8)[1],  # ambiguous: sim < 0.80
    }

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        rows = [self._KNOWN.get(t, np.eye(8)[7]) for t in texts]
        a = np.asarray(rows, dtype=np.float32)
        return a / np.linalg.norm(a, axis=1, keepdims=True)


class FakeNamer:
    model_version = "fake-vlm"

    def __init__(self, names: list[str]):
        self._names = names

    def name_dishes(self, image: bytes, mime: str) -> list[str]:
        return list(self._names)


@pytest.fixture
def table(tmp_path) -> DishTable:
    return DishTable.load(write_table(tmp_path / "dish_table.csv"))


@pytest.fixture(scope="session")
def served() -> dict:
    return pipeline.load_served()


@pytest.fixture(scope="session")
def real_forecaster(served) -> Forecaster:
    try:
        return Forecaster.load(served["forecaster"], REPO_ROOT)
    except ModelUnavailable as exc:  # gitignored artifacts absent on a fresh clone
        pytest.skip(f"served forecaster artifacts not present: {exc.detail}")


@pytest.fixture(scope="session")
def real_ppg_model(served) -> PPGModel:
    spec = served["ppg_model"]
    try:
        return PPGModel.load(REPO_ROOT / spec["path"], spec["version"], spec["error_range_mgdl"])
    except ModelUnavailable as exc:
        pytest.skip(f"served PPG model not present: {exc.detail}")


@pytest.fixture
def deps(served, table, real_forecaster, real_ppg_model) -> Deps:
    return Deps(
        served=served, table=table, embedder=FakeEmbedder(),
        namer=FakeNamer(["nasi putih", "gado-gado"]),
        forecaster=real_forecaster, ppg_model=real_ppg_model, now=lambda: NOW,
    )


def ppg_capture(seconds: float = 33.0, fs: float = 30.0, bpm: float = 78.0, **overrides) -> dict:
    n = int(seconds * fs)
    t = np.arange(n) / fs
    f0 = bpm / 60.0
    samples = 0.5 + 0.2 * np.sin(2 * math.pi * f0 * t) + 0.05 * np.sin(4 * math.pi * f0 * t)
    capture = {
        "samples": samples.tolist(), "fs": fs, "timestamps": t.tolist(),
        "torch_verified": True, "clip_fraction": 0.0,
    }
    capture.update(overrides)
    return capture
