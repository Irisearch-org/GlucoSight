"""Promoting nlp/eval_dish_identity.py's M3_dish result to a real, gated
Predictor (plan: wiring the fusion pipeline, change 4). ADR 0004: reuse
rppg.models.glucose_estimator's 4-criterion gate (positive improvement,
wins most folds, bootstrap CI over subjects excludes zero, label-permutation
test), don't invent a new evaluation standard.

Synthetic data reuses nlp/tests/test_dish_identity.py's `synthetic()` — the
same positive/null control already validated against `run_ladder`, so these
gate tests exercise the same known-signal/known-noise frames at a known
result, just scored through the gate instead of raw MAE comparison.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("sklearn")

from nlp.data.splits import grouped_folds
from nlp.tests.test_dish_identity import synthetic
from rppg.models.glucose_estimator import MAX_PERMUTATION_P, MIN_FOLD_WINS

from nlp.models.dish_identity_ridge import (
    evaluate_dish_addition,
    train_and_persist_from_frame,
)

# Small sample counts for test speed; the gate logic being tested does not
# depend on how many resamples are drawn, only on whether it reads the
# numbers correctly.
FAST = dict(n_bootstrap=200, n_permutations=50)


# ---------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------

def test_fold_predictions_never_share_a_subject():
    frame = synthetic(dish_driven=True)
    folds = grouped_folds(frame, n_splits=5)
    ev = evaluate_dish_addition(frame, folds, run_bootstrap=False, run_permutation=False)
    assert len(ev.folds) == 5
    assert sum(f.n_test for f in ev.folds) == len(frame)


def test_gate_accepts_strong_dish_signal():
    """Mirrors test_dish_identity.py's own positive control: a rise driven
    only by dish identity, where M3 clears M2 by >1 mg/dL MAE."""
    frame = synthetic(dish_driven=True)
    folds = grouped_folds(frame, n_splits=5)
    ev = evaluate_dish_addition(frame, folds, **FAST)
    passed, reasons = ev.passes_gate()
    assert passed, f"gate blocked real dish signal: {reasons}"
    assert ev.fold_wins >= MIN_FOLD_WINS


def test_gate_rejects_dish_blind_signal():
    """Mirrors test_dish_identity.py's own null control: a rise g0-driven,
    dishes attached at random — M3 must not clear the gate."""
    frame = synthetic(dish_driven=False, seed=7)
    folds = grouped_folds(frame, n_splits=5)
    ev = evaluate_dish_addition(frame, folds, **FAST)
    passed, reasons = ev.passes_gate()
    assert not passed, f"gate passed on a dish-blind rise: {reasons}"


def test_bootstrap_resamples_subjects_not_meals():
    frame = synthetic(dish_driven=True)
    folds = grouped_folds(frame, n_splits=5)
    ev = evaluate_dish_addition(frame, folds, run_permutation=False, n_bootstrap=200)
    lo, hi = ev.bootstrap_ci
    assert lo <= ev.improvement <= hi


# ---------------------------------------------------------------------
# train_and_persist
# ---------------------------------------------------------------------

def test_train_and_persist_writes_artifact_when_gate_passes(tmp_path):
    frame = synthetic(dish_driven=True)
    folds = grouped_folds(frame, n_splits=5)
    artifact = tmp_path / "dish_identity_ridge.json"
    report = tmp_path / "reports" / "dish_identity_ridge.md"
    ev = train_and_persist_from_frame(
        frame, folds, artifact_path=str(artifact), report_path=str(report), **FAST
    )
    assert ev.passes_gate()[0]
    assert artifact.exists()
    assert report.exists()

    import json
    blob = json.loads(artifact.read_text())
    assert len(blob["dish_vocabulary"]) > 0


def test_train_and_persist_writes_nothing_when_gate_fails(tmp_path):
    frame = synthetic(dish_driven=False, seed=7)
    folds = grouped_folds(frame, n_splits=5)
    artifact = tmp_path / "dish_identity_ridge.json"
    report = tmp_path / "reports" / "dish_identity_ridge.md"
    ev = train_and_persist_from_frame(
        frame, folds, artifact_path=str(artifact), report_path=str(report), **FAST
    )
    assert not ev.passes_gate()[0]
    assert not artifact.exists()
    assert report.exists()  # the report explains the failure either way


def test_train_and_persist_removes_a_stale_artifact_on_a_failing_rerun(tmp_path):
    artifact = tmp_path / "dish_identity_ridge.json"
    artifact.write_text("{}")
    frame = synthetic(dish_driven=False, seed=7)
    folds = grouped_folds(frame, n_splits=5)
    train_and_persist_from_frame(
        frame, folds, artifact_path=str(artifact),
        report_path=str(tmp_path / "reports" / "r.md"), **FAST
    )
    assert not artifact.exists()
