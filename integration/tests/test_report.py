"""Reporting discipline — finding C3 made structural."""

import numpy as np
import pytest

from integration.report import (
    MissingBaseline, add_stratification, print_table, results_table,
)

rng = np.random.default_rng(11)
Y = rng.uniform(80, 260, 400)
PIDS = np.array([f"P{i % 20:02d}" for i in range(400)])
B0 = np.full(400, Y.mean())
B1 = Y + rng.normal(0, 30, 400)
B2 = Y + rng.normal(0, 25, 400)
MODEL = Y + rng.normal(0, 18, 400)


def test_a_model_without_baselines_is_refused():
    with pytest.raises(MissingBaseline, match="C3"):
        results_table(y_true=Y, horizon="T+60", baselines={},
                      models={"GBDT": MODEL})


def test_a_partial_baseline_set_is_refused():
    with pytest.raises(MissingBaseline, match=r"B1|B2"):
        results_table(y_true=Y, horizon="T+60", baselines={"B0": B0},
                      models={"GBDT": MODEL})


def test_baselines_alone_need_no_model():
    tbl = results_table(y_true=Y, horizon="T+60",
                        baselines={"B0": B0, "B1": B1, "B2": B2}, models={})
    assert len(tbl["rows"]) == 3


def test_escape_hatch_exists_but_must_be_asked_for():
    tbl = results_table(y_true=Y, horizon="T+60", baselines={},
                        models={"GBDT": MODEL}, require_baselines=False)
    assert tbl["rows"][0]["kind"] == "model"


def test_mismatched_sample_counts_are_refused():
    """The Sprint 2 notebook compared a calibrated model on 1534 meals with
    an uncalibrated one on 1669 in the same table. This is that check."""
    with pytest.raises(ValueError, match="same samples"):
        results_table(y_true=Y, horizon="T+60",
                      baselines={"B0": B0, "B1": B1, "B2": B2},
                      models={"GBDT": MODEL[:300]})


def test_model_is_scored_against_the_best_baseline_not_the_worst():
    tbl = results_table(y_true=Y, horizon="T+60",
                        baselines={"B0": B0, "B1": B1, "B2": B2},
                        models={"GBDT": MODEL}, participant_ids=PIDS)
    best = max(r["zone_a"] for r in tbl["rows"] if r["kind"] == "baseline")
    model_row = next(r for r in tbl["rows"] if r["kind"] == "model")
    assert model_row["zone_a_over_best_baseline"] == pytest.approx(
        model_row["zone_a"] - best, abs=0.01)


def test_reference_histogram_is_always_attached():
    tbl = results_table(y_true=Y, horizon="T+60",
                        baselines={"B0": B0, "B1": B1, "B2": B2}, models={})
    assert sum(b["n"] for b in tbl["reference_histogram"]) == len(Y)


def test_per_participant_summary_is_included_when_ids_are_given():
    tbl = results_table(y_true=Y, horizon="T+60",
                        baselines={"B0": B0, "B1": B1, "B2": B2},
                        models={"GBDT": MODEL}, participant_ids=PIDS)
    for r in tbl["rows"]:
        assert "median_zone_a" in r and "iqr_zone_a" in r


def test_table_prints_without_error(capsys):
    tbl = results_table(y_true=Y, horizon="T+60",
                        baselines={"B0": B0, "B1": B1, "B2": B2},
                        models={"GBDT": MODEL}, participant_ids=PIDS,
                        n_held_out_participants=9)
    add_stratification(tbl, Y, MODEL)
    print_table(tbl)
    out = capsys.readouterr().out
    assert "B0 =" in out and "B2 =" in out
    assert "held-out participants=9" in out
    assert "Reference glucose distribution" in out
