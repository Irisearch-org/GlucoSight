"""Tests for the subject-level split (finding N5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nlp.data.splits import (
    SUBJECT_SEPARATOR,
    assert_no_group_leakage,
    ensure_subject_id,
    grouped_folds,
    leakage_report,
    subject_of,
)


def _shaped_like_shanghai() -> pd.DataFrame:
    """100-ish subjects where eight of them contribute a second admission.

    This is the shape that made the old split leak: `participant_id` is the
    period, so grouping on it puts `2001_0_*` in train and `2001_1_*` in test.
    """
    rows = []
    for subject in range(24):
        periods = 2 if subject < 4 else 1
        for period in range(periods):
            for meal in range(6):
                rows.append({
                    "meal_id": f"{subject}-{period}-{meal}",
                    "participant_id": f"{subject:04d}_{period}_20201230",
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# subject id derivation
# --------------------------------------------------------------------------

def test_subject_of_strips_the_period_and_date():
    assert subject_of("2000_0_20201230") == "2000"
    assert subject_of("2069_2_20210104") == "2069"


def test_subject_of_leaves_a_bare_id_alone():
    assert subject_of("2000") == "2000"


def test_subject_separator_is_the_underscore_the_stem_uses():
    assert SUBJECT_SEPARATOR == "_"


def test_ensure_subject_id_derives_the_column_when_absent():
    frame = _shaped_like_shanghai()
    out = ensure_subject_id(frame)
    assert "subject_id" in out.columns
    assert out["subject_id"].eq(out["participant_id"].str.split("_").str[0]).all()


def test_ensure_subject_id_is_a_no_op_when_present():
    frame = ensure_subject_id(_shaped_like_shanghai())
    assert ensure_subject_id(frame) is frame


# --------------------------------------------------------------------------
# the leak itself
# --------------------------------------------------------------------------

def test_grouping_on_subject_leaks_nothing():
    frame = ensure_subject_id(_shaped_like_shanghai())
    folds = grouped_folds(frame, n_splits=5, group_column="subject_id")
    assert_no_group_leakage(frame, folds, group_column="subject_id")
    assert leakage_report(frame, folds, group_column="subject_id") == 0.0


def test_grouping_on_period_leaks_the_repeat_admissions():
    """The old behaviour, kept as a measurement so the fix stays visible."""
    frame = ensure_subject_id(_shaped_like_shanghai())
    period_folds = grouped_folds(frame, n_splits=5, group_column="participant_id")
    leaked = leakage_report(frame, period_folds, group_column="subject_id")
    assert leaked > 0.0
    with pytest.raises(ValueError, match="both sides"):
        assert_no_group_leakage(frame, period_folds, group_column="subject_id")


def test_leakage_report_counts_test_rows_not_groups():
    """Two periods, one subject, both in test: every test row is leaked."""
    frame = ensure_subject_id(pd.DataFrame({
        "meal_id": ["a", "b"],
        "participant_id": ["2000_0_20201230", "2000_1_20210104"],
    }))
    folds = [(np.array([0, 1]), np.array([0, 1]))]
    assert leakage_report(frame, folds, group_column="subject_id") == 1.0


def test_leakage_report_is_zero_for_an_empty_split():
    frame = ensure_subject_id(pd.DataFrame({
        "meal_id": ["a"],
        "participant_id": ["2000_0_20201230"],
    }))
    assert leakage_report(frame, [(np.array([0]), np.array([], dtype=int))]) == 0.0


# --------------------------------------------------------------------------
# fold shape
# --------------------------------------------------------------------------

def test_grouped_folds_returns_the_requested_number_of_folds():
    frame = ensure_subject_id(_shaped_like_shanghai())
    assert len(grouped_folds(frame, n_splits=4)) == 4


def test_grouped_folds_covers_every_row_exactly_once():
    frame = ensure_subject_id(_shaped_like_shanghai())
    folds = grouped_folds(frame, n_splits=5)
    seen = np.concatenate([test for _, test in folds])
    assert sorted(seen.tolist()) == list(range(len(frame)))


def test_grouped_folds_never_splits_a_subject_across_folds():
    frame = ensure_subject_id(_shaped_like_shanghai())
    for _, test in grouped_folds(frame, n_splits=5):
        subjects = set(frame["subject_id"].iloc[test])
        assert len(subjects) == frame["subject_id"].iloc[test].nunique()
