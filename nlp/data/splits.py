"""Subject-level grouped folds.

Finding N5: ``nlp/eval_baseline.py`` grouped ``GroupKFold`` on
``participant_id``, which is the CGM *period* — the workbook stem
``<subject>_<period>_<date>`` — not the patient. The 109 periods belong to 100
subjects, and eight subjects contribute 2–3 periods (2001, 2014, 2015, 2017,
2055, 2074, 2078 twice; 2069 three times), so a repeat admission landed on both
sides of the split and **18.5–21.5% of test meals were a revisit by a subject
the model had already trained on**.

Grouping on ``subject_id`` removes that. What it does *not* remove: two periods
of one subject are still the same person measured twice, so a subject-grouped
split is still not a cross-population estimate. It only stops the same person
being counted as independent evidence.

:func:`leakage_report` exists so the bug is measurable rather than merely
fixed — it returns the share of test rows whose group also appears in train,
which is 0.0 for subject grouping and 0.18–0.22 for period grouping on this
dataset. ``nlp/tests/test_splits.py`` asserts both.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

#: ``participant_id`` is ``<subject>_<period>_<date>``.
SUBJECT_SEPARATOR = "_"


def subject_of(participant_id) -> str:
    """``"2000_0_20201230"`` -> ``"2000"``."""
    return str(participant_id).split(SUBJECT_SEPARATOR)[0]


def ensure_subject_id(
    frame: pd.DataFrame,
    participant_column: str = "participant_id",
    subject_column: str = "subject_id",
) -> pd.DataFrame:
    """Return ``frame`` with a ``subject_id`` column, deriving it if absent."""
    if subject_column in frame.columns:
        return frame
    frame = frame.copy()
    frame[subject_column] = [subject_of(value) for value in frame[participant_column]]
    return frame


def grouped_folds(
    frame: pd.DataFrame,
    n_splits: int = 5,
    group_column: str = "subject_id",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """``(train_index, test_index)`` folds that never split a group."""
    groups = frame[group_column].to_numpy()
    return list(GroupKFold(n_splits=n_splits).split(frame, groups=groups))


def assert_no_group_leakage(
    frame: pd.DataFrame,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    group_column: str = "subject_id",
) -> None:
    """Raise if any group appears on both sides of any fold."""
    groups = frame[group_column].to_numpy()
    for index, (train, test) in enumerate(folds):
        shared = set(groups[train]) & set(groups[test])
        if shared:
            raise ValueError(
                f"fold {index}: {len(shared)} group(s) on both sides, "
                f"e.g. {sorted(shared)[:5]}"
            )


def leakage_report(
    frame: pd.DataFrame,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    group_column: str = "subject_id",
) -> float:
    """Share of test rows whose group also appears in that fold's train set."""
    groups = frame[group_column].to_numpy()
    leaked = 0
    total = 0
    for train, test in folds:
        seen = set(groups[train])
        total += len(test)
        leaked += sum(1 for group in groups[test] if group in seen)
    return leaked / total if total else 0.0
