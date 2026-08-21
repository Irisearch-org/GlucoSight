"""Grouped-split leakage guard and per-subject error reporting.

Pooled error over-weights subjects with many recordings (this dataset ranges
1-7 recordings/subject) and a random split of PPG recordings almost
guarantees the same subject appears in both train and test, at which point
the model memorises per-subject baseline glucose instead of learning
anything optical. assert_group_disjoint makes that failure loud instead of
silent.
"""
import numpy as np
import pandas as pd


def assert_group_disjoint(groups, train_idx, test_idx):
    train_subjects = set(np.asarray(groups)[train_idx])
    test_subjects = set(np.asarray(groups)[test_idx])
    overlap = train_subjects & test_subjects
    assert overlap == set(), f'subject leakage across split: {sorted(overlap)}'


def per_subject_mae(y_true, y_pred, subject_ids):
    df = pd.DataFrame({
        'subject_id': subject_ids,
        'abs_err': np.abs(np.asarray(y_true) - np.asarray(y_pred)),
    })
    grouped = df.groupby('subject_id')['abs_err']
    out = grouped.mean().to_frame('mae')
    out['n_recordings'] = grouped.size()
    return out


def summarize_per_subject(per_subject_df, column='mae'):
    values = per_subject_df[column]
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return {'median': float(median), 'iqr_low': float(q1), 'iqr_high': float(q3)}
