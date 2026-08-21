import numpy as np
import pandas as pd
import pytest

from rppg.evaluate import metrics


def test_assert_group_disjoint_passes_on_disjoint_groups():
    groups = np.array([1, 1, 2, 2, 3, 3])
    train_idx = np.array([0, 1, 2, 3])
    test_idx = np.array([4, 5])
    metrics.assert_group_disjoint(groups, train_idx, test_idx)  # must not raise


def test_assert_group_disjoint_raises_on_leaked_subject():
    groups = np.array([1, 1, 2, 2, 3, 3])
    train_idx = np.array([0, 1, 2, 4])  # subject 3 (idx 4) leaks into train
    test_idx = np.array([3, 5])         # subject 2 (idx 3) and subject 3 (idx 5)
    with pytest.raises(AssertionError):
        metrics.assert_group_disjoint(groups, train_idx, test_idx)


def test_per_subject_mae():
    y_true = np.array([100.0, 110.0, 200.0])
    y_pred = np.array([105.0, 100.0, 190.0])
    subject_ids = np.array([1, 1, 2])
    df = metrics.per_subject_mae(y_true, y_pred, subject_ids)
    assert df.loc[1, 'mae'] == pytest.approx((5.0 + 10.0) / 2)
    assert df.loc[2, 'mae'] == pytest.approx(10.0)
    assert df.loc[1, 'n_recordings'] == 2
    assert df.loc[2, 'n_recordings'] == 1


def test_summarize_per_subject():
    df = pd.DataFrame({'mae': [10.0, 20.0, 30.0, 40.0]})
    summary = metrics.summarize_per_subject(df)
    assert summary['median'] == pytest.approx(25.0)
    assert summary['iqr_low'] <= summary['median'] <= summary['iqr_high']
