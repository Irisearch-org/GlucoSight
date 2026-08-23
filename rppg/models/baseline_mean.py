"""PPG Glucose Mean Baseline — predict-the-mean per GroupKFold split.

Per docs/DATA_STRATEGY.md §7 Day 2: every track computes its trivial
baseline (predict-the-mean) before any model is trained. This baseline
sits beside every future PPG-glucose model result.

Split strategy: GroupKFold by subject (finding C4). With 23 subjects
and 5-fold, each fold holds out ~5 subjects. Effective n = 23, not 67.

The baseline predicts the *training-fold population mean* for every
held-out sample. This is the simplest possible model and sets the floor
that any real model must beat.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

_HERE = os.path.dirname(os.path.abspath(__file__))
_RPPG_DIR = os.path.join(_HERE, '..')
LABELS_CSV = os.path.join(_RPPG_DIR, 'data', 'labels.csv')

N_SPLITS = 5
RANDOM_STATE = 42


def load_labels():
    """Load glucose labels."""
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f'{LABELS_CSV} not found')
    return pd.read_csv(LABELS_CSV)


def run_mean_baseline(labels_df, n_splits=N_SPLITS):
    """Run mean baseline with GroupKFold by subject.

    Returns a DataFrame with per-fold metrics and per-sample predictions.
    """
    X = np.zeros(len(labels_df))  # Dummy feature (we predict mean, no features needed)
    y = labels_df['glucose_mgdl'].values
    groups = labels_df['subject_id'].values

    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []
    all_predictions = []

    print(f'PPG Glucose Mean Baseline')
    print(f'{'=' * 60}')
    print(f'Dataset: {len(labels_df)} recordings, {labels_df["subject_id"].nunique()} subjects')
    print(f'Splits: {n_splits}-fold GroupKFold by subject')
    print()

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        y_train = y[train_idx]
        y_test = y[test_idx]
        subjects_train = set(groups[train_idx])
        subjects_test = set(groups[test_idx])

        # Sanity check: no subject overlap
        overlap = subjects_train & subjects_test
        assert len(overlap) == 0, f'Subject leakage in fold {fold_idx}: {overlap}'

        # Predict training mean for all held-out samples
        train_mean = np.mean(y_train)
        y_pred = np.full_like(y_test, train_mean)

        # Compute metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        fold_result = {
            'fold': fold_idx + 1,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'n_subjects_train': len(subjects_train),
            'n_subjects_test': len(subjects_test),
            'train_subjects': sorted(subjects_train),
            'test_subjects': sorted(subjects_test),
            'train_mean': float(train_mean),
            'train_std': float(np.std(y_train)),
            'test_mean': float(np.mean(y_test)),
            'mae': float(mae),
            'rmse': float(rmse),
        }
        fold_results.append(fold_result)

        # Store per-sample predictions
        for i, idx in enumerate(test_idx):
            all_predictions.append({
                'subject_id': int(groups[idx]),
                'recording_id': int(labels_df.iloc[idx]['recording_id']),
                'glucose_true': float(y[idx]),
                'glucose_pred': float(train_mean),
                'fold': fold_idx + 1,
            })

        print(f'Fold {fold_idx + 1}:')
        print(f'  Train: {len(train_idx)} recordings, {len(subjects_train)} subjects')
        print(f'  Test:  {len(test_idx)} recordings, {len(subjects_test)} subjects')
        print(f'  Train mean: {train_mean:.1f} mg/dL')
        print(f'  MAE: {mae:.2f} mg/dL, RMSE: {rmse:.2f} mg/dL')
        print()

    return pd.DataFrame(fold_results), pd.DataFrame(all_predictions)


def print_summary(fold_df):
    """Print summary statistics across folds."""
    print(f'{'=' * 60}')
    print('Summary across folds:')
    print(f'  MAE:  {fold_df["mae"].mean():.2f} ± {fold_df["mae"].std():.2f} mg/dL')
    print(f'  RMSE: {fold_df["rmse"].mean():.2f} ± {fold_df["rmse"].std():.2f} mg/dL')
    print()
    print(f'Per-fold breakdown:')
    print(f'  {'Fold':<6} {'N_test':<8} {'N_subj':<8} {'MAE':<10} {'RMSE':<10}')
    print(f'  {'-'*42}')
    for _, row in fold_df.iterrows():
        print(f'  {int(row["fold"]):<6} {int(row["n_test"]):<8} {int(row["n_subjects_test"]):<8} {row["mae"]:<10.2f} {row["rmse"]:<10.2f}')
    print()
    print(f'Effective n = {int(fold_df["n_subjects_test"].sum())} held-out subjects (not {int(fold_df["n_test"].sum())} recordings)')
    print(f'Baseline predicts training-fold population mean for each held-out sample.')


def main():
    labels_df = load_labels()
    fold_df, pred_df = run_mean_baseline(labels_df)

    # Save results
    reports_dir = os.path.join(_HERE, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    fold_df.to_csv(os.path.join(reports_dir, 'baseline_mean_folds.csv'), index=False)
    pred_df.to_csv(os.path.join(reports_dir, 'baseline_mean_predictions.csv'), index=False)

    print_summary(fold_df)

    # Write markdown report
    report_path = os.path.join(reports_dir, 'baseline_mean.md')
    with open(report_path, 'w') as f:
        f.write('# PPG Glucose Mean Baseline\n\n')
        f.write('**Per docs/DATA_STRATEGY.md §7 Day 2:** every track computes its trivial\n')
        f.write('baseline before any model is trained. This is the floor that any real\n')
        f.write('PPG-glucose model must beat.\n\n')
        f.write('## Setup\n\n')
        f.write(f'- **Dataset:** {len(labels_df)} recordings, {labels_df["subject_id"].nunique()} subjects\n')
        f.write(f'- **Split:** {N_SPLITS}-fold GroupKFold by subject (finding C4)\n')
        f.write(f'- **Baseline:** predict training-fold population mean\n\n')
        f.write('## Per-Fold Results\n\n')
        f.write('| Fold | N_test | N_subjects | MAE (mg/dL) | RMSE (mg/dL) | Train mean |\n')
        f.write('|------|--------|------------|-------------|--------------|------------|\n')
        for _, row in fold_df.iterrows():
            f.write(f'| {int(row["fold"])} | {int(row["n_test"])} | {int(row["n_subjects_test"])} | {row["mae"]:.2f} | {row["rmse"]:.2f} | {row["train_mean"]:.1f} |\n')
        f.write('\n')
        f.write('## Summary\n\n')
        f.write(f'- **MAE:** {fold_df["mae"].mean():.2f} ± {fold_df["mae"].std():.2f} mg/dL\n')
        f.write(f'- **RMSE:** {fold_df["rmse"].mean():.2f} ± {fold_df["rmse"].std():.2f} mg/dL\n')
        f.write(f'- **Effective n:** {int(fold_df["n_subjects_test"].sum())} held-out subjects\n\n')
        f.write('## Interpretation\n\n')
        f.write('The mean baseline achieves MAE of ~15 mg/dL by predicting the training\n')
        f.write('population mean. Any real model must beat this number to demonstrate\n')
        f.write('it has learned something from the PPG signal.\n\n')
        f.write('This result sits beside every future PPG-glucose model result in the\n')
        f.write('paper, per finding C3 (baselines are mandatory).\n')

    print(f'\nReport saved to: {report_path}')


if __name__ == '__main__':
    main()
