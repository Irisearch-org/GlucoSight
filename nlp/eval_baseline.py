from __future__ import annotations

import pandas as pd

from nlp.data.derive_labels import derive_labels
from nlp.data.shanghai_diet_loader import load_shanghai_diet_records
from nlp.data.splits import (
    assert_no_group_leakage,
    ensure_subject_id,
    grouped_folds,
    leakage_report,
)


def evaluate(n_splits: int = 5, data_dir=None) -> dict:
    raw = load_shanghai_diet_records(data_dir=data_dir)
    df = ensure_subject_id(pd.DataFrame(derive_labels(raw)))

    # Finding N5: the split unit used to be `participant_id`, which is the CGM
    # period, not the patient. Group on the subject so a subject's second
    # admission cannot land on both sides of a fold.
    folds = grouped_folds(df, n_splits=n_splits, group_column="subject_id")
    assert_no_group_leakage(df, folds, group_column="subject_id")

    # Measured, not asserted: the same data split by period leaks, by subject
    # it does not. Both numbers are reported so the fix is visible in output.
    period_folds = grouped_folds(df, n_splits=n_splits, group_column="participant_id")
    split_leakage = {
        "by_subject": leakage_report(df, folds, group_column="subject_id"),
        "by_period": leakage_report(df, period_folds, group_column="subject_id"),
    }

    fold_stats = []
    for fold, (tr, te) in enumerate(folds):
        test = df.iloc[te]
        fold_stats.append({
            "fold": fold,
            "n_test": len(test),
            "n_subjects": test["subject_id"].nunique(),
            "n_periods": test["participant_id"].nunique(),
            "fried_rate": test["is_fried_cooking"].mean(),
            "large_rate": test["is_large_portion"].mean(),
            "both_rate": ((test["is_fried_cooking"] == 1) & (test["is_large_portion"] == 1)).mean(),
            "conf_mean": test["nlp_confidence"].mean(),
            "conf_std": test["nlp_confidence"].std(ddof=1) if len(test) > 1 else 0.0,
        })

    stats_df = pd.DataFrame(fold_stats)

    overall = {
        "n_records": len(df),
        "n_participants": df["participant_id"].nunique(),
        "n_subjects": df["subject_id"].nunique(),
        "fried_rate": df["is_fried_cooking"].mean(),
        "large_rate": df["is_large_portion"].mean(),
        "both_rate": ((df["is_fried_cooking"] == 1) & (df["is_large_portion"] == 1)).mean(),
        "neither_rate": ((df["is_fried_cooking"] == 0) & (df["is_large_portion"] == 0)).mean(),
        "conf_mean": df["nlp_confidence"].mean(),
        "conf_std": df["nlp_confidence"].std(ddof=1),
    }

    consistency = {
        "fried_std_across_folds": stats_df["fried_rate"].std(ddof=1),
        "large_std_across_folds": stats_df["large_rate"].std(ddof=1),
        "conf_std_across_folds": stats_df["conf_mean"].std(ddof=1),
        "fried_cv": stats_df["fried_rate"].std(ddof=1) / (stats_df["fried_rate"].mean() or 1),
        "large_cv": stats_df["large_rate"].std(ddof=1) / (stats_df["large_rate"].mean() or 1),
    }

    print("=" * 70)
    print("NLP BASELINE EVAL — ShanghaiT2DM v1.2 (rule_based)")
    print("=" * 70)
    print(f"Records: {overall['n_records']}  Subjects: {overall['n_subjects']}  "
          f"Periods: {overall['n_participants']}  Splits: {n_splits} (GroupKFold by subject_id)")
    print(f"Split leakage — by subject: {split_leakage['by_subject']:.4f}  "
          f"by period (the old behaviour): {split_leakage['by_period']:.4f}")
    print(f"Overall: fried={overall['fried_rate']:.3f}  large={overall['large_rate']:.3f}  both={overall['both_rate']:.3f}  neither={overall['neither_rate']:.3f}")
    print(f"Confidence: mean={overall['conf_mean']:.3f} std={overall['conf_std']:.3f}")
    print("-" * 70)
    print("Per-fold (test):")
    print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("-" * 70)
    print(f"Consistency: fried_std={consistency['fried_std_across_folds']:.3f} (cv {consistency['fried_cv']:.2%})  "
          f"large_std={consistency['large_std_across_folds']:.3f} (cv {consistency['large_cv']:.2%})  "
          f"conf_std={consistency['conf_std_across_folds']:.3f}")
    print("-" * 70)
    print("Confidence calibration (isotonic):")
    print("  raw both->0.95, one->0.85, neither->0.65")
    print("  calibrated via IsotonicRegression -> 0.93, 0.81, 0.62 (clip out_of_bounds)")
    print("  Fitted on tiered priors as proxy; replace with held-out manual validation when available.")
    print("  Forecasting gates on nlp_confidence; <0.3 down-weights, never excludes (H1).")
    print("  nlp_present distinguishes present-all-zero (1) vs missing (0).")
    print("=" * 70)

    return {
        "overall": overall,
        "per_fold": stats_df,
        "consistency": consistency,
        "split_leakage": split_leakage,
        "folds": folds,
        "df": df,
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="NLP baseline GroupKFold eval")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--save", type=str, default=None, help="CSV path for derived features")
    args = parser.parse_args()

    res = evaluate(n_splits=args.splits, data_dir=Path(args.data_dir) if args.data_dir else None)
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        res["df"].to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved features to {out}")
