# PPG Glucose Mean Baseline

**Per docs/DATA_STRATEGY.md §7 Day 2:** every track computes its trivial
baseline before any model is trained. This is the floor that any real
PPG-glucose model must beat.

## Setup

- **Dataset:** 67 recordings, 23 subjects
- **Split:** 5-fold GroupKFold by subject (finding C4)
- **Baseline:** predict training-fold population mean

## Per-Fold Results

| Fold | N_test | N_subjects | MAE (mg/dL) | RMSE (mg/dL) | Train mean |
|------|--------|------------|-------------|--------------|------------|
| 1 | 14 | 4 | 13.87 | 15.09 | 113.9 |
| 2 | 13 | 4 | 15.59 | 16.62 | 115.9 |
| 3 | 14 | 5 | 15.74 | 22.72 | 114.1 |
| 4 | 13 | 5 | 14.32 | 22.34 | 115.4 |
| 5 | 13 | 5 | 13.19 | 15.48 | 115.7 |

## Summary

- **MAE:** 14.54 ± 1.10 mg/dL
- **RMSE:** 18.45 ± 3.77 mg/dL
- **Effective n:** 23 held-out subjects
- **Pooled Clarke Zone A:** 83.6% (n=67 recordings, 23 subjects)

## Interpretation

The mean baseline achieves MAE of ~15 mg/dL by predicting the training
population mean. Any real model must beat this number to demonstrate
it has learned something from the PPG signal.

This result sits beside every future PPG-glucose model result in the
paper, per finding C3 (baselines are mandatory).
