# PPG Glucose Estimator — held-out evaluation

**Gate: FAILED**

Ridge(alpha=10) on 7 extracted features, 5-fold GroupKFold by subject.
Features extracted at **2190 Hz** — the NATIVE rate. A model trained here will NOT see these feature distributions on a 30 Hz phone camera. Use this run only to measure the sample-rate gap.
n = 67 recordings, 23 subjects.

## Why Zone A is reported but is not the gate

The mean baseline — one constant for everyone — reaches **Zone A 85.1%** on this dataset, against a project target of 70%. Zone A is +/-20%, which at this cohort's median glucose is wider than the label SD, so it cannot separate a working model from a constant here. The gate is MAE against the baseline on held-out subjects.

## Results

| Predictor | MAE | RMSE | Zone A | Zone A+B |
|---|---|---|---|---|
| Mean baseline | 14.57 | 18.91 | 85.1% | 100.0% |
| Ridge | 15.04 | 19.48 | 85.1% | 100.0% |

## Per fold

| Fold | n | subjects | Baseline MAE | Model MAE | Improvement |
|---|---|---|---|---|---|
| 1 | 14 | 4 | 14.79 | 13.92 | +0.87 |
| 2 | 13 | 4 | 14.23 | 14.16 | +0.07 |
| 3 | 14 | 5 | 17.74 | 19.25 | -1.50 |
| 4 | 13 | 5 | 12.62 | 13.15 | -0.53 |
| 5 | 13 | 5 | 13.19 | 14.51 | -1.32 |

## Gate criteria

- FAIL: model MAE 15.04 is not better than the mean baseline 14.57 mg/dL
- FAIL: model beat the baseline in only 2/5 folds (need 4)
- FAIL: bootstrap 95% CI on the MAE improvement [-1.46, 0.57] includes zero
- FAIL: label-permutation p = 0.304 (need <= 0.05) — this improvement is within what shuffled labels produce

## Bound on any claim

n = 23 subjects. Per `agents/rppg/CLAUDE.md` and contract ring-fence rule E1.5, every claim about this model is bounded by that number.
