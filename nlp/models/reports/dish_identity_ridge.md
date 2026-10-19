# NLP Dish-Identity Ridge — held-out evaluation

**Gate: PASSED**

M3 (M2 + per-dish indicators) vs M2 (g0 + total_grams + cooking flags), 5-fold GroupKFold by subject_id, on the rise_t60 target (ShanghaiT2DM).
n = 3045 meals, 100 subjects.

## Results

| Predictor | MAE | RMSE |
|---|---|---|
| M2 (shipped) | 31.148 | 39.696 |
| M3 (+ dish) | 30.500 | 38.781 |

## Per fold

| Fold | n | subjects | M2 MAE | M3 MAE | Improvement |
|---|---|---|---|---|---|
| 1 | 610 | 21 | 28.921 | 27.915 | +1.006 |
| 2 | 610 | 21 | 30.036 | 29.013 | +1.023 |
| 3 | 609 | 20 | 34.501 | 34.372 | +0.128 |
| 4 | 608 | 19 | 31.006 | 30.436 | +0.569 |
| 5 | 608 | 19 | 31.284 | 30.773 | +0.511 |

## Gate criteria

- pass: model MAE 30.50 beats baseline 31.15 by 0.65 mg/dL
- pass: model beat the baseline in 5/5 folds
- pass: bootstrap 95% CI [0.20, 1.07] excludes zero
- pass: label-permutation p = 0.001

## Bound on any claim

n = 100 subjects, ShanghaiT2DM. Per ADR 0004, this is a single-dataset validation, not a claim about the target Indonesian population.
