# CV B0 Population-Mean Baseline — CGMacros

## Summary

- Dataset: CGMacros 1.0.0
- License: CC BY-NC-SA 4.0
- Meals evaluated: 1616
- Participants evaluated: 45
- Ground-truth provenance: `cgmacros_reported_estimate`
- Manifest SHA-256: `064f78df9be342da94bff2761236d0e3b5ac6da7bbd5a5aaac7c7a0109a2c152`
- Images read by predictor: **no**

CGMacros' local data dictionary describes meal macros as reported estimates. This report therefore does not label them as weighed ground truth.

## Fixed Prediction

`carbs_g=45.0`, `protein_g=18.0`, `fat_g=12.0`, `fiber_g=4.0`, `gi_category=1`.

## Pooled Metrics

| Macro | MAE (g) | RMSE (g) | Meals |
|---|---:|---:|---:|
| carbs_g | 27.34 | 31.84 | 1616 |
| protein_g | 19.70 | 28.12 | 1616 |
| fat_g | 12.51 | 18.26 | 1616 |
| fiber_g | 4.18 | 7.27 | 1616 |

## Per-Participant Distribution

| Macro | MAE median [IQR] (g) | RMSE median [IQR] (g) | Participants |
|---|---:|---:|---:|
| carbs_g | 27.06 [25.70, 29.07] | 30.86 [28.96, 33.14] | 45 |
| protein_g | 19.57 [18.32, 21.22] | 27.83 [26.26, 29.94] | 45 |
| fat_g | 12.50 [11.19, 13.41] | 17.45 [16.02, 19.79] | 45 |
| fiber_g | 3.84 [3.60, 4.43] | 5.04 [4.56, 5.62] | 45 |

## Diagnostics by Meal Type

### Breakfast

| Macro | MAE (g) | RMSE (g) | Meals |
|---|---:|---:|---:|
| carbs_g | 22.30 | 22.46 | 426 |
| protein_g | 17.32 | 26.62 | 426 |
| fat_g | 12.67 | 18.82 | 426 |
| fiber_g | 3.81 | 3.83 | 426 |

### Dinner

| Macro | MAE (g) | RMSE (g) | Meals |
|---|---:|---:|---:|
| carbs_g | 28.53 | 36.18 | 452 |
| protein_g | 16.91 | 22.75 | 452 |
| fat_g | 13.58 | 20.33 | 452 |
| fiber_g | 4.23 | 7.14 | 452 |

### Lunch

| Macro | MAE (g) | RMSE (g) | Meals |
|---|---:|---:|---:|
| carbs_g | 31.89 | 35.97 | 431 |
| protein_g | 29.32 | 39.59 | 431 |
| fat_g | 13.56 | 19.53 | 431 |
| fiber_g | 4.63 | 6.56 | 431 |

### Snack

| Macro | MAE (g) | RMSE (g) | Meals |
|---|---:|---:|---:|
| carbs_g | 26.21 | 29.90 | 307 |
| protein_g | 13.59 | 14.68 | 307 |
| fat_g | 9.23 | 10.91 | 307 |
| fiber_g | 3.98 | 11.04 | 307 |

## Manifest Audit

- Participant folders discovered: 45
- Meal-start rows discovered: 1706
- Eligible meals: 1616
- Excluded meals: 90
- Manifest warnings: 3
- Exclusion reasons: `{"macro_out_of_documented_range": 34, "missing_or_invalid_macros": 1, "missing_or_unresolved_before_image": 55}`
- Warning codes: `{"after_image_path_repaired": 3}`

Meal rows require complete macros within the ranges documented by CGMacros (0-176 g per macro) and a resolvable start image. Image-only rows after a meal start are retained as after-meal metadata and never become independent B0 inputs.

## Interpretation

B0 is a deliberately non-informative reference, not a model-quality claim. Every later CV result must be evaluated on the same manifest and printed beside these values.
