# GBDT forecaster trial (ticket #17)

LightGBM glucose forecasters at T+60 and T+120 min on the full CGMacros release (45 participants), evaluated against B0/B1/B2 on identical participant-grouped folds under ADR 0004.

**Verdict.** With nested feature-set selection, the procedure passes every ADR 0004 gate criterion against B2 at both horizons on the CGM and meter conditions. The PPG condition fails (CI crosses zero). Carb sensitivity is pooled and not per-meal. The feature set chosen on the full cohort is `full` at both horizons, but the outer folds disagree about which set wins (see Selection stability).

## Data and cleaning (script: `forecasting/evaluate/gbdt_forecaster.py`)
Source: `/Users/auliamnaufal/Downloads/CGMacros_full/CGMacros/` (45 folders; gaps 024, 025, 037, 040). Counts in `forecasting/gbdt_results/cleaning.json`.

| Rule (#12) | Result | #12 expected |
|---|---|---|
| Core composition all zero (calories, carbs, protein, fat) → drop | 44 | 44 |
| Atwater deviation >20% / >30% / >50% (non-marker rows) | 107 / 85 / 58 | 107 / 82 / 58 |
| Atwater primary ±30% → drop | 85 | 82 |
| Fibre > carbs → fibre NaN, meal kept | 33 raw rows; 24 set NaN | 33 |
| Meals kept | 1,577 (45 participants) | — |

The ±30% count differs from #12 (85 vs 82). I did not tune the rule to match. ±20% and ±50% match exactly. Flagged for the spec owner.

- Target: exact Dexcom glucose at meal time + horizon (C2: no reading at or after t0 is used as input).
- g0: last Dexcom reading strictly before t0.
- Cohort after target/g0 availability: 1,534 meals (T60), 1,531 (T120), all 45 participants.
- Diabetes band from HbA1c: normal 15, prediabetic 16, T2D 14. #13 expects 18/14/13; the rule was not changed.

## Measurement conditions
- **CGM** (gate condition): clean Dexcom g0, no added noise.
- **Meter**: ISO 15197:2013 envelope as zero-mean normal noise on g0: sd = 15/1.96 mg/dL below 100 mg/dL, sd = 0.15·g/1.96 at or above 100; clipped to [20, 600]. This is a conservative stand-in for the meter error distribution. The clause should be verified before citing.
- **PPG**: calibration map fitted on the served PPG model's held-out recordings (`rppg/models/reports/ppg_served_residuals.csv`, branch `wayfinder/ppg-served-model`): `g0_ppg = a + b·g0 + N(0, sigma)`.
  - a = 113.35, b = 0.0101, sigma = 4.20 mg/dL, corr(true, pred) = 0.045.
  - 67 rows, 23 subjects, **but only 7 unique `recording_id` values** (not 67).
  - The file's `residual` column equals true − pred, the opposite sign to the estimate − reference convention. The fit uses `glucose_true`/`glucose_pred` directly, so the sign does not affect it.
  - Implication: the served PPG model carries almost no glucose information. Its predictions are nearly constant around 114 mg/dL.
- **Carb error** (sensitivity only): dish-ID errors from `glucosight-cv-data/results/dish_id_carb_errors.csv` (TKPI-derived; not committed). Per-meal mapping to CGMacros is impossible. The sensitivity draws one error per test meal from the pool of 253 non-abstained test-embed errors (mean −0.69 g, SD 8.19, MAE 3.21 g), clipped at 0 g. Abstained errors are excluded from the pool. Abstained meals are not identifiable per meal, so they cannot be left unperturbed. The run is pooled resampling, not a per-meal estimate.

## Protocol
- Outer: 5 repeats × GroupKFold(5) by participant = 25 outer folds (every participant held out once per repeat).
- Inner (training participants only): GroupKFold(4) selects among feature sets {g0_hour, g0_hour_macros, g0_hour_profile, full} × 16 GBDT configs (offset vs direct × num_leaves {4,8} × min_child_samples {20,60} × n_estimators {100,300}; lr 0.05) by inner MAE.
- Baselines on the same folds and the same noisy g0: B0 = train mean; B1 = g0; B2 = g0 + mean train excursion.
- Training augmentation: 3 meter-noise draws per meal (plus PPG draws when that condition is run). Evaluation: n draws per test meal.
- Gate (ADR 0004): positive MAE gain vs B2; GBDT beats B2 on ≥4 of 5 folds in every repeat; participant-cluster bootstrap 95% CI (2,000 resamples) excludes 0; within-participant macro permutation p ≤ 0.05.
- Deviation: **200 permutations** (not 1,000) to fit runtime. p resolution is 1/201. The sensitivity runs used 0 permutations, so their permutation check is not run and is not evidence of failure.

## Main result (nested design, n = 45 participants, 14 T2D)

| Horizon | Meals | GBDT MAE / RMSE | B0 MAE | B1 MAE | B2 MAE | Gain vs B2 | Fold wins per repeat | Bootstrap 95% CI | Permutation p | Gate vs B2 |
|---|---|---|---|---|---|---|---|---|---|---|
| T+60 | 1,534 | 28.78 / 37.68 | 38.72 | 37.91 | 31.68 | **+2.90** | [5,5,4,5,4] (23/25) | [1.36, 4.42] | 0.005 | **PASS** |
| T+120 | 1,531 | 28.49 / 38.69 | 36.93 | 36.50 | 32.05 | **+3.56** | [5,4,5,4,5] (23/25) | [1.62, 5.51] | 0.005 | **PASS** |

Gain vs B1 and B0 (T60 / T120): +9.13 [4.51, 13.53] / +8.01 [5.06, 11.14] vs B1; +9.94 [6.55, 14.12] / +8.44 [4.82, 11.91] vs B0.

## Per-source table (MAE mg/dL, gain vs B2)

| Source (g0 condition) | Horizon | GBDT | B2 | Gain vs B2 | Bootstrap 95% CI | Gate vs B2 |
|---|---|---|---|---|---|---|
| CGM (clean) | T+60 | 28.78 | 31.68 | +2.90 | [1.36, 4.42] | PASS |
| CGM (clean) | T+120 | 28.49 | 32.05 | +3.56 | [1.62, 5.51] | PASS |
| Meter (ISO 15197 noise) | T+60 | 29.69 | 32.60 | +2.92 | [1.49, 4.30] | PASS |
| Meter (ISO 15197 noise) | T+120 | 29.24 | 33.04 | +3.80 | [1.79, 5.64] | PASS |
| PPG calibration map | T+60 | 36.02 | 37.52 | +1.51 | [−0.80, 4.05] | **FAIL** (CI crosses 0; folds [5,5,4,3,4]) |
| PPG calibration map | T+120 | 32.86 | 35.51 | +2.65 | [−0.42, 5.65] | **FAIL** (CI crosses 0) |

The PPG rows come from the PPG-trained run, so the CGM and meter rows inside that run are not the clean model and are not reported here. The PPG run's CGM values were 29.32 / 29.01 (MAE) with gains +2.37 / +3.04.

Carb sensitivity (clean model; carb features perturbed with pooled dish-ID errors, one draw per test meal; not a per-meal estimate):

| Horizon | GBDT clean | GBDT carb-perturbed | Δ MAE | Gain vs B2 (perturbed) | Bootstrap 95% CI |
|---|---|---|---|---|---|
| T+60 | 28.78 | 28.79 | +0.01 | +2.89 | [1.57, 4.31] |
| T+120 | 28.49 | 28.63 | +0.14 | +3.42 | [1.29, 5.42] |

## Selection stability (nested, 25 outer folds)
- T+60: `g0_hour_macros` 17 folds, `full` 8.
- T+120: `full` 16 folds, `g0_hour_macros` 9.
- The feature-set choice is **unstable across outer folds**. The nested gate is passed by the procedure (selection plus fitting), not by any one feature set. Neither set has a separately gated held-out result.

## Served model (inner selection on all 45 participants)
- Inner selection on all 45 participants chose **`full`** at both horizons. Config: T+60 direct, num_leaves 8, min_child_samples 20, n_estimators 100 (training rows 6,136). T+120 direct, num_leaves 8, min_child_samples 60, n_estimators 100 (training rows 6,124).
- `full` = g0, source flags (meter/PPG), hour sin/cos, carbs, protein, fat, fibre, age, sex, BMI, diabetes band.
- The served model was trained on CGM and meter g0 only. PPG rows are not in the served model. PPG use is not validated (gate fails).
- The served model is the output of a procedure that passes the gate. Its own held-out performance is not separately reported, and at T+60 the outer folds preferred `g0_hour_macros`.

## Clarke error grid (CGM condition, per-participant Zone A, 45 participants, min 5 meals)
- T+60: GBDT pooled Zone A 61.6%, median 66.4% (IQR 20.4). B2 pooled 54.8%, median 58.7% (IQR 19.7).
- T+120: GBDT pooled Zone A 62.4%, median 62.6% (IQR 18.3). B2 pooled 57.1%, median 56.5% (IQR 23.3).
- Zone A by reference range (GBDT, T+60): 0–100 mg/dL 15.1% (n = 545); 100–140 58.9% (n = 2,860); 140–180 82.4% (n = 2,320); 180–250 57.0% (n = 1,560); ≥250 40.0% (n = 385). T+120 in `forecasting/gbdt_results/T120.json`.

## Pre-specified design (earlier run, superseded)
The first run used a single pre-specified `full` design with no nested selection, the old Contour-residual meter pool (n = 1,254, SD 48.8 mg/dL, an upper bound that overstates meter noise), and the old code. Its numbers are not directly comparable with the nested run:
- T+60: gain −0.20 mg/dL, CI [−2.80, 2.35], 3/5 fold wins. **FAIL.**
- T+120: gain +0.40 mg/dL, CI [−3.17, 3.77], 3/5 fold wins. **FAIL.**

A same-fold post-hoc `g0_hour_macros` ablation from that run (+2.28 / +2.49 mg/dL) chose its feature set on the test folds. It is exploratory only and is superseded by the nested run above.

## Recommendation
1. Serve the nested-procedure artifacts (`full`, T+60 and T+120), for CGM and meter g0 only. Their gates pass.
2. Do not claim a PPG benefit. The PPG condition fails the gate, and the served PPG model carries almost no glucose signal (corr 0.045).
3. Feature-set choice is a confirmatory question. Before any claim beyond this cohort, confirm on a held-out cohort: `full` (served) vs `g0_hour_macros` (outer-fold majority at T+60).
4. Carb sensitivity is pooled resampling, so the effect of carb error on the forecast is only bounded (ΔMAE ≤ 0.14 mg/dL), not measured per meal.

## Artifacts
- Script: `forecasting/evaluate/gbdt_forecaster.py`.
- Results (committed): `forecasting/gbdt_results/` (`summary.json`, `T60.json`, `T120.json`, `cleaning.json`); sensitivity: `forecasting/gbdt_results/sensitivity_carb/` and `forecasting/gbdt_results/sensitivity_ppg/` (`T60.json`, `T120.json`).
- Served models (untracked, in the main worktree, gitignored artifacts): `/Users/auliamnaufal/Downloads/glucosight/forecasting/models/gbdt_cgmacros_T60.txt` and `gbdt_cgmacros_T120.txt` (+ `.json` metadata; selected set `full`).
- Carb error pool (not committed, TKPI-derived): `/Users/auliamnaufal/Downloads/glucosight-cv-data/results/dish_id_carb_errors.csv`.
- PPG residuals (not committed): `origin/wayfinder/ppg-served-model:rppg/models/reports/ppg_served_residuals.csv`.

## Caveats and [NOT RUN]
- Permutations: 200 (not 1,000). Sensitivity runs: 0 permutations, so their permutation check is [NOT RUN].
- Meter noise: ISO 15197:2013 zero-mean normal is an assumption; clause to be verified.
- PPG: calibration from a 7-recording, 23-subject file (not 67 recordings); the file's residual sign is reversed. The calibration map is weak (corr 0.045).
- Carb: pooled resampling, not per-meal. Abstained errors excluded from the pool; abstained meals cannot be left unperturbed.
- Atwater ±30% count 85 vs #12's 82; diabetes T2D count 14 vs #13's 13.
- Feature-set selection unstable across outer folds (see Selection stability).
