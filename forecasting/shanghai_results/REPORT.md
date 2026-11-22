# External check on ShanghaiT2DM (ticket #29)

**Answer.** The served CGMacros forecasters do **not** transfer to Shanghai T2D patients: at T+60 they tie the CGMacros B2 baseline (+0.06 mg/dL, CI [-1.44, 1.55]) and at T+120 they are worse than B2 (-5.89 [-9.22, -2.58]). A T2D-native GBDT trained inside Shanghai passes the ADR 0004 gate at both horizons (+2.55 / +3.56 mg/dL vs B2), a gain the same size as in CGMacros, so the *procedure* generalises but the *served artifact* does not.

## Setup
- Data: ShanghaiT2DM, 100 participants (109 CGM periods; participant = `subject_id`). Meals from `nlp.data.shanghai_response` (rebuilt table is byte-identical to the committed `nlp/features/shanghai_nlp_response.csv`). g0 = last CGM reading at or before t0 (staleness <= 15 min); outcome = reading nearest t0+h within 15 min. Cohorts: T+60 3,045 meals, T+120 3,028 meals, 100 participants each.
- Profile (age, sex, BMI) from `Shanghai_T2DM_Summary.xlsx` (inside `~/Downloads/ShanghaiT2DM_raw/data.zip`); diabetes = T2D (109/109). Medication, insulin and HbA1c are **reported only** (`data_summary.json`): 56 of 109 periods list insulin (heuristic regex), 47 metformin, 373 of 3,060 meals have insulin within 30 min. None is a model input.
- Folds: the trial's 5 x 5 participant-grouped outer folds, shared by (a) and (b), so B0/B1/B2 are on identical test meals. For (a) the models are fixed, so the "folds" are test-only pseudo-folds (20 participants each) used for the fold-win criterion.
- (a) Baselines use CGMacros cohort constants: B0 = mean glucose at T+h (155.93 / 151.48 mg/dL), B2 = g0 + CGMacros mean excursion (+30.96 / +26.54). B1 = persistence. Arms: `served_full` (served model, macros NaN, profile from Shanghai); `served_full_noprofile` (diagnostic, profile also NaN); `retrained_g0_hour` and `retrained_g0_hour_profile` (CGMacros, no macros, trial settings: CGM + 3 meter-noise rows per meal, config chosen by the trial's grouped inner CV on CGMacros only). `b2_sh` (B2 with Shanghai's own mean excursion; an oracle that uses test data) is an extra diagnostic.
- (b) Nested CV in Shanghai, CGM g0 only (no noise augmentation). Feature sets g0_hour, g0_hour_profile (age, sex, BMI; dm_band dropped as constant), and both plus `dish_score`. `dish_score` = ridge (dish multi-hot with >= 10 training meals, grams, 2 flags; no g0/hour/profile) on the rise at the horizon, out-of-fold inside every training set (from `nlp/eval_dish_identity.py`). Set and hyper-parameters chosen in inner folds only (trial GRID). S1 = selection among the two text-free sets (the arm without text); S2 = selection among all four.
- Permutation control (200 per arm, p floor 0.005): within-participant shuffle of the rise (y - g0); selected configurations held fixed. The trial's macro permutation is impossible (no macros). For the transfer arms the models are fixed and only the rise is shuffled, so p tests whether predictions track within-participant rise; it says nothing in favour of an arm whose gain is negative (all transfer arms show p = 0.005 even with negative gain). Gate = positive gain, >= 4/5 fold wins in every repeat, participant-bootstrap CI (2,000) excluding 0, permutation p <= 0.05.

## Main results
Gain = B2 MAE - arm MAE (mg/dL, positive is better). "Gate" is vs B2; for transfer arms every row fails at least one criterion.


### Transfer T+60

| Arm | n people / meals | MAE | B2 MAE | Gain vs B2 [95% CI] | Folds won | Perm p | Gate | Zone A median (IQR), arm vs B2 |
|---|---|---|---|---|---|---|---|---|
| served_full | 100 / 3045 | 32.63 | 32.68 | +0.06 [-1.44, 1.55] | 12/25 | 0.005 | FAIL | 55.0 (26.6) vs 54.9 (22.8) |
| served_full_noprofile | 100 / 3045 | 37.65 | 32.68 | -4.97 [-7.46, -2.48] | 1/25 | 0.005 | FAIL | 50.0 (33.9) vs 54.9 (22.8) |
| retrained_g0_hour | 100 / 3045 | 31.35 | 32.68 | +1.34 [0.02, 2.75] | 19/25 | 0.005 | FAIL | 60.3 (19.1) vs 54.9 (22.8) |
| retrained_g0_hour_profile | 100 / 3045 | 32.70 | 32.68 | -0.02 [-2.14, 2.11] | 13/25 | 0.005 | FAIL | 60.0 (22.0) vs 54.9 (22.8) |

Baselines (CGMacros constants): B0 41.04, B1 38.09, B2 32.68 MAE; B0 = 155.93 mg/dL, B2 excursion = +30.96 mg/dL.

Gain of each arm vs the other baselines:

| Arm | vs B1 [CI] | vs B0 [CI] | vs B2-Shanghai-oracle [CI] | MAE RMSE | mean error (pred-y) |
|---|---|---|---|---|---|
| served_full | +5.47 [2.94, 8.15] | +8.41 [6.13, 10.60] | -0.35 [-1.95, 1.20] | 32.63 / 41.62 | -5.7 |
| served_full_noprofile | +0.44 [-1.57, 2.46] | +3.39 [0.31, 6.67] | -5.38 [-7.67, -3.05] | 37.65 / 49.56 | -27.8 |
| retrained_g0_hour | +6.75 [3.86, 9.78] | +9.69 [7.21, 12.19] | +0.93 [-0.67, 2.47] | 31.35 / 39.79 | +3.0 |
| retrained_g0_hour_profile | +5.39 [1.75, 9.01] | +8.34 [5.20, 11.45] | -0.43 [-2.85, 1.94] | 32.70 / 40.90 | +8.7 |

### Transfer T+120

| Arm | n people / meals | MAE | B2 MAE | Gain vs B2 [95% CI] | Folds won | Perm p | Gate | Zone A median (IQR), arm vs B2 |
|---|---|---|---|---|---|---|---|---|
| served_full | 100 / 3028 | 45.23 | 39.34 | -5.89 [-9.22, -2.58] | 0/25 | 0.005 | FAIL | 47.0 (34.3) vs 50.0 (19.9) |
| served_full_noprofile | 100 / 3028 | 49.33 | 39.34 | -9.99 [-13.86, -6.65] | 1/25 | 0.005 | FAIL | 39.6 (28.3) vs 50.0 (19.9) |
| retrained_g0_hour | 100 / 3028 | 37.59 | 39.34 | +1.74 [0.36, 3.13] | 21/25 | 0.005 | FAIL | 49.6 (17.8) vs 50.0 (19.9) |
| retrained_g0_hour_profile | 100 / 3028 | 38.65 | 39.34 | +0.69 [-1.43, 2.76] | 14/25 | 0.005 | FAIL | 50.0 (19.3) vs 50.0 (19.9) |

Baselines (CGMacros constants): B0 43.55, B1 44.91, B2 39.34 MAE; B0 = 151.48 mg/dL, B2 excursion = +26.54 mg/dL.

Gain of each arm vs the other baselines:

| Arm | vs B1 [CI] | vs B0 [CI] | vs B2-Shanghai-oracle [CI] | MAE RMSE | mean error (pred-y) |
|---|---|---|---|---|---|
| served_full | -0.32 [-4.43, 3.45] | -1.68 [-4.05, 0.76] | -5.81 [-9.12, -2.47] | 45.23 / 57.56 | -10.7 |
| served_full_noprofile | -4.42 [-7.71, -1.41] | -5.78 [-9.03, -2.48] | -9.91 [-13.69, -6.37] | 49.33 / 65.12 | -41.1 |
| retrained_g0_hour | +7.31 [4.94, 9.64] | +5.95 [3.69, 8.29] | +1.83 [0.48, 3.20] | 37.59 / 49.03 | -4.2 |
| retrained_g0_hour_profile | +6.26 [3.12, 9.40] | +4.90 [2.24, 7.76] | +0.77 [-1.37, 2.84] | 38.65 / 48.88 | +2.5 |

### Within Shanghai T+60

| Arm | n people / meals | MAE | B2 MAE | Gain vs B2 [95% CI] | Folds won | Perm p | Gate | Zone A median (IQR), arm vs B2 |
|---|---|---|---|---|---|---|---|---|
| S1_nested_no_text | 100 / 3045 | 29.81 | 32.36 | +2.55 [1.52, 3.55] | 25/25 | 0.005 | PASS | 62.0 (25.9) vs 55.9 (23.2) |
| S2_nested_with_text | 100 / 3045 | 29.83 | 32.36 | +2.53 [1.51, 3.53] | 25/25 | 0.005 | PASS | 62.0 (26.5) vs 55.9 (23.2) |
| fixed_g0_hour | 100 / 3045 | 29.79 | 32.36 | +2.57 [1.58, 3.57] | 25/25 | n/a | PASS | 63.0 (27.3) vs 55.9 (23.2) |
| fixed_g0_hour_profile | 100 / 3045 | 29.76 | 32.36 | +2.60 [1.42, 3.79] | 23/25 | n/a | PASS | 61.6 (25.8) vs 55.9 (23.2) |
| fixed_g0_hour_text | 100 / 3045 | 29.84 | 32.36 | +2.52 [1.56, 3.49] | 25/25 | n/a | PASS | 61.8 (26.7) vs 55.9 (23.2) |
| fixed_g0_hour_profile_text | 100 / 3045 | 29.71 | 32.36 | +2.65 [1.46, 3.81] | 24/25 | n/a | PASS | 61.3 (24.1) vs 55.9 (23.2) |

Baselines inside Shanghai folds: B0 41.57, B1 38.09, B2 32.36 MAE.  Selection frequency: {"s1": {"g0_hour": 16, "g0_hour_profile": 9}, "s2": {"g0_hour": 13, "g0_hour_profile": 8, "g0_hour_text": 2, "g0_hour_profile_text": 2}}
- text_increment_S2_vs_S1: gain -0.03 CI [-0.08, 0.03], folds 1/25, perm p 0.9900497512437811, passes False
- text_increment_fixed_g0_hour_profile_text_vs_g0_hour_profile: gain +0.05 CI [-0.04, 0.14], folds 12/25, perm p None, passes False
- text_increment_fixed_g0_hour_text_vs_g0_hour: gain -0.05 CI [-0.15, 0.05], folds 9/25, perm p None, passes False

### Within Shanghai T+120

| Arm | n people / meals | MAE | B2 MAE | Gain vs B2 [95% CI] | Folds won | Perm p | Gate | Zone A median (IQR), arm vs B2 |
|---|---|---|---|---|---|---|---|---|
| S1_nested_no_text | 100 / 3028 | 35.92 | 39.48 | +3.56 [2.20, 4.94] | 25/25 | 0.005 | PASS | 51.9 (15.9) vs 50.0 (20.4) |
| S2_nested_with_text | 100 / 3028 | 35.86 | 39.48 | +3.62 [2.27, 4.99] | 25/25 | 0.005 | PASS | 52.2 (16.6) vs 50.0 (20.4) |
| fixed_g0_hour | 100 / 3028 | 36.25 | 39.48 | +3.23 [1.98, 4.53] | 23/25 | n/a | PASS | 50.8 (18.3) vs 50.0 (20.4) |
| fixed_g0_hour_profile | 100 / 3028 | 35.83 | 39.48 | +3.65 [2.30, 5.06] | 25/25 | n/a | PASS | 52.5 (17.6) vs 50.0 (20.4) |
| fixed_g0_hour_text | 100 / 3028 | 36.09 | 39.48 | +3.39 [2.16, 4.62] | 24/25 | n/a | PASS | 50.7 (19.6) vs 50.0 (20.4) |
| fixed_g0_hour_profile_text | 100 / 3028 | 35.63 | 39.48 | +3.85 [2.50, 5.21] | 25/25 | n/a | PASS | 52.8 (18.0) vs 50.0 (20.4) |

Baselines inside Shanghai folds: B0 44.31, B1 44.91, B2 39.48 MAE.  Selection frequency: {"s1": {"g0_hour": 5, "g0_hour_profile": 20}, "s2": {"g0_hour_profile_text": 14, "g0_hour_profile": 7, "g0_hour": 3, "g0_hour_text": 1}}
- text_increment_S2_vs_S1: gain +0.07 CI [-0.04, 0.19], folds 11/25, perm p 0.14925373134328357, passes False
- text_increment_fixed_g0_hour_profile_text_vs_g0_hour_profile: gain +0.20 CI [0.06, 0.36], folds 20/25, perm p None, passes True
- text_increment_fixed_g0_hour_text_vs_g0_hour: gain +0.16 CI [-0.01, 0.33], folds 18/25, perm p None, passes False


## Reading the tables
- **Served model (a).** T+60: no gain (12/25 folds). T+120: loses to B2 in 25/25 folds. Removing the profile makes it much worse (bias -28 / -41 mg/dL), so it leans on age, BMI and the diabetes band, and Shanghai (age 60 vs 48, BMI 24 vs 31, 44% vs 64% female) lies outside the training profile range (`distribution_shift.json`). All-NaN macros were never seen in training.
- **Retrained, macro-free (a).** `g0_hour` beats B2 by +1.34 [0.02, 2.75] (T+60) and +1.74 [0.36, 3.13] (T+120), but **fails the gate on fold wins** (min 3/5 in one repeat at both horizons) and its T+60 CI lower bound is 0.02. Against the Shanghai-recalibrated oracle B2 the gain is +0.93 [-0.67, 2.47] / +1.83 [0.48, 3.20] at T+60 / T+120, so part of it is cohort calibration of the excursion. Adding profile removes the gain.
- **Zone A.** Per-participant median (IQR) is in the tables. The served model is not better than B2 at either horizon; native S1 is higher than B2 at T+60 (62.0 vs 55.9) and slightly at T+120 (51.9 vs 50.0).
- **Within Shanghai (b).** Both nested arms pass at both horizons (25/25 folds, CI above 0, p = 0.005). Text feature: no benefit at T+60 (S2 vs S1 -0.03 [-0.08, 0.03]); at T+120 +0.07 [-0.04, 0.19] (fails), and the fixed profile+text set beats fixed profile by +0.20 [0.06, 0.36] (20/25 folds, chosen after seeing four comparisons, not replicated by the nested procedure). Conclusion: the dish text adds at most 0.2 mg/dL; treat as no gain. Recommended arm: S1 (text-free). Permutation count for S2 uses the like-for-like pipeline (fixed ridge alpha) observed gain (T+120 3.59 vs 3.62).
- **Sensitivity** (excluding meals with insulin within 30 min or the next meal within the horizon; 93 participants, same predictions): `retrained_g0_hour` +1.72 [0.32, 3.17] / +1.45 [-0.14, 2.97]; `served_full` +0.55 [-1.05, 2.18] / -6.24 [-10.26, -2.64]; native S1 +2.55 [1.45, 3.74] / +3.23 [1.81, 4.79]. Conclusions unchanged.

## What this means for the paper
The generalisation claim must **not** say the served forecaster works on Asian T2D patients. It was trained on 45 mostly non-diabetic US participants; on 100 Chinese T2D patients with macros missing it gives no gain at T+60 and is worse than the baseline at T+120. State it as a limitation: the gate result holds for CGMacros under participant-grouped CV only. The supportable positive statement is weaker: a g0 + clock-hour GBDT retrained on T2D data beats B2 by about 2.6 (T+60) and 3.6 (T+120) mg/dL, comparable to the CGMacros gain, so the method (not the shipped model) is portable. For the app: do not present the served models' forecasts for T2D users as validated; the candidate fix (pool CGMacros + Shanghai, or serve a Shanghai-trained g0+hour model) needs its own ticket and a fresh cohort.

## Caveats
- Shanghai has no weighed macros, so (a) tests a degraded input (macros NaN), not the app's intended carb input. Whether the served model works with real carbs on T2D patients remains untested.
- CGM differs (15-min grid; outcome is the nearest reading within +/-15 min, not exact t0+h; staleness up to 15 min). Meal timestamps are logged, not first-bite times.
- Participants with two periods share one participant id; profile is per period. Shanghai meal times (insulin, next meal) are not modelled; 373 meals have insulin within 30 min (sensitivity above).
- Permutations: 200 for every arm (floor 0.005). A 1,000-permutation extension was started and stopped on instruction; not run. Transfer p-values are weak evidence (see Setup). The T+60 `retrained_g0_hour` result is a near miss and was one of four transfer arms, so no multiplicity correction is claimed.
- `dish_score` uses the same text features as the dish-identity trial (grams, flags, dish names); the dish parser keeps only weighted items (7 meals have no parsed dish).
- Raw data, pickles and models stay outside git (`/tmp/shanghai-external-work`).

## Reproduce
```
python -m nlp.data.shanghai_response --data-dir ~/Downloads/ShanghaiT2DM --output /tmp/shanghai-external-work/response.csv
python -m forecasting.evaluate.shanghai_external prep
python -m forecasting.evaluate.shanghai_external transfer --h 60   # and 120
python -m forecasting.evaluate.shanghai_external within-cv --workers 3
python -m forecasting.evaluate.shanghai_external within-perm --n 200
python -m forecasting.evaluate.shanghai_external report && python -m forecasting.evaluate.shanghai_external tables
```
Compute: <= 3 worker processes, single-threaded BLAS/LightGBM; total wall about 45 min.
