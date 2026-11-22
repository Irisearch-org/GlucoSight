
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
