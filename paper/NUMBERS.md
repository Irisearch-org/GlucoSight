# GlucoSight paper: numbers and sources (single source of truth)

Framing (confirmed with the user): **what a smartphone can and cannot add to post-meal glucose forecasting without a CGM**. It has three findings, all evaluated on people held out from training, and every finding is compared against trivial baselines. Venue format: NeurIPS-style, 9 pages + checklist, anonymous.

## Data
| Dataset | Use | n | Licence |
|---|---|---|---|
| CGMacros v1.0.0 (PhysioNet, DOI 10.13026/3z8q-x658) | forecaster train/eval | 45 participants (data paper: 15 healthy, 16 prediabetes, 14 T2D), 1,706 meal events → 1,534 (T+60) / 1,531 (T+120) after cleaning | CC BY-NC-SA 4.0 |
| ShanghaiT2DM (figshare; Zhao et al. Sci Data 2023) | external check of forecaster | 100 T2D, 3,045 / 3,028 meals; median g0 127.8, median 60-min rise 23.4 mg/dL | CC BY 4.0 |
| PPG_Dataset / MUST (contact fingertip PPG sensor, ≈2 kHz; do not quote the exact rate: Mendeley says 2175 Hz, repo loader 2190 Hz) | PPG glucose | 23 subjects, 67 recordings, glucometer label, 88–183 mg/dL (SD 18.7) | Mendeley Data record (kermani2023dataset) |
| BUT PPG v2.0.0 (PhysioNet DOI 10.13026/tn53-8153) | phone-camera PPG glucose | 30 Hz RGB smartphone; glycaemia for 38 subjects; finger subset 18 subjects, 1,246 still recordings, 35 sessions; 17 with two glucose values | CC BY 4.0 |
| BUT PPG v1 (Brno) | heart rate vs ECG | 12 subjects, 48 recordings | CC BY 4.0 |
| Wibisono / In-TFK Indonesian traditional food | dish identification | dev 164 / test 317 scored (of 329); the released image folders have 35 classes (the Wibisono papers describe 34 food types; say "35 kelas pada rilis yang digunakan") | CC BY 4.0 (article) / CC BY-SA 4.0 (Kaggle) |
| TKPI 2017 (Kemenkes) | dish table | 710 rows built (568 dish, 142 ingredient) | not openly licensed (cited, not redistributed) |

## Cleaning (CGMacros; #12)
Drop meal if Atwater (4C+4P+9F)/kcal outside [0.7,1.3]: 85 of 1,706 dropped (sensitivity ±20% 107, ±50% 58); fibre > carbs → fibre missing (33 meals, 5 participants); "Amount Consumed" unused (count-coded for 8 participants); 44 all-zero meal markers dropped; no clipping/imputation. The dictionary's range gate would have flagged 404 meals (23.7%).

## Forecaster (#17; nested CV, outer GroupKFold(5) × 5 repeats by participant; inner selection of feature set + LightGBM hyperparameters; 200 permutations, p floor 0.005)
Inputs: g0 (last CGM reading strictly before the meal), g0 source (meter/CGM, with ISO 15197:2013 noise for meter), carbs/protein/fat/fibre grams (NaN kept), hour sin/cos, age, sex, BMI, diabetes band. Targets: exact reading at meal + 60 / + 120 min.
Baselines: B0 population mean, B1 persistence, B2 persistence + CGMacros mean excursion.

| Source | Horizon | GBDT MAE | B2 MAE | Gain [95% CI] | Gate |
|---|---|---|---|---|---|
| CGM | T+60 | 28.78 | 31.68 | +2.90 [1.36, 4.42], folds won per repeat [5,5,4,5,4], p 0.005 | PASS |
| CGM | T+120 | 28.49 | 32.05 | +3.56 [1.62, 5.51], [5,4,5,4,5], p 0.005 | PASS |
| Meter (ISO 15197 sim) | T+60 | 29.69 | 32.60 | +2.92 [1.49, 4.30] | PASS |
| Meter | T+120 | 29.24 | 33.04 | +3.80 [1.79, 5.64] | PASS |
| PPG (sim: 113.35 + 0.0101·true, SD 4.20) | T+60 | 36.02 | 37.52 | +1.51 [−0.80, 4.05] | FAIL |
| PPG | T+120 | 32.86 | 35.51 | +2.65 [−0.42, 5.65] | FAIL |
All-baseline MAE (CGM g0): T+60 B0 38.72 / B1 37.91 / B2 31.68 / GBDT 28.78; T+120 B0 36.93 / B1 36.50 / B2 32.05 / GBDT 28.49. Gains vs B1 +9.13 [4.51, 13.53] / +8.01 [5.06, 11.14]; vs B0 +9.94 [6.55, 14.12] / +8.44 [4.82, 11.91].
T2D subgroup (14 participants) MAE: T+60 GBDT 34.44 vs B2 41.15; T+120 41.51 vs 45.81. T2D Clarke zone A per-participant median 63.7 (T+60).
**What the food information adds** (permutation = macros shuffled within participant, 200 draws): gain with macros shuffled averages +1.24 (T+60; 95th pct 1.38) and +1.56 (T+120; 95th pct 1.68) vs observed +2.90 / +3.56, p = 0.005 at both. So about 1.7 / 2.0 mg/dL of the gain over B2 comes from the meal macros. Inner selection never chose a feature set without macros (g0_hour, g0_hour_profile: 0/25 at both horizons).
Clarke pooled (CGM g0, T+60): zone A 61.6%, A+B 98.0%, D 1.96%, E 0. Zone A by reference range (T+60): <100: 15.1%, 100–140: 59.0%, 140–180: 82.4%.
Clarke zone A, per-participant median (IQR): T+60 GBDT 66.4 (20.4) vs B2 58.7 (19.7); T+120 62.6 (18.3) vs 56.5 (23.3). Passes vs B1 and B0 too.
Pre-specified (non-nested, full feature set) design FAILED: T+60 −0.20 [−2.80, 2.35]; T+120 +0.40 [−3.17, 3.77] (profile overfits with 45 people). A post-hoc reduced set passed on the same folds, which motivated the nested procedure; the nested result is the reported one.
Feature-set selection unstable: T+60 g0_hour_macros 17/25 vs full 8/25; T+120 full 16/25. Served refit chose `full` at both horizons.
Carb-error sensitivity (dish-ID errors resampled): ΔMAE +0.01 (T+60), +0.14 (T+120); gain stays significant.
90th-percentile abs error (band): T+60 meter 62.8 / CGM 60.8 / PPG 77.9; T+120 64.0 / 62.6 / 69.1. Coverage of the single band: 74–80% when g0 ≥ 180, 78–84% for T2D, <80% for 9–11 of 45 participants (#30).

## Excursion + monotone carb (#30; pre-registered rule; identical folds)
T+60: 28.69 vs 28.78, paired gain +0.09 [−0.07, 0.27], 16/25 folds; T+120: 28.29 vs 28.49, +0.20 [−0.06, 0.53], 14/25. Not adopted. Second arm (carbs +1, fibre −1): +0.14 / +0.06.

## External check, ShanghaiT2DM (#29)
| Arm | T+60 gain vs B2 [CI] | T+120 | Gate |
|---|---|---|---|
| Served CGMacros model, macros missing | +0.06 [−1.44, 1.55] | −5.89 [−9.22, −2.58] | FAIL |
| CGMacros g0 + hour model | +1.34 [0.02, 2.75] (19/25 folds) | +1.74 [0.36, 3.13] (21/25) | FAIL (fold wins) |
| GBDT trained within Shanghai (nested, no text) | +2.55 [1.52, 3.55], 25/25, p 0.005 | +3.56 [2.20, 4.94], 25/25, p 0.005 | PASS |
| + text dish-identity score | +0.00 increment | +0.06 increment | no gain |
Shanghai B2 MAE 32.36 / 39.48 (within folds); GBDT MAE 29.81 / 35.92. Zone A median 62.0 vs 55.9 (T+60); 51.9 vs 50.0 (T+120).

## Dish identification (#18, #19)
Pipeline: Gemini `gemini-3.5-flash-lite` names dishes (Indonesian prompt, open vocabulary) → embedding matcher (`gemini-embedding-001`) → TKPI dish-table row. 35-row evaluation table (9 exact, 11 approximate, 15 computed).
Test (317 photos): top-1 61.5% (56.1–66.7), family 69.1% (63.8–73.9), abstain 20.2%, false match 18.3% at production 0.75; difflib matcher 60.3 / 65.9 / 16.4 / 23.3.
Vs MobileNetV2 on its 10-class overlap (88 test photos), paired bootstrap: family level embedding +6.8 pp [+1.1, +13.6], difflib −3.4 [−12.5, +5.7]; exact dish level −20.5 [−31.8, −9.1] / −23.9. MobileNet test: 92.0% family on overlap. MobileNetV2 food-10 held-out top-1 88.1% (1051/1193), top-3 98.0, macro-F1 0.874.
VLM self-reported confidence unusable: accuracy plateaus at 74–76% even at ≥ 0.95.
Operating point (chosen on dev: minimise false match subject to coverage ≥ 60%): similarity ≥ 0.80 and top1−top2 margin ≥ 0.05. Dev: coverage 60.4%, 93.9% correct when answered, false match 3.7%. Test: coverage 63.4%, 89.6% correct when answered, false match 6.6% (vs 18.3% at 0.75, coverage 79.8%).
Carb error from identification (test, matched): median |err| 0.0 g, 87.4% within 10 g (portion error not measured; 27/35 serving weights assumed).
Caveat: the operating point was calibrated on the 35-row table; the served table has 710 rows; re-calibration pending.

## PPG glucose (#16, #27, #28)
PPG_Dataset (23 subj, GroupKFold(5) by subject, nested):
| Model | MAE | Gain vs mean (14.55) [CI] | folds | perm p |
|---|---|---|---|---|
| Ridge, 7 handcrafted (served) | 14.69 | −0.14 [−0.52, 0.12] | 2/5 | 0.54 |
| PaPaGei-S + ridge | 14.78 | −0.24 [−1.21, 0.48] | 3/5 | 0.53 |
| RF handcrafted | 14.84 | −0.29 [−1.80, 1.28] | 3/5 | 0.36 |
| PaPaGei-S PCA10 + RF | 14.94 | −0.39 [−1.84, 0.88] | 3/5 | 0.50 |
Served predictions: SD 4.2 vs truth SD 18.7; pred ≈ 113.4 + 0.010·true, r = 0.045; Clarke zone A by range: 98% (100–140), 22% (<100), 33% (140–180), 0% (180–250). 90% abs error 24.7 (shown as ±25).
BUT PPG v2 finger, still (18 subj, 5×5 subject-grouped, 1000 perms):
| Model | MAE | mean MAE | Gain [CI] | folds | perm p | r |
|---|---|---|---|---|---|---|
| 7 handcrafted + ridge | 22.15 | 22.29 | +0.15 [−0.88, +1.14] | 16/25 | 0.12 | −0.18 |
| PaPaGei-S + ridge | 22.25 | 22.29 | +0.04 [−0.54, +0.59] | 15/25 | 0.18 | −0.13 |
Within-person (17 two-session subjects): |Δglucose| median 10.8 (IQR 5.4–27.0); 8/17 < 10 mg/dL. Δ-ridge 17.8 vs carry-over 19.4, gain +1.6 [−1.4, +4.6], sign 12/17 (binomial p 0.14). Calibration (offset on session A + cross-subject model) worse than carry-over: ridge 20.9 vs 19.4 (gain CI [−3.1, −0.1]); PaPaGei 20.3. Model's predicted Δ SD 3.7 vs true 26.8.
Heart rate (BUT PPG v1, 12 subj): pooled MAE 3.76 bpm vs ECG (quality-filtered, 32 recordings); per-subject 1.27–30.85.

## Literature
External numbers and claims: use only what `CITATIONS.md` marks as verified, with its wording (e.g. IDF 20.4 M / 73.2%; Chu 2021 zone A 94.3%; GlucoFM-Bench uses 12 h context by default, TimesFM 2.5 RMSE 16.28 on CGMacros at 30 min; Ramesh trains on VitalDB and tests on MUST; PPG-Bench FMs ≈ random control on BUT PPG glucose). Cite with keys from `references.bib` only.

## Figure-derived numbers (from figures/make_figures.py; quote only in captions or with this exact definition)
- Fig. forecast (a): GBDT beats B2 for 35/45 participants (T+60) and 36/45 (T+120); median per-participant gain +2.50 / +4.37 mg/dL. Unweighted mean of per-participant MAE: 29.40 vs 32.64 (T+60), 29.41 vs 33.43 (T+120) (differs from pooled 28.78/31.68 and 28.49/32.05).
- Fig. forecast (b): Clarke grid, GBDT T+60, 1534 meals, predictions averaged over the 5 CV repeats: zone A 62.6%, A+B 98.0%. (Pooled over all 7670 repeat-level predictions instead: zone A 61.58%.)
- Fig. dish operating: old point 0.75 / margin 0: dev coverage 72.6%, false match 11.6%; test 79.8%, 18.3%. Chosen 0.80 / 0.05: dev 60.4%, 3.7%; test 63.4%, 6.6%. Similarity thresholds 0.60–0.95, margins 0 / 0.02 / 0.05.
- Fig. PPG (a): PPG_Dataset 67 recordings, 23 subjects; prediction SD 4.2 vs truth SD 18.7; r = 0.045.
- Fig. PPG (b): BUT PPG v2 finger, 1246 recordings, 35 sessions, 18 subjects (pooled r of repeat-averaged predictions is −0.23; the table's −0.18 is the mean of per-repeat r; quote −0.18).
- Fig. PPG (c): within-person Δ-ridge (leave-one-subject-out, 17 two-session subjects): predicted Δ SD 7.3 vs true Δ SD 27.6 (sample SD), r = 0.29. This is the Δ-ridge arm (MAE 17.8 vs carry-over 19.4). The "3.7 vs 26.8" SD pair belongs to the calibration arm (cross-subject model Δ).
