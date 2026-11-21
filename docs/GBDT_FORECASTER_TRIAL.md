# GBDT forecaster trial (ticket #17)

Two LightGBM forecasters (T+60, T+120 min) on the full CGMacros release (45 participants), evaluated against B0/B1/B2 on identical participant-grouped folds under ADR 0004.

**Verdict.** The pre-specified served design (`full`: g0 + meter/PPG noise draws + hour + macros + age/sex/BMI/diabetes band) **fails the ADR 0004 gate at both horizons** and does not beat B2. A post-hoc reduced design (`g0_hour_macros`, no participant profile) passes every gate criterion at both horizons, but it was selected on the same folds and is **exploratory only** until confirmed on held-out participants.

## Data and cleaning (script: `forecasting/evaluate/gbdt_forecaster.py`)
Source: `/Users/auliamnaufal/Downloads/CGMacros_full/CGMacros/` (45 folders; gaps 024, 025, 037, 040). Counts in `forecasting/gbdt_results/cleaning.json`.

| Rule (#12) | Result | #12 expected |
|---|---|---|
| Core composition all zero (calories, carbs, protein, fat) → drop | 44 | 44 |
| Atwater deviation >20% / >30% / >50% (computed on non-marker rows) | 107 / 85 / 58 | 107 / 82 / 58 |
| Atwater primary ±30% → drop | 85 | 82 |
| Fibre > carbs → fibre NaN, meal kept (raw rows) | 33 | 33 |
| Meals kept | 1,577 (45 participants) | — |

The ±30% count differs from #12 (85 vs 82). I did not tune the rule to match. ±20% and ±50% match exactly. Flagged for the spec owner.

- Target: exact Dexcom glucose at meal time + horizon (C2: no reading at or after t0 is used as input).
- g0: last Dexcom reading strictly before t0. Age of g0 median 1 min, max 735 min (diagnostic only; no cap applied).
- Cohort after target/g0 availability: 1,534 meals (T60), 1,531 (T120), all 45 participants.
- Diabetes band from HbA1c: normal 15, prediabetic 16, T2D 14 (HbA1c ≥ 6.5 includes one participant at exactly 6.5). #13 expected 18/14/13; I did not change the rule.

## Measurement conditions (g0 noise, #13 design)
- **CGM**: no added noise (clean-sensor condition, the gate condition).
- **Meter**: Contour-minus-Dexcom residuals. The Contour reading date cannot be identified from the release (best-date choice is inconsistent across readings), so residuals are pooled across all days at the same clock time. Pooled residuals: n = 1,254, mean −18.1, SD 48.8, MAD 28.4 mg/dL. **This is an upper bound on meter noise** and overstates it, so meter-condition numbers are pessimistic.
- **PPG**: [NOT RUN] — served PPG residuals (#16) are not available in any branch (`results/ppg_served_residuals.csv` absent). The PPG training rows and condition are therefore absent.
- Per-meal carb error (#18): [NOT RUN] — no #18 data located; carb sensitivity not run.

## Protocol
- Splits: GroupKFold(5) by participant (9 participants per fold). B0 = train mean; B1 = g0; B2 = g0 + mean train excursion. All baselines use the same folds and the same noisy g0 inputs.
- Tuning: nested GroupKFold(4) inside each outer training set; 16 configs (offset vs direct × num_leaves {4,8} × min_child_samples {20,60} × n_estimators {100,300}; lr 0.05).
- Training augmentation: CGM row + 3 meter-noise draws per meal. Evaluation: 5 draws per test meal for the noisy conditions.
- Gate (ADR 0004, constants copied from `rppg/models/glucose_estimator.py`): positive MAE gain; ≥4 of 5 fold wins; participant-cluster bootstrap 95% CI excludes 0 (2,000 reps); within-participant macro shuffle permutation p ≤ 0.05.
- Deviation: **200 permutations** (not 1,000) to fit runtime. p resolution is 1/201; the 0.05 threshold is unaffected.

## Results: pre-specified `full` design (CGM condition, n = 45 participants)

| Horizon | Meals | GBDT MAE / RMSE | B0 MAE | B1 MAE | B2 MAE | Gain vs B2 | Fold wins vs B2 | Bootstrap 95% CI (vs B2) | Permutation p | Gate vs B2 |
|---|---|---|---|---|---|---|---|---|---|---|
| T+60 | 1,534 | 31.77 / 41.15 | 38.41 | 37.91 | 31.58 | −0.20 | 3/5 | [−2.80, 2.35] | 0.005 | **FAIL** (gain, folds, CI) |
| T+120 | 1,531 | 31.74 / 42.53 | 37.30 | 36.50 | 32.14 | +0.40 | 3/5 | [−3.17, 3.77] | 0.005 | **FAIL** (folds, CI) |

Gate detail: vs B1 and B0 both horizons pass the gain, folds, and CI checks. The B2 comparison is the gate that matters: it is not beaten.

Clarke error grid (CGM condition), per-participant Zone A median (IQR) over 45 participants, min 5 meals:
- T+60: GBDT 60.0 (28.0) vs B2 58.6 (19.8). Pooled Zone A: GBDT 56.8% vs B2 54.7%.
- T+120: GBDT 55.6 (25.6) vs B2 57.5 (22.5). Pooled Zone A: GBDT 55.1% vs B2 57.2%.
- Zone A by reference range (GBDT, T+60): 0–100 mg/dL 3.7% (n = 109); 100–140 46.2% (n = 572); 140–180 88.6% (n = 464); 180–250 53.9% (n = 312); ≥250 32.5% (n = 77). T+120 in `T120.json`.

T2D subgroup (n = 14, CGM condition, descriptive): T+60 GBDT MAE 38.2 vs B2 41.1; T+120 44.5 vs 45.8. Per-participant Zone A median 62.8 (IQR 19.9), T+60.

Error by g0 source (MAE, mg/dL, T+60 / T+120): CGM GBDT 31.8 / 31.7 vs B2 31.6 / 32.1. Meter GBDT 35.2 / 33.0 vs B2 50.0 / 51.2 (meter noise is an upper bound; see above). PPG: [NOT RUN].

## Ablation (CGM, same folds; `forecasting/gbdt_results/ablation.json`)

| Feature set | MAE T+60 | MAE T+120 | Gain vs B2 T+60 / T+120 | Fold wins T+60 / T+120 |
|---|---|---|---|---|
| g0 + hour (+ src) | 31.25 | 32.01 | +0.33 / +0.13 | 2 / 2 |
| g0 + hour + macros | **29.30** | **29.66** | **+2.28 / +2.49** | 5 / 5 |
| full (+ profile) | 31.77 | 31.74 | −0.20 / +0.40 | 3 / 3 |

Adding the participant profile (age, sex, BMI, diabetes band) makes the forecast about 2.5 mg/dL worse than macros-only. With 45 participants, per-participant constant features are likely to overfit participant identity. This is a finding about the design, not a tuning issue.

## Post-hoc exploratory gate: `g0_hour_macros` (`forecasting/gbdt_results/posthoc_g0_hour_macros.json`)
The feature set was chosen from the ablation on the same folds, so this is **not a confirmatory claim**. Fold params are the ablation's.

| Horizon | MAE GBDT / B2 | Gain vs B2 | Bootstrap 95% CI | Fold wins | Permutation p | vs B1 / B0 |
|---|---|---|---|---|---|---|
| T+60 | 29.30 / 31.58 | +2.28 | [0.71, 3.83] | 5/5 | 0.005 | pass / pass |
| T+120 | 29.66 / 32.14 | +2.49 | [0.92, 4.18] | 5/5 | 0.005 | pass / pass |

Every ADR 0004 criterion passes at both horizons. Permutation null mean is +0.24 (T+60) and +0.36 (T+120) mg/dL, so the observed gain is well above shuffled-macro models.

## Recommendation
1. **Do not serve the pre-specified `full` design.** It fails the gate, and its gain over B2 is not positive. Its trained artifacts exist, but they should not be presented as an improvement.
2. **Candidate to confirm: `g0_hour_macros`.** The post-hoc gate passes, but the feature set was selected on the same folds. Confirm on held-out participants (a fresh cohort or a pre-registered holdout) before any serving claim.
3. Until confirmation, B2 (g0 + mean excursion) is the reference baseline; no GBDT model is claimed to beat it.

## Artifacts
- Script: `forecasting/evaluate/gbdt_forecaster.py` (run command in its `main()`; post-hoc via `--posthoc-macros`).
- Results (committed): `forecasting/gbdt_results/` (`summary.json`, `T60.json`, `T120.json`, `ablation.json`, `cleaning.json`, `contour_pool.json`, `posthoc_g0_hour_macros.json`).
- Per-meal predictions (not committed, 1.1 MB gzip): `/tmp/gbdt_predictions.csv.gz`.
- Final models (pre-specified `full` design; written to `forecasting/models/` in the main worktree, untracked and not committed): `/Users/auliamnaufal/Downloads/glucosight/forecasting/models/gbdt_cgmacros_T60.txt` and `gbdt_cgmacros_T120.txt` (+ `.json` metadata). Macros-only final model: [NOT RUN] pending confirmation.

## Caveats and [NOT RUN]
- [NOT RUN] PPG condition (#16 artifact absent).
- [NOT RUN] per-meal carb error sensitivity (#18 data not located).
- Meter noise is pooled over all days (Contour date unidentifiable): SD 48.8 mg/dL is an upper bound; meter-condition numbers are pessimistic.
- ±30% Atwater count 85 vs #12's 82; T2D count 14 vs #13's 13.
- Permutations 200 (not 1,000).
- Post-hoc macros gate is selected on test folds; expect optimism.
- Clarke per-participant strata use min 3 meals per stratum and may be sparse.
