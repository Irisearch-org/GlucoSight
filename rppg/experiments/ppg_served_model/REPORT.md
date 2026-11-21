# PPG → glucose: which model to serve (ticket #16)

## Decision

**Serve `ridge_handcrafted_nested`** (Ridge on 7 handcrafted PPG features, alpha tuned by nested CV).
It has the lowest held-out MAE among the non-baseline candidates (14.69 mg/dL vs 14.55 for the mean).
**It does not beat the mean and fails the ADR 0004 gate on all four criteria.** Serve it only as a
non-validated reference estimator, not as a glucose claim.

No PPG model tested (handcrafted or PaPaGei-S embeddings) beats the training-fold mean on the primary protocol.

## Protocol

- Data: `rppg/PPG_Dataset/RawData`, 67 recordings, **23 subjects**, 10 s at 2190 Hz. Glucose 88–183 mg/dL.
- Primary CV: GroupKFold(5) by subject. Effective n = 23 held-out people. No meal/recording-level split.
- Baseline (B-mean): training-fold mean, same folds. Ridge and RF on handcrafted features are the comparison rows.
- Hyperparameters tuned only inside training folds (inner GroupKFold(4)).
- Gate (ADR 0004): MAE gain > 0; ≥ 4/5 fold wins; subject-bootstrap 95% CI of the gain excludes 0 (2000 resamples); label-permutation p ≤ 0.05 (subject-level, re-running nested tuning per permutation).
- Clarke: `utils/clarke_grid.py`; per-subject zone-A median/IQR (min_meals=2), stratified by reference range.

## Primary results (native 2190 Hz, 5-fold, n = 23 people)

| Candidate | MAE | Gain vs mean | Fold wins | Bootstrap 95% CI (gain) | Permutation p | Per-subject MAE median [IQR] | Clarke A (pooled) | Gate |
|---|---|---|---|---|---|---|---|---|
| **B-mean** | **14.55** | – | – | – | – | 13.94 [10.99, 16.63] | 83.6 | – |
| ridge_handcrafted_alpha10 | 15.30 | −0.75 | 1/5 | [−2.22, 0.57] | 0.396 (1000) | 13.64 [10.53, 17.51] | 83.6 | FAIL |
| **ridge_handcrafted_nested (served)** | **14.69** | **−0.14** | **2/5** | **[−0.52, 0.12]** | **0.539 (1000)** | 13.95 [10.96, 16.41] | 82.1 | **FAIL** |
| rf_handcrafted_nested | 14.84 | −0.29 | 3/5 | [−1.80, 1.28] | 0.363 (200) | 14.78 [9.59, 17.02] | 80.6 | FAIL |
| papagei_ridge_nested | 14.78 | −0.24 | 3/5 | [−1.21, 0.48] | 0.534 (1000) | 13.46 [11.67, 15.90] | 83.6 | FAIL |
| papagei_pca10_rf_nested | 14.94 | −0.39 | 3/5 | [−1.84, 0.88] | 0.503 (200) | 14.31 [10.69, 16.86] | 82.1 | FAIL |

Gate criteria for the served model: MAE gain > 0 **fail** (−0.14); fold wins ≥ 4/5 **fail** (2/5); CI lower bound > 0 **fail** (−0.52); permutation p ≤ 0.05 **fail** (0.539).

Served model fold detail (baseline → model MAE): fold 1 13.87→13.85 (win), fold 2 15.59→15.52 (win), fold 3 15.74→16.37 (loss), fold 4 14.32→14.32 (loss), fold 5 13.19→13.31 (loss). Tuned alpha: 1000, 1000, 100, 1000, 1000.

### Clarke (served model, out-of-fold)
- Pooled zone A 82.1%; per-subject zone A median 100%, IQR 33.3 (min 50, n = 22 participants; 1 excluded with < 2 meals).
- By reference range (mg/dL): [0,100) n = 9 → zone A 22.2%; [100,140) n = 53 → 98.1%; [140,180) n = 3 → 33.3%; [180,250) n = 2 → 0%.
- Zone A is not informative here: the mean baseline already sits at 83.6%. The low-glucose and high-glucose strata are where the model fails.

### Residuals (served model)
- Signed residual mean +0.5 mg/dL; |residual| q80 20.3, q90 24.7 mg/dL.
- MAE by SQI tertile: 12.7 / 14.1 / 17.2 (n 22/22/23). Spearman(|residual|, SQI) = 0.012 (p = 0.92): SQI does not predict error.
- Saved: `rppg/models/reports/ppg_served_residuals.csv` (out-of-fold, all 67 rows, with fold id).

## Sensitivity analyses (not gated; no permutation test)

- **Leave-one-subject-out (LOSO, 23 folds)**, mean baseline MAE 14.71: ridge_nested 14.59; rf_nested 14.44; papagei_ridge 14.72; **papagei_pca10_rf 14.07 (−0.64 vs mean)**. The PaPaGei+PCA-RF arm is the only one that beats the mean under LOSO, but its primary 5-fold result is worse (14.94) and it was not gated, so this is an exploratory signal, not a finding.
- **30 Hz (deployment rate)**: papagei_ridge 15.07 (−0.52, 3/5 wins, CI [−2.85, 1.20]); **papagei_pca10_rf 13.72 (+0.83, 4/5 wins, CI [−0.89, 2.55])**. Positive point estimate, CI includes 0, and this arm was chosen after seeing the native results: it cannot be claimed. Handcrafted features are not evaluated at 30 Hz (README: `hrv_rmssd_ms` available for only 13/67 recordings).

## Why this model is served

1. The ticket's rule is lowest held-out MAE among non-baseline candidates → ridge_handcrafted_nested (14.69) is lowest; the others are 14.78–15.30.
2. ADR 0005's fallback says: serve PaPaGei-S only if it passes the ADR 0004 gate; otherwise ridge on existing features. No candidate passes the gate, so the fallback also gives ridge on existing features. **The two rules agree here.** (Rule conflict noted in ADR 0005 text; not triggered.)
3. Ridge is the simplest model and is the most stable choice among candidates with similar MAE.

## Caveats

- Selection optimism: choosing the minimum-MAE candidate among 5 on the same CV folds is optimistic. The served model's own gate fails, so no positive claim rests on the selection; the ranking itself is optimistic.
- Effective n = 23 people; the bootstrap and permutation are at the subject level, so precision is limited.
- Small effects: primary-protocol candidates range from −0.14 to −0.75 mg/dL vs the mean (all negative). The signal in this dataset is weak.
- Domain shift: PPG_Dataset is a contact-PPG dataset, not finger-camera rPPG (the GlucoSight target). None of these results transfer without new validation.
- Permutation p-values for the two forest candidates use 200 permutations (min attainable p = 1/201; the observed p ≈ 0.36–0.50 are far from 0.05, so the decision is unaffected).
- Reconciling reference numbers: Ridge alpha=10 is 15.30 here, 15.04 in `rppg/models/reports/glucose_estimator.md`, 15.64 in `rppg/README.md`. Mean is 14.55 here, 14.57 in the estimator report, 14.54 in the README. Feature extraction differs between these code paths; not reconciled.
- Training-hygiene skill (`ml-training-recipes`) was read after the runs. Applicable points (grouped CV, classical models at n = 67, no leakage) were already met. Its deep-learning recipes (LR schedules, grad clipping, mixed precision) do not apply to ridge/RF.

## Deviations from the spec

1. **Interpreter**: `/tmp/ppg-venv` (system-site-packages + pyPPG 1.0.41 `--no-deps`, dotmap).
2. **Filter order**: resample 2190 → 125 Hz first, then bandpass 0.5–12 Hz (order 4). Filtering at 2190 Hz with pyPPG overflowed (|x| ~ 1e134). PaPaGei's pipeline runs at 125 Hz, so this matches its input.
3. **Permutation budget**: 1000 for ridge and PaPaGei-ridge; 200 for the two forest candidates. Forest nested tuning costs ~14 s per CV run; 1000 permutations per forest candidate would take ~1.5–2 h each on this machine (joblib stalled; a fork pool did 200 in ~20 min per candidate).
4. **Residuals location**: `results/` is gitignored, so the CSV is at `rppg/models/reports/ppg_served_residuals.csv`.
5. **PaPaGei-S weights**: Zenodo 13983110 licence field is null; PaPaGei code is BSD-3-Clause. Weights are gitignored and not committed.
6. **Random-forest leaf size**: nested (2, 5, 10), not fixed.

## Artifacts

- Served model (gitignored, main worktree): `/Users/auliamnaufal/Downloads/glucosight/rppg/models/weights/ppg_served_model.pt` (torch dict: ridge_handcrafted_nested refit on all 67 recordings / 23 subjects; round-trip raw-vs-cached max |diff| = 0.000000 mg/dL).
- PaPaGei-S weights (gitignored, main worktree): `/Users/auliamnaufal/Downloads/glucosight/rppg/models/weights/papagei_s.pt`.
- Out-of-fold residuals: `rppg/models/reports/ppg_served_residuals.csv`.
- Results JSON: `rppg/experiments/ppg_served_model/ppg_served_model.json`.
- Scripts: `rppg/experiments/ppg_served_model/{common,build_cache,candidates,evaluate,serve}.py`.
- Feature cache (not committed): `/tmp/ppg_served_cache/features.npz`.

## [NOT RUN]

- LOSO and 30 Hz sensitivity permutation tests: [NOT RUN] (time; reported as sensitivity only).
- Held-out test beyond the 5-fold CV: [NOT RUN] (no separate test split exists; n = 23).
- Clinical/finger-camera transfer: [NOT RUN] (no finger-camera data in this task).
