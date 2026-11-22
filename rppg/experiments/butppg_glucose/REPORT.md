# BUT PPG v2: can smartphone-camera PPG estimate glucose, and does it track change within a person? (tickets #27, #28)

## Answers

- **#27 (people never seen): No.** On the 18 finger-scan subjects (1,246 still recordings), ridge on the 7 served handcrafted features and PaPaGei-S + ridge are statistically indistinguishable from the training-fold mean (MAE 22.15 and 22.25 vs 22.29 mg/dL; both fail the ADR 0004 gate: bootstrap CIs include 0, permutation p 0.12 / 0.18, 1000 permutations each). Out-of-fold predictions correlate **negatively** with true glucose (r = -0.18 / -0.13). The random-forest primary arm was not finished (see "Not run").
- **#28 (within the same person): No usable signal, so per-user calibration is not supported.** Across 17 two-session finger subjects, the PPG delta-ridge predicts the change in glucose with MAE 17.8 vs 19.4 for "no change" (gain +1.6 mg/dL, 95% CI [-1.4, +4.6], sign right in 12/17, binomial p = 0.14). Calibration (offset from session A + the cross-subject model's A-to-B change) is **worse** than carry-over: ridge on handcrafted features 20.9 vs 19.4 MAE (CI of gain [-3.1, -0.1]); PaPaGei + ridge 20.3 vs 19.4. Carry-over ("glucose unchanged") is the best predictor of session B.

**What it means for the app.** Do not retrain the served PPG model on BUT PPG: a phone-camera dataset gives the same null as the contact-PPG data (ADR 0003 and 0005 stand; the served ridge is still mean-equivalent). Do not build a "calibrate the scan with your glucometer" feature on this evidence: the calibration step only adds noise to the user's own meter reading. This is a negative result with n = 17-18 people, so it excludes large effects, not small ones (see Caveats).

## Protocol

- Primary set: **finger, motion 0** (still), brightest channel (as `frontend/src/ppg.ts`: highest mean; the raw, non-inverted samples the app sends), 30 Hz, full 10 s window as one window. Labels in mg/dL (mmol/L x 18.016).
- Features: (a) the 7 served features from `rppg.features.extractor.extract_features(signal, 30)`; (b) PaPaGei-S 512-d embeddings via the 125 Hz path of `ppg_served_model/common.py::papagei_window`. PaPaGei's input is the **polarity-flipped** brightness signal (camera brightness falls at systole; the model was trained on upright PPG), chosen before running. Polarity-as-recorded is a sensitivity.
- Models: ridge (nested alpha), random forest (nested leaf), PaPaGei + ridge (nested alpha); the ridge/RF fits and grids are those of `ppg_served_model/candidates.py` (the ridge uses a numerically identical Gram-matrix implementation, max |diff| 6e-14 mg/dL against sklearn). Hyperparameters are tuned only inside training folds (inner subject-grouped 4-fold).
- Evaluation: outer **subject-grouped** 5-fold x 5 repeats (random subject-to-fold assignment per repeat); LOSO as sensitivity. Baseline = training-fold mean on identical folds. Recording-level and session-level (predictions averaged within a session, label = session glycaemia). Bootstrap over **subjects** (2000). Permutation: glycaemia profiles shuffled between subjects, full nested fit re-run per permutation, 1 CV repeat each (null is noisier than the 5-repeat observed value, which makes p slightly conservative).
- ADR 0004 gate, recording level: gain > 0; model wins at least 80% of outer folds (20 of 25, the 4-of-5 rule over repeats); subject-bootstrap 95% CI of the gain excludes 0; permutation p <= 0.05. Gain = baseline MAE minus model MAE (positive is better). The session-level gate uses the same fold-win count.
- #28 design: two-value subjects only; session features = mean over the session's still recordings; delta = B minus A (A = lower record numbers). Delta-ridge: leave-one-subject-out, no intercept, trained on both orderings (antisymmetric), alpha by inner leave-one-subject-out. Calibration: `pred_B = model(B) + (glucose_A - model(A))` with the model trained **without that subject** (LOSO over all 18 finger subjects), both orderings, 34 subject-orderings; **note that this equals carry-over + the model's own predicted A-to-B change**, so calibration beats carry-over only if the model tracks within-person change. Gate vs carry-over: gain > 0; calibration better in >= 80% of subjects (14/17); subject-bootstrap CI excludes 0; permutation p <= 0.05.

## Data

- **BUT PPG v2.0.0**, PhysioNet, DOI 10.13026/tn53-8153 (Nemcova et al.). **Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)**, open access, no credentialing. Raw files (CC BY, not committed) are in `/Users/auliamnaufal/Downloads/butppg-2.0.0/`, fetched from the PhysioNet open-data S3 mirror (`s3://physionet-open/butppg/2.0.0/`); every `*_PPG.dat/.hea` file was checked against the release's `SHA256SUMS.txt` (all 7,776 match).
- Recording: Xiaomi Mi9 or Huawei P20 Pro, 10 s, 30 Hz, three channels (R, G, B pixel means, 0–255; the app's samples are the same thing scaled 0–1). Glycaemia: fingerstick glucometer **Fora Diamond Mini** (dataset page); BP: Fora D40; SpO2: Gima pulse oximeter. One glycaemia value per session, not per recording.
- **Measurement site is a property of the subject, not of the recording.** Subjects 112–131 (20 people) are ear-only; 132–149 (18 people) are finger-only. The dataset page says "about half finger, half ear" but the 3,840 glycaemia-labelled recordings split by person, so "finger" = 18 people, not 38. Ear vs finger is therefore confounded with person (and with whatever cohort effects differ between the two groups).
- **Sessions are not documented.** The page says only that the database was "measured in two separate sessions in different time periods" (the two IRB approvals, 2018 and 2021). Within a subject, records 001–054 and 055–108 (001–042 / 043–096 for subjects 119, 121, 126, 145) carry different glycaemia values, so I define a session as (subject, glycaemia value) and order A before B by record number. Time gap and true chronological order are not stated anywhere I could find.
- Counts (`results/data_audit.json`): 3,840 labelled recordings, 38 subjects, 66.7–241.4 mg/dL (3.7–13.4 mmol/L × 18.016). Finger: 18 subjects, 1,318 still recordings (26–38 per session, 35 sessions); 17 have two distinct glycaemia values, subject 138 has one. Ear: 20 subjects, 1,370 still recordings; 15 two-value subjects; 113, 118, 120, 123, 131 have one value (123 and 131 were recorded twice with the same value). 32 two-value subjects in total (17 finger + 15 ear).
- Brightest channel: red in 100% of finger recordings (mean red 0.93 vs green 0.22, blue 0.17) and 98.3% of ear recordings. **So the "red-only" sensitivity is identical to the primary**; I report green as the channel sensitivity instead. 16% of finger recordings have more than 10% of frames at or above the app's clip level (250/255), which the served pipeline's `MAX_CLIP_FRACTION` would reject; the red channel is close to saturated in many finger recordings.
- Features are non-finite (the served pipeline would reject the capture) in 72 of the 1,318 finger-still recordings (mostly `hrv_rmssd_ms`, i.e. fewer than 3 usable beats): the primary set is **1,246 recordings** (18 subjects, 35 sessions). Every model in a comparison uses the same rows and the same folds.

## Results

(Generated by `make_tables.py` from `results/*.json`. Gain = baseline MAE minus model MAE in mg/dL. Folds won: out of 25 outer folds.)

### #27 primary (finger, still), recording level

| model | n subj / rec / sess | MAE | mean MAE | RMSE | r | gain | 95% CI | folds won | perm p (N) | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| hand_raw+ridge | 18 / 1246 / 35 | 22.15 | 22.29 | 34.69 | -0.181 | +0.15 | [-0.88, +1.14] | 16/25 | 0.1239 (1000) | FAIL |
| pap_inv+ridge | 18 / 1246 / 35 | 22.25 | 22.29 | 34.81 | -0.134 | +0.04 | [-0.54, +0.59] | 15/25 | 0.1798 (1000) | FAIL |

### #27 primary, session level (predictions averaged within session)

| model | MAE | mean MAE | RMSE | r | gain | 95% CI | perm p | gate |
|---|---|---|---|---|---|---|---|---|
| hand_raw+ridge | 21.63 | 22.07 | 33.66 | -0.279 | +0.45 | [-0.42, +1.40] | 0.0709 | FAIL |
| pap_inv+ridge | 21.32 | 22.07 | 33.51 | -0.249 | +0.76 | [-0.08, +1.74] | 0.033 | FAIL |

### Clarke (out-of-fold, recording level)

| model | zone A pooled % | per-subject A median [IQR width] | A by reference range (mg/dL): n, A% |
|---|---|---|---|
| hand_raw+ridge | 62.6 | 70.4 [61.0] | [0, 100) n=841 A=61.7; [100, 140) n=330 A=79.0; [180, 250) n=75 A=0.0 |
| pap_inv+ridge | 60.6 | 71.8 [67.3] | [0, 100) n=841 A=61.2; [100, 140) n=330 A=72.7; [180, 250) n=75 A=0.0 |
| mean baseline | 60.6 | 76.5 [83.3] | [0, 100) n=841 A=58.3; [100, 140) n=330 A=80.2; [180, 250) n=75 A=0.0 |

### tracking of OOF predictions

- hand_raw+ridge: {'r_between_subject_means': -0.449, 'r_within_subject_session_means': -0.277}; prediction SD 6.5 mg/dL
- pap_inv+ridge: {'r_between_subject_means': -0.365, 'r_within_subject_session_means': -0.231}; prediction SD 7.82 mg/dL

### #27 sensitivities (ungated)

| subset | features+model | cv | n subj/rec | MAE | mean MAE | gain | 95% CI | folds won | r | session gain [CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| finger_still | hand_inv+ridge | 5x5 | 18/1209 | 22.53 | 22.5 | -0.03 | [-0.96, +0.85] | 16/25 | -0.177 | +0.32 [-0.47, +1.14] |
| finger_still | hand_inv+rf | 5x5 | 18/1209 | 22.58 | 22.5 | -0.08 | [-2.88, +2.78] | 13/25 | 0.136 | +2.25 [+0.38, +4.28] |
| finger_still | hand_red+ridge | 5x5 | 18/1246 | 22.15 | 22.29 | +0.15 | [-0.88, +1.14] | 16/25 | -0.181 | +0.45 [-0.42, +1.40] |
| finger_still | hand_green+ridge | 5x5 | 18/1195 | 22.53 | 22.71 | +0.18 | [-0.72, +1.02] | 20/25 | -0.2 | +0.43 [-0.40, +1.23] |
| finger_still | pap_raw+ridge | 5x5 | 18/1246 | 22.1 | 22.29 | +0.19 | [-0.52, +0.84] | 18/25 | -0.105 | +0.90 [-0.05, +2.06] |
| finger_still | pap_red_inv+ridge | 5x5 | 18/1246 | 22.25 | 22.29 | +0.04 | [-0.54, +0.59] | 15/25 | -0.134 | +0.76 [-0.08, +1.74] |
| finger_still | hand_raw+ridge | LOSO | 18/1246 | 22.74 | 22.52 | -0.23 | [-1.29, +0.85] | 10/18 | -0.243 | +0.18 [-0.81, +1.21] |
| finger_all | hand_raw+ridge | 5x5 | 18/1764 | 22.01 | 22.18 | +0.17 | [-0.53, +0.97] | 15/25 | -0.212 | +0.35 [-0.36, +1.17] |
| ear_still | hand_raw+ridge | 5x5 | 20/1263 | 16.23 | 16.16 | -0.07 | [-0.17, +0.00] | 6/25 | -0.477 | -0.07 [-0.18, +0.01] |
| all | hand_raw+ridge | 5x5 | 38/3532 | 18.98 | 18.97 | -0.00 | [-0.31, +0.33] | 13/25 | -0.226 | +0.08 [-0.21, +0.39] |
| finger_still | pap_inv+ridge | LOSO | 18/1246 | 22.43 | 22.52 | +0.09 | [-0.56, +0.74] | 10/18 | -0.181 | +0.74 [-0.13, +1.72] |
| finger_all | pap_inv+ridge | 5x5 | 18/1764 | 21.99 | 22.18 | +0.19 | [-0.33, +0.71] | 15/25 | -0.148 | +0.69 [+0.05, +1.39] |
| ear_still | pap_inv+ridge | 5x5 | 20/1263 | 16.91 | 16.16 | -0.75 | [-1.74, -0.12] | 3/25 | -0.404 | -0.76 [-1.98, -0.01] |
| all | pap_inv+ridge | 5x5 | 38/3532 | 19.18 | 18.97 | -0.21 | [-0.57, +0.14] | 8/25 | -0.168 | +0.04 [-0.32, +0.41] |
| finger_still | hand_raw+rf | LOSO | 18/1246 | 22.31 | 22.52 | +0.20 | [-1.97, +2.65] | 12/18 | 0.096 | +2.20 [-0.04, +4.46] |
| finger_all | hand_raw+rf | 5x5 | 18/1764 | 22.2 | 22.18 | -0.02 | [-2.15, +2.06] | 14/25 | 0.084 | +1.98 [+0.60, +3.38] |
| ear_still | hand_raw+rf | 5x5 | 20/1263 | 17.79 | 16.16 | -1.64 | [-2.91, -0.71] | 1/25 | -0.32 | -1.59 [-3.24, -0.45] |

## #28 cohort finger_still: 17 two-session subjects, 1246 recordings

delta glucose: {'abs_median': 10.8, 'abs_q25_q75': [5.4, 27.0], 'abs_min_max': [1.8, 57.7], 'n_abs_lt_10': 8, 'n_abs_lt_18': 11, 'n_positive': 9, 'n_negative': 8, 'mean_abs': 19.39}

| feature | rho | p | p Holm |
|---|---|---|---|
| pulse_rate_bpm | -0.417 | 0.0959 | 0.5756 |
| hrv_rmssd_ms | 0.277 | 0.2815 | 0.706 |
| perfusion_index | -0.112 | 0.6698 | 0.706 |
| signal_quality | -0.304 | 0.2353 | 0.706 |
| rise_time_ms | 0.399 | 0.1131 | 0.5756 |
| pulse_width_half_ms | 0.459 | 0.0641 | 0.4486 |
| n_beats_detected | -0.353 | 0.1644 | 0.6575 |

| delta-ridge LOSO | MAE delta | zero-baseline MAE | gain [CI] | subj better | sign acc | majority rate | binom p | perm p sign | perm p gain | N perm |
|---|---|---|---|---|---|---|---|---|---|---|
| hand | 17.77 | 19.39 | +1.62 [-1.39, +4.63] | 12/17 | 12/17 | 0.529 | 0.1435 | 0.1134 | 0.032 | 2000 |
| pap | 21.15 | 19.39 | -1.75 [-6.34, +1.48] | 9/17 | 11/17 | 0.529 | 0.3323 | 0.2059 | 0.8136 | 2000 |

delta-ridge as within-subject calibration: {'mae': 17.77, 'carry_mae': 19.39, 'gain': 1.62, 'ci95': [-1.389, 4.634], 'subjects_better': 12, 'of_subjects': 17}

| calibration sim | pairs | MAE calibrated | MAE carry-over | MAE mean | MAE uncalibrated | vs carry gain [CI] | subj better | perm p | vs mean gain [CI] | perm p | gate vs carry | model-delta r / sign |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hand+ridge | 34 | 20.92 | 19.39 | 22.6 | 22.4 | -1.52 [-3.08, -0.08] | 6/17 | 0.999 | +1.68 [-7.12, +11.90] | 0.998 | FAIL | r=-0.287 sign 14/34 |
| pap+ridge | 34 | 20.29 | 19.39 | 22.6 | 21.85 | -0.89 [-2.33, +0.43] | 7/17 | 0.9801 | +2.31 [-6.47, +12.28] | 0.9402 | FAIL | r=-0.22 sign 14/34 |

## Not run / incomplete

Runs were stopped to bound runtime (the machine gave each worker about a third of a core; total compute passed 4 h). Finished results above are final; these arms are **[NOT RUN: stopped to bound runtime]**:

- #27 primary **random forest on handcrafted features**: 5x5 point estimate only exists in the sensitivities (LOSO and finger-all, ear-still, below); its 200-permutation gate was not finished. Its sensitivity gains are +0.20 (LOSO) and -0.02 (finger-all) mg/dL with CIs spanning about +-2 mg/dL, so no gain is visible, but no gate verdict was computed.
- #27 sensitivity RF on **all 3,840 recordings**.
- #28 calibration simulation with **random forest**, and the whole #28 pipeline on the **ear cohort (15 subjects)** and the pooled finger+ear cohort (32 subjects). The assignment assumed 32 finger-still two-session subjects; only 17 are finger.
- Permutation counts that were reduced: #28 PaPaGei calibration used 300 permutations (not 1000); #28 delta-ridge used 2000.

The finished arms already show: CIs that include 0 for every gain over the mean (#27) and a calibration that is worse than carry-over (#28). Unfinished arms are not needed to state the answers, but they are not evidence either way.

## Reading the numbers

- **Nothing passes the gate.** Session-level PaPaGei + ridge has permutation p = 0.033 (gain +0.76 mg/dL) but its CI [-0.08, +1.74] includes 0 and the recording-level p is 0.18; with about 20 comparisons in this report I treat it as noise. Several ungated sensitivities have a positive session-level CI for random forest (+2.25 [+0.38, +4.28], hand_inv; finger-all +1.98 [+0.60, +3.38]); their recording-level CIs span +-2.9 and none was gated, so these are exploratory flags, not findings.
- **Predictions point the wrong way.** OOF correlation with true glucose is negative (-0.13 to -0.28 across recording- and session-level views), between-person r of person means -0.45 / -0.37. A model trained on 17 people learns something about those people that does not transfer.
- **Clarke zone A is uninformative.** The mean baseline already gets 60.6% pooled (this cohort is mostly 70-100 mg/dL) and the models are 60.6-62.6%. Zone A is 0% for the 75 recordings in 180-250 mg/dL for every model including the baseline: the models cannot see high glucose.
- **The true within-person changes are small.** |delta glucose| median 10.8 mg/dL (IQR 5.4-27.0, range 1.8-57.7); 8 of 17 changes are under 10 mg/dL and 11 under 18 mg/dL (1 mmol/L), the order of a fingerstick meter's own error, so carry-over is hard to beat and many "changes" may be meter noise. The cross-subject models' predicted A-to-B change has SD 3.7 mg/dL against a true SD of 26.8.
- **Delta analysis.** No feature's change correlates with glucose change after Holm correction (smallest Holm p = 0.45; largest |rho| = 0.46 for `pulse_width_half_ms`). Delta-ridge on handcrafted features has a permutation p of 0.032 (MAE gain statistic, 2000 permutations) but sign accuracy is 12/17 (binomial p 0.14, permutation p 0.11; the majority sign already gets 53%), and the CI includes 0; I do not call it a pass. The PaPaGei delta-ridge is worse than zero change (-1.75).
- **Power.** With 17 people, a within-person correlation between predicted and true change would need about rho = 0.63 to be detected with 80% power at two-sided alpha 0.05 (Fisher z approximation, computed, not simulated). Weak tracking (rho below about 0.5) is not excluded.

## Caveats

- **Two sessions only**, with an undocumented gap and unknown chronological order; the A/B labels come from record numbering and glycaemia values. Any difference between sessions in finger pressure, temperature, phone, lighting, or time of day is confounded with the glucose change and could create or hide an effect.
- **Lab conditions**: still, supervised recordings with a chest ECG attached; no free-living variation. This is the easiest case for the app, and it still fails.
- **One glucose reading per session** (fingerstick Fora Diamond Mini), applied to every recording in that session; effective n is 35 sessions / 18 people, not 1,246 recordings. I did not verify the meter's accuracy specification.
- **Site is confounded with person** (finger = 18 people, ear = 20 different people); ear-only and combined sets are sensitivities, not a controlled comparison of sites.
- **30 Hz and 8-bit pixel means**; the red channel is near saturation in many finger recordings (16% of recordings exceed the app's clip limit), reducing pulsatile amplitude. 72 finger-still recordings with non-finite features were dropped, as the served pipeline would reject them.
- Glucose range 67-241 mg/dL with most subjects between 70 and 110: little between-person spread to learn from.
- Multiple comparisons: ~20 arms; only the primary ones carry permutation tests.

## Artifacts

- Code: `common.py`, `build_cache.py` (features cache, outside git at `/tmp/butppg_cache/features.npz`), `data_audit.py`, `eval_lib.py`, `run_ppg27.py`, `run_ppg28.py`, `make_tables.py`.
- Results: `results/data_audit.json`, `results/ppg27_primary.json`, `results/ppg27_sens.json`, `results/ppg28.json`, out-of-fold predictions `results/oof_*.npz`.
- Reproduce: `PAPAGEI_WEIGHTS=<path>/papagei_s.pt /tmp/ppg-venv/bin/python build_cache.py`; `run_ppg27.py primary 1000 200`; `run_ppg27.py sens`; `run_ppg28.py 1000 50 300`; `make_tables.py`.
