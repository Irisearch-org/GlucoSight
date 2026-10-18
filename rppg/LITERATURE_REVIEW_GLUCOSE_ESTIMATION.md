# Literature Review: Non-Invasive Glucose Estimation from Contact PPG

**Scope:** Whether the primary-source literature supports continuing to invest
engineering time in `rppg/models/glucose_estimator.py` (PPG → glucose), given
this project's own result (population-mean baseline beats trained Ridge and
Random Forest on 23 subjects / 67 recordings) and its own documented signal
problems (`docs/PPG_CAMERA_CAPTURE_FINDINGS.md`, P1–P7).

**Bottom line.** The literature does not contradict this project's result —
it predicts it. A peer-reviewed 2026 paper independently re-evaluated
published PPG-glucose models, **including one built on the exact same
dataset this repo uses** (Kermani & Esmaeili's MUST dataset, 23
subjects/67 sessions), under subject-held-out splits, and found every model
collapsed to a mean-baseline-equivalent R² ≈ 0. The field's apparent high
accuracy is overwhelmingly a product of random (leaky) train/test splits and
a clinical metric (Clarke Zone A/B) that cannot detect this failure. Where
the field has found something real, it required either (a) hundreds to
thousands of subjects plus an auxiliary per-subject anchor like HbA1c, or
(b) repeated same-subject measurements enabling personalization/calibration
— not more feature engineering on a single cross-sectional recording. Neither
condition can be met by this dataset or by this project's deployment model
(single finger-scan, no HbA1c, no repeated-visit calibration). **Recommendation: do not invest further engineering time trying to make this
dataset/hardware combination produce a usable `ppg_glucose_estimate`.**
Treat the current result as a valid, citable negative finding; keep the
ring-fence (E1.2) permanent rather than provisional. Details and the specific
change that *would* be load-bearing, if the project ever revisits this, are
in the Recommendation section.

---

## 1. Is contact-PPG glucose estimation an established, working approach, or does the field have a documented reproducibility problem?

**Primary sources found:**

- Jiang H, Yao T, Ding C. "PPG-based glucose sensors: a review." *Artificial
  Intelligence Review* (2025). https://doi.org/10.1007/s10462-025-11379-4 —
  narrative/systematic review of 106 peer-reviewed studies (from 270
  screened, 156 full-text assessed) spanning 2012–2024. Finding: the review
  frames its conclusions entirely as *future research directions* ("near-
  infrared wavelengths (850–940 nm) with reflective mode show better glucose
  sensitivity," "hybrid approaches... show the most potential") rather than
  as an established, working method — i.e., even the most comprehensive
  recent synthesis of the field does not claim the problem is solved.

- Lombardi S, Bocchi L, Francia P. "Photoplethysmography and Artificial
  Intelligence for Blood Glucose Level Estimation in Diabetic Patients: A
  Scoping Review." *IEEE Access* (2024).
  https://doi.org/10.1109/ACCESS.2024.3508467 — of **48** articles identified
  on PPG+AI glucose estimation specifically in diabetic patients, only **5**
  survived full-text eligibility screening. Claim: "all studies used Clarke
  error grid or Parkes error grid, with over 98% of estimates falling into
  clinically acceptable zones A or B." The review's own screening ratio
  (5/48) is itself evidence of a field with a weak base of qualifying
  studies, not a validated one.

- Zanelli S, Ammi M, Hallab M, El-Yacoubi MA. "Diabetes Detection and
  Management through Photoplethysmographic and Electrocardiographic Signals
  Analysis: A Systematic Review." *Sensors* 22(13):4890 (2022).
  https://doi.org/10.3390/s22134890 — 78 studies reviewed (41 on glucose
  estimation specifically). Explicit conclusion (§4, Conclusions): "ECG and
  PPG analysis in diabetes care showed to be very promising. **Clinical
  validation and data processing standardization need to be improved** in
  order to employ these techniques in a clinical environment" — a review
  conclusion that stops short of claiming the approach works, written by
  authors sympathetic to the modality.

- **Ramesh S, Neufeld M, Küttner M, Röddiger T, Beigl M. "Reassessing the
  Feasibility of PPG-Based Non-Invasive Blood Glucose Level Estimation."
  *Proceedings of the 2026 ACM International Symposium on Wearable
  Computers* (ISWC '26). https://doi.org/10.1145/3830727.3834821 (also
  arXiv:2608.01820, https://arxiv.org/abs/2608.01820).** This is the single
  most important source found, for two reasons: (1) it is a direct,
  peer-reviewed-venue reproducibility audit of the exact research claim this
  project is testing, and (2) **one of its two evaluation datasets is this
  project's own dataset** — MUST (Kermani & Esmaeili 2023): "PPG recordings
  sampled at 2175 Hz from 23 individuals, totaling 67 sessions... Blood
  glucose values ranged from 88 mg/dL to 183 mg/dL (M = 115.01, SD = 18.67)"
  — matching this repo's own reported range (88–183 mg/dL, mean 115.0, SD
  18.7) almost exactly. They re-implemented five representative published
  PPG→BGL methods and evaluated each under three split protocols: random
  window-level, participant-aware, and leave-some-participants-out (LSPO).
  Result (Table 2, quoted verbatim): **"Across all five models, MAE
  increases, and R² drops when moving from the random window-level split to
  the participant-aware split, with every model yielding near-zero or
  negative R² under the latter... the central pattern holds: prior exposure
  to the target individual yields no reliable improvement over the unseen
  setting."** Their Global (mean) Baseline under LSPO: MAE 26.39, R² ≈
  −0.00, Zone A+B 97.4%. Every reproduced model lands within noise of that
  baseline once leakage is removed. Their explicit framing of the
  consequence: **"clinical zone metrics systematically conceal model
  failure in this domain."**

- Klonoff DC, Nguyen KT, Xu NY, Arnold MA. "Noninvasive Glucose Monitoring:
  In God We Trust—All Others Bring Data." *J Diabetes Sci Technol*
  15(6):1211–1215 (2021). https://doi.org/10.1177/19322968211046326 (PMC8655290)
  — editorial by the Diabetes Technology Society documenting a specific,
  named track record of commercial non-invasive optical glucose monitors
  that received regulatory marks or heavy press coverage and then failed or
  were withdrawn for lack of accuracy, **without ever publishing validation
  data**: Pendragon's Pendra system (CE mark 2003, "not accurate," company
  dissolved same year), C8 MediSensors' Raman-based monitor (CE mark 2012,
  "poor accuracy," never reached market), Cygnus's GlucoWatch G2 Biographer
  (FDA cleared 2001, "lacked clinical accuracy," withdrawn 2007). Quote: "The
  data reported by these companies is generally too preliminary to justify
  the overhyped announcements commonly released from this industry." This
  is an optical-non-invasive-glucose pattern, not PPG-specific, but
  establishes the base rate of overclaiming in the exact category this
  project sits in.

**Synthesis.** The literature is not silent on this question, and it does
not say "established." Every systematic/scoping review found stops at
"promising, needs validation," one direct reproducibility study
(peer-reviewed, 2026, tested on this project's own dataset) found that
published PPG-glucose results collapse to a mean baseline once evaluated
correctly, and the broader non-invasive-optical-glucose category has a
25-year documented record of regulatory-cleared products failing in the
field. This is a field with an overclaiming problem that has now been
measured directly, not asserted.

---

## 2. What sample sizes do studies reporting genuine success use, and is 23 subjects known-insufficient?

**Primary sources found:**

- Ramesh et al. 2026 (above), Table 1, tabulates subject counts across 20
  representative published studies: from n=1 (single-subject case studies)
  up to n=6388 (Zeynali et al. 2025, pooled across datasets). Studies
  claiming the highest Zone A / R² under a reported **LSPO** (leave-some-
  participants-out, i.e. genuinely held-out people) split are: Prabha et al.
  2022, n=217, R²=0.99, Zone A 98.97%; Gupta et al. 2021, n=26, Zone A
  96.15% (LOPO+random personalization — not a pure stranger test); Lee et
  al. 2023, n=18, Zone A 100% (small, 2-channel). Under Ramesh et al.'s own
  re-implementation of these same methods with a corrected LSPO protocol,
  only three of five (Prabha, Gupta, Islam) retained *any* positive R², and
  those were small and close to zero (R² 0.04–0.05).

- Chu J, Yang WT, Lu WR, Chang YT, Hsieh TH, Yang FL. "90% Accuracy for
  Photoplethysmography-Based Non-Invasive Blood Glucose Prediction by Deep
  Learning with Cohort Arrangement and Quarterly Measured HbA1c." *Sensors*
  21(23):7815 (2021). https://doi.org/10.3390/s21237815 — this paper is
  itself a direct test of the scale question. At the full recruited n=2538
  subjects, a single universal CNN model reaches only **Zone A 60.6%, R²=0.06,
  MAE 25.4 mg/dL** (Table 3) — i.e. scaling up subject count alone made the
  Zone A result *worse* than many small-cohort papers, not better, because
  "physiological diversity... might originate from medication." Only by
  stratifying into a medication-free cohort (n=856) **and** adding quarterly
  HbA1c as a feature did they reach Zone A 94.2%, MAE 8.9 mg/dL, R²=0.71.
  Without HbA1c, even the clean cohort (n=856, no medication) scored R²=−0.05.

- Zhang H, Dromard E, Tsang KCH, Güemes A, Guo Z, Baldeweg SE, Li K.
  "Non-invasive glucose monitoring vs iCGM: a systematic review and
  meta-analysis of accuracy and methodological challenges." *medRxiv*
  (2026). https://doi.org/10.64898/2026.04.24.26351680 — meta-analysis of 32
  studies / 38 cohorts, N=1,693 pooled. Pooled MARD for non-invasive glucose
  monitoring (not PPG-specific, but the same accuracy-claim category):
  10.21% (95% CI 8.73–11.69%), with **I²=95.2%** heterogeneity — meaning the
  pooled number is not a stable estimate; individual-study results vary far
  more than sampling noise explains. This is a second, independent signal
  that the field's reported accuracy numbers are not generalizable.

**Synthesis.** 23 subjects / 67 recordings is low even by this field's own
(already low) standards — the smallest study in Ramesh et al.'s LSPO-evaluated
set with any retained signal was n=18–26, and Prabha et al.'s n=217 retained
the most signal. But the more important finding from Chu et al. is that
**scale alone does not fix this**: going from small cohorts to n=2538
*degraded* naive accuracy, because between-subject physiological variance
(medication status, baseline HbA1c) swamps any glucose-waveform signal
unless that variance is explicitly modeled. So the honest framing is not "23
is too small, collect more" — it is "23 is too small to detect a signal that,
per the field's own larger-n experiments, requires an auxiliary covariate
(HbA1c, medication status) to even become visible at any n."

---

## 3. What feature/signal-processing choices separate real validated success from the rest, and what does `rppg/features/extractor.py` lack?

**Primary sources found:**

- Second-derivative (SDPPG / APG) morphology features. D'Acquisto L.
  "Analysis of the Second Derivative of the PPG Signal as..." (2025), MDPI,
  https://www.mdpi.com/2673-4591/118/1/72, and Jiang et al. 2025 (§ above)
  both describe the standard SDPPG feature set: five fiducial points (a, b,
  c, d, e waves) on the second derivative of the PPG waveform, and ratios
  between them (b/a, c/a, d/a, e/a) that are the field's standard
  vascular-stiffness descriptors, following Elgendi's PPG analysis
  framework (cited by both as the source convention).
  Bereksi Reguig MA, Labdelli N. "Blood glucose prediction using
  non-invasive optical system based on photoplethysmography." *IJECE*
  14(5):5200–5208 (2024). https://doi.org/10.11591/ijece.v14i5.pp5200-5208 —
  built a calibration model directly on SDPPG's b/a ratio and systolic
  amplitude (Sa), reporting R² between 0.89 and 0.98. **Caveat, stated
  plainly:** the paper does not report a subject-held-out split; features
  and glucose labels were paired per-recording for calibration, which is
  exactly the "random/same-source" evaluation regime Ramesh et al. (§1)
  show inflates apparent accuracy. Treat this number as unvalidated by
  the same standard this project already applies to itself.

- Multi-wavelength / near-infrared sensing. Jiang et al. 2025 (review):
  "near-infrared wavelengths (850–940 nm) with reflective mode show better
  glucose sensitivity" than the visible-light (commonly green, 550 nm)
  single-wavelength PPG most smartphone-camera and low-cost-sensor studies
  use — including the MUST/Kermani dataset this project uses, which was
  collected with a **550 nm green LED + APDS-9008 photodiode** (per
  Kermani & Esmaeili's Mendeley dataset description, DOI
  10.17632/37pm7jk7jn.3). Chinchanikar A, Dale M. "Multimodal Approach for
  Non-Invasive Blood Glucose Estimation Using Fingertip Video." *Mathematical
  Modelling of Engineering Problems* 12(11) (2025).
  https://doi.org/10.18280/mmep.121108 — used a dedicated 940 nm NIR
  illumination board (not a bare phone flash) plus a hybrid feature set (46
  handcrafted PPG features + 2048 ResNet-50 deep features) on n=243
  subjects, 5-fold CV (not explicitly subject-disjoint — another caveat),
  reporting R²=0.88–0.89, MAE 14.50–15.13 mg/dL. Their own ablation: models
  trained on handcrafted PPG features *alone* scored markedly lower (R²
  0.52–0.83) than the hybrid or deep-feature-only sets — i.e. even within
  this one paper, hand-built PPG morphology features (the category this
  project extracts) were the weakest of the three feature sets tried.

- Per-subject auxiliary anchors (HbA1c) and personalization/calibration.
  Chu et al. 2021 (§2) and its 2025 follow-up — Chu J, Hsieh TH, Chang YT,
  Liaw SK, Yang FL. "Improving non-invasive glucose estimation with monthly
  calibrated photoplethysmography and implicit HbA1c." *Communications
  Medicine* (2025). PMC12738703, https://www.nature.com/articles/s43856-025-01210-0
  — show the feature that actually moves these numbers is not a better PPG
  morphology descriptor, it is an auxiliary per-subject physiological
  anchor (HbA1c) plus either cohort stratification by medication status or
  a monthly single-calibration-point protocol, reaching MARD 9.59–16.40%
  across three medication cohorts with no per-reading retraining.

**Cross-reference against `rppg/features/extractor.py`.** The extractor
computes exactly seven scalars per 10 s recording: `pulse_rate_bpm` (Welch
PSD peak), `hrv_rmssd_ms`, `perfusion_index` (AC/DC ratio ×100),
`signal_quality` (autocorrelation SQI), `rise_time_ms`, `pulse_width_half_ms`,
and `n_beats_detected`. This is a first-derivative-level
time/frequency-domain feature set. It does **not** compute: (a) any
second-derivative (SDPPG a/b/c/d/e) morphology ratio — the feature class the
one calibration-regime paper above reported its strongest correlations
from; (b) anything from a second wavelength — the dataset itself is
single-wavelength (550 nm green), and the literature's own stated
mechanism (glucose-dependent light absorption) is reported as stronger in
the 850–940 nm band this sensor and this project's smartphone-camera
deployment path cannot reach; (c) any auxiliary per-subject covariate
(HbA1c, medication status, age/BMI) — the extractor and `labels.csv`
carry only signal-derived features and the glucose label itself, no
HbA1c; (d) any repeated-measures/calibration mechanism — each of the 67
recordings is one independent glucose value with no earlier same-subject
reading to calibrate against, so the "monthly pretest" or "deduction
learning" (paired-difference) approaches above are structurally
inapplicable to this dataset as collected.

---

## 4. Given P1–P7, is the failure a feature bug, a model-choice problem, or a smartphone-camera-vs-dedicated-sensor hardware ceiling?

**Primary sources distinguishing smartphone-camera PPG from dedicated-sensor
PPG specifically:**

- A TU Delft MSc thesis, "Adaptive Real-Time PPG Signal Qualification and
  Pulse Segmentation" (https://repository.tudelft.nl/file/File_60d07a34-4e0d-4e12-8a0d-716214b222a4),
  is the only source found that directly measures smartphone-camera PPG
  quality against a dedicated sensor side by side. **Flagged explicitly as
  a lower-trust, non-peer-reviewed source** — cited only for its wavelength
  table, not its accuracy claim. Its Table 2.1 states the mechanism
  precisely: dedicated pulse oximeters use narrowband LEDs chosen at
  specific wavelengths (e.g. 670 nm red / 779 nm IR) matched to known
  absorption differences between oxy- and deoxyhemoglobin; a smartphone
  instead uses a broadband white flash LED and a Bayer RGB sensor, so "red
  light reaches much deeper and captures more information, but is more
  chaotic" while green has a cleaner signal-to-noise ratio but shallower
  penetration and less physiological information. This is exactly P1's
  finding in this project's own data (`docs/PPG_CAMERA_CAPTURE_FINDINGS.md`):
  green sat at the sensor floor (DC 0.0021) while red carried the usable
  signal — an instance of a general, documented smartphone-camera-specific
  limitation, not a bug unique to this pipeline.

- Jiang et al. 2025 (§1, §3): near-infrared 850–940 nm reflective-mode
  sensing is identified as the more glucose-sensitive band across the
  reviewed literature. No consumer smartphone RGB camera + visible-light
  flash can access this band; it requires a dedicated NIR LED (as used by
  Chinchanikar & Dale 2025, Sumaiya et al.'s 850/1070 nm transmission setup,
  and essentially every dedicated-hardware paper in Ramesh et al.'s Table 1
  that reports multi-channel/2–3-channel sensing). This is a **hardware
  ceiling**, not a software defect: it cannot be fixed by better filtering
  or feature extraction on the signal a phone's RGB camera can physically
  capture.

**Synthesis — answering (a)/(b)/(c) directly.** All three contribute, but
not equally:

- **(a) Bad input features, fixable in software — partially true, and
  already partially fixed.** P1–P7 document real, specific bugs: no
  sub-bin interpolation or confidence gate on pulse rate (quantized to a
  1.5 bpm FFT grid, P3), an HRV estimate that never enters the physiological
  range even on a paper negative control (P4), and a perfusion-index unit
  mismatch between extractor and contract validator (P6). These are
  legitimate, fixable defects and should be fixed regardless of whether the
  glucose task succeeds, because they also corrupt `ppg_signal_quality`-
  adjacent trust in the pipeline.
- **(b) Wrong model choice (Ridge/RF) — not well supported.** Ramesh et
  al. 2026 re-implemented five different published model classes (ResNet,
  CNN, XGBoost, PLS, GB) on this general task and found **all of them**,
  including non-linear/deep architectures, collapse to the mean baseline
  under a correct subject-held-out split. The model family is not the
  binding constraint; `rppg/models/glucose_estimator.py`'s own documented
  reasoning for using Ridge over a forest (n=67, p=7, a forest memorizes
  training subjects) is sound and not contradicted by anything found.
- **(c) Task is outside what finger-contact smartphone-camera PPG can
  support, vs. a dedicated sensor — well supported.** The literature's own
  best-replicated positive LSPO results (Prabha et al. 2022, Castro-Pimentel
  et al. 2023) use dedicated wrist-worn or multi-channel sensors, not bare
  smartphone cameras, and the review literature identifies a specific
  spectral band (850–940 nm NIR) as more glucose-sensitive than the
  visible-light band any phone camera + flash can reach. Combined with
  Ramesh et al.'s direct demonstration that even the field's positive
  results evaporate under correct evaluation on datasets including this
  one, (c) is the best-supported explanation for why fixing (a) would not
  be expected to flip this project's own result.

---

## 5. Is there a credible reason to keep trying, and if so, what specific change would be load-bearing?

**What the primary sources say would actually move the needle** (not
"bigger model," not "better filtering"):

1. **An auxiliary per-subject physiological anchor (HbA1c or medication
   status), not a better PPG feature.** Chu et al. 2021 and 2025 (§2, §3)
   are the two papers in this review with the largest validated sample
   sizes (n=2538, n=856-cohort) and the clearest ablation: adding HbA1c as
   a model input is what took their R² from ≈0.06–0.09 to 0.71, not any
   change to PPG signal processing. GlucoSight's PPG dataset (Kermani &
   Esmaeili's MUST) has no HbA1c field, and the Indonesian deployment
   protocol does not currently plan to collect one.
2. **Repeated same-subject measurements enabling calibration/personalization,
   not a single cross-sectional reading.** The 2025 "monthly calibrated"
   approach and the earlier "deduction learning" paired-difference model
   (Chu/Hsieh et al., *Scientific Reports* 2022,
   https://doi.org/10.1038/s41598-022-10360-3 — "DL model achieved 80% of
   test prediction in zone A... with 12 rounds of training data... a dozen
   rounds") both require multiple same-subject readings over time. MUST
   provides a single session per glucose label per subject-visit; there is
   no within-subject trajectory to calibrate against, and nothing in the
   current GlucoSight capture protocol (one finger-scan per meal) creates
   one.
3. **A dedicated NIR emitter at 850–940 nm, not better software on the
   phone's visible-light flash.** This is a hardware addition outside
   "no CGM, no extra hardware, just a smartphone" — the project's own
   stated design constraint (CLAUDE.md: "three input modalities from a
   smartphone"). Adopting it would mean abandoning the smartphone-only
   premise for this modality specifically.

**None of these three is available to this project without either changing
the dataset (impossible — it's a fixed, already-collected public dataset),
changing the deployment protocol to add HbA1c/longitudinal calibration
(a scope change bigger than a feature-engineering task), or adding dedicated
NIR hardware (contradicts the "just a smartphone" premise).** Given that,
and given that a peer-reviewed 2026 paper has already run this exact
experiment — model class after model class, properly evaluated, on this
exact dataset — and found the same null result this project found
independently, the primary-source evidence supports treating this as a
negative/limitation finding, not a software-engineering backlog item.

---

## Recommendation

**No. Do not invest further engineering time trying to produce a usable
`ppg_glucose_estimate` from the `rppg/PPG_Dataset/` + smartphone-camera
combination for the primary fusion signal.**

This is not hedged. The evidence for it:

- This project's own result (mean baseline 14.54 mg/dL MAE beats Ridge
  15.64 and RF 16.52, GroupKFold by subject, n=23) is not an unlucky run or
  a software bug — it is the exact outcome Ramesh et al. (2026, ACM ISWC)
  independently obtained re-evaluating five different published model
  architectures, on this same dataset plus a second (VitalDB, n=4872
  participants), under a correct subject-held-out split: near-zero or
  negative R² indistinguishable from a mean-prediction baseline, with Zone
  A/B appearing to "pass" regardless (>90% for every model including the
  baseline) because Clarke's zones are wider than the cohort's own label
  spread — the identical mechanism `rppg/models/glucose_estimator.py`'s own
  docstring already documents for this cohort.
- P1–P7 show the capture pipeline has real, independently fixable bugs
  (pulse-rate quantization with no confidence gate, a fictitious HRV
  estimate, a unit-mismatched perfusion-index validator). Fix these for
  pipeline integrity and honest `ppg_signal_quality` reporting — but do not
  expect fixing them to change the glucose result: Ramesh et al.'s
  reassessment used five *different* feature-extraction and modeling
  pipelines, none sharing this project's specific bugs, and all of them
  still collapsed to baseline.
- The field's genuine positive results, where they exist (Chu et al.'s
  medication-stratified + HbA1c cohort, R²=0.71 at n=856; the monthly-
  calibration follow-up, MARD 9.6–16.4%), are load-bearing on an auxiliary
  covariate (HbA1c) and/or repeated-measures personalization — neither of
  which this dataset contains or this deployment protocol currently
  collects. Chasing better PPG morphology features (SDPPG, multi-wavelength)
  without also adding one of those two things is pursuing the part of the
  literature that is *not* where the field's own ablations say the signal
  comes from.
- A smartphone RGB camera cannot reach the 850–940 nm NIR band the review
  literature identifies as more glucose-sensitive; this is a hardware
  ceiling, not a software one, and is out of scope for a "just a
  smartphone" product premise.

**If the project ever revisits this**, the literature says the one change
worth trying first, before any new model architecture or feature set, is:
add a self-reported auxiliary covariate at capture time — most cheaply,
ask participants for their most recent known HbA1c or diabetes-medication
status (both already collected in this project's intake per `labels.csv`'s
gender/age/height/weight pattern, so the precedent for a demographic/
clinical covariate field exists) — and re-run the existing Ridge pipeline
with that one field added, before touching the signal-processing code at
all. That is the cheapest test of the one lever the primary sources
consistently identify as load-bearing, and it requires no new hardware, no
new data collection protocol, and no new dataset.

Keep the ring-fence (`integration/glucose_source.py`, rule E1.2) as a
permanent design decision for this modality, not a provisional one pending
more engineering. The honesty framing already in
`agents/rppg/CLAUDE.md` ("any signal you find is more likely autonomic and
vascular than optically-glucose") is corroborated, not merely hedged
against, by this literature.

---

## References

1. Jiang H, Yao T, Ding C. "PPG-based glucose sensors: a review." *Artificial
   Intelligence Review* (2025). https://doi.org/10.1007/s10462-025-11379-4
2. Lombardi S, Bocchi L, Francia P. "Photoplethysmography and Artificial
   Intelligence for Blood Glucose Level Estimation in Diabetic Patients: A
   Scoping Review." *IEEE Access* (2024).
   https://doi.org/10.1109/ACCESS.2024.3508467
3. Zanelli S, Ammi M, Hallab M, El-Yacoubi MA. "Diabetes Detection and
   Management through Photoplethysmographic and Electrocardiographic
   Signals Analysis: A Systematic Review." *Sensors* 22(13):4890 (2022).
   https://doi.org/10.3390/s22134890
4. Ramesh S, Neufeld M, Küttner M, Röddiger T, Beigl M. "Reassessing the
   Feasibility of PPG-Based Non-Invasive Blood Glucose Level Estimation."
   *Proceedings of the 2026 ACM International Symposium on Wearable
   Computers* (ISWC '26). https://doi.org/10.1145/3830727.3834821 ;
   preprint: arXiv:2608.01820, https://arxiv.org/abs/2608.01820
5. Klonoff DC, Nguyen KT, Xu NY, Arnold MA. "Noninvasive Glucose Monitoring:
   In God We Trust—All Others Bring Data." *Journal of Diabetes Science and
   Technology* 15(6):1211–1215 (2021).
   https://doi.org/10.1177/19322968211046326 (PMC8655290)
6. Chu J, Yang WT, Lu WR, Chang YT, Hsieh TH, Yang FL. "90% Accuracy for
   Photoplethysmography-Based Non-Invasive Blood Glucose Prediction by Deep
   Learning with Cohort Arrangement and Quarterly Measured HbA1c." *Sensors*
   21(23):7815 (2021). https://doi.org/10.3390/s21237815
7. Chu J, Hsieh TH, Chang YT, Liaw SK, Yang FL. "Improving non-invasive
   glucose estimation with monthly calibrated photoplethysmography and
   implicit HbA1c." *Communications Medicine* (2025). PMC12738703;
   https://www.nature.com/articles/s43856-025-01210-0
8. Zhang H, Dromard E, Tsang KCH, Güemes A, Guo Z, Baldeweg SE, Li K.
   "Non-invasive glucose monitoring vs iCGM: a systematic review and
   meta-analysis of accuracy and methodological challenges." *medRxiv*
   (2026). https://doi.org/10.64898/2026.04.24.26351680
9. Kermani A, Esmaeili H. "The dataset of photoplethysmography signals
   collected from a pulse sensor to measure blood glucose level" (MUST
   dataset). *Mendeley Data*, V3 (2023/2024).
   https://doi.org/10.17632/37pm7jk7jn.3 — this project's own `rppg/PPG_Dataset/`
   source dataset.
10. Bereksi Reguig MA, Labdelli N. "Blood glucose prediction using
    non-invasive optical system based on photoplethysmography."
    *International Journal of Electrical and Computer Engineering (IJECE)*
    14(5):5200–5208 (2024).
    https://doi.org/10.11591/ijece.v14i5.pp5200-5208
11. Chinchanikar A, Dale M. "Multimodal Approach for Non-Invasive Blood
    Glucose Estimation Using Fingertip Video." *Mathematical Modelling of
    Engineering Problems* 12(11) (2025). https://doi.org/10.18280/mmep.121108
12. D'Acquisto L. "Analysis of the Second Derivative of the PPG Signal as..."
    *MDPI* (2025). https://www.mdpi.com/2673-4591/118/1/72
13. Chu J, Yang WT, et al. "Deduction learning for precise noninvasive
    measurements of blood glucose with a dozen rounds of data for model
    training." *Scientific Reports* (2022).
    https://doi.org/10.1038/s41598-022-10360-3
14. Zeynali M, Alipour K, Tarvirdizadeh B, Ghamari M. "Non-invasive blood
    glucose monitoring using PPG signals with various deep learning models
    and implementation using TinyML." *Scientific Reports* (2025).
    https://doi.org/10.1038/s41598-024-84265-8
15. "Adaptive Real-Time PPG Signal Qualification and Pulse Segmentation."
    TU Delft MSc thesis (non-peer-reviewed; cited only for its
    smartphone-vs-dedicated-sensor wavelength/channel table, §4 of this
    review). https://repository.tudelft.nl/file/File_60d07a34-4e0d-4e12-8a0d-716214b222a4
