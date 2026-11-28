# GlucoSight — every number, and whether it is real

**Date:** 2026-10-18; v2 update 2026-11-21 (retired numbers marked, new
reproducible numbers added)
**Purpose:** one table per track, so a number can be quoted without opening five
documents and hoping they agree.
**Status:** Every figure below was checked against the file named beside it on
2026-10-18, not copied from prose. Where a check found a figure that does not
reproduce or does not agree with its twin, it is recorded in §7 rather than
quietly fixed. The 2026-11-21 update retires the figures listed in
[`EVALUATION_PLAN.md`](EVALUATION_PLAN.md) (tickets
[#11](https://github.com/Irisearch-org/GlucoSight/issues/11),
[#16](https://github.com/Irisearch-org/GlucoSight/issues/16),
[#17](https://github.com/Irisearch-org/GlucoSight/issues/17),
[#24](https://github.com/Irisearch-org/GlucoSight/issues/24)).

---

## Why this document exists

The integration audit's central complaint was not that a model was wrong. It was
that **a number improved while the model learned nothing** — finding F15, where
Ridge trained on pure noise beat the constant baseline on Zone A, and would have
been reported as "88.1% Zone A, exceeding the 70% clinical target".

A number in a sentence has no such defence. A number with a committed file and a
command behind it can be re-derived by the person asking the question, which is
the only reason to believe it.

So every figure here carries a **tag** for how it is backed.

## The tags

| tag | means |
|---|---|
| **artefact** | a committed file contains the number as data, and a command in this repo regenerates it with no download |
| **artefact · gated** | a committed file contains it, but regenerating needs a dataset that is **not** in the repo |
| **derivable** | committed files contain the inputs, but no committed command computes the number |
| **prose** | the number appears only inside a sentence. No file holds it as data |
| **uncommitted** | its source file exists in the working tree but is not in git |

The distribution is the finding. **43 of the 55 figures below sit in a committed
file; 12 do not** — 9 living only in a sentence, 3 not in git at all. And 16 of
the committed ones cannot be re-derived without a dataset the repo does not ship,
so a reader with a clone can check 27 of them unaided.

**v2 rows.** Rows added on 2026-11-21 (PP12–PP14, CV12–CV16, FC11–FC14) are
backed by files committed on the unmerged evidence branches named beside each
(`wayfinder/*`, never merged to `main`), so they are tagged **artefact · gated**:
regenerating them needs the branch checked out plus a dataset the repo does not
ship. Rows marked **retired** are struck through and must not be quoted; the
replacement row is named in the same line. Retirement list:
[`EVALUATION_PLAN.md`](EVALUATION_PLAN.md) and
[#11](https://github.com/Irisearch-org/GlucoSight/issues/11),
[#16](https://github.com/Irisearch-org/GlucoSight/issues/16),
[#17](https://github.com/Irisearch-org/GlucoSight/issues/17),
[#24](https://github.com/Irisearch-org/GlucoSight/issues/24).

---

## 1 · Contact PPG (`rppg/`)

| # | metric | value | tag | where it lives | reproduce |
|---|---|---|---|---|---|
| PP1 | finger-scan glucose, mean baseline MAE (PP12 supersedes it as the comparison number: the nested training-fold mean is **14.55**; **14.54** is retired from paper tables, [#24](https://github.com/Irisearch-org/GlucoSight/issues/24)) | **14.54 ± 1.10** mg/dL; folds 13.87 / 15.59 / 15.74 / 14.32 / 13.19; 67 recordings, 23 subjects | **artefact** | `rppg/models/reports/baseline_mean.md`, `baseline_mean_folds.csv` | `python3 -m rppg.models.baseline_mean` |
| PP2 | mean-baseline Clarke Zone A | **83.6%** (n=67, 23 subjects) | **artefact** | `rppg/models/reports/baseline_mean.md`; computed from `baseline_mean_predictions.csv`; also stated in `docs/INTEGRATION_AUDIT.md`, `integration/README.md`, `rppg/models/glucose_estimator.py` | `python3 -m rppg.models.baseline_mean` |
| PP3 | ~~trained model MAE vs that baseline: baseline 14.54, Ridge 15.64, Random Forest 16.52~~ **retired** ([#16](https://github.com/Irisearch-org/GlucoSight/issues/16), [#24](https://github.com/Irisearch-org/GlucoSight/issues/24)): superseded by the nested run, PP12 (ridge 14.69 vs mean 14.55); the earlier ridge figures 15.64 / 15.04 / 15.30 disagreed across docs | ~~Ridge 15.64, RF 16.52~~ | retired | `rppg/README.md` | — |
| PP4 | pulse rate vs simultaneous ECG | **MAE 3.76 bpm**, bias +0.00, LoA [−14.82, +14.82]; over **32** recordings that yield a rate error, from 35 annotated good-quality of 48; 12 subjects | **artefact · gated** | `rppg/reports/brno_validation.md`, `brno_validation_raw.csv` | `python3 -m rppg.brno_validation` (needs BUT PPG) |
| PP5 | the same agreement, per subject | MAE spans **1.27 → 30.85 bpm**, computed over **all** recordings per subject — see §7 | **artefact · gated** | same table | same — read the caveat in §7 |
| PP6 | HRV RMSSD vs ECG | **MAE 79.36 ms**, bias **+50.32 ms** (31 recordings) | **artefact · gated** | same | same |
| PP7 | a verified live capture | perfusion index **0.0284** (red channel), quality **0.765**, **55.5 bpm** against a manual 30-s count | **prose** (one-off device observation) | `docs/PPG_CAMERA_CAPTURE_FINDINGS.md` P7 | none — the five captures have no in-repo reproduction |
| PP8 | paper as negative control | perfusion index **0.000345**, quality **0.131**, modality weight **0.493** | **artefact** | P2 | `python3 -m rppg.check_ppg_findings` |
| PP9 | rate is quantised to the FFT grid | 1.5 bpm step at 30.3 Hz / 10 s; a rate is returned even for paper | **artefact** | P3 | `python3 -m rppg.check_ppg_findings` |
| PP10 | perfusion index units disagree | contract declares percent, extractor emits percent, service divides by 100 → an impossible value passes validation | **artefact** | P6 | `python3 -m rppg.check_ppg_findings` |
| PP11 | how a 53% perfusion index was manufactured | same synthetic finger, green channel: DC **0.0021** of full scale, AC/DC **0.2568**, far above the 0.002–0.03 physiological band; red: DC 0.5501, AC/DC 0.0161 | **artefact** | P1 | `node integration/web/tests/ppg_signal_check.js` |
| PP12 | **served PPG model, nested CV** (ridge on 7 handcrafted features, nested α; GroupKFold(5) by subject; 67 recordings, 23 subjects) vs training-fold mean | ridge MAE **14.69** vs mean **14.55**; gain −0.14, CI [−0.52, 0.12], wins 2/5, perm p 0.54. **Gate fails**; served per ADR 0005, not a validated glucose estimator. Other candidates: PaPaGei-S + ridge 14.78, RF 14.84, PaPaGei-S PCA10 + RF 14.94 | **artefact · gated** (branch `wayfinder/ppg-served-model`) | `rppg/experiments/ppg_served_model/{REPORT.md,ppg_served_model.json}` | `rppg/experiments/ppg_served_model/evaluate.py` (needs gitignored `PPG_Dataset/RawData`) |
| PP13 | what the served model does | predictions 82–117 mg/dL (SD 4.2) vs reference SD 18.7 (88–183); fit pred ≈ 113.4 + 0.010·true, **r = 0.045**; Clarke zone A by reference range: 98% for 100–140, **22% below 100, 33% for 140–180, 0% for 180–250** | **artefact · gated** (same) | `rppg/models/reports/ppg_served_residuals.csv` | same |
| PP14 | error range shown in the UI | absolute held-out error 80th percentile **±20**, 90th percentile **±25 mg/dL** (shown); signal quality does not predict error (Spearman 0.012) | **artefact · gated** (same) | same | same |

---

## 2 · Computer vision (`cv/`)

| # | metric | value | tag | where it lives | reproduce |
|---|---|---|---|---|---|
| CV1 | ~~food-10 test accuracy: 0.8659 / 0.866 / 0.8801~~ **retired** ([#11](https://github.com/Irisearch-org/GlucoSight/issues/11)); the reproducible number is **88.1%**, CV12 | ~~three different numbers~~ | retired | `cv/models/training_metadata.json` (0.8801 reproduces as 88.0% on the training transform), `cv/MODEL_CARD.md`, `cv/models/glucosight-cv-baseline.ipynb` | — |
| CV2 | best validation accuracy | 0.8674 | **artefact** | `cv/models/training_metadata.json` | idem |
| CV3 | ~~food-10 top-1 on a 99-image sample, MobileNetV2 70.71% (70/99)~~ **retired** ([#11](https://github.com/Irisearch-org/GlucoSight/issues/11)): raw records missing, matches no reconstruction (§7.3); reconstructed value 89.9%, CV14 | ~~**70.71%**~~ | retired | `cv/reports/vlm_vs_mobilenet.md` | — |
| CV4 | ~~Gemini VLM top-1, same 99 images, 95.96% (95/99)~~ **retired** ([#11](https://github.com/Irisearch-org/GlucoSight/issues/11)): VLM raw records missing; replaced by CV15 | ~~**95.96%**~~ | retired | same | — |
| CV5 | B0 population-mean macro MAE, CGMacros | carbs **27.342 g**, protein 19.699, fat 12.509, fiber 4.178; 1,616 meals, 45 participants | **artefact · gated** | `cv/reports/b0_population_mean/metrics.json` | `python3 cv/scripts/evaluate_mean_baseline.py` (needs CGMacros) |
| CV6 | classifier macro MAE on the same 1,616 meals (out-of-domain) | carbs **35.019 g** (**+7.677** vs B0), protein 20.696 (**+0.997**), fat 13.099 (**+0.590**), fiber 4.277 (**+0.098**) | **artefact · gated** | `cv/reports/indonesian_classifier_ood/20260908T053051_450315Z/metrics.json` | idem |
| CV7 | photos forced to the largest class | **531 / 1,616 = 32.9%** predicted `pempek` | **artefact · gated** | `.../predicted_class_distribution.csv` | idem |
| CV8 | low-confidence predictions | **46.9%** below 0.6; mean confidence 0.636 | **artefact · gated** | `.../metrics.json` → `confidence` | idem |
| CV9 | manifest | 45 participants, 1,706 meal rows, **1,616 eligible**, 90 excluded (34 macro range, 1 missing macros, 55 no usable before-photo) | **artefact · gated** | `.../metrics.json` → `manifest_build` | idem |
| CV10 | the class list itself | **10** classes | **artefact** | `cv/models/class_to_idx.json` | — |
| CV11 | the breadth that list has to cover | **455** distinct dishes in the BIG IDEAs logs | **uncommitted** | `docs/BIGIDEAS_MEAL_RESPONSE_FINDINGS.md` | `python3 -m forecasting.evaluate.bigideas_response --data-dir ~/Downloads/Bigideas-dataset` |
| CV12 | MobileNetV2 food-10 test top-1 (n = 1,193; `mobilenetv2_food10.pt`, sha256 `394c8869…`) | **88.1%** (1,051/1,193, app path `cv/cv_baseline`); 88.0% (1,050/1,193) on the training transform. **The number the paper may quote** | **artefact · gated** (branch `wayfinder/cv-accuracy-reconciliation`) | `cv/reports/cv_accuracy_reconciliation.md` | `cv/scripts/eval_food10_reconcile.py` (needs the Mendeley food-10 test split) |
| CV13 | same, top-3 and macro-F1 | top-3 **98.0%**, macro-F1 **0.874** | **artefact · gated** (same) | same | same |
| CV14 | MobileNetV2 on the reconstructed 99-image sample (seed 42, minus `pic_338`) | **89/99 = 89.9%**, keyed by full path | **artefact · gated** (same) | same | same |
| CV15 | **dish identification, test (317 of 329 photos scored; 12 hit HTTP 429 quota and are unscored)**, VLM `gemini-3.5-flash-lite` + `gemini-embedding-001` matcher (no longer served, ADR 0006), 35 classes | top-1 **61.5%** (56.1–66.7), family-level **69.1%** (63.8–73.9), abstain 20.2%, false match **18.3%** at the old 0.75 setting. Pre-registered VLM + difflib: 60.3% / 65.9% / 16.4% / 23.3%. Overlap with MobileNet's 10 classes (n = 88), paired bootstrap Δ vs MobileNet: embedding family **+6.8 pp (+1.1 to +13.6)**; difflib family −3.4 pp (−12.5 to +5.7); embedding exact dish **−20.5 pp (−31.8 to −9.1)**; difflib exact dish −23.9 pp (−35.2 to −12.5). Dev: 164 photos | **artefact · gated** (branch `wayfinder/dish-id-trial`) | `cv/reports/dish_id_trial/{dev,test}/{summary.json,per_image.csv}` | `cv/experiments/dish_id_trial/evaluate.py` (needs the Wibisono/In-TFK photos; VLM responses cached) |
| CV16 | manual-entry operating point of the Gemini embedding matcher (accept when similarity ≥ **0.80** and margin ≥ **0.05**; no longer served), on the 35-row evaluation table | test: coverage **63.4%**, **89.6% correct when answered**, false match **6.6%**; dev: coverage 60.4%, 93.9% correct, false match 3.7%. Old production point (0.75, no margin): coverage 79.8%, false match 18.3% | **artefact · gated** (same) | `cv/reports/dish_id_trial/threshold_embedding.csv` | same |
| CV17 | **served matcher** (local hashed char 2–4-gram cosine, ADR 0006): operating point chosen on dev by the CV16 rule, scored by the served code on the **served 710-row table**, strict row-id truth, the trial's cached VLM names | accept when similarity ≥ **0.85**, margin ≥ **0**. Test: coverage **61.5%**, **85.6% correct when answered**, false match **8.8%**, top-1 52.7% (n = 317); dev: 60.4% / 88.9% / 6.7% (n = 164). Not like-for-like with CV16 (different table) | **artefact · gated** (needs the dev/test `per_image.csv` from `wayfinder/dish-id-trial`, the trial's class table and the built dish table) | `cv/reports/charngram_matcher/{summary.json,threshold_sweep.csv}` | `python -m cv.experiments.charngram_matcher.evaluate` (see its docstring) |

**Say this with CV6:** the report is explicit that it is **not an accuracy
number**. CGMacros has no class annotations, so the classifier is being asked
for a class it was not trained to name. The finding is "out-of-domain
transfer fails and here is how it fails", not "CV is 22% worse".

---

## 3 · NLP (`nlp/`)

| # | metric | value | tag | where it lives | reproduce |
|---|---|---|---|---|---|
| NL1 | does dish identity beat total grams? (T+60 rise, MAE) | shipped features **31.148** → **+dish identity 30.500**, worth **0.648 mg/dL**; 3,045 meals, 100 subjects | **artefact** | `nlp/features/dish_identity.json`, `docs/NLP_DISH_IDENTITY_TEST.md` | `python3 -m nlp.eval_dish_identity --json nlp/features/dish_identity.json` |
| NL2 | the same gain, with an uncertainty | **−0.555 ± 0.252** mg/dL across 5 independent subject-grouped resplits | **artefact** | same | same |
| NL3 | the permutation control | **+0.064 ± 0.063** — earns nothing | **artefact** | same | same |
| NL4 | robustness: no insulin dose within 30 min | **−0.681** pooled, **−0.585 ± 0.189** across resplits; 2,673 meals, 93 subjects | **artefact** | same | same |
| NL5 | what the shipped meal features (`M2`) are worth over `g0` alone (`M1`) | **M2 − M1 = −0.082** pooled; **−0.048 ± 0.057** across resplits; **+0.032 ± 0.079** unmedicated — no reliable gain | **artefact** | same | same |
| NL6 | `total_grams` correlation with the rise | **+0.011** (median per-subject **−0.023**, 87 subjects) | **artefact** | same → `diagnostics` | same |
| NL7 | `g0` correlation with the rise | **−0.296** (median per-subject **−0.386**) | **artefact** | same | same |
| NL8 | crude OLS R² ladder (not comparable to NL1) | constant −0.0025, `g0` 0.0756, shipped 0.0790, +23 dishes **0.1145**, both 0.1152 | **artefact** | same | same |
| NL9 | response coverage | **3,045 / 3,060 = 99.51%** at T+60; 3,028 (98.95%) at T+120 | **artefact · gated** | `nlp/features/shanghai_nlp_response.summary.json` | `python3 -m nlp.data.shanghai_response --data-dir ~/Downloads/ShanghaiT2DM ...` |
| NL10 | median pre-meal glucose and rise | `g0` **127.8** mg/dL, T+60 rise **+23.4**, T+120 **+25.2** | **artefact · gated** | same | same |
| NL11 | share of meals that rise | **75.1%** | **prose** | `docs/SHANGHAI_RESPONSE_REBUILD.md` R3 | — |
| NL12 | records are rich | 3,060 meals, **2,762 distinct texts**, **1,825 distinct dish names**, median 3 items | **artefact** | `docs/NLP_PATH_A_GATE.md` N1 | `python3 -m nlp.eval_path_a_gate` |
| NL13 | the pipeline throws the richness away | 2,762 meal texts → **661 feature states**; 1,825 dish names → **one** carb density | **artefact** | N2/N4 | same |
| NL14 | `nlp_confidence` is not a confidence | takes 4 values (0.62 / 0.6745 / 0.93 / 0.9053) by flag pair; **65.6%** at "neither present" | **artefact** | N3 | same |
| NL15 | portion share of `log(grams)` variance, weighed | **0.168** within participant × dish; **0.453** pooled by dish name | **artefact** | N8 | same |
| NL16 | leakage of the old period-level split | **0.2154** of test meals | **artefact** | N5 | same |
| NL17 | leakage of the subject-level split | **0.000** | **artefact** (asserted in tests) | `nlp/data/splits.py`, `nlp/tests/test_splits.py` | `python3 -m pytest nlp/tests/test_splits.py -q` |
| NL18 | ~~the Indonesian dish table~~ **retired**: `cv/data/tkpi_macro_lookup.csv` is deleted from git in v2 (spec §5); the dish table is built locally, never committed | ~~**10 rows**, one per food class~~ | retired | — | — |

**Sign convention for every delta here:** `arm − baseline`, so **negative is an
improvement**. NL6–NL8 come from the `diagnostics` block of the same JSON.

**NL15 must not be compared to BIG IDEAs' 2.5%.** They are different quantities
at different dish granularity, and the gate document says so explicitly. Quote
both or neither.

---

## 4 · Forecasting (`forecasting/`)

| # | metric | value | tag | where it lives | reproduce |
|---|---|---|---|---|---|
| FC1 | ~~CGMacros Zone A at T+60, by model~~ **retired** ([#24](https://github.com/Irisearch-org/GlucoSight/issues/24)): different n and no nested selection; replaced by FC11–FC13 | ~~B0 45.7% / B1 51.0% / B2 54.7% / Ridge 53.0% / RF 58.7% / GBDT 57.5%~~ | retired | `forecasting/notebooks/glucosight_forecasting_sprint2.ipynb` (stored outputs) | — |
| FC2 | ~~the same, MAE~~ **retired** | ~~38.45 / 37.07 / 31.39 / 62.00 / 30.00 / 31.44 mg/dL~~ | retired | same | — |
| FC3 | ~~calibrated GBDT, T+60~~ **retired**: Zone A 62.5% on n = 1,534, never comparable to FC1 (§7.5) | ~~Zone A 62.5%, MAE 28.11~~ | retired | same | — |
| FC4 | ~~T+120~~ **retired** | ~~B0 47.8 / B1 52.1 / B2 56.3 / RF 61.3 / GBDT 61.4 / calibrated 61.5%~~ | retired | same | — |
| FC5 | the dataset behind it | **1,669** valid meal events, 45 participants; mean excursion T+60 **29.83** (SD 40.83), T+120 25.82 (43.55) | **artefact · gated** | same | same |
| FC6 | exact-target counts under the implemented rule | 1,662 at T+60, 1,659 at T+120, 1,652 paired | **artefact · gated** | `docs/CGMACROS_PIPELINE_SPEC.md`; computed by `forecasting/evaluate/run_cgmacros.py` | `python3 -m forecasting.evaluate.run_cgmacros` (needs CGMacros) |
| FC7 | BIG IDEAs meal-response ladder, T+60 MAE | B0 24.92 / B1 25.93 / B2 22.23 / **B2g 21.01** / B3 19.52 | **uncommitted** | `docs/BIGIDEAS_MEAL_RESPONSE_FINDINGS.md` (untracked) | `python3 -m forecasting.evaluate.bigideas_response --data-dir ~/Downloads/Bigideas-dataset` |
| FC8 | the meal's own contribution | **B3 − B2g = 1.49 mg/dL** at T+60, 4.23 at T+120 | **uncommitted** | B4 | same |
| FC9 | break-even carbohydrate correlation | **r ≈ 0.41** | **prose** | B5 | — |
| FC10 | carb sensitivity | **+0.190 mg/dL per g** → a 30 g error is **5.7 mg/dL** | **prose** | B6, B11 | — |
| FC11 | **GBDT forecaster, nested CV** (outer GroupKFold(5) × 5 by participant, 45 people, 14 T2D), CGM g0 | T+60 GBDT **28.78** vs B2 31.68, gain **+2.90 [1.36, 4.42]**, wins 23/25, perm p 0.005 (200 perms); T+120 **28.49** vs 32.05, **+3.56 [1.62, 5.51]**, wins 23/25, p 0.005. Gate **PASS** | **artefact · gated** (branch `wayfinder/gbdt-forecaster`) | `forecasting/gbdt_results/{T60,T120,summary}.json`; `docs/GBDT_FORECASTER_TRIAL.md` | `forecasting/evaluate/gbdt_forecaster.py` (needs CGMacros full release) |
| FC12 | forecaster by g0 source (per-source table; meter and PPG are **simulated**) | Meter (ISO 15197 noise): T+60 29.69 vs 32.60 (**+2.92** [1.49, 4.30]), T+120 29.24 vs 33.04 (**+3.80** [1.79, 5.64]), PASS. PPG (113.35 + 0.0101·true, SD 4.20): T+60 36.02 vs 37.52 (**+1.51** [−0.80, 4.05]), T+120 32.86 vs 35.51 (**+2.65** [−0.42, 5.65]), **FAIL**; no PPG benefit claimed | **artefact · gated** (same branch) | `forecasting/gbdt_results/{T60,T120}.json`, `sensitivity_ppg/{T60,T120}.json` | same |
| FC13 | per-participant Clarke zone A, median (IQR) | T+60 GBDT **66.4 (20.4)** vs B2 58.7 (19.7); T+120 **62.6 (18.3)** vs 56.5 (23.3) | **artefact · gated** (same branch) | `forecasting/gbdt_results/summary.json` | same |
| FC14 | sensitivity and the failed pre-specified design | dish-ID carb perturbation changes MAE by +0.01 (T+60) / +0.14 (T+120); Atwater ±30% drops 85 meals; pre-specified design **fails** (T+60 −0.20 [−2.80, 2.35], T+120 +0.40 [−3.17, 3.77]); feature-set selection unstable (T+60 `g0_hour_macros` 17/25, `full` 8/25) | **artefact · gated** (same branch) | `forecasting/gbdt_results/{cleaning.json,sensitivity_carb/*}` | same |

**FC7 is a ceiling, not a result.** BIG IDEAs carries oracle macronutrients, so
every model there has the answer to the part the phone must guess. And **FC3
must never be quoted against FC1** — different sample counts (F3).

---

## 5 · Integration and utilities

| # | metric | value | tag | where it lives | reproduce |
|---|---|---|---|---|---|
| IN1 | integration tests | **108 pass** | **artefact** | `integration/tests/` | `python3 -m pytest integration/tests -q` |
| IN2 | utility tests | **47 pass** | **artefact** | `utils/tests/` | `python3 -m pytest utils/tests -q` |
| IN3 | ~~modality down-weighting floor~~ **retired** (contract v2: `MIN_MODALITY_WEIGHT` dropped; `*_present` masks and down-weighting superseded, see [`INTERFACE_CONTRACT_v2.md`](interface/INTERFACE_CONTRACT_v2.md) §7) | ~~absent **or** zero-confidence modality → weight **0.1**, never 0 (H1)~~ | retired | ~~`integration/contract.py::modality_weight`~~ | — |
| IN4 | the noise-model trap | mean baseline MAE 14.54 / Zone A **83.6%**; Ridge on pure noise MAE 14.78 / Zone A **88.1%** | **prose** | `docs/INTEGRATION_AUDIT.md` §F15 | gate refuses it: permutation p = 0.486 |
| IN5 | the 70% floor | the project's own Zone A target | **prose** | `docs/INTEGRATION_AUDIT.md` | — |

### Test counts, whole repo

| track | tests | command |
|---|---|---|
| integration | 108 | `python3 -m pytest integration/tests -q` |
| utils | 47 | `python3 -m pytest utils/tests -q` |
| nlp | 73 | `python3 -m pytest nlp -q` |
| rppg | 38 | `python3 -m pytest rppg/tests -q` |
| forecasting | 33 | `python3 -m pytest forecasting/tests -q` |
| cv | 7 | `python3 -m pytest cv/tests -q` |
| **total** | **306** | `python3 -m pytest --ignore=rppg/tests/test_convert_labels.py -q` |

`rppg/tests/test_convert_labels.py` is excluded because it needs `matio` from
the PPG virtualenv. A bare `pytest` fails collection on it.

---

## 6 · What the tags add up to

| tag | count | what it means for a claim |
|---|---|---|
| artefact | 27 | re-derivable from the repo today |
| artefact · gated | 16 | re-derivable, but only with a dataset the repo does not ship |
| derivable | 0 | the inputs are committed; nobody wrote the command |
| prose | 9 | a sentence. Treat as a lead, not a result |
| uncommitted | 3 | not in git at all yet |
| **total** | **55** | |

Almost a third of the tags depend on a dataset the repository does not ship (16 of
55), which is unavoidable — but it means most of this project's evidence cannot be
re-derived by a reader with a clone. Only the 27 `artefact` rows travel.

Three of the best numbers in this repository are `uncommitted` (the BIG IDEAs
ladder, the meal contribution, and the 455-dish breadth behind everything CV
cannot cover), and one of the most-quoted is a single `prose` sentence (the
CGMacros 62.5%).

**v2 update (2026-11-21).** The counts above are the 2026-10-18 snapshot and are
not re-tallied. The 11 new rows (PP12–PP14, CV12–CV16, FC11–FC14) are all
**artefact · gated** and live on unmerged evidence branches; rows struck through
as retired no longer count as figures a paper may quote.

---

## 7 · Numbers that disagree with themselves

These were found by building this page. None is fixed here; each is recorded so
it cannot be discovered for the first time by a reviewer.

**7.2 — ~~CV test accuracy has three values~~ (resolved 2026-11-21, [#11](https://github.com/Irisearch-org/GlucoSight/issues/11)).**
Re-running `cv/models/mobilenetv2_food10.pt` on the full food-10 test split
(n = 1,193) gives **88.1%** (1,051/1,193, app path) and 88.0% (1,050/1,193) on
the training transform; the 7-image difference is preprocessing. 0.8801 in
`training_metadata.json` is real (reproduced exactly) but `train_mobilenet.py`
never computes it. **0.8659 / 86.6% and macro-F1 0.858 are retired**: they do not
reproduce from the saved checkpoint [INFERENCE: they came from other weights].
Quote CV12–CV14.

**7.3 — ~~CV top-1 on a 99-image sample is 70.71%~~ (resolved 2026-11-21, retired).**
The VLM raw records and the MobileNet 99-image features are missing, so
"VLM 95.96% vs MobileNet 70.71%" cannot be reproduced. The reconstructed
MobileNet score on those 99 images is 89/99 (89.9%); 70/99 matches no
reconstruction [UNEXPLAINED]. A related bug: 130 filename stems (310 images)
occur in more than one class folder, so scripts that key by stem
(`run_gas_vlm.py`, `evaluate_real_vs_gt*.py`) give invalid scores. Key by full path.

**7.4 — The pooled pulse MAE is filtered, and two things hide inside it.** The
pooled **3.76 bpm** is computed over 32 recordings; the report's per-subject table
computes MAE over **all** recordings and spans **1.27 to 30.85 bpm**. Two effects
make the pooled number small:

* The **quality filter removes the biggest errors.** 13 of 48 recordings are
  annotated bad-quality, and the per-subject MAEs include them — subject 108 is
  30.85 bpm over its four recordings but 14.04 bpm over its good ones.
* **Per-subject biases cancel.** They range from −30.85 to +11.75 bpm while the
  pooled bias is +0.004, so pooling opposite-signed errors shrinks the average.

**Quote 3.76 bpm with both attached** — the quality filter and the per-subject
range — or quote a per-subject median instead.

**7.5 — The 62.5% and the 57.5% are not comparable.** Both are committed
notebook outputs, and §5 of the audit is right that they are computed on
different sample sets (1,534 vs 1,669) — the calibrated number drops each
participant's first three meals and participants with fewer than four, removing
the sparsest and plausibly hardest participants. The like-for-like figure was
computed and never printed. **Never present the +5.0-point calibration gain as
established.**

**7.6 — ~~The mean baseline appears as both 14.51 and 14.57~~ (fixed
2026-10-18).** It did: 14.51 was `rppg/models/reports/baseline_mean.md`'s
committed summary, stale against the committed `rppg/data/labels.csv`
(predates `8936fb1`, which added the file after an unanchored `data/`
ignore rule was fixed); 14.57 was `rppg/README.md` and the audit's Zone A
table, a second, different stale draw. `python3 -m rppg.models.baseline_mean`
is deterministic — re-running it regenerates **14.54 ± 1.10** mg/dL, Zone A
**83.6%** (previously an uncommitted, hand-derived 85.1% — PP2 is now
**artefact**). Updated everywhere that quoted either figure: PP1/PP2/PP3/IN4
above, `integration/glucose_source.py`, `integration/api.py`,
`integration/README.md`, `docs/INTEGRATION_AUDIT.md` (three sites),
`docs/PPG_CAMERA_CAPTURE_FINDINGS.md`, `rppg/README.md`, and
`rppg/models/glucose_estimator.py`'s docstring. The committed report also
had a mojibake `�7`/`�` (should be `§7`/`±`) from the same stale write,
fixed by the same regeneration.

---

## 8 · Numbers never to say

Not because they are embarrassing — because each one is **a metric that moved
while nothing was learned**, and each already has a documented rebuttal.

| number | why not |
|---|---|
| **"Zone A 88.1%, exceeding the 70% clinical target"** | It is a Ridge regression trained on pure noise. MAE 14.78, worse than the 14.54 constant it beat on Zone A. The gate refused it (permutation p = 0.486) |
| **"Zone A 84%, exceeds the 70% target"** | That is the constant predictor (83.6%, PP2). On this cohort a Zone A figure cannot distinguish a working finger-scan model from a flat line — glucose SD is 18.6 mg/dL and the Zone A band is ±22 mg/dL |
| **"the meal contributes 2.70 mg/dL"** | Quoted against B2 instead of B2g. The meal's own contribution is **1.49** mg/dL at T+60. The 2.70 figure does not survive review |
| **"30 g of carbohydrate error moves glucose 40–60 mg/dL"** | Measured at **5.7** mg/dL (B11 / FC10). Amended in place in `agents/cv/CLAUDE.md` and recorded in `docs/REVIEW_FINDINGS.md` |
| **"53% perfusion index"** | Manufactured: the page read the green channel under a fingertip, where haemoglobin absorbs, so AC/DC divided one near-zero number by another (P1) |
| **"62.5% Zone A, a +5-point calibration gain"** | Different sample set from the 57.5% it is compared to (7.5 / F3) |
| **"portion is the dominant term in carbohydrate load"** | Measured at **2.5–5.0%** of variance. Which dish is **97.5%** (C5 amendment) |
| **"CV classifies food at 88%"** | Say it only as: MobileNetV2 food-10 top-1 **88.1%** (1,051/1,193, 10 classes, studio photos, CV12). It is no longer served and cannot name the ~75% of dishes outside its 10 classes; the served path is VLM + embedding matcher (CV15). The old 86.6% and "95.96% vs 70.71%" are retired ([#11](https://github.com/Irisearch-org/GlucoSight/issues/11)) |
| **"NLP has a glucose model"** | NLP has **no** glucose model. It has a validated *unit of prediction* worth 0.65 mg/dL of MAE on someone else's cohort, and a 10-row dish table |
| **"PPG gives four usable fields"** | Of the four contract fields only `ppg_signal_quality` discriminated cleanly (0.765 on a real capture, 0.131 on paper — P2/P7). Pulse rate is quantised to a 1.5 bpm grid and is returned even for paper (P3), `ppg_hrv_rmssd` is not measuring heart-rate variability (P4), and `ppg_perfusion_index`'s units disagree between extractor and contract (P6). `glucose_estimate` is permanently `null` by ring-fence E1.2 |

---

## 9 · Reproduce everything in one place

```bash
# contact PPG — no download needed
python3 -m rppg.models.baseline_mean          # PP1, PP2
python3 -m rppg.check_ppg_findings            # PP8, PP9, PP10

# NLP — reads committed tables only
python3 -m nlp.eval_path_a_gate               # NL12–NL16
python3 -m nlp.eval_dish_identity \
    --json nlp/features/dish_identity.json    # NL1–NL8

# every test in the repository
python3 -m pytest --ignore=rppg/tests/test_convert_labels.py -q
```

Gated commands (a committed artefact exists; regeneration needs data the repo
does not ship):

```bash
python3 -m rppg.brno_validation                                   # BUT PPG
python3 -m nlp.data.shanghai_response --data-dir ~/Downloads/ShanghaiT2DM
python3 -m forecasting.evaluate.bigideas_response --data-dir ~/Downloads/Bigideas-dataset
python3 -m forecasting.evaluate.run_cgmacros                      # CGMacros
python3 cv/scripts/evaluate_mean_baseline.py                      # CGMacros
```

New v2 numbers (rows PP12–PP14, CV12–CV16, FC11–FC14) live on their evidence
branches, never merged to `main`. The scripts below are read from those branches
and need the data named beside each:

```bash
git show wayfinder/cv-accuracy-reconciliation:cv/scripts/eval_food10_reconcile.py   # CV12–CV14 (food-10 test split)
git show wayfinder/ppg-served-model:rppg/experiments/ppg_served_model/evaluate.py   # PP12–PP14 (PPG_Dataset)
git show wayfinder/dish-id-trial:cv/experiments/dish_id_trial/evaluate.py           # CV15–CV16 (Wibisono photos, cached VLM responses)
git show wayfinder/gbdt-forecaster:forecasting/evaluate/gbdt_forecaster.py          # FC11–FC14 (CGMacros full release)
```

---

## Open threads for this document

1. ~~**Fix 7.1**~~ — done 2026-10-18 (§7.6).
2. ~~**Resolve 7.2 / 7.3**~~ — done 2026-11-21 ([#11](https://github.com/Irisearch-org/GlucoSight/issues/11)); see §7.
3. **Move FC7 and FC8 off `uncommitted`** — the BIG IDEAs findings and their code
   are the strongest meal-response evidence in the repository and are not in git.
4. ~~**Make PP2 `artefact`**~~ — done 2026-10-18 (§7.6); `baseline_mean.py` now
   prints and writes the Zone A figure itself.
5. **Re-check this page when the contract is signed.** A number that is
   `derivable` or `prose` today should be `artefact` before it goes in a paper.
