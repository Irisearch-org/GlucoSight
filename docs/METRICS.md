# GlucoSight — every number, and whether it is real

**Date:** 2026-10-18
**Purpose:** one table per track, so a number can be quoted without opening five
documents and hoping they agree.
**Status:** Every figure below was checked against the file named beside it on
2026-10-18, not copied from prose. Where a check found a figure that does not
reproduce or does not agree with its twin, it is recorded in §7 rather than
quietly fixed.

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

The distribution is the finding. **42 of the 55 figures below sit in a committed
file; 13 do not** — 9 living only in a sentence, 3 not in git at all, and 1
derivable but never computed. And 16 of the committed ones cannot be re-derived
without a dataset the repo does not ship, so a reader with a clone can check 26 of
them unaided.

---

## 1 · Contact PPG (`rppg/`)

| # | metric | value | tag | where it lives | reproduce |
|---|---|---|---|---|---|
| PP1 | finger-scan glucose, mean baseline MAE | **14.51 ± 2.00** mg/dL; folds 14.79 / 14.23 / 17.74 / 12.62 / 13.19; 67 recordings, 23 subjects | **artefact** — but see §7, **it does not reproduce** | `rppg/models/reports/baseline_mean.md`, `baseline_mean_folds.csv` | `python3 -m rppg.models.baseline_mean` |
| PP2 | mean-baseline Clarke Zone A | **85.1%** (n=67, 23 subjects) | **derivable** | inputs in `rppg/models/reports/baseline_mean_predictions.csv`; stated in `docs/INTEGRATION_AUDIT.md`, `integration/README.md`, `rppg/models/glucose_estimator.py` | none committed |
| PP3 | trained model MAE vs that baseline | baseline 14.57, Ridge 15.64, Random Forest 16.52 | **prose** | `rppg/README.md` | `python3 -m rppg.models.glucose_estimator --train` (needs gitignored `RawData/*.mat`) |
| PP4 | pulse rate vs simultaneous ECG | **MAE 3.76 bpm**, bias +0.00, LoA [−14.82, +14.82]; over **32** recordings that yield a rate error, from 35 annotated good-quality of 48; 12 subjects | **artefact · gated** | `rppg/reports/brno_validation.md`, `brno_validation_raw.csv` | `python3 -m rppg.brno_validation` (needs BUT PPG) |
| PP5 | the same agreement, per subject | MAE spans **1.27 → 30.85 bpm**, computed over **all** recordings per subject — see §7 | **artefact · gated** | same table | same — read the caveat in §7 |
| PP6 | HRV RMSSD vs ECG | **MAE 79.36 ms**, bias **+50.32 ms** (31 recordings) | **artefact · gated** | same | same |
| PP7 | a verified live capture | perfusion index **0.0284** (red channel), quality **0.765**, **55.5 bpm** against a manual 30-s count | **prose** (one-off device observation) | `docs/PPG_CAMERA_CAPTURE_FINDINGS.md` P7 | none — the five captures have no in-repo reproduction |
| PP8 | paper as negative control | perfusion index **0.000345**, quality **0.131**, modality weight **0.493** | **artefact** | P2 | `python3 -m rppg.check_ppg_findings` |
| PP9 | rate is quantised to the FFT grid | 1.5 bpm step at 30.3 Hz / 10 s; a rate is returned even for paper | **artefact** | P3 | `python3 -m rppg.check_ppg_findings` |
| PP10 | perfusion index units disagree | contract declares percent, extractor emits percent, service divides by 100 → an impossible value passes validation | **artefact** | P6 | `python3 -m rppg.check_ppg_findings` |
| PP11 | how a 53% perfusion index was manufactured | same synthetic finger, green channel: DC **0.0021** of full scale, AC/DC **0.2568**, far above the 0.002–0.03 physiological band; red: DC 0.5501, AC/DC 0.0161 | **artefact** | P1 | `node integration/web/tests/ppg_signal_check.js` |

---

## 2 · Computer vision (`cv/`)

| # | metric | value | tag | where it lives | reproduce |
|---|---|---|---|---|---|
| CV1 | food-10 test accuracy | **three different numbers** for one quantity: 0.8659 (notebook print), 0.866 (`MODEL_CARD.md`), **0.8801** (`training_metadata.json` → `test_acc`) | **artefact — self-contradictory**, see §7 | `cv/models/training_metadata.json`, `cv/MODEL_CARD.md`, `cv/models/glucosight-cv-baseline.ipynb` | notebook needs the Kaggle food dataset |
| CV2 | best validation accuracy | 0.8674 | **artefact** | `cv/models/training_metadata.json` | idem |
| CV3 | food-10 top-1 on a 99-image sample of the same held-out split | MobileNetV2 **70.71%** (70/99) | **prose** | `cv/reports/vlm_vs_mobilenet.md` | needs the food dataset |
| CV4 | Gemini VLM top-1, same 99 images | **95.96%** (95/99) | **prose** | same | idem |
| CV5 | B0 population-mean macro MAE, CGMacros | carbs **27.342 g**, protein 19.699, fat 12.509, fiber 4.178; 1,616 meals, 45 participants | **artefact · gated** | `cv/reports/b0_population_mean/metrics.json` | `python3 cv/scripts/evaluate_mean_baseline.py` (needs CGMacros) |
| CV6 | classifier macro MAE on the same 1,616 meals (out-of-domain) | carbs **35.019 g** (**+7.677** vs B0), protein 20.696 (**+0.997**), fat 13.099 (**+0.590**), fiber 4.277 (**+0.098**) | **artefact · gated** | `cv/reports/indonesian_classifier_ood/20260908T053051_450315Z/metrics.json` | idem |
| CV7 | photos forced to the largest class | **531 / 1,616 = 32.9%** predicted `pempek` | **artefact · gated** | `.../predicted_class_distribution.csv` | idem |
| CV8 | low-confidence predictions | **46.9%** below 0.6; mean confidence 0.636 | **artefact · gated** | `.../metrics.json` → `confidence` | idem |
| CV9 | manifest | 45 participants, 1,706 meal rows, **1,616 eligible**, 90 excluded (34 macro range, 1 missing macros, 55 no usable before-photo) | **artefact · gated** | `.../metrics.json` → `manifest_build` | idem |
| CV10 | the class list itself | **10** classes | **artefact** | `cv/models/class_to_idx.json` | — |
| CV11 | the breadth that list has to cover | **455** distinct dishes in the BIG IDEAs logs | **uncommitted** | `docs/BIGIDEAS_MEAL_RESPONSE_FINDINGS.md` | `python3 -m forecasting.evaluate.bigideas_response --data-dir ~/Downloads/Bigideas-dataset` |

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
| NL18 | the Indonesian dish table | **10 rows**, one per food class | **artefact** | `cv/data/tkpi_macro_lookup.csv` | — |

**Sign convention for every delta here:** `arm − baseline`, so **negative is an
improvement**. NL6–NL8 come from the `diagnostics` block of the same JSON.

**NL15 must not be compared to BIG IDEAs' 2.5%.** They are different quantities
at different dish granularity, and the gate document says so explicitly. Quote
both or neither.

---

## 4 · Forecasting (`forecasting/`)

| # | metric | value | tag | where it lives | reproduce |
|---|---|---|---|---|---|
| FC1 | CGMacros Zone A at T+60, by model | B0 45.7% / B1 51.0% / B2 54.7% / Ridge 53.0% / RF **58.7%** / GBDT 57.5% | **artefact · gated** | `forecasting/notebooks/glucosight_forecasting_sprint2.ipynb` (stored outputs) | re-run the notebook (needs CGMacros) |
| FC2 | the same, MAE | 38.45 / 37.07 / 31.39 / 62.00 / 30.00 / 31.44 mg/dL | **artefact · gated** | same | same |
| FC3 | calibrated GBDT, T+60 | **Zone A 62.5%**, MAE 28.11 — on **n = 1,534**, not 1,669 | **artefact · gated**, and **not comparable** to FC1, see §7 | same | same |
| FC4 | T+120 | B0 47.8 / B1 52.1 / B2 56.3 / RF 61.3 / GBDT 61.4 / calibrated **61.5%** | **artefact · gated** | same | same |
| FC5 | the dataset behind it | **1,669** valid meal events, 45 participants; mean excursion T+60 **29.83** (SD 40.83), T+120 25.82 (43.55) | **artefact · gated** | same | same |
| FC6 | exact-target counts under the implemented rule | 1,662 at T+60, 1,659 at T+120, 1,652 paired | **artefact · gated** | `docs/CGMACROS_PIPELINE_SPEC.md`; computed by `forecasting/evaluate/run_cgmacros.py` | `python3 -m forecasting.evaluate.run_cgmacros` (needs CGMacros) |
| FC7 | BIG IDEAs meal-response ladder, T+60 MAE | B0 24.92 / B1 25.93 / B2 22.23 / **B2g 21.01** / B3 19.52 | **uncommitted** | `docs/BIGIDEAS_MEAL_RESPONSE_FINDINGS.md` (untracked) | `python3 -m forecasting.evaluate.bigideas_response --data-dir ~/Downloads/Bigideas-dataset` |
| FC8 | the meal's own contribution | **B3 − B2g = 1.49 mg/dL** at T+60, 4.23 at T+120 | **uncommitted** | B4 | same |
| FC9 | break-even carbohydrate correlation | **r ≈ 0.41** | **prose** | B5 | — |
| FC10 | carb sensitivity | **+0.190 mg/dL per g** → a 30 g error is **5.7 mg/dL** | **prose** | B6, B11 | — |

**FC7 is a ceiling, not a result.** BIG IDEAs carries oracle macronutrients, so
every model there has the answer to the part the phone must guess. And **FC3
must never be quoted against FC1** — different sample counts (F3).

---

## 5 · Integration and utilities

| # | metric | value | tag | where it lives | reproduce |
|---|---|---|---|---|---|
| IN1 | integration tests | **108 pass** | **artefact** | `integration/tests/` | `python3 -m pytest integration/tests -q` |
| IN2 | utility tests | **47 pass** | **artefact** | `utils/tests/` | `python3 -m pytest utils/tests -q` |
| IN3 | modality down-weighting floor | absent **or** zero-confidence modality → weight **0.1**, never 0 (H1) | **artefact** | `integration/contract.py::modality_weight` | `python3 -m pytest integration/tests -q` |
| IN4 | the noise-model trap | mean baseline MAE 14.57 / Zone A **85.1%**; Ridge on pure noise MAE 14.78 / Zone A **88.1%** | **prose** | `docs/INTEGRATION_AUDIT.md` §F15 | gate refuses it: permutation p = 0.486 |
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
| artefact | 26 | re-derivable from the repo today |
| artefact · gated | 16 | re-derivable, but only with a dataset the repo does not ship |
| derivable | 1 | the inputs are committed; nobody wrote the command |
| prose | 9 | a sentence. Treat as a lead, not a result |
| uncommitted | 3 | not in git at all yet |
| **total** | **55** | |

Almost a third of the tags depend on a dataset the repository does not ship (16 of
55), which is unavoidable — but it means most of this project's evidence cannot be
re-derived by a reader with a clone. Only the 26 `artefact` rows travel.

Three of the best numbers in this repository are `uncommitted` (the BIG IDEAs
ladder, the meal contribution, and the 455-dish breadth behind everything CV
cannot cover), and one of the most-quoted is a single `prose` sentence (the
CGMacros 62.5%).

---

## 7 · Numbers that disagree with themselves

These were found by building this page. None is fixed here; each is recorded so
it cannot be discovered for the first time by a reviewer.

**7.1 — The PPG baseline report does not reproduce.**
`python3 -m rppg.models.baseline_mean` regenerates
`rppg/models/reports/baseline_mean.md` with **different numbers than are
committed**: summary **14.54 ± 1.10** against the committed **14.51 ± 2.00**, and
every per-fold MAE different. The script is deterministic — two consecutive runs
agree exactly — so the committed report is **stale relative to the committed
data**. It predates the commit that first tracked `rppg/data/labels.csv`
(`8936fb1`), which added the file only after an unanchored `data/` ignore rule
was fixed. **The 14.51 quoted in `integration/glucose_source.py`,
`integration/api.py`, `integration/README.md` and `docs/INTEGRATION_AUDIT.md`
(twice) is a stale draw.** The difference is immaterial (0.03 mg/dL), but the
staleness is not, and the SD more than halves.

**7.2 — CV test accuracy has three values.** 0.8659, 0.866 and 0.8801, all for
"food-10 test accuracy" of `cv-baseline-v0.1`. `training_metadata.json` records
`test_acc: 0.8801`, while `MODEL_CARD.md` states 0.866 and the notebook prints
0.8659 — and the card's "Best val accuracy 0.867 (epoch 10)" matches the
metadata's `best_val_acc`, suggesting the card's test figure was copied from the
notebook's run while the metadata came from a later one. **Quote the range, or
re-run and pick one.**

**7.3 — The CV top-1 on a 99-image sample is 70.71%, and on the full split it
claims 86.6–88.0%.** Both are described as the same held-out test split. If both
were estimates of one quantity, 70/99 would be roughly four standard deviations
low. Something differs — the sample, the label mapping, or the inference path —
and it is unresolved. **Until it is resolved, do not quote the food-10 accuracy
at all.**

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

**7.6 — The mean baseline appears as both 14.51 and 14.57.** 14.51 is the
committed report's summary; 14.57 is in `rppg/README.md` and the audit's Zone A
table. Same model, two values, neither annotated. See 7.1.

---

## 8 · Numbers never to say

Not because they are embarrassing — because each one is **a metric that moved
while nothing was learned**, and each already has a documented rebuttal.

| number | why not |
|---|---|
| **"Zone A 88.1%, exceeding the 70% clinical target"** | It is a Ridge regression trained on pure noise. MAE 14.78, worse than the 14.57 constant it beat on Zone A. The gate refused it (permutation p = 0.486) |
| **"Zone A 85%, exceeds the 70% target"** | That is the constant predictor. On this cohort a Zone A figure cannot distinguish a working finger-scan model from a flat line — glucose SD is 18.6 mg/dL and the Zone A band is ±22 mg/dL |
| **"the meal contributes 2.70 mg/dL"** | Quoted against B2 instead of B2g. The meal's own contribution is **1.49** mg/dL at T+60. The 2.70 figure does not survive review |
| **"30 g of carbohydrate error moves glucose 40–60 mg/dL"** | Measured at **5.7** mg/dL (B11 / FC10). Amended in place in `agents/cv/CLAUDE.md` and recorded in `docs/REVIEW_FINDINGS.md` |
| **"53% perfusion index"** | Manufactured: the page read the green channel under a fingertip, where haemoglobin absorbs, so AC/DC divided one near-zero number by another (P1) |
| **"62.5% Zone A, a +5-point calibration gain"** | Different sample set from the 57.5% it is compared to (7.5 / F3) |
| **"portion is the dominant term in carbohydrate load"** | Measured at **2.5–5.0%** of variance. Which dish is **97.5%** (C5 amendment) |
| **"CV classifies food at 88%"** | Self-contradictory, and contradicted by its own comparison report (7.2, 7.3) |
| **"NLP has a glucose model"** | NLP has **no** glucose model. It has a validated *unit of prediction* worth 0.65 mg/dL of MAE on someone else's cohort, and a 10-row dish table |
| **"PPG gives four usable fields"** | Of the four contract fields only `ppg_signal_quality` discriminated cleanly (0.765 on a real capture, 0.131 on paper — P2/P7). Pulse rate is quantised to a 1.5 bpm grid and is returned even for paper (P3), `ppg_hrv_rmssd` is not measuring heart-rate variability (P4), and `ppg_perfusion_index`'s units disagree between extractor and contract (P6). `glucose_estimate` is permanently `null` by ring-fence E1.2 |

---

## 9 · Reproduce everything in one place

```bash
# contact PPG — no download needed
python3 -m rppg.models.baseline_mean          # PP1  (rewrites the committed report; see 7.1)
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

---

## Open threads for this document

1. **Fix 7.1** — regenerate the PPG baseline report against the committed data
   and update the five files that quote 14.51. One atomic change.
2. **Resolve 7.2 / 7.3** — re-run the CV baseline once and decide which split
   and which number is the reported one, then delete the others.
3. **Move FC7 and FC8 off `uncommitted`** — the BIG IDEAs findings and their code
   are the strongest meal-response evidence in the repository and are not in git.
4. **Make PP2 `artefact`** — 85.1% is a one-line computation over a committed
   predictions file that nobody wrote down. It should have a command.
5. **Re-check this page when the contract is signed.** A number that is
   `derivable` or `prose` today should be `artefact` before it goes in a paper.
