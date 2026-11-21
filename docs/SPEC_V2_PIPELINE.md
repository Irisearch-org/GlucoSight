# GlucoSight v2 pipeline spec

**Status:** ready to implement (2026-11-21). Destination of the wayfinder map
[Wayfinder: GlucoSight v2 pipeline spec](https://github.com/Irisearch-org/GlucoSight/issues/5).
Every statement below was decided in a linked ticket; this document decides nothing new.
Audience: implementers now, paper reviewers (December 2026) later. Research prototype, not a medical device.

## 0. Invariants

- **The user flow is fixed.** (1) Current glucose: glucometer entry, CGM reading, or finger PPG scan. (2) Meal: a photo, or manual entry item by item (dishes or single ingredients, each with grams or a portion). Items the photo can't identify confidently must be entered manually. (3) Forecast at T+60 and T+120. Changes live inside these steps, never around them.
- **Every stage serves a real model.** Baselines B0/B1/B2 and the PPG mean are comparison rows in evaluation only ([Glossary](../GLOSSARY.md): *Baseline*, *Served model*).
- **A stage claims a gain only by the ADR 0004 gate** (positive gain vs baseline on held-out people, most folds won, bootstrap CI excluding zero, permutation control). A stage that fails is still served, reported honestly, and shown with its error.
- The silent-failure rules C1 (timing), C2 (causality), C3 (baselines) and C4 (grouped splits) stand. Root `CLAUDE.md` rules 8–9 (`*_present` masks, down-weighting) are retired with contract v2 ([contract v2](https://github.com/Irisearch-org/GlucoSight/issues/23)).

## 1. Step 1: current glucose (`integration/stages/glucose.py`)

`current_glucose(input) -> G0Reading(value, source, error_range, quality)`

| Source | Value | Error shown | Evidence |
|---|---|---|---|
| Glucometer | user entry | ISO 15197 (±15 mg/dL below 100, ±15% above) | [features](https://github.com/Irisearch-org/GlucoSight/issues/13) |
| CGM | single latest reading (no history) | none | [glucose options](https://github.com/Irisearch-org/GlucoSight/issues/21) |
| PPG scan | served PPG model output | **±25 mg/dL** (90% of held-out errors) | [PPG served model](https://github.com/Irisearch-org/GlucoSight/issues/16), ADR 0005 |

- **Reading age ≤ 60 min**, otherwise step 1 asks to re-measure. No population fallback ([contract v2](https://github.com/Irisearch-org/GlucoSight/issues/23)).
- **PPG served model:** nested ridge on 7 handcrafted features, `rppg/models/weights/ppg_served_model.pt`. Held-out MAE 14.69 vs mean 14.55; it fails the gate. Predictions barely track glucose (pred ≈ 113.4 + 0.010·true, r = 0.045), and it under-reads high glucose. ADR 0005 records why it is served anyway and how it is kept honest.
- **The scan screen shows** the glucose as a range ("≈ 112 mg/dL (87–137)", *Estimasi kasar dari kamera*, with a note that a glucometer is more accurate) and heart rate as a secondary line. It does **not** show HRV, perfusion index or a quality score ([scan outputs](https://github.com/Irisearch-org/GlucoSight/issues/14)).
- **The capture gate checks validity only:** finger on the lens; torch verified (lit/dark luma delta > 0.03); clipping under the limit; HR 40–180 bpm; server window rules (fs 10–240 Hz, jitter ≤ 5%, ≥ 1 × 10 s window after a 3 s settle). A fail shows "Ulangi scan".
- **Server feature extraction matches training exactly:** resample to 125 Hz, pyPPG bandpass, the same 7 features.
- **One capture module:** `frontend/src/ppg.ts` absorbs the `integration/web/ppg-capture.html` protocol (torch check, brightest-channel record, POS pulse detector, clipping).

## 2. Step 2: meal (`integration/stages/meal.py`, `integration/dishes/`)

`resolve_meal(photo | items) -> Meal(items[row_id, grams, confidence], totals, needs_manual_entry)`

**Photo path:** Gemini `gemini-3.5-flash-lite` (paid tier), prompted in Indonesian for a list of dishes, then the **embedding matcher** (`gemini-embedding-001`), then a dish-table row ([VLM choice](https://github.com/Irisearch-org/GlucoSight/issues/9), [dish-ID trial](https://github.com/Irisearch-org/GlucoSight/issues/18)). MobileNetV2 leaves the serving path.

**Confidence rule, in one place** ([manual-entry rule](https://github.com/Irisearch-org/GlucoSight/issues/19)): accept an item when top-1 similarity is **≥ 0.80** and the top-1 − top-2 margin is **≥ 0.05**; otherwise the item needs manual entry. Held-out test: coverage 63.4%, 89.6% correct when accepted, 6.6% false match (vs 18.3% at the old 0.75 setting). VLM self-reported confidence is never used, since it is uncalibrated.

**Confirm screen (prototype variant B,** branch `prototype/dish-gate`):
- Accepted items show "Dari foto" with the Sedikit/Sedang/Banyak picker.
- Each other item shows "Foto menebak '<name>'. Mana yang benar?" with the matcher's top suggestions plus "Cari lainnya…".
- Items can be removed and missed items added.
- "Lihat prediksi" is enabled only when every item is resolved.
- No confidence percentage is shown; `CV_CONFIDENCE_GATE` (75) is deleted.

**Manual entry:** structured items `[{name, grams | portion}]` go through the same matcher and table. The free-text `note_text` → NLP path is retired ([layout](https://github.com/Irisearch-org/GlucoSight/issues/20)).

**Dish table** ([row schema](https://github.com/Irisearch-org/GlucoSight/issues/15)):
- One row = one food **as eaten**, `kind` = dish | ingredient.
- Columns: `id`, `name`, `aliases`, per-100 g `energy_kcal`, `carbs_g`, `protein_g`, `fat_g`, nullable `fibre_g`, `serving_g`, `serving_source`, `source`/`source_ref`, `confidence` (exact | approximate | computed).
- No GI column and no fibre heuristic.
- Raw (*mentah*) rows only for foods eaten raw.
- **Grams per item** = entered grams, or portion scale (½ / 1 / 1.5) × `serving_g`. Meal macros = sum over items.
- **Ingredient serving weights** cite Kemenkes KepDirjen HK.02.02/B/576/2025. **Dish `serving_g` values are assumed defaults**, labelled as such ([serving weights](https://github.com/Irisearch-org/GlucoSight/issues/22)).
- **Licence (repo is public; TKPI 2017 has no open licence):** git holds the parser, aliases, recipes and class mapping (`cv/experiments/dish_id_trial/` on `wayfinder/dish-id-trial`). Values are built locally (`~/Downloads/glucosight-cv-data/dish_table_35.csv` today) and never committed. The demo serves rows individually with a citation.

## 3. Step 3: forecast (`integration/stages/forecast.py`)

`Forecaster.predict(g0, meal, profile, t0) -> {t60, t120, model_version}`

- **Served:** two LightGBM models, `forecasting/models/gbdt_cgmacros_T60.txt` / `_T120.txt` (+ `.json`), the `full` feature set chosen by inner selection on all 45 CGMacros participants ([GBDT trial](https://github.com/Irisearch-org/GlucoSight/issues/17)). They were trained with meter and CGM g0 only: the `src_ppg` input was never set, so a PPG-started forecast runs the model on a near-constant g0.
- **Features:** g0, g0 source, carbs/protein/fat/fibre grams (null stays null), hour of day (sin/cos), age, sex, BMI, diabetes status. Tipe 1 and Lainnya are flagged `outside_training` and the result screen notes it ([features](https://github.com/Irisearch-org/GlucoSight/issues/13)).
- **Targets:** the exact reading at meal time + 60 / + 120 min. Horizons are fixed; there is no Δt input.
- **Training data:** CGMacros cleaned by plausibility rules ([cleaning](https://github.com/Irisearch-org/GlucoSight/issues/12)):
  - drop a meal if macros and calories disagree by more than ±30% (Atwater);
  - fibre larger than carbs → fibre set to null;
  - Amount Consumed unused;
  - all-zero meal markers dropped;
  - no clipping or imputation.

**Evidence (nested CV, outer GroupKFold(5) × 5 by participant):**

| g0 source | Horizon | GBDT MAE | B2 MAE | Gain [95% CI] | Gate |
|---|---|---|---|---|---|
| CGM | T+60 | 28.78 | 31.68 | +2.90 [1.36, 4.42] | PASS |
| CGM | T+120 | 28.49 | 32.05 | +3.56 [1.62, 5.51] | PASS |
| Meter (simulated) | T+60 | 29.69 | 32.60 | +2.92 [1.49, 4.30] | PASS |
| Meter (simulated) | T+120 | 29.24 | 33.04 | +3.80 [1.79, 5.64] | PASS |
| PPG (simulated) | T+60 | 36.02 | 37.52 | +1.51 [−0.80, 4.05] | FAIL |
| PPG (simulated) | T+120 | 32.86 | 35.51 | +2.65 [−0.42, 5.65] | FAIL |

- Dish-ID carb errors change MAE by ≤ 0.14 mg/dL.
- Feature-set selection was unstable across folds, so the gate validates the procedure, not one fixed set.
- **The result screen** shows each horizon with its error band per g0 source (from held-out residuals), replacing `prediction_confidence`.

## 4. Data contract v2.0 (`integration/contract.py`)

Envelope, `G0Reading`, `Meal`, `Profile` and `ForecastResult` exactly as resolved in [contract v2](https://github.com/Irisearch-org/GlucoSight/issues/23). That resolution closes audit defects F6 (fallback macros), F7 (Δt) and F12 (stale g0). It also drops every NLP field, `gi_category`, `food_class`, `cv_confidence`, `portion_reported`, HRV/PI/SQI, `CONFIDENCE_FLOOR` and `MIN_MODALITY_WEIGHT`. Write it up as `docs/interface/INTERFACE_CONTRACT_v2.md`.

## 5. Code layout ([layout](https://github.com/Irisearch-org/GlucoSight/issues/20))

```
integration/
  api.py              HTTP only (auth/user routers, POST /predict, GET /dishes, /health, static)
  pipeline.py         forecast_meal(request) -> ForecastResult; used by the API AND the evaluation scripts
  stages/glucose.py   stages/meal.py   stages/forecast.py
  dishes/             DishTable.load(build), match(name)
  models/served.json  PPG model, T+60/T+120 GBDTs, VLM + embedding model ids, dish-table build hash
  contract.py         schema only
frontend/src/
  App.tsx             screen state machine
  steps/GlucoseStep.tsx  steps/MealStep.tsx  steps/ForecastStep.tsx
  ppg.ts              the one capture module
```

**Delete:**
- `cv-baseline-v0.1/`;
- the `DISHES` list in `frontend/src/foodCatalog.ts`;
- in `adapters/cv.py`: `GI_BY_CLASS`, the 45/18/12/4 fallback macros and `cv_items_to_contract`;
- `cv/data/tkpi_macro_lookup.csv` (from git);
- the `vlm_stub` runner and the MobileNet serving path;
- the `experimental_fusion` and `nlp_dish_identity_forecast` modes;
- `MeanBaselineG0Estimator` as a serving path, and B2 as the serving default;
- `integration/web/ppg-capture.html` and the `/ppg` route;
- the duplicated `_PROVENANCE`/`strip_derived` list;
- broad `except Exception` in the stage runners.

## 6. Evaluation and paper ([evaluation plan](https://github.com/Irisearch-org/GlucoSight/issues/24))

- **Results:** tables T1–T8 and figures F1–F2 as resolved there. There is no end-to-end accuracy number (ADR 0004). The final forecaster run uses 1,000 permutations.
- **Negative results reported as such:** PPG glucose; the pre-specified GBDT design; dish identification at exact-dish level (−20.5 pp vs MobileNet on its overlap).
- **Retired numbers:** listed in the evaluation plan; update `docs/METRICS.md` and fix `cv/MODEL_CARD.md` to food-10 top-1 88.1% ([CV numbers](https://github.com/Irisearch-org/GlucoSight/issues/11)).

## 7. Demo ([demo](https://github.com/Irisearch-org/GlucoSight/issues/25))

- **Runs locally:** uvicorn + Vite, `npm run dev:phone` for phone cameras over LAN HTTPS.
- **Reviewers get** a recorded walkthrough and an anonymised code mirror of a release tag.
- **Gemini key:** backend-only, paid tier.
- **Photos:** not stored; history keeps dish names and macros only.
- **TKPI table:** built locally, never shipped.

## 8. Open items (known, not blocking implementation)

- **TKPI permission:** send the drafted request to Kemenkes (in the [row schema](https://github.com/Irisearch-org/GlucoSight/issues/15) resolution) before publishing the table as a paper artifact. Owner: user.
- **Dish serving weights:** obtain clean copies of the SKMI 2014 food-photo book and the BDD 2014 conversion guide to replace the assumed dish `serving_g`. Until then the paper states that portion error is unmeasured.
- **Data limits that only new data fixes:**
  - phone-camera PPG paired with meter readings;
  - a fresh cohort to confirm one forecaster feature set;
  - mixed-plate photo evaluation;
  - Indonesian participants.
- **12 test photos** hit Gemini quota and are unscored; rerun `vlm_run.py --manifest manifest_test.csv`.

## Evidence branches (never merged)

`wayfinder/cv-accuracy-reconciliation`, `wayfinder/ppg-served-model`, `wayfinder/gbdt-forecaster`, `wayfinder/dish-id-trial`, `prototype/dish-gate`, and `research/{ppg-pretrained-models, glucose-forecasters, tkpi-dish-table-source, dish-naming-vlm, indonesian-food-photo-datasets, urt-serving-weights}`.
