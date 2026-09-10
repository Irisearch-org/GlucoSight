# GlucoSight Integration Layer

The PM-owned fusion spine. Joins the CV, NLP and contact-PPG tracks into one
contract-compliant record and serves it through an MVP endpoint.

**Read [`docs/INTEGRATION_AUDIT.md`](../docs/INTEGRATION_AUDIT.md) first** if
you are about to quote a number from Sprint 2.

## Quick start

```bash
pip install -r integration/requirements.txt
python -m pytest utils/tests integration/tests rppg/tests/test_glucose_estimator.py -q     # 165 tests
python -m uvicorn integration.api:app --reload        # http://127.0.0.1:8000
```

## What is here

| Module | Responsibility |
|---|---|
| `contract.py` | Interface Contract v1.2 — the single validator for the whole repo |
| `adapters/` | Track-native output → contract records |
| `fusion.py` | Assemble one validated meal record; causal history features |
| `features.py` | Three feature tiers + train-only normalization |
| `glucose_source.py` | Where `g0` comes from: fingerstick / CGM / PPG / fallback |
| `predictor.py` | The prediction step — **currently a labelled baseline** |
| `report.py` | Results tables that refuse to omit B0/B1/B2 |
| `api.py` | `POST /predict`, `GET /contract`, `GET /health` |
| `web/` | Demo page |

## The one thing to understand before demoing

**No trained forecasting model exists in this repository.** The Sprint 2
estimators were fitted in a Kaggle notebook and never serialised. So
`/predict` serves **baseline B2** — last causal reading plus the cohort mean
excursion — the same B2 that any real model must be reported against.

Every response says so (`is_trained_model: false`, plus a `basis` string),
and the demo page leads with a banner. Because B2 uses only `g0`, the
prediction is currently **the same whether or not the modalities are
present**: the spine carries, validates and weights them; the baseline does
not consume them. That is the true state of the system and the demo shows
it rather than hiding it.

To serve a real model, implement the `Predictor` protocol and return it from
`predictor.default_predictor()`. Nothing else changes.

## Rules enforced here rather than trusted

Each of these produces a clean run, a good-looking loss curve and a
worthless result. Each has a test that fails if it regresses.

- **C1 timing** — `delta_t_minutes` is carried as a feature, never assumed
  to be 60 or 120.
- **C2 causality** — `glucose_history` is filtered strictly before `t0`;
  derived features use only what survives; drops are counted and reported.
  Plus a **staleness** guard (proposed for v2.0): a causal-but-238-day-old
  reading is not a pre-meal glucose value.
- **C4 splits** — `participant_id` is mandatory so the grouping key always
  exists downstream.
- **C3 baselines** — `report.py` raises `MissingBaseline` rather than build
  a table with a model and no B0/B1/B2, and raises `ValueError` when rows
  cover different sample counts.
- **H1 missingness** — every modality has a `*_present` mask separate from
  its confidence; contradictory combinations are rejected.
- **H1 down-weighting** — low confidence scales a modality toward its
  fallback; it never removes a column, and the weight never reaches 0.
- **H7 encoding** — `gi_category` and `portion_reported` are one-hot.
- **H7 leakage** — `TrainOnlyScaler` refuses a second `fit` and refuses
  `transform` before `fit`.
- **E1 ring-fence** — `ppg_glucose_estimate` is excluded from every default
  feature tier and stripped from every API response.

## Where pre-meal glucose comes from

A user with a glucometer types their reading in; a user with a CGM supplies
a trace; a user with neither gets the population fallback. Each is a
different epistemic situation and `glucose_source.py` keeps them distinct.

| Source | Trust | Status |
|---|---|---|
| `fingerstick` | 1.00 | **Live.** User-entered, range-checked, C2-checked, staleness-checked |
| `cgm` | 0.95 | **Live.** From `glucose_history` |
| `ppg_estimate` | 0.30 | **Gated off** — see below |
| `population_fallback` | 0.10 | Always available, not personalised |

Direct measurements compete on recency: a fingerstick taken 5 minutes ago
beats a CGM reading from an hour ago, and vice versa. A PPG estimate is
used only when no direct measurement is available *and* the path is
explicitly enabled *and* an informative estimator is registered.

### The finger scan cannot be validated on the current dataset

Before training anything, know this: **predicting one constant for every
subject already reaches Clarke Zone A 85.1%** on the PPG glucose dataset
(n=67 recordings, 23 subjects, GroupKFold by subject), against a project
target of 70%. Zone A is ±20%, which at this cohort's median glucose of 110
mg/dL is ±22 mg/dL — wider than the label SD of 18.6.

A Zone A number here cannot separate a working model from a constant. So
`rppg/models/glucose_estimator.py` reports Zone A but gates on MAE against
the mean baseline, and requires **all four**: positive improvement, wins in
≥4 of 5 folds, a subject-level bootstrap 95% CI excluding zero, and a
label-permutation p ≤ 0.05.

```bash
python -m rppg.models.glucose_estimator --train
```

Gate fails → no artifact written → the finger-scan path stays off. That is
the intended behaviour, not an error. See `docs/INTEGRATION_AUDIT.md` F15.

### Why the PPG path is off by default

**Ring-fence E1.2** says `ppg_glucose_estimate` is "never used as a sole
prediction and never surfaced to a user". In `forecast = g0 + excursion`, a
PPG-derived `g0` *is* the sole driver of a user-visible number. Enabling it
needs a contract v2.0 amendment, not a config change.

**And there is nothing to enable yet.** The only committed PPG-glucose
model predicts the training-fold mean for every subject (MAE 14.51, n=23),
so it carries no more information than the population fallback.
`G0Estimator.provides_information` records that, and `resolve_g0` refuses
to prefer a constant predictor over the fallback — so the seam cannot be
lit up by accident with a model that does not deserve it.

A trained artifact is picked up automatically —
`glucose_source.autoload_ppg_g0_estimator()` runs at API startup and is a
no-op when no artifact exists. To register an estimator by hand:

```python
from integration import glucose_source as gsrc
gsrc.register_ppg_g0_estimator(MyEstimator())   # provides_information=True
```

## Feature tiers

`features.py` splits features by what a phone can actually produce:

| Tier | Contents | Use |
|---|---|---|
| `deployable` | Contract fields only — CV macros, NLP context, PPG, causal glucose history, timing | **The headline.** What the API can serve |
| `deployable_plus_askable` | + age, BMI, sex | A product decision away |
| `oracle` | + venous labs + commercial gut panel + dataset-weighed macros | **Upper bound**, never fusion |

The gap between `deployable` and `oracle` separates what CV can approximate
from what participant physiology contributes. Report all three.

## Not a medical device

Research prototype. Not for any treatment decision.
