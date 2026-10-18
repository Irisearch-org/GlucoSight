# Model-level fusion: two separate dataset validations, not one unified test

**Status:** accepted (2026-10-19) — decision only, not yet built.

The next step after dish identification (ADR 0002) is to make the actual
forecast consume what CV and NLP produce, instead of `BaselineB2Predictor`
ignoring all of it (`integration/predictor.py` — the live app currently
predicts `g0 + cohort_mean_excursion`, full stop). The natural plan —
train a model on `carbs_g`, `cv_confidence`, and `is_fried_cooking`
together — ran into the same shape of mistake as ADR 0002's CGMacros
finding: a feature list assembled without checking what each candidate
dataset actually contains.

**No single dataset has photo + text + glucose together.** Checked, not
assumed:

| dataset | real photos | text (→ `is_fried_cooking`) | glucose | carbs |
|---|---|---|---|---|
| CGMacros | yes | **no** — no text field at all | yes | weighed (ground truth) |
| BIG IDEAS | **no** | yes — `logged_food`, free text | yes | oracle (self-reported) |

So "carbs, cv, and is_fried_cooking" as one feature vector, live-derived,
cannot be built on either dataset alone today.

We decided: **run two separate validations, not one unified pipeline
test**, each answering the question its dataset can actually support —

1. **Carbs/CV contribution, on CGMacros**, using its ground-truth weighed
   macros directly rather than CV's real output — CV structurally cannot
   identify American food against an Indonesian dish table (ADR 0002), so
   running the real pipeline here would just reproduce CV6's already-known
   out-of-domain failure, not test whether a carbs signal helps the
   forecast at all. The ground truth isolates the right question.
2. **`is_fried_cooking` contribution, on BIG IDEAS**, using its real
   `logged_food` text through the existing `_is_fried()` keyword matcher.

Claiming a single end-to-end "proof" across both would misrepresent what
was actually tested, the same way the original CGMacros/Zone-A acceptance
criterion did.

## Considered options

| option | rejected because |
|---|---|
| wait for a dataset with photo+text+glucose together | none exists; the literature reviews already confirmed no such Indonesian-cuisine dataset exists either (ADR 0002) |
| run CV's real pipeline on CGMacros anyway, accept near-100% no-match | tests a known failure, not a new question — wastes the validation, not just API quota |
| build one new model/eval harness from scratch | `forecasting/evaluate/baselines.py` already does GroupKFold-by-participant, already produces B0/B1/B2 comparison rows — extending it is the smaller diff |
| **two separate validations, reported as such** | *chosen* |

## Consequences

- `forecasting/evaluate/baselines.py`'s `FEATURE_COLUMNS` gains
  `is_fried_cooking` as one more column; no new evaluation harness.
- Whatever gets trained is evaluated **offline first**, against the same
  four-criterion gate `rppg/models/glucose_estimator.py` already uses
  (positive improvement, wins in most folds, bootstrap CI excluding zero,
  permutation test) — `integration/api.py`'s `_predictor` stays
  `BaselineB2Predictor` until a model actually clears that gate, not
  speculatively.
- PPG is not an input to this work (ADR 0003) — `g0` keeps coming from a
  direct measurement or the population fallback only.
- This ADR records the decision; the build itself (feature column, BIG
  IDEAS loader wiring, the gated evaluation, the eventual `_predictor`
  swap) is not done yet and is tracked as the next spec when picked up.
