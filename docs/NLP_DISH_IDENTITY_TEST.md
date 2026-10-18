# NLP Path A — does dish identity beat total grams?

**Date:** 2026-10-18
**Dataset:** ShanghaiT2DM — 100 T2D patients, Shanghai, China (Zhao et al.,
*Scientific Data* 10:35, 2023; `PMC9849330`). Same cohort as the Day 1 gate and
the response rebuild; no new download.
**Question:** the one this track exists to answer. On a weighed cohort, does
**dish identity** carry information about the post-meal glucose rise that
**total weight** does not?
**Status:** Answered, and the answer is yes — small in absolute terms,
consistent across every split tried, and it survives its control. The shipped
meal features, by contrast, are indistinguishable from carrying no signal at
all on this cohort.

Artifacts: `nlp/eval_dish_identity.py`, `nlp/features/dish_identity.json`,
`nlp/data/splits.py`, `nlp/data/diet_items.py`,
`nlp/features/shanghai_nlp_response.csv` (the response),
`nlp/features/shanghai_nlp_features.csv` (the shipped features).
Findings are referenced by `DI` number. Related: N1–N8
(`docs/NLP_PATH_A_GATE.md`), R1–R6 (`docs/SHANGHAI_RESPONSE_REBUILD.md`),
B1–B11 (`docs/BIGIDEAS_MEAL_RESPONSE_FINDINGS.md`).

---

## Why this was runnable only today

Two blockers stood in front of this test, and both were removed in the last two
days.

**N6 — there was no response variable.** The Path A feature table has no glucose
column; it could not be scored at all. The rebuild
(`57d10fc`, `docs/SHANGHAI_RESPONSE_REBUILD.md`) produced a strictly causal `g0`
and a T+60 rise for **3,045 of 3,060 meals (99.51%)**.

**N5 — the split was by CGM period, not patient.** `nlp/eval_baseline.py` grouped
`GroupKFold` on `participant_id`, which is the workbook stem
`<subject>_<period>_<date>`. The 109 periods belong to 100 subjects, so a
subject's repeat admission landed on both sides of the fold. This is now fixed:
`nlp/data/splits.py` groups on `subject_id`, and `leakage_report` keeps the old
behaviour measurable — **0.000 by subject, 0.185–0.215 by period** on this
dataset. The bug is a regression test, not a memory
(`nlp/tests/test_splits.py`).

The ordering matters and is recorded deliberately: the split was fixed *before*
the first number was produced, so no Path A figure has to be withdrawn.

---

## The design

A ladder of ridge regressions on the T+60 rise, all scored on the same
subject-grouped folds. The arms differ only in features:

| arm | features |
|---|---|
| `M0_constant` | the training-set mean rise |
| `M1_g0` | the pre-meal level |
| `M2_shipped` | `g0` + everything the shipped pipeline produces: `total_grams`, `is_fried_cooking`, `is_large_portion` |
| `M3_dish` | `M2` + one shrunk coefficient per dish |
| `M3p_permuted` | `M2` + the **same** dish matrix, rows permuted |

**`M3 − M2` is the answer.** `M3p` is why it can be believed.

Each arm's Ridge penalty is chosen by an **inner subject-grouped CV** over
`alpha ∈ {0.1, 1, 10, 100, 1000}` — so the comparison is not decided by a
penalty that happens to suit one arm's width. Dish coefficients are only
estimated for dishes seen at least **10 times in the training fold**, and folds
are grouped by subject, so a dish that a test subject ate but no training
subject ate contributes nothing. That is the real generalisation cost, and its
size is reported rather than hidden (**DI1**, **DI4**).

---

## DI1 — The result

Primary: 3,045 meals, 100 subjects, 5 `GroupKFold` folds by subject. Target SD
41.4 mg/dL; mean rise 27.7.

| arm | MAE | RMSE | fold SD | vs M0 | vs M2 |
|---|---|---|---|---|---|
| `M0_constant` | 32.326 | 41.411 | 2.820 | — | +1.178 |
| `M1_g0` | 31.230 | 39.748 | 2.134 | −1.096 | +0.082 |
| `M2_shipped` | 31.148 | 39.696 | 2.090 | −1.178 | — |
| `M3_dish` | **30.500** | 38.781 | 2.448 | −1.825 | **−0.648** |
| `M3p_permuted` | 31.252 | 39.770 | 2.121 | −1.074 | +0.103 |

Dish identity is worth **0.65 mg/dL of MAE** over the shipped feature set, on the
same meals, the same folds, and the same tuning procedure.

**It is not a fold artefact.** Every one of the five folds moves the same way
(−1.006, −1.023, −0.128, −0.569, −0.511), and so does every one of five
independent subject-grouped 80/20 resplits (−0.646, −0.769, −0.398, −0.767,
−0.195) — **`M3 − M2 = −0.555 ± 0.252` mg/dL**. Ten evaluations, ten
improvements, no reversal.

**It is not the medicated subgroup.** Dropping the 372 meals in this set that
sit within 30 min of an insulin dose (R4; 373 across all 3,060) leaves 2,673
meals and 93 subjects. The effect is unchanged.

| arm | MAE | RMSE | fold SD | vs M0 | vs M2 |
|---|---|---|---|---|---|
| `M0_constant` | 31.924 | 41.031 | 3.182 | — | +1.228 |
| `M1_g0` | 30.726 | 39.113 | 3.251 | −1.197 | +0.030 |
| `M2_shipped` | 30.696 | 39.098 | 3.183 | −1.228 | — |
| `M3_dish` | **30.015** | 38.305 | 3.226 | −1.909 | **−0.681** |
| `M3p_permuted` | 30.706 | 39.148 | 3.130 | −1.217 | +0.011 |

All five folds move the same way (−1.166, −1.195, −0.495, −0.417, −0.131);
**−0.681** pooled, **−0.585 ± 0.189** across independent resplits, and the
control still earns nothing (+0.077 ± 0.085). Test-meal coverage is 84.3% here
against 86.2% on the full set.

Full per-arm numbers for every run: `nlp/features/dish_identity.json`.

---

## DI2 — The control, and why it is not optional

This repository has already been burned once by a metric that improved while
the model learned nothing. Finding F15 recorded Ridge trained on **pure noise**
beating the constant baseline on Zone A.

Dish one-hot features add parameters, so a gain from `M3` means nothing on its
own. `M3p_permuted` runs the identical procedure — same matrix, same shape, same
column count, same inner tuning, same folds — with only the dish-to-meal pairing
destroyed.

| comparison | mean ± SD across 5 independent splits |
|---|---|
| `M3 − M2` (real dishes) | **−0.555 ± 0.252** |
| `M3p − M2` (permuted dishes) | **+0.064 ± 0.063** |

The real arm's gain is more than twice its own spread away from zero; the
control's is not distinguishable from it. The control's per-seed values straddle
zero entirely (+0.050, +0.115, −0.011, +0.023, +0.142).

There is a second, quieter piece of evidence. Each arm picks its own penalty by
inner CV, so the noise arm was free to shrink itself to nothing if the data said
so — and it did. The permutation arm chose the **maximum** penalty in 5 of 5
primary folds, versus 10 or 100 for the real dish arm. The harness's own tuning
switched the placebo off.

---

## DI3 — The uncomfortable finding: the shipped meal features carry no signal here

Look again at `M1_g0` versus `M2_shipped`:

> `g0` alone: **31.230**. `g0` + total grams + fried flag + large-portion flag:
> **31.148**. Difference: **0.082 mg/dL**, and across the five independent
> resplits it is **−0.048 ± 0.057** — straddling zero, i.e. indistinguishable
> from no effect. On the unmedicated subset it flips sign: **+0.032 ± 0.079**,
> the shipped features doing marginally *worse* than `g0` alone.

This is not a quirk of feeding the features to a model. Directly:

| feature | correlation with the T+60 rise |
|---|---|
| `g0_mgdl` | **−0.296** |
| `total_grams` | **+0.011** |
| `is_fried_cooking` | −0.048 |
| `is_large_portion` | −0.047 |

`total_grams` correlates **+0.011** with the rise. This is not a pooling
artefact: across the 87 subjects with at least 10 meals, the median per-subject
correlation is **−0.023** (against **−0.386** for `g0`). Whether a meal was
300 g or 500 g tells you essentially nothing about this cohort's T+60 rise.

So the track's current representation — one global carbohydrate density applied
to summed grams (`total_grams × 0.28 × 73/100`, N2) — is built on a quantity
that is informationally empty in the one cohort where the pipeline's own input
can be tested against a measured rise.

The precision matters, because BIG IDEAs is not a counterexample: B1–B4 measure
**oracle macronutrients** (`carbs_g`, +0.246 median per-participant `r`), which
the pipeline does not have and cannot produce from text. Summed weight — the
quantity the pipeline *does* have — was not measured there. Shanghai is where it
is, and it is worth +0.011.

**A consistency check from a different estimator.** This is *not* the headline
number and is not comparable to the table above: it is unregularised OLS, also
on subject-grouped folds, used only to see whether the ladder's shape repeats.

| features | held-out R² |
|---|---|
| constant | −0.0025 |
| `g0` | 0.0756 |
| `g0` + grams + both flags (**shipped**) | 0.0790 |
| `g0` + 23 dish indicators (dishes ≥ 50 meals) | **0.1145** |
| shipped + those dishes | 0.1152 |

The shipped feature set adds **+0.34 percentage points** of variance beyond
`g0`. Twenty-three dish indicators add **+3.9**. The shape of the ladder is the
same under a cruder, unregularised estimator, which is the point of including
it. Both the correlations above and this table are emitted by the same command
into the `diagnostics` block of `nlp/features/dish_identity.json`, so neither
number lives only in this document.

---

## DI4 — What this does *not* say

Four limits, stated before anyone else has to.

**It is a small effect in absolute terms.** 0.65 mg/dL against a target SD of
41.4. This is a statement about **which feature carries signal**, not a
clinically useful model, and no CGM device cares about 0.65 mg/dL. Reading it as
a performance claim would be wrong.

**It says nothing about a density *number*.** The test shows that *which dish*
predicts the rise better than *how much of it*. It does **not** show that a
carbohydrate-density value transfers between cohorts, which is precisely what
the Indonesian bridge needs (`docs/NLP_INDONESIAN_BRIDGE.md`, IB3). Shanghai has
no macronutrient ground truth — this cohort can validate the *unit* of
prediction and nothing about the *content* of a dish table.

**Coverage is real but partial.** 86.2% of test meals contain at least one dish
seen ≥10 times in training; the modelled vocabulary is 106–122 dishes out of
1,821. The long tail is dropped, not bucketed, so those meals fall back to `M2`.
The estimate is therefore **conservative** — the ceiling with a larger cohort
and a shared dish vocabulary is unmeasured.

**The cohort is not the deployment cohort.** 100 Chinese T2D patients, medicated
(372 of these meals sit near an insulin dose, R4). BIG IDEAs is 15 pre-diabetic
unmedicated US participants. Do not put the two MAE ladders side by side.

---

## What this changes

**The question the gate left open is closed, and the gate's suspicion was
right.** N1's finding was that the records are rich in *dish names*; N2's was
that the pipeline throws them away. This is the first measurement showing that
the discarded part is the informative part.

**The NLP track has its first number.** It previously had none — every tracked
metric was "none" for NLP. There is now a headline with error bars, a control,
and a reproduce command.

**It is evidence for the bridge's shape, and only its shape.** The bridge
(ADR `docs/adr/0001-dish-density-bridge.md`) claims the Indonesian join is on
dish identity → density, with Shanghai validating the method. `DI1` supports the
"dish identity is the right unit" half. `DI4` is the half still unvalidated.

**It sharpens the case against the current feature vector.**
`(total_grams, is_fried, is_large, nlp_confidence)` cannot express any of this
by construction: it collapses 1,821 distinct dish names onto one global density,
and one of its inputs is now measured to be uncorrelated with the outcome.
(The rebuild counts 1,838 dish-name *strings* over the same meals under a looser
parser that does not merge case and whitespace variants; the gate's stricter
parser gives 1,825. This analysis builds its vocabulary with the gate's rule, so
1,821 is the figure it uses throughout.)

---

## Reproduction

```bash
python3 -m nlp.eval_dish_identity                    # the ladder, ~5 s
python3 -m nlp.eval_dish_identity --seeds 5 --json nlp/features/dish_identity.json
python3 -m nlp.eval_dish_identity --min-dish-meals 20 # sensitivity to the floor
python3 -m pytest nlp/tests/test_dish_identity.py nlp/tests/test_splits.py -q
```

No download and no network: both input tables (`shanghai_nlp_response.csv`,
`shanghai_nlp_features.csv`) are committed, and they join on `meal_id` with
100% agreement. The 28 tests include a **positive control** (a rise generated
from dish identity alone — the dish arm must win) and a **null control** (a
dish-blind rise with random dish labels — it must not). Without those, the
real-data run would not be interpretable.

---

## Open questions

1. **Does the ceiling hold up with the whole vocabulary?** The floor of 10
   training meals is a deliberate conservative choice. A hierarchical or
   embedding treatment of the 1,821-dish vocabulary would test whether the
   long tail adds more.
2. **Does the dish *combination* matter, or only the individual dishes?** The
   design is additive. Meal composition — rice *with* fish versus rice *with*
   milk — is the obvious next term.
3. **Can the fitted dish effects be compared to real carbohydrates?** If the
   per-dish coefficients rank dishes by glycaemic load in a way that matches
   published food composition tables, the bridge's content half gets its first
   (indirect) support. This is the cheapest way to test IB3 without a new
   dataset.
4. **Does the same test reproduce on CGMacros**, where macronutrients *are*
   measured, so dish identity and carbohydrate content can be separated
   directly? That experiment is currently blocked by the over-broad
   `meal_feature_gate` (2.0% of meals are out of range on the four macro fields,
   not the 34 the gate rejects).
