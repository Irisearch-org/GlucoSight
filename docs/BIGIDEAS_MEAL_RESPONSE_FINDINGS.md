# BIG IDEAs — meal response, and what a CV estimate must achieve to matter

**Date:** 2026-10-17
**Dataset:** BIG IDEAs, PhysioNet v1.1.2, ODC-BY 1.0
**Question:** does meal composition carry glucose information the history does
not, and what would a computer-vision estimate have to achieve to contribute?
**Status:** Findings below are measured, not proposed. B7 retracts a claim made
in an earlier reading of this work.

Artifacts: `forecasting/data/bigideas.py`,
`forecasting/evaluate/bigideas_response.py`,
`forecasting/tests/test_bigideas.py`.

---

## Why this exists

`docs/CGMACROS_PIPELINE_SPEC.md` §1 excludes BIG IDEAs from the primary cohort
with an explicit escape clause:

> "PPG and NLP remain separate component studies **unless a real event-level
> pairing is demonstrated**."

This work demonstrates a real event-level pairing — meal composition paired
with a glucose outcome, 688 meals across 15 participants. That makes BIG IDEAs
usable as a component study rather than an isolated track.

**Scope of the pairing.** What is demonstrated is *food log ↔ CGM*. It does not
pair a meal photograph or a text note with glucose. The photo and NLP
modalities remain unpaired; what is established is that the meal composition
channel is real, measurable, and bounded.

---

## Data and hygiene

16 participants downloaded; **15 usable**, 688 meals, ~8 days each, Dexcom G6
at 5-minute cadence.

| Issue | Participants | Handling |
|---|---|---|
| Food log and CGM windows do not overlap (015: log 2020-07-05..07-13, CGM 2020-07-19..07-27 — two weeks apart) | 015 | Excluded, reported with a reason |
| Food log ships with no header row and 11 columns instead of 14 | 003 | Layout rebuilt positionally and verified against content |
| Food log uses `time_of_day` instead of `time` | 007, 013, 015, 016 | Both accepted |

Design constraints applied throughout:

- **`g0` is strictly causal.** It is the last reading *at or before* the meal,
  within 15 minutes. A reading taken after the meal can never be the pre-meal
  level.
- **Missing stays missing.** A `sum()` over an all-NaN nutrient column returns
  `0.0` in pandas, which would turn "this export has no protein column" into
  "this meal contained no protein". Aggregation uses `min_count=1`.
- **Every score is leave-one-out.** No model is fitted and scored on the same
  meal.
- **The macros are Oracle values.** `total_carb` is the number the
  participant's food-search app looked up for the logged amount. It has zero
  measurement error. **This is the ceiling, not a result.**

---

## Findings

### B1 — The pairing is dense and complete

688 meals aligned. 666 of them (97%) have a glucose reading at T+0, T+60 and
T+120. One participant-day gap exceeded 10 minutes across the whole cohort.

### B2 — Glucose rises after a meal, and peaks at T+60

| horizon | n meals | mean | median | sd |
|---|---|---|---|---|
| T+30 | 670 | +11.2 | +8.0 | 21.3 |
| T+60 | 666 | **+17.9** | +14.0 | 31.5 |
| T+120 | 662 | +7.8 | +7.0 | 30.2 |

The interface contract's T+60 horizon is correct. (An earlier single-
participant reading suggested T+30; that was an artefact of one unusually
well-controlled participant.)

### B3 — Starting glucose predicts the rise better than carbohydrate does

Median per-participant correlation with the T+60 rise:

| feature | median r | pooled r |
|---|---|---|
| `g0_mgdl` | **−0.355** | −0.384 |
| `carbs_g` | **+0.246** | +0.290 |
| `kcal` | +0.202 | +0.199 |
| `protein_g` | +0.152 | +0.097 |
| `fat_g` | +0.015 | +0.057 |
| `fiber_g` | +0.023 | +0.055 |

Pooled figures treat every meal as independent, which it is not — meals within
a participant are correlated. **The median across participants is the
reportable statistic; pooled is shown only to make the difference visible.**
No p-values are reported for this reason.

The confound is visible in the terciles. `g0` is held roughly flat across the
bands, and the rise climbs monotonically with carbohydrate:

| carb band | median carbs | median `g0` | rise T+30 | rise T+60 | rise T+120 |
|---|---|---|---|---|---|
| low | 5.9 g | 108.5 | +2.0 | +3.0 | +1.0 |
| mid | 33.0 g | 105.0 | +11.0 | +16.0 | +7.0 |
| high | 73.5 g | 106.0 | +14.0 | +27.0 | +14.0 |

### B4 — Most of B2's advantage is mean reversion, not the meal

Leave-one-out MAE at T+60, mean of per-participant MAE (mg/dL):

| model | MAE | vs B2g |
|---|---|---|
| B0 constant (participant mean) | 24.92 | +3.91 |
| B1 persistence (reading at the meal) | 25.93 | +4.92 |
| B2 `g0` + mean rise *(contract baseline)* | 22.23 | +1.22 |
| **B2g `g0` + fitted slope — no meal term at all** | **21.01** | **0.00** |
| B3 `g0` + fitted slope + carbs | 19.52 | **−1.49** |

**B2g contains no meal information.** It is B2 with one change: instead of
assuming the rise is independent of `g0`, it fits that slope. Because
`corr(g0, rise)` is negative, that slope is negative, and fitting it is worth
**1.22 mg/dL for free**.

**The meal's own contribution is the B3 − B2g gap: 1.49 mg/dL at T+60, and
4.23 mg/dL at T+120.** Quoting the 2.70 mg/dL gap against the contract's B2 as
"the meal's contribution" would be wrong, and would not survive review.

**B2g is not in the interface contract. It should be.** It dominates the
contract's B2, needs no food input, and costs one regression coefficient.

B1 is worth noting on its own: persistence is *worse than a constant* here.
Echoing the last reading fails because glucose reverts.

### B5 — There is a break-even carb correlation of 0.41

How much the meal is worth depends on how well carbohydrates are known. When
the carb estimate is corrupted to have a controlled correlation with the truth:

| carb correlation | meal gain vs B2g | participants improved |
|---|---|---|
| 1.00 | −1.49 | 8 / 15 |
| 0.80 | −0.90 | 8 / 15 |
| 0.60 | −0.49 | 7 / 15 |
| **0.40** | **+0.04** | 6 / 15 |
| 0.20 | +0.20 | 3 / 15 |
| **0.00** | **+0.34** | 1 / 15 |

**Break-even: r ≈ 0.41.** Below it the meal term stops paying for itself, and
at zero correlation it is **actively harmful** — it makes predictions worse.
A CV that names dishes but cannot size them does not merely fail to help.

### B6 — Portion *precision* is not the binding constraint

Corrupting the gram estimate multiplicatively:

| portion error | mean carb error | carb correlation left | meal gain |
|---|---|---|---|
| 0% | 0 g | 1.00 | −1.49 |
| ±30% | 12 g | 0.91 | −1.17 |
| ±50% | 19 g | 0.80 | −0.81 |
| ±100% | 31 g | 0.61 | −0.38 |

The meal survives ±100% portion error, because multiplicative noise barely
reduces the *correlation* between estimate and truth. Precision is cheap.
Correlation is what matters.

### B7 — Portion *size* is 2.5% of carbohydrate variance. Which dish is 97.5%.

**This retracts the earlier reading of this analysis**, which concluded that
portion estimation was CV's missing capability. It is not.

Decomposition of `log(carbs)` variance, measured on the 628 items belonging to
dishes a participant logged at least twice (the only place portion variation
is observable at all):

| source | variance | share |
|---|---|---|
| portion size (within dish) | 0.0568 | **2.5%** |
| which dish (between dishes) | 2.2276 | **97.5%** |

**51% of repeated dishes are logged at exactly one amount** (`amount.nunique()
== 1`), which forces their within-dish variance to zero and biases the portion
share *downward*. Restricting to the 310 items in dishes where the amount
actually moved gives **5.0%**. **The portion share is therefore 2.5–5.0%, and
97.5% of the variance is which dish.** The bracket is reported rather than the
single figure because the measurement floor is real and cannot be removed from
this data.

Supporting measurements:

- **`amount` is a genuine user entry, not a UI default.** Only 19.3% of items
  are logged as exactly `1`; values span 0.25 to 20 across real household units
  (cup 244, ounce 224, fluid ounce 110, gram 53, tablespoon 32); and the share
  logged as `1` ranges from 5% to 36% across participants. So a dish logged
  twice at one amount is a repeated serving, not a default left untouched.
- **The within-dish `corr(amount, carbs_g) = 1.00` is tautological and is not
  evidence of accuracy.** The within-dish coefficient of variation of
  carbs-per-unit is 1.2%, i.e. the food-search app computes
  `carbs = fixed_density × amount`. That establishes the app's arithmetic, not
  whether the entry reflects what was eaten. The real limitation is sharper
  than that; see Limitations.
- Amount varies within a dish in only 48% of repeated-dish groups, with a
  coefficient of variation of 0.25.
- There are **455 distinct dishes**, median 39 per participant.

The meal's carbohydrate content is overwhelmingly a question of *which dish*
was eaten, not *how much of it*.

### B8 — Coarse portion buckets add almost nothing

Textbook carbohydrate values learned per dish and per size band, then fed to
the same model. Two textbook sources, because the cold-start version is
starved by data sparsity:

| textbook source | identity only (1 band) | best k bands | gain |
|---|---|---|---|
| `self` — participant's own earlier meals | r = 0.617 | k = 8, r = 0.630 | **+0.013** |
| `cohort` — other participants (cold start) | r = 0.472 | k = 3, r = 0.475 | **+0.003** |

Adding portion bands moves the carb correlation by ~0.01 either way. This is
B7 arriving from a second direction, and the two agree.

### B9 — Identification recovers most of the meal's value, but needs breadth

| | carb correlation | meal gain vs B2g | share of oracle's 1.49 |
|---|---|---|---|
| oracle macros | 1.000 | −1.49 | 100% |
| dish identified, personalised | 0.617 | −1.06 | **71%** |
| dish identified, cold start | 0.472 | −0.57 | **38%** |

Both clear the 0.41 break-even — **but both assume perfect identification of
~455 distinct dishes.** `cv/MODEL_CARD.md` documents **10 Indonesian classes**
at 0.866 accuracy with no portion output.

**The gap is class breadth and personalisation, not portion estimation.**
A 10-class model cannot reach this operating point by construction, no matter
how high its accuracy on those 10 classes.

### B10 — The two sensitivity analyses are not interchangeable

At a *matched* carb correlation (r ≈ 0.64), a textbook dish estimate is worth
more than a correlation-matched noisy oracle:

| construction at r ≈ 0.64 | mean per-participant gain |
|---|---|
| textbook dish estimate | **−1.255** |
| correlation-matched noisy oracle | **−0.838** |

10 of 12 participants show the effect. The reason: the dish estimate carries the
whole meal-composition channel — fat, protein and fibre co-vary with dish — while
the reliability sweep deliberately isolates carbohydrate alone.

**Use B5's break-even as a threshold on the carbohydrate channel. Use B8's
table for what identity actually delivers.** The two `meal_gain` columns must
not be compared to each other.

### B11 — A carbohydrate error costs ~8× less than the CV track assumes

`agents/cv/CLAUDE.md` justifies its accuracy target with:

> "If the CV module gets carbs wrong by 30g, it can shift predicted glucose by
> 40–60 mg/dL. Accuracy here is critical."

That figure is not reproducible. Fitting `delta ~ 1 + g0_mgdl + carbs_g` per
participant, so the starting glucose is controlled:

| horizon | median slope | range across participants | a 30 g error moves glucose |
|---|---|---|---|
| T+60 | **+0.190 mg/dL per g** | −0.073 .. +0.638 | **+5.7 mg/dL** |
| T+120 | +0.128 mg/dL per g | −0.072 .. +0.502 | +3.8 mg/dL |

**The measured figure is 5.7 mg/dL at T+60, not 40–60.** The stated number is
roughly 8× too large, and it is the premise on which "accuracy here is
critical" and (via C5) "portion is the dominant term in carbohydrate load" both
rest.

A separate caveat: the slope is **negative for at least one participant**. For
that person a larger meal is associated with a *smaller* rise, once `g0` is
controlled. Carbohydrate is not a uniformly signed effect across people, which
is consistent with B3's weak median correlation of +0.246.

---

### B12 — `is_fried_cooking` adds nothing measurable, and it's a power problem, not a physiology one (ADR 0004)

`history_baselines()` (`forecasting/evaluate/bigideas_response.py`) gained a
fourth model, **B4: `g0 + fitted slope + carbs + is_fried_cooking`**, with
`is_fried_cooking` derived per meal from BIG IDEAs' own `item_names` text via
the same rule-based matcher the NLP track already ships
(`utils/cooking_method.py::is_fried` — not a new keyword list).

**B4 is numerically identical to B3** on the real data: 19.52 mg/dL MAE,
both, to two decimal places. **This is not "frying doesn't matter."** The
matcher works correctly — checked directly, not assumed: 12 of 688 meals
(1.7%) are flagged fried, and reading them confirms it: *"Air Fried Chicken
Wings," "Hibachi Steak + ... + Fried Rice," "Fried Potato."* The problem is
sample size, not signal: `history_baselines` fits one coefficient per
participant via leave-one-out regression, and across 15 participants, 12
fried meals works out to under one per participant on average — there is
essentially no within-participant variation left to fit a third coefficient
against. A real effect at 1.7% prevalence is underpowered by construction
here, independent of whether frying genuinely changes the glucose response
(the published literature — `rppg/LITERATURE_REVIEW_GLUCOSE_ESTIMATION.md`
cites Bo et al. 2017 on fried-food fat content — says it plausibly does).

**What would actually answer the question:** either a cohort where fried
meals are logged densely enough per participant to fit a coefficient
(BIG IDEAs is not that cohort), or a pooled (not per-participant) model
willing to spend degrees of freedom differently — not re-running this exact
test and expecting a different number.

Reproduce: `python3 -m forecasting.evaluate.bigideas_response --data-dir
~/Downloads/Bigideas-dataset` (gitignored raw data, not shipped).

---

## Relationship to review finding C5

`docs/REVIEW_FINDINGS.md` C5 states:

> "a single 2D photo with no scale reference is geometrically incapable of
> recovering portion volume, and **portion is the dominant term in carbohydrate
> load**."

The **mechanism** in C5 stands and is unaffected by this work. C5's architecture
argument — that a classification-labelled set gives zero within-class variance,
so the regression head learns a lookup table rather than volume — is a fact
about the training data and remains true. `portion_reported`, the in-app
question C5 introduced, is still a sensible input.

What this work qualifies is the **magnitude**, in two ways:

1. **The 40–60 mg/dL premise is not reproducible** (B11: measured 5.7 mg/dL per
   30 g at T+60). C5's "dominant term" is downstream of that figure.
2. **Portion is 2.5–5.0% of carbohydrate variance, not the dominant term**
   (B7). Which dish is 97.5%.

Consequence for the contract: `portion_reported` should stay — it is cheap, it
is the only portion signal CV can have, and B5 shows the meal term is sensitive
to *carbohydrate* accuracy in general. But it should not be treated as the
highest-value field in the CV schema, and B5's break-even should not be read as
a portion-accuracy target. The higher-value change to the CV track is **class
breadth** (B9: 455 dishes in these logs against 10 classes in
`cv/MODEL_CARD.md`).

This section is a finding for the PM, not a rewritten C5. Amending
`docs/REVIEW_FINDINGS.md` and `agents/cv/CLAUDE.md` is a PM decision and has not
been made here.

---

## What this changes

1. **The headline claim is real but modest.** "Meal features beat history
   baselines" is measured at **1.49 mg/dL at T+60 and 4.23 at T+120** over a
   properly specified history baseline (B2g), on real paired data. It is not
   2.70 mg/dL, which is the gap against a baseline that is itself beatable
   without any food input.

2. **Recommend adding B2g to the contract and to
   `forecasting/evaluate/baselines.py`.** It is the strongest meal-free
   baseline and the contract's B2 does not contain it.

3. **The CV track's bottleneck is dish breadth and personalisation.** Portion
   estimation is worth 2.5% of the carbohydrate variance (B7, B8). Ten classes
   cannot reach the operating point B9 measures. Retract the earlier
   recommendation that portion estimation was the missing capability.

4. **CV may not need to get carbohydrates exactly right.** B10 shows the dish
   estimate outperforms a correlation-matched carbohydrate estimate, because
   dish identity proxies fat, protein and fibre. Getting the *dish* right may
   be worth more than getting the *grams* right.

5. **B9's cold-start column is the honest deployment number** (38% of the
   meal's value) until a system has seen enough of a user's meals. The 71%
   personalised figure assumes that warm-up has already happened.

6. **Two published figures should be corrected.** `agents/cv/CLAUDE.md`'s
   "30 g → 40–60 mg/dL" is not reproducible; the measured value is 5.7 mg/dL at
   T+60 (B11). And C5's "portion is the dominant term in carbohydrate load" is
   contradicted by B7 (2.5–5.0% of variance). Both are premises for how the CV
   sprint has been prioritised. See "Relationship to review finding C5" above.

---

## Limitations

- **Cohort mismatch.** 15 pre-diabetic participants, not the 14 T2D in
  CGMacros. Findings may not transfer.
- **The PPG channel is untested here.** BIG IDEAs ships wristband BVP at
  64 Hz; deployment is finger-on-lens at 30 Hz. Out of domain on the right
  axis, and not analysed in this work. The RAW PPG contribution remains
  unmeasured.
- **Self-reported macros — the load-bearing limitation.** Amounts are what
  participants typed into a food-search app. B7 measures the variance in
  *logged* carbohydrate, which is exactly the quantity the app's own arithmetic
  produces (`fixed density × amount`, CV 1.2%). If people eat more variably
  than they log, the true portion share is **larger** than 2.5–5.0% and this
  work understates it. A weighed ground truth would settle it; the dataset that
  has one is CGMacros, and that is the dataset whose meal-feature modelling is
  currently blocked (see `docs/CGMACROS_DATA_QUALITY_REPORT.md`). **The
  direction of this bias is known and it runs against the headline.**
- **The 455-versus-10 comparison is cross-cultural.** BIG IDEAs is a US cohort
  — the logs contain chicken nuggets, not nasi goreng — and the model card's 10
  classes are Indonesian. The *argument* survives: identifying the dish rather
  than sizing it is the bottleneck, and 10 conceptual classes cannot cover a
  real diet. The specific numbers are not a like-for-like comparison and must
  not be presented as one.
- **The 0.41 break-even comes from simulated correlation loss**, not from real
  classifier errors. A classifier's mistakes are systematic — confusing rice
  with noodles every time — which is more damaging than the random corruption
  modelled in B5. Treat 0.41 as indicative.
- **Single dataset, no replication.** 15 participants, one cohort, one
  country. ShanghaiT2DM (100 T2D, dietary text paired with CGM) is available
  and would be a genuine independent test of B7.
- **Within-participant leave-one-out.** This measures personalisation, not
  cross-participant generalisation.
- **The `self` textbook is optimistic.** It assumes the system has already
  logged enough of the user's meals to know their dishes. The `cohort` column
  is the cold-start equivalent.
- **No p-values.** Meals within a participant are correlated; independence
  assumptions do not hold. Medians across participants are the reportable
  statistic and no significance is claimed.
- **Bucket bands are population-relative.** "Small / medium / large" in B8 are
  cohort quantiles of the logged amount, not absolute gram bands, and a
  different cohort would shift them.
- **Portion measurement floor.** 51% of repeated dishes are logged at one
  amount. Since `amount` is a genuine entry (see B7), this is mostly real
  repetition — the same breakfast most mornings — not an untouched default. It
  is still a floor: self-reported amounts are coarser than weighed ones, so the
  true within-dish variance is *at least* this large, never smaller.

---

## What would change this answer

Ordered by how much each would move the conclusion.

1. **Weighed macros.** B7 is measured on *logged* carbohydrate derived from a
   self-reported amount, so the self-report bias runs against the headline.
   CGMacros has weighed macros for 45 participants and would settle whether
   portion is 2.5% or substantially more. Its meal-feature modelling is blocked
   by the range violations already written up in
   `docs/CGMACROS_DATA_QUALITY_REPORT.md` and not yet sent to the maintainers.
   **Resolving those violations is the single change that would most improve
   this work**, and it is the one item that turns a caveat into a number.
2. **An independent cohort.** ShanghaiT2DM (100 T2D, dietary text paired with
   CGM) needs no new permissions and is already listed in
   `docs/DATA_STRATEGY.md`. It would test whether dish-dominance holds outside a
   15-person US pre-diabetic group — and it is the only available source of T2D
   meal data.
3. **Real classifier errors instead of simulated ones.** B5 corrupts the
   carbohydrate estimate with random noise at a controlled correlation. A real
   classifier confuses the same foods every time, which is worse. Replacing the
   simulation with an actual confusion matrix would make the 0.41 break-even
   defensible.
4. **A like-for-like class comparison.** Re-express the 455 dishes at the
   granularity a classifier would actually output, instead of comparing a US
   food log against the Indonesian 10-class set.
5. **The PPG channel.** BIG IDEAs ships wristband BVP. Nothing here touches it.
   The tri-modal claim is still untested end to end.

Items 1 and 2 are dataset work, not modelling work, and both are unblocked by
anything except a decision.

---

## Reproduction

```bash
python -m forecasting.evaluate.bigideas_response \
    --data-dir ~/Downloads/Bigideas-dataset \
    --json findings.json

pytest forecasting/tests/test_bigideas.py
```

15 participants, 688 meals, ~25 s. Raw data is not in the repository
(ODC-BY 1.0; download from PhysioNet v1.1.2). Set `BIGIDEAS_DIR` or pass
`--data-dir`.

An interactive version of B5, B7 and B8 is at
`forecasting/prototypes/portion-breakeven.html` — open it in a browser, no
server required.

## Open questions

1. Does a coarse portion estimate help *more* in a cohort with wider carb
   ranges or less consistent logging? B7's floor could be pressing the
   estimate down.
2. Is dish identification from a photograph good enough across 455 dish
   concepts, or does the personalisation gap (B9: 38% → 71%) mean the order
   should be personalisation first?
3. Does the composition channel in B10 hold when the estimate comes from a
   real classifier rather than a log lookup?
4. Does BIG IDEAs' wristband BVP carry any glucose signal, and does it survive
   transfer to 30 Hz contact PPG? Untested; this is the PPG track's question.
