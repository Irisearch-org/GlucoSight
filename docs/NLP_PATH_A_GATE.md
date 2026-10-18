# NLP Path A — the Day 1 gate, answered

**Date:** 2026-10-17
**Dataset:** ShanghaiT2DM — 100 T2D patients, Shanghai, China (Zhao et al.,
*Scientific Data* 10:35, 2023; `PMC9849330`)
**Question:** `agents/nlp/CLAUDE.md` §"The Day 1 gate" — *are the dietary
records rich enough to model, or are they uniformly `米饭 100g`?*
**Status:** Gate answered. The data answers "rich". The shipped pipeline
answers "grams only". Both are findings, and they disagree.

Artifacts examined:
`nlp/features/shanghai_nlp_features.csv` (and `.xlsx`),
`nlp/data/shanghai_diet_loader.py`, `nlp/data/derive_labels.py`,
`nlp/nlp_baseline/{classifier,predict}.py`, `nlp/eval_baseline.py`,
`nlp/docs/DOMAIN_KNOWLEDGE.md`, `nlp/scripts/create_dataset.py`,
`nlp/data/processed/context_{labeled,train,val,test}.csv`.

---

## Why this exists

The gate is the track's first instruction and its answer has been
outstanding since the strategy change:

> "**Download ShanghaiT2DM and read the actual dietary strings.** … If the
> strings carry real variation … Path A is a genuine extraction task.
> Proceed. If they are uniformly `米饭 100g` … Path A is a dictionary
> lookup with extra steps. It is the same failure mode as finding C5 on
> the CV side: a model that learns `feature = f(food_name)` is a lookup
> table wearing a neural network. **Say so, collapse Path A into a
> documented lookup**, and move the week's remaining effort to Path B."
> — `agents/nlp/CLAUDE.md` §"The Day 1 gate"

The data required to answer it is already in the repository as a derived
feature file, so the gate can be closed without a new download. What
follows is that closure, plus what reading the surrounding code turned up.

**Note on scope.** This answers the gate and audits the artifacts. It does
not run any glucose model. No result here is a claim about predictive
accuracy.

---

## Findings

### N1 — The records are rich, and the richness is *dish names*

3,060 meal records across 109 CGM periods (100 subjects), 365 distinct
days, 2020-09-07 → 2022-01-29, median 30 meals per period.

The gate's feared case does not hold. The strings are not uniformly
`米饭 100g`:

| measure | value |
|---|---|
| meal records | 3,060 |
| distinct meal texts | 2,762 (90% unique) |
| item segments | 9,496 |
| distinct item segments | 4,763 |
| distinct dish names (weight stripped) | 1,825 |
| items per meal | median 3, max 10; 2,570 meals list more than one item |
| segments occurring exactly once | 3,618 / 4,763 (76%) |

There is a small high-frequency core and a long tail: the top 20 dish
names cover 39% of occurrences (`rice`, `egg`, `coarse grain`,
`vegetable`, `milk`, …) while the top 20 *segments* cover only 13%. Real
composite dishes are present, with preparation and sauce:

- `Braised fish in brown sauce 120 g`
- `Stir fried pork, bamboo shoot, peanut with sauce 50 g`
- `Minced pork steamed with egg 80 g`
- `Shrimp 50 g | Steamed salted pork in wine 30 g | Fried egg with towel gourd 30 g | Cucumber 30 g | Vegetable 30 g`

**By the gate's own criteria this is the "real variation" branch.** Path A
proceeds as an extraction task, not as a documented lookup.

Cooking method is recoverable, but as a property of the dish name:
`is_fried_cooking` fires on 11.9% of meals (364), and when it fires it
fires on the name — `Fried rice`, `Pan fried bun`, `Fried noodles`,
`Deep-fried dough stick`. `is_large_portion` fires on 27.1%.
The dataset paper describes this field as *"Self-reported time and
weighed food intake"* — the names and the weights are recorded, and the
weights are measured rather than typed.

### N2 — The pipeline discards the dish name and keeps only summed grams

The feature vector Path A actually produces is
`(total_grams, is_fried_cooking, is_large_portion, nlp_confidence)`.

`total_grams` is the regex sum of every `\d+ g` / `\d+ ml` in the string
(`derive_labels.py:_total_grams`), so
`Rice 150 g | Vegetable 100 g` → 300 g — a bowl of rice and a pile of
cabbage contribute identically.

Composition is then a **single global constant**
(`nlp/docs/DOMAIN_KNOWLEDGE.md` §1.4, `_gl_from_grams`):

```
GL = total_grams × 0.28 × 73/100
```

One carbohydrate density (0.28) and one glycemic index (73) for every
food. The dish name never enters the arithmetic except to set the fried
flag. So:

| | count |
|---|---|
| distinct dish names in the data | 1,825 |
| distinct carbohydrate densities the pipeline can represent | **1** |
| distinct meal texts | 2,762 |
| distinct `(total_grams, is_fried, is_large)` states | **661** |

The collapse is measurable: **2,762 distinct meals become 661 distinct
feature states**, and the 1,825 dish names are gone.

This matters because of `docs/BIGIDEAS_MEAL_RESPONSE_FINDINGS.md` B7,
measured independently on paired meal-and-glucose data: **97.5% of the
variance in a meal's carbohydrate is which dish was eaten, and 2.5–5.0%
is portion size.** The pipeline's feature set retains the portion term —
summed grams — and discards the dish term. It keeps the small one.

### N3 — `nlp_confidence` is a re-encoding of the two flags

`nlp_confidence` is required by the contract as a **calibrated probability,
used by Forecasting as a gating weight** ("an uncalibrated score makes
that gate meaningless" — `INTERFACE_CONTRACT_v1.md` §"Notes").

In the shipped artifact it is (near-)deterministically the two labels:

| `(is_fried, is_large)` | rows | share | mean confidence |
|---|---|---|---|
| (0, 0) | 2,007 | 65.6% | 0.6200 |
| (0, 1) | 689 | 22.5% | 0.9300 |
| (1, 0) | 225 | 7.4% | 0.6745 |
| (1, 1) | 139 | 4.5% | 0.9053 |

Within a cell the remaining spread is a function of `total_grams`
(`fried_confidence = 0.60 + fat/50 × 0.18`, same doc §2). So confidence
carries no information beyond the two binaries and the gram total. A
field that is a deterministic function of the fields beside it cannot gate
anything, and `0.62` — the value on two thirds of rows — is the
documented "neither present" prior, not a measurement.

### N4 — The Path A artifact is orphaned and cannot be regenerated

`nlp/features/shanghai_nlp_features.csv` and `.xlsx` are **git-tracked**,
but:

- **No code in the repository references them.** A grep for the filename
  across `*.py`, `*.md`, `*.ipynb` returns nothing.
- **No code in the repository can produce them.** `derive_labels.py` and
  `predict.py` both hardcode `nlp_confidence: 0.0`, and
  `nlp/tests/test_derive_labels.py` asserts exactly that
  (`test_predict_does_not_claim_uncalibrated_confidence`). The artifact's
  `0.62 / 0.93 / 0.6745` cannot come from the code that exists.
- **The source data is not on disk.** `nlp/data/Shanghai_T2DM/` (the
  loader's `DEFAULT_DATA_DIR`) does not exist, and no copy of the
  ShanghaiT2DM Excel files was found outside the repository either.

So the repository's only Path A result is a committed artifact that no
code reads, that no code can rebuild, and whose input is absent. The
`0.62`-style values are consistent with the calibration scheme described
in `DOMAIN_KNOWLEDGE.md` §3, which suggests an earlier generator that has
since been removed — and the current code deliberately refuses to emit
that confidence. The two states are not reconciled anywhere.

### N5 — The evaluation split unit is the CGM period, not the patient

`participant_id` is the Excel **file stem**, i.e. `<subject>_<period>_<date>`.
The 109 ids resolve to **100 subjects**; 8 subjects contribute 2–3
periods:

| subject | ids |
|---|---|
| 2001, 2014, 2015, 2017, 2055, 2074, 2078 | 2 each |
| 2069 | 3 |

This matches the dataset paper, which reports patients with 1, 2 and 3 CGM
periods — so the structure is faithful to the source, and the loader is
not at fault.

`nlp/eval_baseline.py`, however, groups `GroupKFold` on `participant_id`:

```python
groups = df["participant_id"].values      # file stem → period, not subject
gkf = GroupKFold(n_splits=n_splits)
```

Grouped that way, a subject's second visit is a *different group* from
their first. `2069`'s three periods are three groups; so are `2001`'s two.

What leaks is therefore fixed, but how much depends on the partition:
**9 of the 109 period-groups are a repeat visit of a subject who also
appears under another id** (7 subjects × 1 extra + 1 subject × 2 extra).
Across two different 5-fold partitions the leak landed at **566 meals
(18.5%)** and **659 meals (21.5%)** of the corpus. Either way, **one meal
in five is a test meal from a subject the model trained on**. The
grouping unit should be the subject (the first token), 100 groups.

This is the same class of error as review finding C4 ("effective sample
size is participants, not meals"), one level down.

### N6 — The extracted features contain no response variable

Each row carries the CGM value **at that row's timestamp**. Per the
dataset paper, the ShanghaiT2DM sheet is an event log: `<Date>` is the
recording time, `<CGM>` is the reading at that time (recorded every 15
minutes), and a row is a meal when it carries a `<Dietary intake>` entry.

So `cgm_mgdl` is glucose *at* the meal — roughly the pre-meal level — and
there is no T+60 or T+120 outcome anywhere in the extracted features.
96 of 3,060 rows have no CGM value at all; the column has 189 distinct
values, median 126.0, range 39.6–399.6, median 20 distinct values per
participant.

`docs/BIGIDEAS_MEAL_RESPONSE_FINDINGS.md` establishes that the meal's
value is only visible as a **rise**, fitted against a strictly causal
`g0` (B3, B4). Path A as currently extracted has a predictor and no
outcome, so it cannot be evaluated for glucose contribution at all — not
because the data is missing, but because only the meal rows were kept.
The 15-minute series needed to reconstruct `g0` and the T+60 rise exists
in the raw files, which are not present.

> **Update 2026-10-18 — N6 answered.** The raw v3 workbooks were downloaded
> (3.4 MB, CC BY 4.0) and the response rebuilt: 3,045 of 3,060 meals now
> carry a T+60 rise (99.51%), with `g0` strictly causal and every exclusion
> named. `corr(g0, T+60 rise) = −0.296` — the same structure as BIG IDEAs.
> The rebuild also found the old loader silently lost CGM for 96 meals
> (3.1%), all in the two workbooks whose header spells the column `"CGM "`
> with a trailing space. See `docs/SHANGHAI_RESPONSE_REBUILD.md` (R1–R6) and
> `nlp/data/shanghai_response.py`. No model was fitted; N2 stands.

### N7 — Path B's evaluation set leaks, and its labels *are* the rules

`nlp/data/processed/context_labeled.csv`: 400 rows, 400 unique strings —
but only a few hundred distinct base sentences. `create_dataset.py` takes
~120 hand-written templates and emits each one under one of five prefixes
(`Tadi …`, `Hari ini …`, `Barusan …`, `Aku tadi …`, or bare), so strings
differing only by prefix are near-duplicates. Depending on how the
prefixes are stripped, the set collapses to **124–143 distinct base
sentences**, with **355–379 of the 400 strings sharing their base with
another string**.

The train/val/test split is by string, not by base sentence, so:

| split | base sentences | also present in train |
|---|---|---|
| train | 116–124 | — |
| val | 50–51 | **42–46 (84–90%)** |
| test | 50–51 | **42–46 (84–90%)** |

A model scored on this split is scored on paraphrases of its own training
sentences. Any reported test accuracy from it is inflated and should not
be quoted. The range is the prefix-stripping convention, not noise in the
measurement — but every convention tried leaves at least 84% of test
sentences with a near-duplicate in train.

Worse than the leak, the labels have no independent source.
`create_dataset.py:annotate` assigns each of the 8 labels from hand-written
keyword lists (`"goreng"`, `"begadang"`, `"porsi jumbo"`, …), and
`nlp/nlp_baseline/classifier.py` re-implements the same lists. A
classifier trained on this set and evaluated on it measures **agreement
between two copies of one rule list** — not the rule list's agreement with
human judgement, and not its agreement with glucose. The set contains no
glucose column, consistent with `agents/nlp/CLAUDE.md`'s statement that
anything built on Indonesian text has no glucose link.

The 8 labels also exceed the contract: `stress`, `poor_sleep` and
`high_activity` were removed in v1.2, and `sweet_drink`, `late_meal`,
`high_carb_hint` were never in it.

### N8 — Weighed data in the same repository tests B7's portion floor, and the floor looks real

`docs/BIGIDEAS_MEAL_RESPONSE_FINDINGS.md` B7 measures portion at
**2.5–5.0%** of carbohydrate variance, and its first caveat says the
figure is a *floor* because it is computed on **self-reported** amounts
("A weighed ground truth would settle it"). Item 1 of "What would change
this answer" is exactly that test.

ShanghaiT2DM is **weighed** food intake, and it is already local. It has
no carbohydrate table, so carbohydrate cannot be measured — but if
carbohydrate density is constant within a dish (which is what B7
establishes), then within-dish variance in `log(carbs)` equals within-dish
variance in `log(grams)`. So the portion share is comparable in kind.

Decomposition of `log(grams)`, dishes with ≥2 servings, using **B7's own
grouping unit** (participant × dish):

| source | BIG IDEAs, self-reported (B7) | ShanghaiT2DM, weighed (N8) |
|---|---|---|
| within dish — **portion** | 2.5% (5.0% restricted) | **16.8%** |
| between dishes — **which dish** | 97.5% | 83.2% |
| repeated dishes logged at exactly one amount | 51% | 27.3% |

Pooled by dish name rather than within participant — which mixes real
person-to-person differences into the "portion" term — the portion share
rises to **45.3%**, with 15.0% single-amount. The two figures measure
different things and **must not be compared to each other**: 16.8% is the
like-for-like B7 comparison; 45.3% is what a cold-start system faces when
it does not yet know the person.

The spread is stark in the raw data. `rice` alone: 548 servings, 49
distinct weights, 10 g to 500 g, CV 0.55, and across the 40 participants
with at least five servings each, the **median serving ranges from 25 g to
255 g** — a ten-fold between-person spread for the same food name.

**Interpretation.** B7's direction-of-bias caveat was correct and its
magnitude is understated. Portion is small in BIG IDEAs partly because
self-reported amounts are coarse; measured portions vary several times
more. B7's "2.5–5.0%, and it is a floor" survives — but the floor is
substantially below the true value, and 16.8% is the better current
estimate in the like-for-like case.

**What N8 does not establish.** Shanghai's dish names are coarser than
BIG IDEAs'. `vegetable`, `coarse grain` and `fish` are categories, not
dishes, so some of the 16.8% is dish-name granularity rather than portion
choice — and this data cannot separate the two. The cohorts differ too
(Chinese hospital T2D patients on weighed meals vs US pre-diabetic users
self-logging). N8 moves the estimate; it does not settle it.

---

## Relationship to the BIG IDEAs findings

- **B7** — the "2.5–5.0%, a floor" caveat is confirmed as a real bias and
  now has a number attached in the like-for-like case (**16.8%** while
  self-reported is 2.5%). B7's *conclusion* is unaffected: which dish
  dominates carbohydrate, and portion estimation is not CV's missing
  capability. But "portion is a small term" should be qualified as
  "portion is a small term *as logged*, and roughly 7× larger when
  weighed".
- **B9** — N8's 16.8% vs 45.3% split is the personalisation gap again.
  Between-person portion differences are the larger component, which is
  the same argument B9 makes about dish identity (38% cold start → 71%
  personalised).
- **C5** — nothing here revises C5 further. The mechanism stands, the
  magnitude was already amended on 2026-10-17.
- **BIG IDEAs "What would change this answer" item 2** ("an independent
  cohort … ShanghaiT2DM … would test whether dish-dominance holds outside
  a 15-person US pre-diabetic group") is **partially answered**: at 16.8%
  / 83.2% weighed, dish dominance holds but is less extreme than B7's
  97.5% implied.

---

## What this changes

1. **The gate is closed, and its answer is "rich" — but the pipeline
   implements the other branch.** The strings carry 1,825 dish names; the
   feature vector keeps one carbohydrate density. Path A is not currently
   the extraction task the gate authorised; it is the lookup the gate
   warned about, reached by discarding the input rather than by lacking
   it.

2. **Decide which branch Path A is on, explicitly.** The gate offers two.
   Either (a) derive composition from the dish name against a composition
   table — TKPI/DKBM or USDA, which is *the same dictionary CV already
   looks up*, computed from text instead of pixels — or (b) declare Path A
   a documented lookup, set `nlp_feature_source=lookup`, and move the
   effort to Path B, as the gate's second branch instructs.
   Recommendation: (a), because the dish names demonstrably exist and the
   dictionary already exists for CV. But it should be a recorded decision,
   not a default.

3. **The honest contribution statement.** With the pipeline as it stands,
   NLP contributes **meal weight** and **cooking method**, and on
   ShanghaiT2DM the weight is the only portion signal in the project (the
   contract already says so, §"Notes"). The dish identity that B7 shows
   carries 97.5% of carbohydrate variance **is present in the text and is
   currently discarded**. That sentence is the answer to "what does NLP
   add that CV doesn't", and it is more useful stated plainly than
   defended.

4. **Fix the split unit before any Path A number is reported.**
   `eval_baseline.py` should group on the subject, not the file stem.
   18.5–21.5% of the corpus is currently a repeat visit of a training
   subject, depending on the fold partition.

5. **Path A needs a response variable.** Reconstruct `g0` and the T+60
   rise from the raw 15-minute CGM series, exactly as
   `forecasting/data/bigideas.py` does. Without it there is no outcome to
   model. This requires the raw ShanghaiT2DM files, which are absent.

6. **Re-scope Path B's evaluation.** Split by base sentence, not by
   sentence, and stop reporting any score derived from
   rule-with-rule agreement. If Jev-style zero-shot decision models are
   used here (see Open questions), that changes the *extractor*, not the
   absence of an independent label source.

7. **Correct B7's caveat in the paper's Limitations.** It currently says
   the portion share "may be larger". N8 says it is larger, by roughly 7×
   in the like-for-like case, with a measured alternative figure.

---

## Limitations

- **No glucose modelling was done here.** Nothing in this document is a
  claim about predictive accuracy. N1–N8 are property and provenance
  findings.
- **N8's dish granularity confound.** Shanghai dish names are coarser than
  a food-search app's. Part of the 16.8% is granularity, not portion
  variability. Not separable with this data, and the direction is unknown.
- **N8's 16.8% and 45.3% are different quantities.** Within-participant is
  the like-for-like comparison to B7; pooled is the cold-start view. They
  must never be quoted against each other.
- **N8 compares across cohorts and diets** — Chinese hospital T2D patients
  on weighed meals vs US pre-diabetic participants self-logging. A
  difference in portion variance is expected for reasons unrelated to
  measurement.
- **N8 measures grams, not carbohydrate.** The equivalence to B7's
  log(carbs) decomposition holds only while carbohydrate density is
  constant within a dish, which is B7's own claim and is itself
  measured on self-reported data.
- **The Shanghai raw files were not examined.** All Shanghai findings come
  from the derived feature file plus the dataset paper. Row-level parsing
  choices in the original loader could not be re-verified.
- **N5 and N6 concern `eval_baseline.py` and the extracted feature set**,
  not the loader. The loader faithfully reports the source's period
  structure; the grouping and outcome construction downstream of it are
  the problem.
- **The `0.62` generator is unidentified.** N4's conclusion is that the
  artifact is unreproducible, which rests on the absence of a matching
  generator in the repository — not on proof that none ever existed.

---

## What would change this answer

1. **The raw ShanghaiT2DM files.** They would allow N6 to be resolved
   (`g0` + T+60 rise) and the loader's parsing to be re-verified. This is
   the single highest-value item, and it unblocks Path A entirely.
2. **A dish-name-to-composition table applied to this data.** It would
   replace the single 0.28/73 constant and immediately test whether Path A
   carries carbohydrate information the gram total does not. The
   dictionary CV already intends to use would serve.
3. **A hand-annotated sample of Shanghai records.** 150–200 manually
   labelled meals would give `is_fried_cooking` and `is_large_portion` an
   error rate, which they currently lack entirely.
4. **Dish-name granularity control for N8.** Re-run the decomposition at
   the granularity a classifier would output, to separate portion
   variability from name coarseness.
5. **The CGMacros weighed macros**, still blocked by
   `docs/CGMACROS_DATA_QUALITY_REPORT.md`. CGMacros measures carbohydrate
   directly, which would settle what N8 can only bound.

---

## Reproduction

```bash
python -m nlp.eval_path_a_gate
pytest nlp/tests/test_path_a_gate.py
```

`nlp/eval_path_a_gate.py` prints every number quoted above (N1–N8) and
reads only committed files — no download, no network, no glucose model.
Two figures are partition- or convention-dependent and are reported as
ranges in the text rather than point values: the N5 fold leak (which
depends on how the folds are cut) and the N7 base-sentence count (which
depends on how the prefix variants are stripped). The script prints one
convention of each.

Three readings of the code are needed for N2 and are not reproducible
from the feature file alone:

```bash
grep -n "0.28" nlp/docs/DOMAIN_KNOWLEDGE.md     # the single carb density
grep -n "_total_grams" nlp/data/derive_labels.py  # summed, not per item
grep -n "GroupKFold\|groups =" nlp/eval_baseline.py  # the split unit (N5)
```

N8's decomposition is a log-variance split of `log(grams)` (within dish vs
between dish), restricted to dishes with ≥2 servings, run twice — grouped
by dish name, and grouped by (participant, dish). The second is the
B7-comparable figure. `total_grams` must **not** be used for this; per-item
weights are recovered by parsing each `|`-separated segment.

ShanghaiT2DM is not redistributable from this repository; the raw Excel
files are absent from disk, which is itself finding N4.

---

## Open questions

1. Does a composition table applied to Shanghai's dish names lift the
   carbohydrate channel measurably over summed grams? This is the whole of
   Path A's remaining value and it is a small, bounded experiment.
2. Is `cgm_mgdl` in the raw sheet the value *at* the meal, or the meal's
   postprandial peak? The paper's field description (`<Date>` = recording
   time, CGM every 15 min) says the former, but the raw files were not
   available to confirm against the published example tables.
3. Does the pattern in N2 generalise — i.e. do other tracks also build a
   feature that retains a small term and discards a large one available in
   the input? `forecasting/evaluate/bigideas_response.py` is the tool for
   asking it of the CV path.
4. Zero-shot decision models (TypeSafe Jev, Sept 2026) remove the need for
   a *training set* for the label extraction in Path B. They do not
   address N7: the labels still have no independent source, and Bahasa
   text still has no glucose link. Worth recording, because the question
   recurs, but it does not change the track's status.
