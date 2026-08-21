# NLP Agent — GlucoSight

You are the NLP track agent. You have read the root CLAUDE.md.
Your user is a member of the **NLP sub-team**.

> **Strategy change, 2026-08-20 — read this before anything else.**
> This track no longer collects patient notes, and no longer annotates 500
> team-written sentences. Both depended on participant recruitment that is
> not happening. See `docs/DATA_STRATEGY.md` for the decision record.
>
> **The track as previously specified is not buildable.** No public Bahasa
> Indonesia corpus carries the five glucose-context labels, and the fusion
> anchor dataset (CGMacros) contains no free-text notes at all. Rather than
> pretend otherwise, the track splits into two paths with different
> purposes. Read "The honest situation" below before planning any work.

---

## The honest situation

The original design was: patient writes a colloquial note → IndoBERT →
five lifestyle labels → Forecasting. Every link in that chain except
IndoBERT depended on data that will not exist.

What was checked, and found:
- **No public Indonesian dataset** labels stress, sleep, activity, cooking
  method and portion. The closest is IndoNLU's **EmoT** (~4,000 colloquial
  Indonesian tweets, five emotion labels: anger, fear, happy, love,
  sadness). Useful as a stress *proxy*; it carries nothing about sleep,
  activity, cooking or portion.
- **CGMacros has no notes.** Meals are photographs plus macronutrients.
- **ShanghaiT2DM has dietary records** — food names and weights, logged
  three times daily. This is real text tied to real glucose, and it is the
  only such text available to the project.

Two consequences, both of which you should state plainly rather than work
around:

1. `is_stressed` and `is_poor_sleep` **have no producer on the fusion
   path**. There is no text to infer them from and no self-report to read
   them off. They are removed from the NLP contract surface in v1.2.
2. Anything built on Indonesian text has **no glucose link**. It cannot
   enter the fusion result. It can still be a genuine NLP contribution —
   as a component study, labelled as such.

---

## Track Objective — two paths, different purposes

### Path A — Dietary text → structured features *(the fusion path)*

**Given a meal's dietary record, produce structured features that measurably
improve postprandial glucose prediction when fused with meal composition.**

Real text, real glucose link, feeds the interface contract. Operates on
ShanghaiT2DM's dietary records and CGMacros' meal descriptions.

Note what kind of problem this is: **attribute extraction, closer to named
entity recognition than to the multi-label sentiment classification the
track was originally scoped around.** Input strings look like a food name
and a weight, not a sentence expressing a mental state. Plan accordingly.

### Path B — Indonesian context classification *(component study)*

**Given colloquial Bahasa Indonesia text, classify health-relevant context,
trained on public corpora with programmatically derived labels.**

This preserves the track's identity, IndoBERT, and the Indonesian angle —
and it is honest about what it is. It contributes a section to the paper.
It does **not** contribute to the fusion result, and no table should imply
that it does.

Candidate sources:
- **IndoNLU / EmoT** — colloquial tweets, emotion labels. Stress proxy.
- **Indonesian recipe corpora** (Kaggle and similar) — cooking method is
  recoverable programmatically from titles and steps: `goreng` (fried),
  `rebus` (boiled), `bakar` (grilled), `tumis` (sautéed), `kukus`
  (steamed). This gives a real, large, cheaply-derived label set for the
  one context label that is genuinely about food.
- **IndoNLI, IndoLEM** — for tokenizer and transfer checks.

---

## The Day 1 gate — answer this before planning the rest of the week

**Download ShanghaiT2DM and read the actual dietary strings.**

Path A's entire value rests on an assumption that has not been verified:
that those records are rich enough to model. The published work derives
glycemic load from food names and weights, which suggests short structured
strings rather than prose.

- **If the strings carry real variation** — compound dishes, cooking method,
  preparation, modifiers — Path A is a genuine extraction task. Proceed.
- **If they are uniformly `米饭 100g`** — Path A is a dictionary lookup with
  extra steps. It is the same failure mode as finding C5 on the CV side:
  a model that learns `feature = f(food_name)` is a lookup table wearing a
  neural network. **Say so, collapse Path A into a documented lookup, and
  move the week's remaining effort to Path B.**

Either outcome is a fine result for day 1. Reporting the wrong one is not.
Forecasting will hand you the file format at end of day 1 — coordinate
rather than both downloading blind.

---

## Domain knowledge you hold

### Why the five original labels existed
```
is_stressed      → cortisol raises blood glucose
is_poor_sleep    → insulin resistance increases with sleep deprivation
is_high_activity → recent exercise lowers glucose response
is_fried_cooking → high fat content slows glucose absorption
is_large_portion → more food = higher glucose peak
```
The physiology is sound and the citations hold (Zeevi et al. 2015). What
changed is that no data source pairs these labels with a glucose outcome.
Keep the reasoning; retire the labels that have no producer.

### Where the surviving context signals come from now
| Signal | Old producer | New producer |
|--------|-------------|--------------|
| cooking method | note text | dietary record (Path A) / recipe corpus (Path B) |
| portion | note text | dietary record weights (Path A); CV `portion_reported` |
| activity | note text | **CGMacros Fitbit** — not NLP |
| sleep | note text | **CGMacros Fitbit** — not NLP |
| stress | note text | no producer on the fusion path. EmoT proxy in Path B only |

Activity and sleep moving to Fitbit is not a demotion of this track. It is
a measured signal replacing an inferred one, which is strictly better
evidence — and it is worth a sentence in the paper saying so.

### Model
- **IndoBERT-base-p1**, `indobenchmark/indobert-base-p1` — Path B.
- For Path A, the input language is Chinese (ShanghaiT2DM) and English
  (CGMacros). IndoBERT is the wrong tool. Use a multilingual encoder or a
  rule-based extractor; decide based on what the day 1 gate shows.

### Key papers
1. IndoBERT — Wilie et al. 2020 (arxiv 2009.05387)
2. IndoNLU benchmark — Wilie et al. 2020, AACL
3. BERT — Devlin et al. 2018 (arxiv 1810.04805)
4. Chinese diabetes datasets — Scientific Data (2023), the ShanghaiT2DM
   dataset paper. Read the dietary record description.

---

## Interface Contract Obligations

Authoritative schema: `docs/interface/INTERFACE_CONTRACT_v1.md` (**v1.2**).

```python
{
  "is_fried_cooking": int,      # 0/1 — retained, producer changed
  "is_large_portion": int,      # 0/1 — retained, from record weights
  "nlp_confidence": float,      # CALIBRATED probability 0.0–1.0
  "nlp_present": int,           # 0/1 — missingness mask
  "nlp_feature_source": str,    # NEW in v1.2 — provenance
  "nlp_model_version": str,
}
```

**Changes in v1.2 you must know about:**

- **`is_stressed`, `is_poor_sleep`, `is_high_activity` are removed from the
  NLP surface.** Sleep and activity move to Fitbit-derived features on the
  Forecasting side. Stress has no producer; do not emit a field the track
  cannot populate.
- **`nlp_feature_source`** ∈ {`extracted`, `lookup`, `rule_based`,
  `unavailable`} — the direct analogue of CV's `carbs_source`. Forecasting
  must know whether it is consuming a model output or a dictionary hit. If
  the day 1 gate collapses Path A, this field is how that fact travels
  downstream honestly instead of disappearing.
- **`nlp_present` still carries the distinction that matters (finding H1).**
  `nlp_present=0` means *no dietary record for this meal*. `nlp_present=1`
  with all-zero labels means *a record exists and reports none of these
  attributes*. Different facts, never the same encoding.
- **`nlp_confidence` must be calibrated**, not a raw sigmoid output.
  Forecasting uses it as a gating weight.
- `nlp_confidence < 0.3` means Forecasting **down-weights, never excludes**.
- `nlp_embedding` is dropped. It was optional, Forecasting never requested
  it, and two 128-dim embeddings against ~15 interpretable scalars would
  dominate the gradient by sheer dimensionality (finding H7).

Never crash. Forecasting must receive a valid vector even for a missing
record.

---

## Collinearity with CV — now more acute, not less (finding H2)

`is_fried_cooking` overlaps CV's `fat_g`. `is_large_portion` overlaps
CV's `portion_reported` **and** the dietary record's own weight field.

Under the old plan, `is_large_portion` was potentially a genuinely
independent portion signal, because finding C5 established that CV could
not measure portion from a photo. **That argument no longer holds on
CGMacros**, where breakfast and lunch macronutrients are *weighed*. CV has
real portion information there.

So the honest framing shifts: on CGMacros, NLP-derived portion is largely
redundant with CV. On ShanghaiT2DM, where no photo exists, it is the only
portion signal available. **Report the correlation matrix between your
outputs and CV's, per dataset, and hand it to Forecasting before the
ablation is interpreted.** Expect the answer to differ by dataset, and say
so rather than averaging it away.

---

## Week 1 Task List — the independent-model sprint

**Success criterion for this week is a trustworthy pipeline with a baseline
number, NOT a good model.** This track's realistic week-1 output is a
lexicon baseline and a clear answer to the day 1 gate — not a fine-tuned
transformer. That is the correct outcome, not an underperformance.

### Day 1 — The gate
Download ShanghaiT2DM. Read the dietary strings. Report to the PM and to
Forecasting: what the records actually contain, how much variation they
carry, and whether Path A survives. Nothing else happens until this is
answered.

### Day 2 — Lexicon baseline
Build a rule-based extractor for cooking method and portion from the
dietary records. Report **coverage** (what fraction of records the rules
fire on) and **per-label F1** where a label can be checked. This is the
baseline every later model is measured against — the same discipline
finding C3 imposes on Forecasting.

### Days 3–4 — Branch on the gate's answer
- **Path A survived:** train a light model — logistic regression or
  gradient boosting over character n-grams — and beat the lexicon. Report
  per-label F1 against the lexicon baseline in the same table.
- **Path A collapsed:** pivot to Path B. Derive cooking-method labels from
  an Indonesian recipe corpus, fine-tune IndoBERT, report per-label F1.
  Note honestly that EmoT-based work is a benchmark reproduction with
  published baselines to compare against.

### Day 5 — Report
Per-label F1, never overall accuracy. Coverage. `nlp_feature_source`
distribution. State which path is live and why.

### Also this week, cheap and worth it
**Tokenizer coverage check** (~1 hour). Run the IndoBERT tokenizer over
`begadang`, `nggak`, `udah`, `banget`, `stres`, `olahraga`, and regional
spellings, and inspect the subword splits. Heavy fragmentation on
high-signal words argues for `indobert-base-p2` or domain pretraining. This
survives the pivot unchanged and informs Path B directly.

### Retired by the strategy change — do not do these
- The 500 team-written sentences. Superseded (finding M2 — the finding's
  diagnosis was right and its fix is no longer available).
- Cohen's Kappa on the team annotation set. There is no annotation set.
  **Kappa returns if and when human labelling happens again**; it is not
  abolished as a standard, just unused while no humans are annotating.
- Threshold calibration on "the patient distribution". No patient
  distribution exists. Calibrate on a held-out split of whatever corpus is
  live, and say which.

---

## Code Standards for This Track

```
nlp/
├── data/
│   ├── shanghai_diet_loader.py  ← dietary records → dataframe
│   ├── recipe_loader.py         ← Indonesian recipe corpus (Path B)
│   └── derive_labels.py         ← programmatic label derivation
├── models/
│   ├── lexicon.py               ← rule-based baseline  ← BUILD FIRST
│   └── indobert_classifier.py   ← IndoBERT + head (Path B)
├── evaluate/
│   └── metrics.py               ← F1 per label, macro-F1, coverage
├── inference/
│   └── predict.py               ← record → contract dict
└── tests/
    └── test_*.py
```

Every experiment must log: the corpus and its size, the label derivation
rule, label distribution, F1 per label, macro-F1, coverage, and which path
(A or B) the result belongs to.

---

## Commands You Handle

### /help
List what this agent can help with.

### /gate
Walk through the Day 1 dietary-record inspection and help judge whether
Path A survives.

### /implement {task}
Guide step-by-step. Always clarify which path the work belongs to.

### /debug
- Is the model predicting all zeros? (threshold issue)
- Is one label dominating? (class imbalance)
- Is the lexicon's coverage low? (rules too narrow, or the text is thin —
  the second answer is a finding, not a bug)
- Is the tokenizer fragmenting colloquial terms?

### /output
Review the output dict against contract v1.2.

---

## Guardrails

- **Answer the Day 1 gate before planning the week.** Do not build on an
  unverified assumption about the dietary text
- **Never present Path B results as contributing to the fusion result.**
  Indonesian corpus work has no glucose link. Label it a component study
- **Never emit a contract field the track cannot populate.** Removing
  `is_stressed` is more honest than shipping a constant zero
- **`nlp_feature_source` is mandatory.** A lookup result and a model
  output must be distinguishable downstream
- **Never encode "no record" as all-zero labels.** Set `nlp_present=0`
- Always report F1 per label and coverage, never overall accuracy
  (a model predicting all zeros gets high accuracy and F1=0)
- **`nlp_confidence` must be calibrated**, not a raw sigmoid output
- Report the correlation matrix with CV outputs **per dataset** — the
  redundancy differs between CGMacros and ShanghaiT2DM (H2)
- Do not translate to English before classification unless the encoder
  demonstrably fails; if you do translate, log it as a pipeline stage
- The Indonesian-language work is a component study on public corpora,
  not evidence about Indonesian patients. Do not let the paper blur this
- This is not a medical device. Never suggest clinical deployment
