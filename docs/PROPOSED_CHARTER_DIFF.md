# Proposed diff — CV/NLP re-scope (companion to ADR 0002)

**Not applied.** This is the reviewable diff the Further Notes section of
`docs/SPEC_DISH_IDENTIFICATION.md` and ADR `0002` both call for. `CLAUDE.md`,
`agents/cv/CLAUDE.md`, `agents/nlp/CLAUDE.md`, and the interface contract
are edited only after the team agrees to this — not by this ticket, and not
by anyone acting on this document alone without that review.

---

## `CLAUDE.md`

```diff
- 📷 CV           — food recognition + macro estimation from meal photo
+ 📷 CV           — dish identification from meal photo (macros via a shared lookup table)
- 📝 NLP          — context classification from Bahasa Indonesia text
+ 📝 NLP          — owns the shared dish table; cooking-method correction from text where it exists
```

---

## `agents/cv/CLAUDE.md`

Current (Track Objective, opening lines):

> **Given a smartphone photo of an Indonesian meal, produce a per-meal
> composition vector — macros, GI category, and a calibrated confidence —
> that measurably improves postprandial glucose prediction when fused with
> context and optical signals.**

Proposed:

> **Given a smartphone photo of an Indonesian meal, identify every distinct
> dish on it, each with a calibrated confidence.** Macros and GI category
> come from the shared dish table via the matcher (`integration/
> dish_table.py`), not from this track's model — see ADR 0002. A mixed
> plate (Task 5, below) returns a list of dishes, not one label.

And resolve Task 5 in place (it currently poses the question this ADR
answers):

```diff
  ### Task 5 — Handle mixed plates
  Nasi padang is several distinct components on one plate. Single-label
  classification cannot represent it. Decide and document: multi-label
- classification, or detection + per-component macro summation. Flag to the PM
- which Indonesian dishes in the target list are multi-component.
+ classification, or detection + per-component macro summation.
+ **Decided (ADR 0002): detection + per-component summation.** The VLM
+ returns a list of dishes; each is matched against the dish table and the
+ macros sum. No multi-label classifier is trained.
```

---

## `agents/nlp/CLAUDE.md`

Path A's objective line gains an explicit deliverable (it already does
this work per ADR 0001; this names it as the track's ongoing job, not a
one-off finding):

```diff
  ### Path A — Dietary text → structured features *(the fusion path)*

  **Given a meal's dietary record, produce structured features that measurably
  improve postprandial glucose prediction when fused with meal composition.**
+
+ **Standing deliverable (ADR 0002): own and grow the shared dish table**
+ (`cv/data/tkpi_macro_lookup.csv`), consumed by both tracks through the
+ matcher (`integration/dish_table.py`). Grow it reactively — add a row when
+ an evaluation run hits a name the matcher can't resolve, not pre-built
+ toward a larger published list.
```

Path B's "does not contribute to fusion" claim needs a named exception —
right now it is a blanket statement that the cooking-method signal
contradicts:

```diff
  ### Path B — Indonesian context classification *(component study)*

  **Given colloquial Bahasa Indonesia text, classify health-relevant context,
  trained on public corpora with programmatically derived labels.**

  This preserves the track's identity, IndoBERT, and the Indonesian angle —
  and it is honest about what it is. It contributes a section to the paper.
- It does **not** contribute to the fusion result, and no table should imply
- that it does.
+ It does **not** contribute to the fusion result, **with one named
+ exception (ADR 0002): cooking-method extraction** (`goreng`/`rebus`/
+ `bakar`/`tumis`/`kukus`, recoverable from text the same way as today)
+ feeds `is_fried_cooking` as a correction on CV's already-identified
+ dish, gated by confidence and checked for redundancy against CV's own
+ signal (H2) before being trusted as independent. Every other Path B
+ signal (the stress proxy) stays out of the fusion result, no table should
+ imply otherwise.
```

---

## `docs/interface/INTERFACE_CONTRACT_v1.md`

```diff
- **`carbs_source` ∈ {`weighed`, `cv_regressed`, `class_lookup`, `population_mean`}**
+ **`carbs_source` ∈ {`weighed`, `cv_regressed`, `class_lookup`,
+ `class_lookup_multi`, `population_mean`}** — `class_lookup_multi` is a
+ mixed-plate result: macros summed across more than one matched dish-table
+ item (ADR 0002), kept distinct so it isn't mistaken for a single-dish
+ lookup downstream.
```

This would be a v1.3 changelog entry (one new enum value, additive, no
existing field removed or retyped — the smallest contract bump this
redesign needs).

---

**Status of the code these docs describe:** already built and tested on
this branch — `integration/dish_table.py` (the matcher), `integration/
adapters/cv.py::cv_items_to_contract`, `integration/adapters/
nlp.py`'s cooking-method correction, and `integration/contract.py`'s
`class_lookup_multi` value. The code runs today; these four files are the
only things still waiting on review.
