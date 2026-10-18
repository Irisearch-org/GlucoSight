# The Indonesian bridge — how Shanghai reaches a Bahasa menu

**Date:** 2026-10-18
**Question:** the sprint brief names IndoBERT. ShanghaiT2DM is Chinese and
glucose-linked. Make it Indonesian — a translation lexicon, or what?
**Status:** decided. The bridge is **dish identity → carbohydrate density**,
not text → text.
**Depends on:** `docs/NLP_PATH_A_GATE.md` (N1–N8),
`docs/SHANGHAI_RESPONSE_REBUILD.md` (R1–R6),
`docs/DATA_STRATEGY.md`.

---

## IB1 — Two different mappings are being conflated

The question "can we make it Indonesian" sounds like one problem. It is two,
and a translation lexicon only touches the first:

| mapping | from | to | solved by |
|---|---|---|---|
| **name** | `nasi goreng` | `fried rice` | a lexicon |
| **food** | `nasi goreng` | 27 g carbohydrate / 100 g | a composition table |

A lexicon that produces `fried rice` has given you an English string. The
model still has no number for it. The mapping that carries predictive weight
is the second one, and no amount of translation produces it.

This is the whole argument in one line: **translation moves the problem
between languages; it does not attach a number to a dish.**

## IB2 — The glucose response does not transfer. The density does.

A glucose response belongs to a *person on a day*: their fasting level, their
insulin sensitivity, what they did that morning. The same bowl of rice gives
different rises to two people and to the same person on two days. It is not a
property of the food, so it cannot be carried across a cohort, let alone
across a language.

Carbohydrate density — grams of carbohydrate per 100 g of a prepared dish —
*is* a property of the food. It survives the trip.

So the transferable quantity, the only one worth building a bridge for, is
density. Shanghai's glucose data is how the density claim gets *tested*; it is
not the thing being transferred.

## IB3 — Shanghai's language barrier is already gone

The dataset ships both columns, and the English one is the authors'
translation:

```
dietary_intake      Rice 150 g | Vegetable 100 g | Sliced lotus root 50 g
dietary_intake_cn   米饭150g | 青菜100g | 藕片50g
```

Every meal in the response table already parses without any Chinese handling.
Over the **3,045 meals that carry a T+60 rise: 9,445 item segments, 1,838
distinct dish names, median 3 items per meal** — counting every segment,
including the few with no weight written. The most common are `Rice` (546),
`Egg` (520), `Coarse grain` (450), `Vegetable` (445), `Milk` (300).

> The Day 1 gate reports **1,825** distinct dish names in
> `docs/NLP_PATH_A_GATE.md`. Not a contradiction: that parser requires a
> weight and `grams > 0`, and counts across all 3,060 meals. Same data,
> stricter filter. Quote whichever, with its definition.

There is no translation step to insert on the Shanghai side. The barrier is on
the *output* side — the Indonesian menu — and it is a breadth barrier.

## IB4 — The Indonesian ceiling is breadth, not language

`cv/data/tkpi_macro_lookup.csv` is the Indonesian food-composition lookup in
this repository. It has **10 rows**, one per `food_class`:

`bakso · bebek_betutu · gado_gado · gudeg · nasi_goreng · pempek · rawon ·
rendang · sate · soto`

Fold in a translation lexicon and you land exactly where CV already is: the
same 10-class ceiling. The CV out-of-domain report says what that ceiling
costs — the classifier loses to the population-mean baseline on all four
macros, 32.9% of photos are forced to a single class, and 46.9% come back
low-confidence. CV's own conclusion is that the gap is **class breadth**;
portion was retracted as the gap (portion is 2.5–5.0% of carbohydrate
variance, which dish is 97.5%).

So a 10-entry lexicon is not a small first step toward something. It is the
current state, relabelled.

## IB5 — What the bridge actually is

```
   ShanghaiT2DM                      Indonesian menu
   ────────────                      ───────────────
   dish ──▶ glucose response         dish ──▶ density
        (validates the method)              (supplies the content)
                     ╲                    ╱
                      ╲   dish identity  ╱
                       ╲   is the join  ╱
```

- **Shanghai supplies the method test.** Does knowing *which dish* beat
  knowing only *total grams*? That is now runnable — 3,045 meals with a
  causal `g0` and a T+60 rise.
- **Indonesia supplies the content.** A fixed list of dishes and their
  densities. This is the deliverable, and it is data work, not modelling.
- **The join is the concept**, not the language. Two tables meet on dish
  identity.

The honest consequence: this design deliberately makes the Indonesian table
**unvalidatable on its own terms.** There is no Indonesian glucose-linked
dataset. A density entry can be checked against a composition source, or the
*method* can be checked against Shanghai — but "this density is right for this
Indonesian dish" cannot be tested against glucose anywhere. Anyone asking for
that number is asking for something the data cannot currently produce.

## IB6 — Where IndoBERT actually goes

The brief names IndoBERT and it does have a job — just not the one the
question assumed.

| stage | input | output | built with |
|---|---|---|---|
| parse the log entry | `nasi goreng 200 gram, extra telur` | `(dish, grams, method)` | **IndoBERT / rules** |
| attach the number | `(dish, grams)` | carbohydrate grams | **the density table** |

IndoBERT turns free Indonesian text into structured slots. The density table
turns slots into a number. Swapping the second for a transformer would mean
training an encoder to memorise a lookup — with 3,060 Chinese meals as the
only supervision, and none of them Indonesian.

So: IndoBERT stays, scoped to **parsing**. The encoder is not the model that
carries the glucose link.

## IB7 — Options considered, and why each was rejected

| option | why not |
|---|---|
| **Machine-translate Indonesian dish names** into a generic composition table (e.g. USDA) | fails *silently* on composite regional dishes — `rendang`, `gudeg`, `soto` have no equivalent entry, and a nearest-neighbour fallback looks like a real answer |
| **Multilingual embeddings, zero-shot** (IndoBERT ↔ a Chinese/English encoder) | needs no training data, but the alignment cannot be *scored* — no labels exist to say a match is right, so the error rate is unknowable |
| **Grow the lexicon only** | 10 → N requires N verified composition entries. The lexicon is not the hard part; the entries are |
| **Fine-tune an encoder on the 3,060 meals** | the shipped feature vector already collapses to `total_grams × 0.28` (one global density); there is no dish supervision to fine-tune *toward* |
| **chosen: a fixed Indonesian dish list + curated densities, method-validated on Shanghai** | small, auditable, and the only option whose failure mode is visible |

## IB8 — What is explicitly *not* claimed

- This does **not** say the density table is a model. It is a look-up with
  provenance per row.
- This does **not** make a Shanghai result an Indonesian result. The cohorts
  differ in at least three ways that matter: Type 2 vs pre-diabetic, medicated
  (373 of 3,060 meals have an insulin dose within 30 minutes), and Chinese
  menu vs Indonesian menu.
- This does **not** close N2. The shipped Path A features still extract no
  dish. Nothing here extracts one.
- This does **not** say "never use translation". It says translation is a
  *tool inside the bridge*, applied to a fixed dish list, not the bridge.

## Settled vocabulary

| term | means | is not |
|---|---|---|
| **dish** | a named prepared food, e.g. `nasi goreng` | a food class (`rice dish`), a raw ingredient |
| **dish identity** | *which* dish | not *how much* |
| **carbohydrate density** | g carbohydrate per 100 g of the prepared dish | not glycaemic index; not portion size |
| **portion** | how much of the dish was eaten | 2.5–5.0% of carbohydrate variance |
| **meal response** | the glucose rise attributable to the meal | not the absolute glucose level |
| **`g0`** | the last CGM reading at or before the meal, strictly causal | not an average, not a projection |
| **the bridge** | connecting dish identity to a density, validated through glucose | not a translation layer |

## What would change this decision

1. **An Indonesian CGM + food-log dataset.** Then the table becomes directly
   validatable and the argument for it gets much stronger. None is known to be
   public.
2. **A dish-level macro ground truth for Shanghai.** The response would no
   longer be the only supervision, so a dish table could be checked against
   macros rather than inferred through glucose.
3. **A published Indonesian food-composition source with prepared dishes
   already derived.** That removes the data-entry cost, which is the real
   blocker in IB7.

## Where the numbers come from

```bash
# dish counts, item segments, top dishes — from the committed response table
python3 - <<'PY'
import pandas as pd, re, collections
d = pd.read_csv("nlp/features/shanghai_nlp_response.csv")
v = d[d.valid_t60 == 1]
items = [re.sub(r"\s*\d+(?:\.\d+)?\s*(?:g|ml)\s*$", "", s.strip(), flags=re.I).strip()
         for t in v.dietary_intake.astype(str) for s in t.split(" | ")]
items = [i for i in items if i]
print(len(items), len({i.lower() for i in items}))
PY

# the Indonesian side of the join — 11 lines, i.e. a header plus 10 dishes
wc -l cv/data/tkpi_macro_lookup.csv
```

The Shanghai response table is rebuilt by
`python3 -m nlp.data.shanghai_response` (see
`docs/SHANGHAI_RESPONSE_REBUILD.md`). The CV numbers are in
`cv/reports/indonesian_classifier_ood/*/report.md`.
