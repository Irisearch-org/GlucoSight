# CV's objective is dish identification; macros come from the dish table, not CV's model

**Status:** proposed (2026-10-18) — not yet reviewed by the team.

CV's objective was "food recognition + macro estimation" (two co-equal
goals, `CLAUDE.md`/`agents/cv/CLAUDE.md`). Finding C5 already showed
within-class carbohydrate variance is zero on the Indonesian set (macros
come from a class lookup, not a measurement), and `cv/LITERATURE_REVIEW.md`
independently confirms the macro-estimation half from outside this
project's own data: Nutrition5k's carb-MAE roughly triples once mass is
inferred from a photo instead of given as ground truth, and every
production food-logging VLM found (Dexcom Stelo/G7, LogMeal) solves
portion the same way this project already does — ask the user — not from
the photo.

We decided: **CV's objective is dish identification — which dish(es), with
what confidence, across as many real dishes as breadth allows.** Macro and
portion numbers come from the shared dish table (via the matcher) and the
existing `portion_reported` in-app question, never from CV's model. A
closed 10-class classifier is replaced by an open-vocabulary VLM for the
same reason: the field's long-tailed/fine-grained benchmarks top out around
100–160 classes even with dedicated architecture, short of this project's
measured 455-dish breadth, and a general VLM already beats this project's
own trained classifier by a wide margin on a measured sample
(`cv/vlm_vs_mobilenet.md`: 95.96% vs. 70.71%).

A mixed plate (nasi padang: rice plus several side dishes — `agents/cv/
CLAUDE.md` Task 5, previously open) is handled the same way: the VLM
returns a list of dishes, not one label; each is matched against the dish
table and the macros sum.

## Considered options

| option | rejected because |
|---|---|
| keep macro estimation as a CV goal, invest in a better regression head | the ceiling is external (no depth/scale reference in a flat photo), confirmed by Nutrition5k and this project's own C5; no amount of CV modelling clears it |
| a bigger closed-set classifier (more Indonesian classes) | tops out around 100–160 classes in the published literature, still short of the measured 455-dish breadth, and no Indonesian dataset at that scale carries macro ground truth to train on |
| multi-label classification for mixed plates | forces a fixed label set onto an open-ended item list; a VLM returning a list is the same mechanism already chosen for single dishes, not a second model |
| **open-vocabulary VLM, dish identification only, macros from the dish table** | *chosen* — matches what this project's own measurement and outside production systems both already point at |

## Consequences

- **Portion/macro-from-photo work is closed**, not just deprioritized.
  Reopening it needs a stated reason the literature review's gap-finding
  didn't anticipate (e.g. a depth-capable phone becomes the baseline
  device), not a hunch that more training data would help.
- **The dish table and its matcher become shared infrastructure**, read by
  both CV (VLM-detected item names) and NLP (cooking-method/dish mentions
  in text) — one matcher, not two independent implementations.
- **`carbs_source` gains a new value, `class_lookup_multi`**, so a
  mixed-plate result is distinguishable downstream from a single-dish
  lookup instead of looking identical to it.
- **This contradicts `CLAUDE.md`'s and `agents/cv/CLAUDE.md`'s current
  wording** and is not applied to either file by this ADR — see the
  companion diff in `docs/SPEC_DISH_IDENTIFICATION.md`'s Further Notes.
  Those files change only after the team reviews this decision.

Full reasoning, user stories, and the mixed-plate/matcher mechanics:
`docs/SPEC_DISH_IDENTIFICATION.md`. Primary sources:
`cv/LITERATURE_REVIEW.md`, `nlp/docs/LITERATURE_REVIEW.md`.

## Addendum, 2026-10-19 — the matcher itself is still an open problem

Building this surfaced a second, smaller decision worth recording. The
matcher (how a VLM-listed or text-mentioned name resolves to a dish-table
row) was built with plain `difflib` string-ratio matching, on the
assumption — stated in the code at the time — that a learned embedding
"buys nothing a string ratio doesn't already give" at 10 rows. That
assumption is **false, measured live**: a real VLM call against
`cv/data/raw/images/gudeg_005.jpg` correctly said "nasi gudeg," and the
difflib matcher resolved it to `nasi_goreng` — the shared `"nasi "` prefix
dominates a whole-string character ratio over the word that actually
distinguishes the dish.

Three approaches were tried, in order: plain difflib, naive symmetric
Gemini embeddings, and task-type-aware (query/document) Gemini embeddings.
**None of them cleanly separates true dish matches from superficially
similar garnishes and condiments** ("kacang goreng," fried peanuts, scored
*higher* against `nasi_goreng` than a true match, "soto mie," scored
against `soto`, under naive embeddings). The shipped default
(`dish_table.EMBEDDING_MATCH_THRESHOLD = 0.75`) is a deliberately
conservative threshold from that measurement — it rejects every measured
false positive at the cost of one known false negative (a harder synonym,
"ayam betutu" for `bebek_betutu`) — chosen because a wrong match silently
pollutes the macro sum, while a dropped item is already a safe, visible
fallback in this project's design.

This is **not a solved problem**, just a safely-defaulted one. It was
measured against 8 real queries from one photo set — not a validated
benchmark. Revisit before trusting it beyond a smoke test; the literature
review's retrieval recommendation (`nlp/docs/LITERATURE_REVIEW.md`) may
need a sharper mechanism than top-1 cosine similarity (e.g. the VLM
reasoning about a short candidate list directly, rather than a separate
embedding-similarity step) to actually close this.
