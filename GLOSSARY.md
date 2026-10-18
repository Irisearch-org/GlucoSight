# GlucoSight — shared vocabulary

Cross-track terms used by more than one sub-team. Track-local terms stay in
that track's own docs (`agents/{track}/CLAUDE.md`, `nlp/docs/`, etc.) unless
they start getting used across tracks, at which point they move here.

## Language

**Dish table**:
The single shared file mapping a dish name to its macro composition (carbs,
fat, protein, fiber per 100 g) and a standard serving weight. The only place
a dish's macros live — no track computes macros itself, both just read this.
Currently `cv/data/tkpi_macro_lookup.csv`, sourced from TKPI (Tabel Komposisi
Pangan Indonesia).
_Avoid_: TKPI lookup, macro lookup — those name the data's source, not the
project's shared artifact.

**Matcher**:
The shared retrieval function that maps a free-text or VLM-generated
food-name string to its nearest row in the dish table, or to "no match."
Similarity-based, not exact-key — the same function is called by CV (on
VLM-detected item names) and NLP (on cooking-method/dish mentions in text).
_Avoid_: classifier, lookup — "classifier" implies a fixed trained label
set, "lookup" implies an exact key; this is neither.

**Dish identification**:
The CV track's objective: naming which dish(es) appear in a meal photo, with
a confidence per dish. Deliberately excludes macro and portion estimation —
those come from the dish table and the `portion_reported` tap, not from CV.
_Avoid_: food recognition — the project's earlier framing, which bundled
macro estimation into the same objective and is being retired.

**Mixed plate**:
A meal photo containing more than one distinct dish (e.g. nasi padang: rice
plus several side dishes), as opposed to a single-dish photo. Dish
identification on a mixed plate returns a list, not one label; the dish
table and matcher are then applied per item and the macros summed.
_Avoid_: composite plate, multi-item meal — "mixed plate" is the term
already used in `agents/cv/CLAUDE.md`'s Task 5; this just canonicalizes it
instead of introducing a synonym.
