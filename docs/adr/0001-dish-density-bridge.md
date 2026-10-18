# The Indonesian bridge is dish identity → carbohydrate density, not translation

**Status:** accepted (2026-10-18)

Sprint 1 planned Indonesian food text → IndoBERT → a glucose contribution.
The only glucose-linked text corpus available is ShanghaiT2DM (Chinese, with
the authors' English translation), so the track needs a principled way to
reach a Bahasa menu. We decided the bridge is **dish identity → carbohydrate
density per 100 g**: ShanghaiT2DM validates the *method* (does knowing which
dish beat knowing only total grams?), a fixed Indonesian dish list supplies
the *content*, and the two join on the concept rather than the language.

A translation lexicon was the obvious candidate and was rejected as
insufficient in principle, not just in effort. A lexicon maps
`nasi goreng → "fried rice"` — it renames a dish and attaches no number to
it. The quantity that actually transfers across cohorts and languages is a
food property; the glucose response is a person-on-a-day property and does
not travel. Translation remains a *tool inside the bridge*, applied to a
fixed small dish list, not the bridge itself.

## Considered options

| option | rejected because |
|---|---|
| machine-translate dish names into a generic composition table (USDA) | fails silently on composite regional dishes — `rendang`, `gudeg`, `soto` have no equivalent entry, and a nearest-neighbour fallback looks like a real answer |
| multilingual embeddings, zero-shot (IndoBERT ↔ a ZH/EN encoder) | needs no training data, but the alignment cannot be scored — no labels exist to say a match is right |
| grow the lexicon only | N verified composition entries are required; the lexicon is not the hard part |
| fine-tune an encoder on the 3,060 Shanghai meals | the shipped features already collapse to one global density (`total_grams × 0.28`), so there is no dish supervision to fine-tune toward |
| **dish identity → density, method-validated on Shanghai** | *chosen* — small, auditable, and its failure mode is visible |

## Consequences

- **`cv/data/tkpi_macro_lookup.csv` is 10 rows**, one per food class, so a
  10-entry lexicon is the current state relabelled. Breadth is the ceiling,
  not language. The NLP track's deliverable becomes a dish list with densities
  — data work, not modelling.
- **The resulting table is deliberately unvalidatable on its own terms.** There
  is no Indonesian glucose-linked dataset, so "this density is right for this
  dish" cannot be tested against glucose anywhere. Only the method can be
  tested, and only on Shanghai.
- **IndoBERT is scoped to parsing** — free text → `(dish, grams, method)` slots.
  It is not the model that carries the glucose link.
- Shanghai and BIG IDEAs results are **not directly comparable**: Type 2 vs
  pre-diabetic, and 373 of 3,060 Shanghai meals sit within 30 minutes of an
  insulin dose.

Full reasoning and the rejected-alternative detail:
`docs/NLP_INDONESIAN_BRIDGE.md` (IB1–IB8).
