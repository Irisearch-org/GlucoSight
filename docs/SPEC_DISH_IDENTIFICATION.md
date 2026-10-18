# Spec — Dish identification replaces food recognition + macro estimation

**Status:** drafted 2026-10-18, not yet published to an issue tracker (none
configured for this repo — see Further Notes), not yet agreed by the team.
**Builds on:** ADR `0001-dish-density-bridge.md`, `cv/LITERATURE_REVIEW.md`,
`nlp/docs/LITERATURE_REVIEW.md`, `GLOSSARY.md` (dish table, matcher, dish
identification, mixed plate).

---

## Problem Statement

A GlucoSight user photographs a meal so the app can estimate its
carbohydrate and fat content for the glucose forecast. Right now, two
things are wrong with how that photo gets turned into numbers.

First, the CV track tries to do two jobs at once — say what the dish is,
*and* estimate how much of it there is from the photo alone — and the
second job cannot be done well. A flat photo with no depth or scale
reference cannot recover portion weight, no matter how the model is
trained; the project's own measurements and outside published results agree
on this. Spending CV effort on it produces a confident-looking number that
isn't more accurate than a lookup table.

Second, real Indonesian meals are routinely more than one dish on a plate
(nasi padang: rice plus several side dishes). A model trained to output one
label per photo cannot represent that, and currently doesn't try to.

Separately, the NLP track was asked to classify a patient's free-text note
into health-context labels, but no such note exists in any dataset
available to the project, and none will exist from the real pilot either —
the in-app note is optional, one to three sentences, not a guaranteed food
log. The track's actual, defensible contribution (that *which dish* a meal
is predicts the glucose rise better than *how many grams* it weighed) has
no home in the current contract.

## Solution

Narrow CV's job to **dish identification**: given a meal photo, name every
distinct dish on it, each with a confidence, using an open-vocabulary model
rather than a fixed small classifier. Macro and portion numbers no longer
come from CV's model — they come from the **dish table**, a shared lookup
of dish name to macro composition, via a shared **matcher** that maps a
detected (or mentioned) dish name to its nearest table row.

NLP stops trying to identify dishes from a free-text note that doesn't
exist. Its contribution becomes: own and grow the dish table (the asset
both tracks now depend on), and — only where live text actually exists —
extract a cooking-method correction (fried vs. not) that adjusts the
dish table's default macros, checked against CV's own signal for
redundancy before being trusted as independent.

Both tracks' outputs still land in the existing contract shape Forecasting
already consumes — this changes what produces the numbers, not the shape
of the numbers.

## User Stories

1. As a CV sub-team member, I want the track's objective to be "identify
   the dish(es) in this photo," not "identify the dish and also estimate
   its macros," so that I'm not building toward a portion-estimation
   accuracy ceiling the published literature says a flat photo cannot
   clear.
2. As a CV sub-team member, I want to replace the closed 10-class
   classifier with an open-vocabulary model, so that the track isn't
   capped at whatever class list was hand-picked in advance, and so that
   real-world dish breadth (hundreds of distinct dishes) has somewhere to
   go.
3. As a CV sub-team member, I want the model to return a list of detected
   dishes for a mixed plate, not a single label, so that a nasi padang
   photo doesn't get forced into one of its ten components.
4. As a CV sub-team member, I want each detected dish name resolved through
   the shared matcher against the dish table, so that I'm not writing my
   own string-matching logic that silently diverges from NLP's.
5. As a CV sub-team member, I want the per-meal output to still be the
   scalar `carbs_g`/`fat_g`/`cv_confidence` shape Forecasting already
   consumes, with macros summed across every matched item, so that
   Forecasting's code doesn't have to change to consume a list.
6. As a CV sub-team member, I want the per-meal confidence to be the
   minimum confidence across all matched items, so that one shaky
   identification on a mixed plate pulls the whole meal's confidence down
   instead of being hidden by an average.
7. As a CV sub-team member, I want a new `carbs_source` value distinguishing
   "summed across more than one matched item" from the existing single-item
   lookup case, so that a downstream consumer can tell the two apart instead
   of a mixed-plate result looking identical to a single-dish one.
8. As a CV sub-team member, I want portion/macro-from-photo work explicitly
   marked closed, so that future effort isn't spent re-deriving a result
   the literature already settled.
9. As an NLP sub-team member, I want the track's deliverable reframed as
   "own and grow the dish table," so that my actual contribution (dish
   identity as the right unit of prediction) has a concrete, ongoing home
   instead of being a one-off finding in a markdown file.
10. As an NLP sub-team member, I want new dish-table rows added reactively —
    only when an evaluation run actually hits a name the table can't
    match — rather than pre-populated toward a larger published list, so
    that effort tracks demonstrated need, not speculative coverage.
11. As an NLP sub-team member, I want the cooking-method signal to be
    checked for redundancy against CV's own lookup-implied default before
    it's treated as an independent contribution to the fusion result, so
    that a correlated signal doesn't get double-counted.
12. As an NLP sub-team member, I want it to be unambiguous in the track's
    own docs that there is no live dish-identification-from-text pipeline,
    so that the next person working on this track doesn't rebuild one
    assuming it's needed.
13. As a Forecasting/Integration consumer, I want the CV and NLP adapters'
    output contract unchanged in shape, so that nothing downstream of the
    adapters has to be touched by this redesign.
14. As a Forecasting/Integration consumer, I want the new `carbs_source`
    value and the min-confidence rule documented in the interface contract,
    so that I can reason about a mixed-plate result without reading the
    CV adapter's source.
15. As the project's PM/reviewer, I want this entire redesign to land as a
    reviewable proposal (this spec, plus an ADR, plus a diff against
    `CLAUDE.md` and the two track `CLAUDE.md` files) rather than as a
    silent edit to those files, so that other sub-teams aren't surprised by
    a changed track charter they didn't see coming.
16. As the project's PM/reviewer, I want the redesign's first proof point to
    run against CGMacros, the only dataset with both real meal photos and
    weighed macro ground truth, so that "does this actually work" can be
    answered before any Indonesian pilot data exists.
17. As a future maintainer, I want the matcher implemented once, in one
    place, called identically by both tracks, so that I'm not maintaining
    two near-identical string-matching implementations that drift apart.
18. As a future maintainer, I want the matcher's "no match" case to be an
    explicit, handled outcome (not an exception or a silent default), so
    that an unmatched dish name is visible in the data rather than quietly
    turning into a population-mean fallback that looks like a real
    identification.
19. As someone writing the eventual paper, I want the Zone A evaluation of
    this redesign reported against the existing B0/B1/B2 baselines on
    subject-grouped folds, so that an improvement claim can't be mistaken
    for the same baseline-beats-everything trap this project already caught
    once (finding F15).
20. As someone writing the eventual paper, I want it stated plainly that
    dish-table rows are population-level assignments, not per-user
    measurements, so that the paper's limitations section doesn't have to
    be reconstructed after the fact.

## Implementation Decisions

- **CV's objective** changes from "food recognition + macro estimation" to
  **dish identification**: name the dish(es) present, with a confidence
  each. Macro and portion numbers are explicitly not CV's model's job.
- **Model choice**: an open-vocabulary vision-language model, prompted to
  list every distinct dish visible in the photo, replaces the closed-set
  classifier. No new object-detection model (no bounding boxes) — the list
  comes from the VLM's text output, matched the same way a single label
  would have been.
- **The matcher**: one shared function, `food name (string) -> matched
  dish-table row, or no-match`, similarity-based, not an exact-key lookup
  and not a trained classifier. Called identically by CV (on each
  VLM-listed item) and by NLP (on cooking-method mentions in whatever text
  exists). Lives in one module shared by both tracks' adapters, not
  duplicated. **Correction (ADR 0002 addendum, 2026-10-19):** built two
  ways — a default, offline, dependency-free difflib matcher, and an
  optional embedding-based one (real Gemini embeddings, dependency-injected
  so either can be swapped in) — and neither cleanly separates true dish
  matches from similar-sounding garnishes/condiments, measured live. The
  shipped embedding threshold trades recall for precision deliberately;
  this is an open problem, not a solved one. See the ADR addendum for the
  measurement.
- **The dish table**: unchanged in role (the single source of a dish's
  macro composition and standard serving weight) but now read by both
  tracks via the matcher, not owned/read by NLP alone. NLP continues to own
  growing its row count, reactively.
- **CV adapter output**: extends from "one matched dish" to "a list of
  matched dishes," but the adapter's output contract is unchanged in shape:
  `carbs_g`/`fat_g` sum across every matched item in the list; `cv_confidence`
  is the minimum confidence across matched items; a new `carbs_source` value
  is added to mark "this result summed more than one item" distinctly from
  the existing single-item lookup case. Any unmatched item in the list
  (the matcher returned no-match) is dropped from the sum, not defaulted to
  a population mean silently — the resulting lower item count should be
  visible in whatever logging/diagnostics the adapter already has.
- **GI category for a mixed plate**: no existing rule covers more than one
  dish. Default to the GI category of the item contributing the most carbs
  in the matched list; revisit if this turns out to matter for a real
  result.
- **NLP adapter**: adds the cooking-method correction path — when live text
  exists and a cooking-method mention is found, it sets `is_fried_cooking`
  on the already identified dish (not a second, independent dish
  identification). That is the only contract mechanism available for this
  signal; there is no field through which NLP can adjust CV's `carbs_g`/
  `fat_g` directly, so this spec doesn't ask for that. This correction's
  calibrated confidence still gates the existing down-weight behavior; it
  does not change when the correction is absent.
- **Portion**: unchanged. The in-app `portion_reported` tap remains the
  primary portion signal. A dish-table standard-serving default is only
  the fallback when that tap is skipped, not a competing estimate.
- **Scope of dish-table growth**: no pre-population toward the larger
  published Indonesian food-name lists found in the literature review.
  Grow only against names actually encountered in evaluation runs.
- **Governance**: this spec, once agreed, needs a dated ADR (successor to
  ADR 0001) recording the re-scope decision, plus a diff against
  `CLAUDE.md`, `agents/cv/CLAUDE.md`, `agents/nlp/CLAUDE.md`, and
  `docs/interface/INTERFACE_CONTRACT_v1.md` (for the new `carbs_source`
  value). None of those four files are edited as part of implementing this
  spec; the ADR and diff are reviewed by the team first.

## Testing Decisions

A good test here asserts the adapter's output contract (what Forecasting
actually receives) or the matcher's input/output behavior — never the
VLM call itself, which is an external dependency and not something this
project's test suite should be asserting the internals of.

- **The matcher**: tested directly, as a pure function, against a small
  fixture dish table — an exact-name case, a near-miss that should still
  match, and a genuinely unmatched name that must return no-match rather
  than a nearest-neighbor guess dressed up as a match. Prior art: this
  project already tests pure lookup/classification functions this way
  (`utils/tests/test_clarke_grid.py` pins boundary behavior the same way).
- **The CV adapter**: extended tests feed it a list of (name, confidence)
  pairs standing in for the VLM's output — a single-item case (unchanged
  behavior, regression-tested against today's output), a multi-item case
  (sums correctly, confidence takes the minimum, `carbs_source` reflects
  multi-item), and a case with one unmatched item (dropped from the sum,
  not defaulted silently). Prior art: `integration/tests/test_adapters.py`
  already tests this adapter's single-item path; extend it rather than
  starting a new test file.
- **The NLP adapter**: extended tests cover the cooking-method correction
  applied and not applied, and the redundancy check against CV's signal.
  Prior art: same file, alongside the existing NLP adapter tests.
- **Not unit-tested here**: the VLM's own dish-identification accuracy and
  the open-vocabulary coverage claim — those are evaluated empirically
  against CGMacros (per the acceptance criterion below), not asserted in
  the test suite.

## Out of Scope

- Single-photo portion or macro estimation. Closed, per C5 and the
  literature review; not reopened by this spec.
- Pre-building the dish table out to the larger published Indonesian
  food-name lists. Grows reactively only.
- Choosing a specific VLM provider, or building the CLIP-adapter fallback
  the literature review names. Use a VLM by default; the fallback is only
  built if a concrete, stated reason rules the VLM option out.
- Editing `CLAUDE.md`, `agents/cv/CLAUDE.md`, `agents/nlp/CLAUDE.md`, or
  the interface contract directly. This spec and its companion ADR are the
  thing to review; the doc edits follow agreement, not this ticket.
- Stress, sleep, and activity NLP signals. Already settled elsewhere (no
  text producer for stress; sleep/activity now come from CGMacros Fitbit
  data) and untouched by this redesign.
- Re-running the existing CV6–CV9 out-of-domain CGMacros evaluation under
  the old single-label classifier. That evaluation is superseded by the
  new CGMacros proof point in this spec's acceptance criterion, not
  patched.

## Further Notes

- **Acceptance criterion**: this redesign counts as working when it's
  wired into the actual fusion forecast and Clarke Zone A is reported
  against the existing B0/B1/B2 baselines on subject-grouped held-out
  folds — not as an isolated feature-level or adapter-level metric.

  **Correction, 2026-10-19: CGMacros cannot prove this.** The line above
  originally said to prove it first on CGMacros, "the only dataset with
  both real meal photos and weighed macro ground truth today" — true, but
  it went unchecked that CGMacros is an American-diet dataset. A live
  photo pulled from it mid-session was a Chipotle burrito order. No
  Indonesian dish will ever match against it; a VLM+matcher run against
  CGMacros can only reproduce the same out-of-domain finding CV6 already
  measured for the old classifier, not test anything new. Also found the
  same session: B0/B1/B2 don't consume meal macros at all (they're
  glucose-history-only baselines — population mean, persistence,
  persistence+excursion); only one candidate model in
  `forecasting/evaluate/baselines.py` uses `carbs_g`/`fat_g`, and that
  module computes MAE/RMSE, not Zone A — FC1's Zone A table came from a
  separate notebook. **There is currently no dataset and no wired pipeline
  that can produce this acceptance criterion's number.** What a smoke test
  *can* check — and did, live, this session — is matcher/dish-identification
  accuracy against real Indonesian food photos
  (`cv/data/raw/images/`, 6 labeled samples: 5/6 correct with the
  difflib matcher, with two diagnosed false-match bugs; see the ADR 0002
  addendum). Closing the real acceptance criterion needs: Zone A scoring
  added to `forecasting/evaluate/baselines.py`, and either an Indonesian
  photo+glucose dataset that doesn't currently exist, or a different
  proof structure entirely. Open, not solved.
- **Primary sources behind this spec**: `cv/LITERATURE_REVIEW.md` (VLM vs.
  closed-set classifier, portion-estimation ceiling, Indonesian dataset
  gap) and `nlp/docs/LITERATURE_REVIEW.md` (retrieval vs. classification
  for dish-name matching, TKPI scaling path). Both cite primary sources
  directly; this spec doesn't re-derive their findings.
- **Vocabulary**: this spec uses `GLOSSARY.md`'s terms (dish table,
  matcher, dish identification, mixed plate) throughout. Read that file
  before reading this one if any of those four terms are unfamiliar.
- **No issue tracker is configured for this repo** (`docs/agents/issue-
  tracker.md` doesn't exist), so this spec is a file, not a published
  issue, and carries no `ready-for-agent` label yet. Run
  `/setup-matt-pocock-skills` to configure one, then publish this spec
  (and run `/to-tickets` against it) through that tracker.
