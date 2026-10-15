# CGMacros Forecasting Pipeline Specification

**Status:** Implementation specification, version 0.1 (target definition remains gated)
**Anchor dataset:** CGMacros v1.0.0, participant-level CSV records
**Scope:** Auditable offline glucose forecasting, not medical use

## 1. Research question and scope

Test whether meal-related features available in CGMacros improve prediction of observed Dexcom glucose after a dataset-reported meal event, compared with causal glucose-history baselines, on participants excluded from model fitting.

This study is limited to the 45 CGMacros participant CSVs and their own event, nutrition, image-reference, activity, and Dexcom fields. The public archive is licensed CC BY-NC-SA 4.0. Cite the PhysioNet dataset and its DOI; record the source version and license in any distributed report. Raw source data, images, and derived artifacts remain outside the repository unless redistribution terms are explicitly checked.

PPG glucose data, BIG IDEAs, ShanghaiT2DM, synthetic NLP records, and Indonesian food-classification outputs are excluded from the primary training/evaluation cohort. Do not manufacture participant or meal joins across datasets. PPG and NLP remain separate component studies unless a real event-level pairing is demonstrated.

This is a research prototype. Do not present its output as clinical guidance, a glucose-monitor replacement, or evidence of safety outside observed participants/ranges.

## 2. Dataset audit baseline (CGMacros v1.0.0)

The participant CSVs were read from the official archive using ranged reads and stored only in the session scratchpad; no CGMacros data were added to Git. Reproduce these values with the audit command before training and treat the audit output, not this snapshot, as authoritative for the downloaded copy.

- 45 participant CSVs, with source IDs 1–23 and 26–49; filenames encode IDs as `CGMacros-###`.
- `bio.csv` has 45 unique values in its `subject` key, matching the participant CSV IDs exactly.
- `gut_health_test.csv` has 47 rows; source IDs 24 and 25 have no participant time-series CSV. Never positional-join participant-level files. Join on `subject`; unmatched rows must be reported and must not enter the cohort silently.
- Participant CSVs contain 1,706 rows with non-empty `Meal Type` (17–64 per participant; median 38). The inherited forecasting notebook reports 1,669 examples but does not account for all 37 omitted event rows.
- The data dictionary names `Timestamp`, Libre and Dexcom glucose, activity/heart rate, meal type, nutrition, amount consumed, and image path. Participant CSV timestamps are minute-indexed. Dexcom non-null readings were present at a one-minute median interval, with gaps that must be audited per participant rather than assumed absent.
- An exploratory ±7.5-minute diagnostic found readings for 1,674/1,706 T+60 events and 1,669/1,706 T+120 events; this is not the implemented label rule. The accepted implementation uses exact observed Dexcom timestamps only, yielding 1,662 T+60, 1,659 T+120, and 1,652 paired events. It flags events with a later event before each horizon; the isolated-only sensitivity set has 1,496 exact T+60 and 1,409 exact T+120 targets (1,403 paired). These isolation counts depend on the explicit event-marker rule and are not a claim that every marker is an independent meal.
- 250 of 1,706 consecutive event pairs were less than 120 minutes apart. Some event markers are separated by only 1–2 minutes; do not assume every marker is an independent meal. Retain all source events for the base cohort and report isolated-only sensitivity separately until a source-backed event-curation rule is agreed.
- Nutrition fields were populated for nearly all event rows, but fiber, amount consumed, and image path each have missing values. Preserve missingness; never convert missing nutrition to zero. The 45 CSVs contain inconsistent capitalization/pluralization of meal labels.
- **Blocking data-quality finding:** 494 meal-field values across the 1,706 event rows violate the ranges in the official data dictionary (v1.0.0). Examples include protein below the stated 3 g minimum, calorie values above 1,180, and CGMacros-016 fiber values as high as 2,830 despite the listed 0–176 range. Do not clip, transform, impute, or train meal-feature models until the dataset owner’s encoding/units are reconciled and a documented correction is tested. The runner must report history-only evaluation separately and mark meal-feature analysis blocked.
- The source documentation describes meal timestamps as derived from photographs, and the meal-time field is not verified to mean first bite. Use `event_timestamp`/`dataset_event_time` in code and reports. Do not call an endpoint “T+60/T+120 after first bite” unless source documentation confirms this interpretation.

These are data-audit observations, not model results. Recompute target eligibility by horizon and event curation rule, including all excluded-row reasons.

## 3. Event, target, and cohort rules

### Event identity

- Canonical participant key is the numeric `subject` ID matched to the participant filename ID; retain the original value and derive a stable string such as `cgmacros-001` only after the equality check.
- A meal/event candidate is a participant CSV row with a non-empty `Meal Type` and a valid timestamp.
- Canonical event identity is `(dataset_version, participant_id, source_row_id)`, where `source_row_id` is the explicit CSV index only if verified unique and stable within each participant file; otherwise use a deterministic participant-file row number plus source-file checksum. Do not create a synthetic cross-source match.
- Normalize meal-type whitespace/case and known plural aliases for analysis, while retaining the untouched source label.
- Repeated entries close in time are not merged automatically. Audit their timestamps, source-row IDs, image references, meal types, and nutrition values. Version a curation rule before model evaluation. Until then, include all source events as candidates, report close-event incidence, and maintain a separate isolated-event sensitivity set.

### Candidate endpoints

- Candidate target horizon is event time +60 and +120 minutes. The anchor is the dataset-reported event timestamp, not assumed first bite.
- Primary labels use exact timestamps in the Dexcom series, with finite observed values only. No forward-fill, back-fill, or interpolation. Store target timestamp, actual `delta_t_minutes`, absolute offset from nominal horizon, sensor/source, and eligibility reason.
- Score each horizon on its own eligible event set and additionally report a paired T+60/T+120 cohort. Report counts at each exclusion step; no complete-case filtering without an explicit reason.
- Before claiming a post-meal forecast, confirm in primary dataset documentation whether this event timestamp denotes meal start/first bite. If not established, call this an event-indexed glucose prediction and do not use first-bite or postprandial wording.
- Target curation must decide whether an intervening event invalidates attribution. Report at least: (a) all eligible events with subsequent-event flags and (b) events with no later event before the target horizon. Lock the primary policy before model selection; do not use target glucose to decide event merging or eligibility.

## 4. Prediction-time feature contract

Define versioned schemas separately for source rows, derived event records, model features/targets, and API transport. Each field must include type, units, provenance, missingness, availability time, transformation, and one of: deployable, askable, or oracle-only.

- **Causal glucose inputs:** finite Dexcom readings with timestamp strictly before event time. Filter first; only then compute summaries. Store observation count, latest-reading age, and source. No post-event data may enter history or input preprocessing.
- Current exploratory sparse schedule is fixed at 07:00, 12:00, and 20:00 local dataset time, selecting distinct observed values within 15 minutes before each scheduled time. It is capped at three values/day and must be sensitivity-tested; it is not observed fingerstick behavior. Readings older than six hours are unavailable for B1/B2/history-model inputs.
- Current API `baseline_demo` supports exactly T+60 and T+120; it rejects other requested horizons. This limitation is separate from the offline target builder.
- **Meal composition:** distinguish dataset-provided nutrition and amount consumed from CV model inference. CGMacros values are available as study features but cannot be labelled as image-model estimates. Missing values remain missing and are accompanied by masks or explicit unavailable states.
- **Images:** path existence and image availability are separate from nutrition fields. An absent image does not erase dataset nutrition; it does prevent CV-inference claims for that event.
- **Fitbit/activity:** use only fields available at or before event time. Any post-event window summaries are labels/analysis-only, never prediction inputs.
- **Participant physiology:** demographics/labs/gut-health data are oracle/askable tiers, not deployable meal-time features. Join by `subject`; unmatched IDs and missingness remain explicit.
- Exclude post-event CGM values, future meal records, post-event Fitbit readings, participant IDs, source row order, image paths/filenames, and any label-derived feature from model inputs.
- Imputation, scaling, feature selection, calibration, and thresholding fit only on the training participants in each fold.

## 5. Splits, baselines, and evaluation

- Use participant-held-out GroupKFold (up to five folds when the available unique participant count supports it). Assert participant sets are disjoint and persist each fold assignment. No random row/meal split.
- Primary cold-start evaluation has no participant calibration. Any per-participant calibration is a separate protocol; calibration events cannot also be scored.
- Compare identical eligible events at each horizon against:
  - B0: training-participant mean target glucose.
  - B1: latest causal pre-event glucose.
  - B2: latest causal pre-event glucose plus training-fold mean target excursion over that latest glucose.
  - Glucose-history-only learned model.
  - Incremental meal-feature models, one feature group at a time.
- If no valid causal pre-event glucose exists, mark B1/B2 and history-based predictions unavailable for that event; report their denominators rather than substituting a global value. A separate B0-only cohort is clearly labelled.
- Report MAE and RMSE at each horizon, per-participant metric medians/IQRs, participant and event counts, target distributions, missingness, exclusions, target offsets, all baselines, and paired participant-level bootstrap uncertainty for candidate-minus-baseline differences. Use multiple fixed seeds if the selected estimator is stochastic.
- Clarke grid metrics are secondary, and only after validating the implementation against boundary fixtures. Zone A alone is never a success criterion. Report reference-glucose distribution and range strata, and state ranges with insufficient support.
- A model is not accepted as providing added value unless it reproducibly improves over the prespecified same-case baseline on held-out participants with uncertainty supporting the improvement. Null/negative results are valid.

## 6. API and artifact behavior

The current integration API serves B2 and explicitly labels it as a baseline; it is not an integrated CGMacros trained model. Keep that status explicit until a forecasting artifact is trained on the specified features, evaluated participant-held-out, serialized with its preprocessing state, and loaded by the prediction path.

A trained prediction response must include endpoint definition and horizon, units, actual target timing semantics, schema/feature/model versions, model identity/hash, input availability/provenance summary, and research-only disclaimer. Reject unsupported schema, invalid event timestamps, non-causal history, unavailable/incompatible models, and malformed provenance. Do not emit fabricated values or silently convert missing features to zero. PPG/NLP data from unrelated cohorts and `ppg_glucose_estimate` are excluded from this endpoint.

## 7. Stage gates

1. **Repository gate:** selected base commit recorded; tests run from a clean branch snapshot; local modifications preserved; each capability labelled code-present, test-covered, data-validated, or unimplemented.
2. **Data-rights gate:** exact source/license/version/citation and intended storage/sharing are documented; restricted data/artifacts stay out of Git.
3. **Data-audit gate:** all 45 participant files parse; keys and duplicates reconcile; audit report includes participant/event counts, field missingness, cadence/gaps, event type normalization, target coverage/offsets, close-event statistics, and exclusion reasons.
4. **Event/target gate:** source event-time semantics and event curation are explicit; target selection is deterministic and has no interpolation; exact-vs-tolerance rule is fixed before modeling.
5. **Join/schema gate:** no positional joins; all source identifiers reconcile; unknown versions, invalid units/types, and absent provenance fail validation; missing is distinguishable from zero/negative.
6. **Causality gate:** tests prove timestamps at or after event time do not affect model inputs; preprocessing is fit on training participants only; future-data mutation leaves features unchanged.
7. **Baseline gate:** participant-disjoint splits; B0/B1/B2 and history-only evaluated on defined same-case cohorts; sample accounting is exact; Clarke boundary tests pass before zone reporting.
8. **Incremental-model gate:** each eligible feature group is added separately and compared to history-only; no model/result is labelled multimodal unless real paired source features were consumed.
9. **Prediction gate:** full source-shaped fixture passes ingestion → event/target construction → causal features → fold/model transform → prediction response; model artifact and versions are traceable; all invalid-input behaviors are tested.
10. **Reproducibility gate:** documented command reproduces audit/evaluation from permitted local data; deterministic manifests and split IDs are saved; CI runs data-independent tests, while private-data tests are clearly reported.

## 8. Explicit non-goals for this milestone

- Tri-modal CV+NLP+PPG prediction.
- Joining CGMacros, ShanghaiT2DM, BIG IDEAs, PPG Dataset, or synthetic records to fill missing modalities.
- Claiming model-derived nutrition from weighed CGMacros values.
- Mobile phone PPG capture, app/clinical deployment, treatment decisions, or glucometer replacement.
- LSTM/TFT/deep modeling before data audit and required baselines pass.
