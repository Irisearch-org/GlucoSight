# Integration Audit — Sprint 2 Forecasting Result and Cross-Track Pipeline

**Date:** 2026-09-10
**Author:** Integration layer (PM-owned)
**Scope:** the Sprint 2 forecasting notebook, the three track outputs on
`main`, and the fusion spine built to join them.
**Reproduce:** `python -m pytest utils/tests integration/tests rppg/tests/test_glucose_estimator.py -q` (165 tests)

---

## 1. Verdict in one paragraph

The Sprint 2 forecasting result is **methodologically sound in its
skeleton and not yet credible in its claim**. The grouped split is real and
asserted, the causal filter is real and asserted, the three trivial
baselines are present, and the per-participant median and IQR are reported.
That is more discipline than most of this literature shows and it should be
said plainly. But the headline is labelled "Multimodal" when no modality
track contributed to it, the best model clears the best baseline by about
4 points of Zone A, the Clarke implementation used is wrong outside Zone A,
the calibration gain is measured against the wrong denominator, and the
figure that makes the result look convincing is an in-sample fit with no
split at all. **Zone A 62.5% is also below the project's own 70% floor**,
and the honest reading is that the pipeline is nearly trustworthy while the
result is not yet a finding.

---

## 2. What the notebook got right

Worth recording, because the next sprint should not regress on any of it.

| Rule | Status | Evidence |
|---|---|---|
| **C4 grouped splits** | Correct | `GroupKFold(n_splits=5)` on `participant_id`, with `assert len(set(trn) & set(val)) == 0` inside the fold loop |
| **C2 causality** | Correct for the input history | `history = df_dexcom[df_dexcom['Timestamp'] < t0]`, then `assert history['Timestamp'].max() < t0` |
| **C3 baselines** | Present | B0 population mean, B1 persistence, B2 persistence + train-split mean excursion, all in the same table |
| **C3 per-participant reporting** | Present | Median and IQR of Zone A per participant, not only the pooled figure |
| **C3 stratification** | Present | Zone A by reference range, and it is the most informative output in the notebook |
| **Leakage in preprocessing** | Correct | `SimpleImputer` and `StandardScaler` are fitted on the training fold only, inside the loop |
| **Delta formulation** | Good judgement | Predicting the excursion and adding `g0` back is the right target for this problem |

The stratification table is the single most valuable thing in the notebook:

```
Range < 100      (N = 110 ): Zone A = 36.4%
Range 100 - 140  (N = 594 ): Zone A = 57.1%
Range 140 - 180  (N = 452 ): Zone A = 76.8%
Range 180 - 250  (N = 305 ): Zone A = 65.9%
Range > 250      (N = 73  ): Zone A = 43.8%
```

Zone A is a ±20% band, so it is ±18 mg/dL at 90 and ±50 mg/dL at 250. The
36.4% at the bottom of the range is largely the metric's geometry, not the
model failing there specifically. Any Zone A headline that does not carry
this table beside it is misleading, and the notebook is to its credit for
producing it.

---

## 3. Findings, most severe first

### F1 — The showcase figure is an in-sample fit (blocking)

Cell 6 builds a 24-horizon trajectory model and plots four "real cases":

```python
X_all_imp = imputer.fit_transform(df_multi[features_to_use])   # all rows
X_all     = scaler.fit_transform(X_all_imp)                    # all rows
model.fit(X_all, y_h)                                          # all rows
predicted_trajectories[:, i] = df_multi['g0'].values + model.predict(X_all)
```

There is no split. The model is fitted on every row and then predicts the
rows it was fitted on. The figure showing "GlucoSight Forecast (Multimodal)"
tracking the Dexcom trace inside the Clarke Zone A band is showing a GBDT
memorising 24 targets on ~1600 rows.

This is the most damaging item in the notebook because it is the most
persuasive. It is exactly the artefact that survives into a slide deck.

**Action:** delete the figure, or regenerate it with the fold's held-out
participants only. Never show it as-is.

### F2 — "Multimodal" describes a feature set with no modality in it

The column labelled `GBDT (Delta + Multimodal)` uses:

- CGMacros' own **weighed** macros — read from the dataset, not predicted by
  the CV track;
- clinical labs — HbA1c, insulin, triglycerides, HDL, cholesterol, fasting
  glucose;
- a commercial **gut-microbiome panel** — butyrate production pathways,
  digestive efficiency, gut lining health, inflammatory activity;
- age, BMI, sex.

No CV model output. No NLP. No PPG. Nothing from any of the three tracks
this project is organised around.

Worse for the prototype: nothing after the macros can be obtained from a
phone. A user cannot photograph their butyrate production pathways. So the
number is not reproducible by the thing being built.

This does not make it worthless — it bounds what meal-composition modelling
could achieve on this cohort, which is genuinely useful. It makes it an
**oracle upper bound**, and it has to be labelled as one.

**Action:** `integration/features.py` now defines three tiers —
`deployable` (contract fields a phone produces), `deployable_plus_askable`
(adds age/BMI/sex, which the app *could* ask for), and `oracle` (the
notebook's current set). Report all three. The gap between tier 1 and tier 3
is itself a result: it separates what CV can approximate from what
participant physiology contributes.

### F3 — The calibration gain is measured against the wrong denominator

The table reports:

| Row | n | Zone A (T+60) |
|---|---|---|
| GBDT (Delta + Multimodal) | 1669 | 57.5% |
| ★ GBDT Calibrated (H9) | 1534 | 62.5% |

These are different sample sets. The 1534 excludes each participant's first
3 meals and drops participants with ≤3 meals entirely — which removes the
sparsest, and plausibly the hardest, participants.

The like-for-like number **was computed and never printed**:
`calibrated_eval[h]['cold']` in cell 4 holds the uncalibrated predictions on
exactly those 1534 meals. It is populated and then never used.

**Action:** print it. Until then the +5.0 pt gain from per-participant
intercept calibration is not established. `integration/report.py` now
raises `ValueError` on any table whose rows cover different sample counts,
so this cannot recur silently.

### F4 — Clarke Error Grid: two wrong implementations, now one right one

This is finding M7, carried as a blocker since Sprint 1 and still open in
ClickUp. Both implementations in the repo were wrong. Measured by sweeping a
1 mg/dL grid over [20, 400]² (144,400 cells):

| Implementation | Cells wrong | Nature |
|---|---|---|
| `utils/clarke_grid.py` (old) | 1,734 (1.20%) | Lower Zone C coded as a rectangle instead of the (130,0)–(180,70) line, over-claiming C from B on 1,664 cells; Zone A's `ref >= 70` guard sent 70 A cells to D |
| notebook `evaluate_clarke` | 15,899 (11.01%) | Calls any `70 ≤ ref ≤ 180, pred > 180` Zone C (9,904 cells B→C); has no `pred ≥ ref + 110` rule at all (5,995 cells C→B) |

**Zone A membership is identical in all three implementations on all
144,400 cells.** The ±20% band is the one boundary the notebook got right.

So: **the notebook's Zone A column stands. Every A+B, C, D and E figure it
reports must be recomputed.** The old `__main__` "sanity check" generated
`y_pred = y_true * U(0.85, 1.15)` — inside the Zone A band by construction,
a test that could not fail.

`utils/clarke_grid.py` is rewritten against the published 1987 boundaries
and pinned by 47 tests including the two grid sweeps above.

### F5 — The participant ID join is positional and can silently misalign

```python
df_bio['participant_id'] = [f"CGMacros-{i+1:03d}" for i in range(len(df_bio))]
df_gut['participant_id'] = [f"CGMacros-{i+1:03d}" for i in range(len(df_gut))]
```

Both clinical files are keyed by **row position**, not by the subject ID in
the file. If `bio.csv` or `gut_health_test.csv` is not in exact 1..45 order
with no gaps — a missing participant, a different sort, a header row — every
clinical and gut feature is attached to the wrong person.

Nothing errors. The model still trains, the loss still falls, and the
result is scrambled physiology. This is precisely the class of failure
CLAUDE.md's four rules exist to catch, and it is not one of the four.

**Action:** join on the file's own subject-ID column and assert the join is
total before proceeding. Because these are participant-level features under
a participant-level split, a misalignment would not even show up as
leakage — it would just quietly destroy signal.

### F6 — Missing macros are encoded as zero

```python
carbs = float(row['Carbs']) if pd.notna(row.get('Carbs')) else 0.0
```

A meal with an unrecorded macro becomes a meal with 0 g of it. That is not
missingness, it is a strong and wrong assertion — a zero-carbohydrate meal
is a real and very informative thing.

CLAUDE.md rule 8 is explicit: missing ≠ negative. Contract v1.2 gives the
fallbacks (45/18/12/4 g) and requires `cv_present=0` alongside. The CV
track's own code already gets this right — its `_error_record` comment says
a 0 "akan terbaca downstream sebagai 'makanan tanpa kalori'". The
forecasting loader undoes that care.

### F7 — Δt is computed, used as a filter, and then discarded

The loader accepts a target within ±7.5 min of exactly 60 or 120 minutes,
then stores neither the actual offset nor the exclusion count. Every meal is
subsequently treated as if it were at exactly T+60 or T+120.

Finding C1 and the contract both require `delta_t_minutes` as a carried
feature. Glucose moves 1–3 mg/dL per minute during the rise, so ±7.5 min is
±8–23 mg/dL of label noise being thrown into the residual rather than
modelled. `dt_last_reading` is a different quantity — the age of the last
CGM reading — and does not substitute.

### F8 — No overlapping-meal guard

Meals less than 120 minutes apart contaminate each other's T+120 target:
the T+120 reading after meal A may be the T+30 reading after meal B. Snacks
make this common. Nothing excludes or flags it, and
`time_since_last_meal_hours` — a contract field — is never computed.

This is the most likely single explanation for T+120 being no better than
T+60 despite the extra hour of information.

### F9 — Ridge blows up and nobody looked

T+60: **RMSE 218.36, MAE 62.00.** An RMSE 3.5× the MAE means a handful of
catastrophic predictions. A ridge regression with `alpha=20` on standardised,
median-imputed features should not do this. Something in the feature matrix
is pathological — most likely an extreme value surviving in
`carb_to_fiber_ratio` or `Cho/HDL Ratio`, or a fold where an imputed
constant column gives the solver nothing to work with.

Leaving an unexplained 218 in a results table costs more credibility than
the row is worth. Either explain it or drop the model.

### F10 — The NLP track cannot process Bahasa Indonesia

Found by running the MVP demo. Entering `nasi goreng porsi besar, digoreng`
returns `is_fried_cooking=0, is_large_portion=0`.

- `_is_fried()` in `nlp/data/derive_labels.py` checks `FRIED_EN` and
  `FRIED_CN` only. There is no Bahasa lexicon. "digoreng" is not matched.
- `_is_large()` works purely by parsing gram amounts out of the text. Bahasa
  portion language ("porsi besar") carries no grams, so `is_large_portion`
  is **structurally impossible** to detect from an Indonesian note.

This is defensible for ShanghaiT2DM, which is what the track was pivoted to
under `DATA_STRATEGY.md`. It is not defensible for the MVP, whose entire
premise is an Indonesian user typing a note. **The demo's NLP path has no
working producer for its target language**, and until it does, the note
input is decorative.

### F11 — Three tracks, three schemas, no shared validator

Not the notebook's fault, but the reason nothing was integrated:

| Track | Emits | Contract v1.2? |
|---|---|---|
| CV | `sample_id`, `confidence`, `feature_status`, `calories_kcal` | **No** — missing `cv_present`, `carbs_source`, `portion_reported`, `gi_category`, `cv_model_version` |
| NLP | contract field names directly | Close |
| PPG | `to_contract_dict()` + `validate_contract_output()` | Yes — the most mature |

`cv/cv_baseline/schema.py` says in its own docstring that it should be
*replaced* by the shared validator when one exists, "bukan dipertahankan
berdampingan". `integration/contract.py` is that validator.

Four contract fields have **no producer anywhere in the CV track**:
`gi_category`, `carbs_source`, `portion_reported`, `cv_present`. The adapter
derives the first two, takes `portion_reported` as an argument (it is the
in-app question, not a CV output), and sets the mask from `feature_status`.

### F12 — Staleness: the mirror image of C2 (found by running the demo)

C2 stops glucose readings that are too **new** from entering the history.
Nothing stopped readings that are too **old**.

With a history from January and a first bite in September, the causality
filter passed every reading happily — they are all before t0 — and handed
the predictor a **238-day-old reading as the pre-meal glucose value**. B1
persistence and B2 persistence-plus-excursion are both meaningless on a
reading that old. Nothing errored; the prediction looked entirely
reasonable.

**Action:** `MAX_G0_AGE_MINUTES = 360` added to
`integration/contract.py`, rejected values reported explicitly rather than
silently substituted. **Proposed as a contract v2.0 amendment** — it is not
in v1.2.

### F13 — The "no CGM required" claim rests on a CGM reading

The root `CLAUDE.md` states the research question as Zone A > 70%
**"without continuous glucose monitoring"** and the project summary as
prediction from a smartphone, **"no CGM required"**.

The Sprint 2 result obtains its pre-meal glucose like this:

```python
df_dexcom = df.dropna(subset=['Dexcom GL'])
history   = df_dexcom[df_dexcom['Timestamp'] < t0]
g0        = history.iloc[-1]['Dexcom GL']
```

That is the Dexcom G6 trace, sampled every 5 minutes, so `g0` is a CGM
reading taken at most about 5 minutes before the first bite. B1
(persistence) scoring 51.0% Zone A and B2 scoring 54.7% is what a
near-instantaneous CGM reading buys on its own — the baselines are strong
*because* they stand on a CGM.

**The headline result depends on precisely the instrument the project
claims not to need.** The notebook does compute `dt_last_reading`, but
never reports its distribution; that distribution is the most important
unreported number in Sprint 2.

**Action taken:** `integration/glucose_source.py` makes the origin of `g0`
explicit and pluggable — `cgm`, `fingerstick`, `ppg_estimate`,
`population_fallback` — with the source, its age and a trust weight
travelling downstream, the same discipline the contract already applies to
`carbs_source`. Accuracy can now be reported stratified by source.

**Action still needed:** re-run Sprint 2 with `g0` artificially aged
(5 / 30 / 60 / 120 minutes, and absent) and publish the decay curve. That
single experiment decides whether the project's claim is
"no CGM required" or "one cheap fingerstick required", and the second is
still a good project.

### F14 — The PPG fallback for `g0` is the population fallback in disguise

Proposed product behaviour: a user with a glucometer types their reading;
a user without one gets the finger scan instead. The seam is right and is
now built. The PPG branch cannot ship yet, for two independent reasons:

1. **Contract.** Ring-fence rule E1.2 says `ppg_glucose_estimate` is
   "never used as a sole prediction and never surfaced to a user". In
   `forecast = g0 + excursion`, a PPG-derived `g0` is the sole driver of a
   user-visible number. This is a contract violation, and belongs on the
   v2.0 agenda rather than in a code change.
2. **Evidence.** The only committed PPG-glucose model
   (`rppg/models/baseline_mean.py`) predicts the training-fold population
   mean for every subject — MAE 14.51 +/- 2.00 mg/dL over 23 subjects.
   Predicting one constant for everyone carries the same information as
   the population fallback. Registering it would change the label on the
   number, not the number.

`G0Estimator.provides_information` encodes reason 2 in the type, and
`resolve_g0` refuses to prefer a constant predictor over the fallback, so
this cannot be lost by accident. When the PPG track has an estimator that
beats its own mean baseline on held-out subjects, registering it lights the
path up.

### F15 — This dataset cannot demonstrate that the finger scan works

Attempting to fix the finger-scan path surfaced a blocker that no amount of
modelling clears.

**Predicting a single constant for every subject achieves Clarke Zone A
85.1% on the PPG glucose dataset.** Computed with the corrected
`utils.clarke_grid` from the committed
`rppg/models/reports/baseline_mean_predictions.csv` — the mean baseline,
GroupKFold by subject, n=67 recordings, 23 subjects. Zone A+B is 100.0%,
with zero points in C, D or E.

The project target is Zone A > 70%.

So on this dataset, **a Zone A figure cannot distinguish a working
PPG-glucose model from a constant.** Anyone who trains a model here and
reports "Zone A 85%, exceeds the 70% clinical target" will have reported the
baseline. This is finding C3 in its purest form.

The cause is the cohort, not the metric:

| | |
|---|---|
| glucose range | 88–183 mg/dL |
| median | 110 mg/dL |
| SD | 18.6 mg/dL |
| Zone A band at the median | **±22 mg/dL — wider than the label SD** |
| between-subject share of variance | 40.8% |
| within-subject SD | 15.4 mg/dL (larger than the 13.4 between-subject SD) |
| recordings per subject | median 2, min 1 |

Two further consequences worth stating:

- An **oracle that knew each subject's own mean** would reach MAE 10.89
  against the constant predictor's 14.41 — a 24% improvement, and that is
  the ceiling for anything that merely identifies the subject. Under
  GroupKFold by subject that path is closed anyway, which is correct.
- Most of the variance a model would have to explain (59%) is *within*
  subject: the same person at different times, from a 10-second waveform.

**A second trap, found while building it:** the recordings are 10 s at
**2190 Hz**; a phone camera runs at **~30 Hz**. Training at the native rate
and deploying at 30 Hz is a silent domain shift — one sample at 30 Hz is
33 ms, and a pulse rise time is 100-200 ms, so `rise_time_ms` and
`pulse_width_half_ms` arrive at deployment with a fraction of their training
resolution and no way to signal it. Features are therefore extracted at the
**deployment rate by default**; `--native` exists only to measure the gap,
which is itself a number worth publishing.

**Action taken:** `rppg/models/glucose_estimator.py` reports Zone A but does
not gate on it. The gate is MAE against the mean baseline on held-out
subjects, and it requires all four of: a positive improvement, wins in at
least 4 of 5 folds, a subject-level bootstrap 95% CI excluding zero, and a
label-permutation p ≤ 0.05. If the gate fails, no artifact is written and
the finger-scan path stays off. `integration/glucose_source.py` autoloads
the artifact only if it exists, so `provides_information` is set by
evidence, never by hand.

**Demonstrated, not argued.** Running the pipeline on synthetic signals
with no relationship to the real glucose labels (a negative control, since
the labels are real and the signals are noise) produced:

| Predictor | MAE | Zone A |
|---|---|---|
| Mean baseline | 14.57 | 85.1% |
| Ridge on **pure noise** | 14.78 | **88.1%** |

The noise model is *worse* on MAE and *better* on Zone A, and would have
been reported as "88.1% Zone A, exceeding the 70% clinical target". All four
gate criteria correctly refused it (permutation p = 0.486). This is what a
Zone A headline on this dataset actually buys.

**Action still needed:** the honest conclusion may be that this dataset
cannot support the claim at all. If so, that is a publishable negative
result bounded by n=23 subjects, and BIG IDEAs (raw 64 Hz PPG + CGM, 16
pre-diabetic subjects, listed in `DATA_STRATEGY.md` §2) is the only public
data that could test it properly.

---

## 4. Is the result credible?

### The claim as it stands

> GBDT Calibrated reaches Zone A 62.5% at T+60 and 61.5% at T+120,
> against B2 at 54.7% / 56.3%.

### What survives scrutiny

- The Zone A figures themselves are correctly computed (F4).
- The split is grouped by participant and asserted (C4).
- The input history is causally filtered and asserted (C2).
- The baselines are real and in the same table (C3).

### What does not

1. **The margin is small.** Best cold model over best baseline is **+4.0 pts
   Zone A at T+60** (RF 58.7 vs B2 54.7) and **+5.1 at T+120** (GBDT 61.4 vs
   B2 56.3). With ~9 held-out participants per fold and no confidence
   interval or repeated-seed variance reported, a 4-point gap is not
   distinguishable from fold noise. **No error bars anywhere** is the single
   biggest gap between this and a publishable result.
2. **The calibration gain is not established** (F3).
3. **62.5% is below the project's own 70% floor**, and the floor is
   explicitly described in CLAUDE.md as "a floor, not the research claim".
4. **It is not a fusion result** (F2). No modality track contributed.
5. **The persuasive figure is invalid** (F1).
6. **A silent join bug could have scrambled the clinical features** (F5) —
   until that join is fixed and asserted, the contribution attributed to
   physiology is unverified.

### Verdict

**The pipeline is close to trustworthy. The result is not yet a finding.**

The right way to state it today:

> On CGMacros (n=1669 meals, 45 participants, 5-fold GroupKFold by
> participant), gradient boosting on weighed meal macronutrients plus
> participant clinical and gut-panel features reached Zone A 58.7% at T+60
> against a persistence-plus-mean-excursion baseline at 54.7%. The margin is
> ~4 points with ~9 held-out participants per fold and no variance estimate,
> so it should be treated as preliminary. The feature set includes venous
> labs and a commercial microbiome panel and is therefore an upper bound on
> what a phone-only pipeline could achieve, not a measurement of one.

That is a defensible Sprint 2 outcome. It is a trustworthy pipeline
producing one real number beside a baseline — which is exactly what
`DATA_STRATEGY.md` §7 set as the success criterion for the week. It is not
"multimodal fusion achieves clinical accuracy", and the gap between those
two sentences is the entire risk to this paper.

---

## 5. What was built in this pass

```
utils/clarke_grid.py            rewritten against published boundaries (M7 closed)
utils/tests/test_clarke_grid.py 47 tests incl. two 144,400-cell grid sweeps

integration/contract.py         contract v1.2: the single validator
integration/features.py         three feature tiers + train-only normalization
integration/fusion.py           assemble one validated meal from three tracks
integration/adapters/{cv,nlp,ppg}.py   track-native -> contract
integration/glucose_source.py   pluggable g0: fingerstick / CGM / PPG / fallback
rppg/models/glucose_estimator.py  PPG->glucose training + evidence gate
rppg/tests/test_glucose_estimator.py  15 tests: the gate must reject noise
integration/predictor.py        B2 baseline, labelled as a baseline
integration/report.py           results tables that refuse to omit baselines
integration/api.py              FastAPI /predict /contract /health
integration/web/index.html      demo page
integration/tests/              103 tests

forecasting/notebooks/          Sprint 2 notebook, committed with audit header
```

Run: `python -m pytest utils/tests integration/tests rppg/tests/test_glucose_estimator.py -q` → **165 passed**
Serve: `python -m uvicorn integration.api:app --reload` → http://127.0.0.1:8000

### What the MVP does and does not do

It accepts a meal photo path, a Bahasa note, a PPG capture, the in-app
portion answer and a glucose history; runs each track; degrades any missing
or failing modality to contract fallbacks with `*_present=0`; applies
down-weighting; filters the history causally; and returns a contract-valid
forecast.

**It serves baseline B2, not a model**, because no trained forecasting
estimator is committed to this repository — the Sprint 2 estimators were
never serialised. The response carries `is_trained_model: false` and a
plain-language `basis` string, and the demo page leads with a banner saying
so. When Forecasting commits a serialised estimator,
`integration/predictor.py::default_predictor` is the only line that changes.

A consequence worth stating: because B2 uses only `g0`, the prediction is
currently **identical whether or not the modalities are present**. The
plumbing carries, validates and weights them; the baseline does not consume
them. That is the honest state of the system today and the demo shows it
rather than hiding it.

---

## 6. ClickUp is stale — verified against the repo

Not updated, per instruction. Recorded here so the board can be corrected.

| Task | ClickUp | Actual |
|---|---|---|
| `[W1] BLOCKER clarke_grid.py` | to do | **Was genuinely not done.** Done now |
| `[PPG] brno ECG validation` | in progress | **Done** — `rppg/brno_validation.py`, MAE 3.76 bpm pooled |
| `[PPG] predict-the-mean baseline` | in progress | **Done** — MAE 14.51 ± 2.00, 5-fold by subject |
| `[PM] Merge cv/nlp baselines into monorepo` | to do | **Done** |
| `[CV] Indonesian classifier on CGMacros (OOD gap)` | complete | Done |
| `[Forecasting] CGMacros loader (causal filter)` | to do | **Done in the notebook, never committed** |
| `[Forecasting] B0/B1/B2 baselines` | to do | **Done in the notebook, never committed** |
| `[Forecasting] Ridge/RF vs baselines` | to do | **Done in the notebook** (see F9) |
| `[Forecasting] shared contract validator` | to do | **Done now** — `integration/contract.py` |
| `[PM] Scaffold integration layer` | to do | **Done now** |
| `[PM] MVP website` | to do | **Done now**, serving a baseline |
| `[Forecasting] fusion ablation harness` | to do | **Partial** — tiers and reporting exist; the CGMacros runner does not |
| `[NLP] Calibrate nlp_confidence` | to do | Not done — confirmed uncalibrated (F10) |

---

## 7. Recommendations for Sprint 3

Ordered. The first three are cheap and unblock the rest.

### Must (week 1)

1. **Fix the participant join and re-run** (F5). Join on the subject-ID
   column, assert the join is total. Half a day. Until this is done, no
   number involving clinical or gut features means anything.
2. **Print the like-for-like calibrated comparison** (F3). One line — the
   array already exists. It either confirms or kills the +5 pt claim.
3. **Delete or re-generate the trajectory figure** (F1). It cannot appear
   anywhere.
4. **Recompute every A+B / C / D / E figure** with `utils.clarke_grid` (F4).
5. **Publish the g0-ageing decay curve** (F13). Re-run with `g0` aged
   5/30/60/120 min and absent. This decides the project's central claim.
6. **Add error bars.** Repeat the 5-fold `GroupKFold` over 5–10 seeds and
   report Zone A mean ± SD across seeds, plus a per-participant bootstrap CI.
   Without this the +4 pt margin is not a claim. **This is the single
   highest-value item in the sprint.**

### Should (weeks 1–2)

7. **Report the three feature tiers side by side** (F2), using
   `integration.features`. Headline the deployable tier. The
   tier-1-to-tier-3 gap is a publishable observation in its own right.
8. **Carry Δt as a feature and report exclusions** (F7).
9. **Flag or exclude overlapping meals** and compute
   `time_since_last_meal_hours` (F8). Most likely fix for the flat T+120.
10. **Population-mean fallbacks instead of zero-fill** (F6). Route the loader
   through `integration.fusion.assemble` so the contract does this for free.
11. **Explain or drop Ridge** (F9).
12. **Serialise the fitted estimator** so the MVP can serve a model rather
    than a baseline. Persist the scaler alongside it, per the contract's
    normalization spec.

### Should (the tracks)

13. **Bahasa lexicon for NLP** (F10), or state plainly that the MVP's note
    input is non-functional for the target language. `is_large_portion`
    needs a portion-word lexicon (kecil/sedang/besar/porsi jumbo), not a
    gram parser. This is a small, well-defined, high-visibility task.
14. **Calibrate `nlp_confidence`** (isotonic or Platt) on a held-out split.
    Forecasting uses it as a gating weight; uncalibrated, that gate is
    meaningless.
16. **Route CV through the adapter** and retire `cv/cv_baseline/schema.py`,
    as its own docstring asks. Give `gi_category` a real owner — the table
    in `integration/adapters/cv.py` is an integration-layer assignment from
    published GI values, not a CV-track measurement, and it should not stay
    that way.

### Contract v2.0 agenda (the meeting is already scheduled)

16. **Decide E1.2 vs the fingerstick-or-scan product design** (F14). If the
    app is to offer the finger scan to users without a glucometer, the
    ring-fence must be amended to permit a PPG-derived `g0` under stated
    conditions — at minimum a signal-quality floor, a confidence ceiling,
    and a user-visible label distinguishing an inference from a
    measurement. The code is built and gated; only the decision is missing.
17. Promote **`g0_source`** into the contract proper (F13), alongside
    `carbs_source` and `nlp_feature_source`. Results should be reportable
    stratified by it.
18. Add the **staleness rule** (F12) — `MAX_G0_AGE_MINUTES`, currently an
    integration-layer constant.
19. Decide whether **`ppg_embedding`** is requested at all — open item 2 in
    the contract, still unanswered.
20. Settle **open item 4**: the root `CLAUDE.md` still states the tri-modal
    research question and an Indonesian target population, which
    `DATA_STRATEGY.md` §3–4 supersede in practice. This audit's F2 and F10
    are both downstream of that unresolved divergence. It should be decided,
    not carried into Sprint 3 a third time.
21. Tell CV that on CGMacros `carbs_source = weighed`, which is better than
    `agents/cv/CLAUDE.md` currently assumes (open item 5).

### One thing to stop doing

Reporting a number before its baseline, its n, and its variance are in the
same table. `integration/report.py` now refuses to build such a table, but
the discipline matters more than the guard rail.
