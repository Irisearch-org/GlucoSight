# GlucoSight — Architecture Review Findings

**Date:** Sprint 1
**Scope:** End-to-end review of data flow, integration points, and failure
modes, conducted before any model training code was written.
**Status:** Findings accepted by PM. Contract v1.1, the data collection
protocol, and all track docs were updated against this list.

Every finding has a stable ID. Other documents cite these IDs. If you
disagree with a change in the contract, argue against the finding here.

---

## How to read this

Reviews like this read as an attack on the project. This one is not. The
process design here — an interface contract in Sprint 1, temporal splits
mandated up front, an ablation study planned before the model, an honest
"this may not work" written into the PPG track — is better than most funded
work. See §5.

The findings concentrate in **measurement and evaluation design, not
modelling**, and that is good news: every critical item is cheap to fix now
and impossible to fix after data collection starts.

---

## 1. CRITICAL — fix before Sprint 3

### C1 — Ground truth had no timing protocol
**Where:** Local data collection; ground-truth labels.
**Problem:** T=0 was undefined (photo? first bite?) and nothing enforced
when the T+60/T+120 pricks actually happen. Glucose moves 1–3 mg/dL per
minute postprandially, so a realistic 20-minute timing error is 20–60 mg/dL
of label noise — larger than the target RMSE. Worse, eating duration
correlates with food type, so the offset correlates with `carbs_g`: a
confound, not noise.
**Fix:** T=0 = first bite via an explicit in-app button. Device timestamps
on every reading. `delta_t_minutes` becomes an explicit model input;
evaluate on Δt ∈ [50,70] and [110,130] and report exclusions.
**Landed in:** `DATA_COLLECTION_PROTOCOL.md` §1, contract §0.

### C2 — Glucose interpolation leaked the labels
**Where:** Forecasting, sparse glucose handling (both linear and GP).
**Problem:** The T+60 and T+120 readings *are the prediction targets*. Any
interpolation fitted across the participant's series uses readings on both
sides of a gap, so the `glucose_history` handed to the model was computed
from the values it is asked to predict. A GP is worse — fitted globally, so
the posterior at T=0 is conditioned on every future observation. Nothing
crashes; the loss curve looks excellent; the model is worthless.
**Fix:** Strictly causal interpolation. Filter to `timestamp < t0` *before*
interpolating. Hard assertion in the loader. GP ablation becomes GP
extrapolation with growing uncertainty — which will look worse, correctly.
**Landed in:** contract "Causality rule", `agents/forecasting/CLAUDE.md`.

### C3 — Zone A > 70% is likely achievable by a model that learned nothing
**Where:** Evaluation standard.
**Problem:** Clarke Zone A is ±20% of reference. For a medicated T2D cohort
clustering 120–220 mg/dL, a **constant predictor of 160** plausibly lands
50–70% in Zone A (ref=140 → 14% error → A; ref=180 → 11% → A; ref=200 →
20% → A). The stated pass threshold therefore does not distinguish working
multimodal fusion from a lookup table.
**Fix:** Three mandatory baselines, run first, in every results table:
B0 population constant, B1 persistence (last reading), B2 persistence +
mean training excursion. Zone A > 70% is a floor for clinical plausibility,
**not evidence of a contribution**. The real claim must be "fusion beats B2
and beats history-only, with a CI excluding zero."
**Landed in:** `agents/forecasting/CLAUDE.md`, `agents/pm/CLAUDE.md`.

### C4 — Effective sample size is participants, not meals
**Where:** Phase 2 fine-tuning, 30–100 participants.
**Problem:** ~3,500 meals looks adequate; it is not. Meals within a
participant are heavily correlated. For any claim generalising to *new
people*, effective n is the **number of test participants** — roughly
10–30. Detecting a few percentage points of Zone A difference at that n is
likely underpowered.
**Fix:** Simulation-based power analysis before recruiting (~200 lines of
NumPy). Report per-participant Zone A as **median and IQR across
participants**, never pooled over meals — pooling lets one enthusiastic
participant dominate the headline.
**Landed in:** `agents/forecasting/CLAUDE.md` Sprint 2 tasks.

### C5 — The CV regression head cannot learn portion size
**Where:** CV dual head + DKBM lookup.
**Problem:** The Indonesian set is classification-labeled; macros come from
`food_class` → DKBM lookup. Within-class variance in the regression target
is **zero** — every nasi goreng photo carries the same `carbs_g`. The head
learns a lookup table. Meanwhile a single 2D photo with no scale reference
is geometrically incapable of recovering portion volume, and portion is the
dominant term in carbohydrate load. The CV track's own doc notes a 30 g
carb error shifts glucose 40–60 mg/dL.
**Fix (PM decision):** in-app portion question `portion_reported` becomes
the primary portion signal. `carbs_source` field added so Forecasting knows
`carbs_g` is a class-conditional estimate. Weighed ground truth and
in-frame scale reference documented as upgrade paths, **not committed**.
**Landed in:** contract CV schema + honesty note, `agents/cv/CLAUDE.md`.

### C6 — PURE/UBFC are face datasets; the input is a fingertip
**Where:** PPG track training data.
**Problem:** PURE and UBFC-rPPG are *remote* photoplethysmography
benchmarks — face video at ~1 m, pulsatile component 0.1–1% of intensity.
The deployment input is a finger against the lens with the flash on:
contact PPG, 1–2 orders of magnitude better SNR, different ROI, different
noise sources, no face. A model trained on face video has learned facial
attention and ambient-light statistics. None of it transfers.
**Fix:** Track renamed to **contact PPG** (variables `rppg_*` → `ppg_*`;
directories unchanged). Training moves to the datasets already in
`rppg/PPG_Dataset/`. PURE/UBFC retained only as an explicitly out-of-domain
literature comparison.
**Note:** this correction is *in the project's favour* — contact PPG is the
easier version of the problem.
**Landed in:** contract PPG schema, `agents/rppg/CLAUDE.md`.

### C7 — Auto-exposure will cancel the signal being measured
**Where:** PPG capture; no camera configuration was specified anywhere.
**Problem:** A camera's AE loop exists to hold brightness constant. The
pulsatile blood-volume signal *is* a small periodic brightness variation.
With AE on, the camera servos it out, and the residual is a nonlinear
function of the AE time constant — which varies by phone model. Separately,
smartphone video is often variable-rate in the low light of a covered lens,
so the assumed uniform (900,3) series is wrong and FFT results are biased in
a device-correlated way.
**Fix:** Lock AE/AWB/ISO/focus; record per-frame timestamps and resample;
discard first 2–3 s; keep red channel off saturation; log device metadata.
Promoted to a **contract-level app requirement**, since the PPG track
cannot enforce it alone.
**Landed in:** contract "Shared Capture Protocol",
`DATA_COLLECTION_PROTOCOL.md` §2.

---

## 2. HIGH — fix before the fusion layer

### H1 — Missing modality and negative modality shared an encoding; contract self-contradicted
Contract v1.0 said `signal_quality < 0.3` ⇒ "exclude rPPG entirely";
`agents/forecasting/CLAUDE.md` said "down-weight, do not exclude". Two
teams, two implementations, uninterpretable ablations. Separately,
"no text" and "text reporting no stress" were both encoded as all-zeros.
**Fix:** resolved to **down-weight, never exclude**. Added explicit
`cv_present` / `nlp_present` / `ppg_present` masks — missingness is a fact,
confidence is a belief, they are different fields. Train with **modality
dropout** at the rate measured in the pilot, so graceful degradation is
learned rather than an inference-time hack.

### H2 — NLP labels are collinear with CV outputs
`is_fried_cooking` ≈ `fat_g`; `is_large_portion` ≈ `portion_reported`.
Makes "NLP contribution" partly a restatement of CV, and feature-attribution
claims unreliable. Given C5, `is_large_portion` may be a genuine independent
portion signal — which would be a headline NLP result, but only if isolated.
**Fix:** report the feature correlation matrix before interpreting
ablations; add ablations isolating the overlap.

### H3 — 30-second HRV at 30 fps is quantization-limited
SDNN conventionally needs 5-minute windows; 30 s gives ~35–40 beats and is
dominated by respiratory sinus arrhythmia. At 30 fps, inter-beat
quantization is **33 ms** while the SDNN effect size is 20–50 ms — the
measurement resolution equals the signal.
**Fix:** `ppg_hrv_sdnn` → `ppg_hrv_rmssd`; mandatory sub-sample
(parabolic/cubic) peak interpolation; drop the feature entirely if pilot
test–retest shows it is unstable.

### H4 — `ppg_glucose_estimate` had no training data *(resolved by the provided dataset)*
Originally raised because PURE/UBFC contain no glucose labels, leaving only
the sparse local set — which would have leaked into the fusion labels.
**Resolved:** `rppg/PPG_Dataset/` provides 23 subjects / 67 recordings with
a glucometer value per recording. The output **stays in the contract**, ring-
fenced (see contract §"Ring-fence rules") so it cannot contaminate the
primary fusion result. Bounded by n=23.

### H5 — Pretraining and fine-tuning have different input schemas
OhioT1DM has insulin/carbs/exercise; the local set does not. Unstated which
phase consumes what. If Phase 1 is glucose-only, every fusion weight is
randomly initialised in Phase 2 and pretraining contributes nothing to the
part the paper is about.
**Fix:** pretrain a **glucose-dynamics encoder** on history + time features
(the shared schema), attach the fusion pathway fresh in Phase 2, and ablate
with/without the pretrained encoder — which directly measures whether
OhioT1DM helped at all.

### H6 — Feature-extractor versioning: the fusion set silently rots
Fusion trains on frozen track outputs. Any track model improvement
invalidates the fusion training distribution with no error raised.
**Fix:** archive raw inputs forever; version stamps on every dict; rule
that any track model change triggers full re-extraction; use DVC.

### H7 — Scale and dimensionality in concatenation fusion
`carbs_g` ∈ [0,300] beside binary flags beside 256 embedding dims: the
embeddings dominate the gradient by sheer dimensionality and wash out the
interpretable features the paper is about. Normalization was marked "TBD".
**Fix:** normalization spec written into the contract — z-score from the
**training split only**, one-hot `gi_category` and `portion_reported`,
binaries untouched, **embeddings omitted from the Sprint 6 baseline** and
projected to 8–16 dims if later added.

### H8 — The ablation as designed is confounded and seed-noise limited
Each experiment changed input dimensionality *and* effective capacity. 8
experiments × 2 horizons × 2 architectures = 32 numbers; something will look
significant by chance.
**Fix:** one architecture, fixed input width, **ablate by setting the
`*_present` masks** — capacity held constant by construction. ≥5 seeds per
config, paired bootstrap **over participants**, and a pre-registered
primary endpoint.

### H9 — Participant-level holdout removes personalization
T2D response is dominated by between-individual variation. A strict
participant-level split gives the model a stranger with zero calibration
data — the hardest possible setting, and possibly not the product being
built.
**Fix:** report **both** — cold-start (no calibration) and calibrated
(first 3–5 meals fit a per-participant offset). The gap between them is
itself an interesting result and informs prototype onboarding.

---

## 3. MEDIUM — later sprint or Limitations section

- **M1 — Domain gaps, ranked:** (1) contact vs. facial PPG data [C6];
  (2) Western/lab food data with no weighed macros [C5]; (3) T1D → T2D
  pathophysiology; (4) dense CGM → sparse fingerstick, where real
  missingness is **behavioural (missing-not-at-random)**, not the uniform
  downsampling the plan simulates; (5) lab → real-world capture [C7];
  (6) Japanese/Western food taxonomy → Indonesian; (7) formal corpora →
  colloquial Surabaya Bahasa.
- **M2 — 500 team-written sentences will not represent patient language.**
  The team writes, labels, and trains on its own phrasing. Participants are
  older, use Javanese-inflected colloquial Surabaya Indonesian, SMS
  abbreviation, code-mixing. The 40%-positive annotation target also
  guarantees over-prediction in deployment. **Fix:** collect real notes from
  the 5 pilot participants, label them, hold them out as a realistic-
  distribution test set, and calibrate thresholds on it. The gap between
  team-set and patient-set F1 is publishable.
- **M3 — Glucose range imbalance makes Zones C/D/E untestable.** A medicated
  cohort logging ordinary meals produces mostly 110–200 mg/dL. "Zone E:
  0.0%" will mean "no data there", not "safe there". **Fix:** publish the
  reference histogram beside every Clarke result; report Zone A stratified
  by reference range; state explicitly when D/E are not evaluable.
- **M4 — DKBM vs USDA, and Indonesian cooking.** Keep DKBM; log any
  substitution. Santan and palm-oil frying shift fat and energy density.
  Cooked-and-cooled rice develops **resistant starch**, materially lowering
  effective GI — no static table captures this. Treat `gi_category` as a
  coarse prior, not a measurement.
- **M5 — Ramadan** likely intersects collection (~Feb 2027). See protocol §7.
- **M6 — Device heterogeneity** across the Android price range affects both
  CV colour rendition and PPG amplitude. Log device model; check whether
  error correlates with device tier — that is a fairness issue, not just a
  limitation.
- **M7 — `utils/clarke_grid.py`** is a simplified reimplementation with
  hand-written boundaries, unvalidated against a reference; and the
  `__main__` sanity check generates `y_pred = y_true * U(0.85,1.15)`, which
  is within ±20% **by construction** — a self-test that cannot fail.
  **Fix:** validate `_classify_zone` against published test vectors,
  especially for reference values in 70–180 where the fall-through to Zone B
  is least certain; replace the demo with fixed boundary-case pairs.
  *(Deferred by PM — not this sprint.)*

---

## 4. DESIGN SUGGESTIONS

- **D1 — Predict the excursion, not the absolute value.** Target Δglucose
  from the pre-meal reading, add the baseline back for Clarke. Removes the
  dominant between-participant intercept, stops the model echoing the last
  reading, and makes "learned nothing" immediately visible.
- **D2 — Add a mechanistic baseline.** A two-parameter gamma-shaped
  absorption curve driven by `carbs_g` and `gi_category`, fitted to the
  training set. A day's work, interpretable, physiologically grounded. If
  the TFT cannot beat it, that is essential information.
- **D3 — Run the whole pipeline on synthetic data first.** Catches
  C2-class leakage (inject a known future-leak, confirm detection),
  validates the eval code, and produces the C4 power analysis. Highest-value
  engineering task available right now.
- **D4 — Add `ppg_perfusion_index`** (AC/DC ratio). Trivial to compute, a
  genuine physiological quantity, better motivated than a raw embedding.
  *(Adopted into contract v1.1.)*
- **D5 — Version the contract properly.** `schema_version` on every dict so
  mismatches fail loudly. *(Adopted.)*
- **D6 — Write the Limitations section now**, in Sprint 2, while the
  reasoning is fresh. It doubles as a running honesty check.

---

## 5. THINGS DONE WELL

Stated plainly, because the list above is long and the process design here
is genuinely better than most published work:

- **An interface contract existed in Sprint 1**, with types, ranges,
  required flags and per-field fallbacks. Most multi-team ML projects
  discover their integration schema in month five, in a panic.
- **Random splits banned outright**, temporal splitting specified at two
  granularities. This is the most common fatal flaw in glucose-prediction
  papers, and the team pre-empted it.
- **The ablation study was designed before the model**, with "always run
  baseline before fusion" as a rule. H8 is about statistical rigour, not
  about the intent, which was right.
- **The PPG track's honesty is exemplary** — "frontier research… it may not
  work. The paper is still valid either way. Never oversell accuracy to the
  team." More intellectually honest than most published PPG-glucose work.
- **Confidence scores mandatory on every modality**, with "never return
  None, never silently fail" enforced per-track. H1 completes this instinct
  rather than replacing it.
- **Clarke Error Grid outranks RMSE** project-wide from day one.
- **Catastrophic-forgetting guardrail** in CV (≤5pp Food101 degradation) —
  a specific, checkable criterion on a real risk.
- **Cohen's Kappa required before NLP training.** Most teams skip
  inter-annotator agreement entirely.
- **"Bukan medical device" is a hard UI requirement** and clinical
  deployment claims are prohibited project-wide.

---

## 6. VERDICT

**Is the architecture viable?** Yes as a research contribution. The
three-modality decomposition is sensible, the tracks are cleanly separated,
and the contract discipline is strong. The failure modes are concentrated in
**measurement and evaluation design, not modelling** — and every critical
item was cheap to fix at Sprint 1 and would have been impossible to fix
after collection. Two of them (C2 label leakage, C3 trivial-baseline-passes)
would have produced a paper reporting success while demonstrating nothing,
which is the worst available outcome.

**Highest-risk assumption.** Not the one the team worries about — everyone
knows contact PPG glucose is speculative, and it is hedged. The unacknowledged
one is **that a self-administered finger-prick, taken at home by a T2D
participant asked to test "about an hour after eating", constitutes a
measurement of glucose at T+60.** Every metric and the entire research claim
rest on that label being what it says it is. C1 addresses it; the pilot must
verify it.

**Validate first.** The 5-participant pilot (protocol §6) and the synthetic
power analysis (D3), concurrently, before Sprint 3. Plus — newly possible
with the provided data — the **30 Hz decimation experiment** in
`agents/rppg/CLAUDE.md`, which tests the project's central novelty claim
using data already on disk, before recruiting anyone.
