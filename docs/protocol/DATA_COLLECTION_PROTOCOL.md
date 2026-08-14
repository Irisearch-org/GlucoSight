# GlucoSight Data Collection Protocol v1.0

**Status:** DRAFT — to be finalized before participant recruitment
**Owner:** Project Manager
**Applies to:** All tracks + mobile app team
**Companion documents:** `docs/interface/INTERFACE_CONTRACT_v1.md`,
`docs/REVIEW_FINDINGS.md`

---

## Why this document exists

Almost everything that can permanently ruin this project happens here, not
in the modelling. Model choices are reversible for the price of a rerun.
A mistimed label, a missing timestamp, or an unlocked camera is baked into
the data forever and cannot be recovered by any architecture.

**Read this before writing app code. Read it again before recruiting.**

---

## 1. Meal timing — the single most important section

### T=0 is the FIRST BITE

Not the photo. Not the note. Not when the food arrived.

The app must present an explicit **"Mulai makan"** button. Pressing it
writes `t0_timestamp` to the second. Everything else is measured from there.

**Why:** the photo is taken before eating. Nasi padang is eaten slowly;
soto is eaten quickly. If T=0 is the photo, the label offset differs by
food type — which means the timing error is *correlated with `carbs_g`*.
That is a confound, not noise, and no amount of data fixes it.

### Every glucometer reading carries a device timestamp

The app timestamps the reading at the moment of entry. Participants must
**not** be able to backfill a reading from memory later. If a reading is
entered more than 10 minutes after its stated time, flag the record.

Recommended: have the participant photograph the meter display; the photo's
EXIF is an independent check on the entered time.

### Δt is recorded, not assumed

Real home readings will land anywhere from T+45 to T+90 when a participant
is asked to test "about an hour after eating." Glucose moves **1–3 mg/dL
per minute** during the postprandial rise, so a 20-minute timing error is
**20–60 mg/dL of label noise** — larger than any RMSE the project would be
proud to report, and comparable to the full width of Clarke Zone A.

Therefore:

- `delta_t_minutes` is computed from actual timestamps and passed to the
  model as an **explicit input feature**.
- The model predicts *glucose at Δt*, not *glucose at 60*.
- Evaluation restricts to Δt ∈ **[50, 70]** and **[110, 130]** minutes, and
  **always reports how many samples were excluded**.

### Per-meal capture sequence

| Order | Action | Timestamp field |
|-------|--------|-----------------|
| 1 | Fasting / pre-meal finger prick | glucose_history entry |
| 2 | Meal photo | `photo_timestamp` |
| 3 | Portion question (kecil/sedang/besar) | — |
| 4 | **Contact PPG capture (30 s)** | `ppg_timestamp` |
| 5 | Text note in Bahasa Indonesia | `note_timestamp` |
| 6 | **"Mulai makan" button** | **`t0_timestamp`** |
| 7 | Prick ~T+60 | label, with actual timestamp |
| 8 | Prick ~T+120 | label, with actual timestamp |

**The PPG capture happens BEFORE the first bite.** A recording taken after
eating measures postprandial heart-rate elevation, which is a different
physiological state. Mixing pre- and post-meal captures in one variable
makes it uninterpretable.

If the participant captures out of order, record it — do not silently
reorder. `ppg_timestamp` relative to `t0_timestamp` tells Forecasting
whether the recording is pre- or post-prandial.

---

## 2. Camera capture requirements (contact PPG)

These are **app requirements**, not PPG-track implementation details.

| # | Requirement | Rationale |
|---|-------------|-----------|
| 1 | `CONTROL_AE_MODE_OFF`, fixed `SENSOR_EXPOSURE_TIME` + `SENSOR_SENSITIVITY` | Auto-exposure exists to hold brightness constant. The pulse signal *is* a small periodic brightness variation. AE actively cancels it. |
| 2 | `CONTROL_AWB_MODE_OFF`, fixed `COLOR_CORRECTION_GAINS` | Auto white balance drifts the channel ratios the model depends on |
| 3 | `CONTROL_AF_MODE_OFF` | Refocus events cause step discontinuities |
| 4 | Record **per-frame presentation timestamps** | Smartphone video is often variable-rate in low light; naive FFT on unevenly-spaced samples is biased, and the bias correlates with device model |
| 5 | Discard first **2–3 seconds** | LED thermal settling |
| 6 | Tune exposure so red channel ≈ **70% of full scale** | With flash on, red frequently clips, flattening the pulsatile component |
| 7 | Log `device_model`, `os_version`, `mean_fps`, `fps_jitter` | Makes device-driven bias detectable later |
| 8 | Capture duration **30 s** | Yields three 10-second windows; see §3 |

**Verification test before shipping the app:** record a static scene
(finger removed, lens covered) for 30 s and confirm the per-frame pixel
means are flat. If they drift, a control loop is still active.

---

## 3. Why 30 seconds, and how it is used

The training data (`rppg/PPG_Dataset/`) consists of **10-second**
recordings. The deployment capture is 30 seconds.

This is deliberate and it is an advantage:

- Train the model on **10-second windows** (matching the training data).
- At inference, extract **three non-overlapping 10-second windows** from
  the 30-second capture.
- Report the **median** across windows as the feature value, and the count
  of valid windows as `ppg_n_windows`.

Median-across-three is free variance reduction and gives a within-capture
reliability check at no extra cost to the participant.

---

## 4. Portion size

**Committed approach: the in-app question only.**

> "Porsi: kecil / sedang / besar" → `portion_reported` ∈ {0, 1, 2}

One extra tap, high signal, and it does not depend on the CV model working.

**What this means, stated plainly:** a single 2D photo with no scale
reference cannot recover portion volume. Combined with the fact that the
Indonesian food set is classification-labeled (so `carbs_g` is a
class-conditional lookup), **`portion_reported` is the project's primary
portion signal**, not `carbs_g`.

This must appear in the paper's Limitations. It is an honest constraint,
not a flaw to hide — and `portion_reported` being informative would itself
be a reportable result.

**Not committed this cycle** (upgrade paths if capacity appears):
- Scale reference object in frame (a 10,000 rupiah note has fixed dimensions)
- Weighed ground truth on a 200–300 photo subset

---

## 5. Text notes

- Free-form Bahasa Indonesia, 1–3 sentences, optional.
- **An empty note is recorded as empty** (`nlp_present=0`). It is never
  silently converted to all-negative labels. "The user wrote nothing" and
  "the user wrote that they are not stressed" are different facts.
- Do not prompt with a template or checkbox list. The research question
  requires natural patient language; a template would produce text that
  looks like the team's annotation set rather than real notes.

---

## 6. Pilot before scale — run this first

**5 participants, 2 weeks, before any full recruitment.**

The pilot exists to answer questions that no amount of design can:

| Question | Why it matters |
|----------|----------------|
| What is the real distribution of Δt? | Determines whether the [50,70] window keeps enough samples to be usable |
| What fraction of meals arrive with all three modalities? | Sets the modality-dropout rate used in training |
| What does real patient Bahasa actually look like? | The team-written annotation set will not match it |
| What is the real `ppg_signal_quality` distribution on participants' own phones? | **If `< 0.3` is the common case rather than the exception, the degradation rule stops being an edge case and becomes the entire system** |
| Do participants press "Mulai makan"? | If compliance is poor, the whole timing design needs rework |
| Test–retest: 3 back-to-back PPG captures from one person | Any feature whose within-person variance approaches its between-person variance is noise |

Pilot data is **protocol validation, not training data**. Do not pool it
into the main set.

---

## 7. Ramadan

With Sprint 1 starting August 2026 and a 12-sprint / 6-month timeline,
collection lands roughly December 2026 – February 2027. **Ramadan 1448 is
expected to begin around mid-February 2027** (subject to sighting).

In a predominantly Muslim Surabaya cohort this means participants shift to
two meals with radically altered timing — sahur before dawn, iftar at
sunset, 12+ hour fasts between. `time_since_last_meal_hours` will take
values that appear nowhere in OhioT1DM.

**Preference:** schedule collection to complete before Ramadan begins.

**If it overlaps:** add `is_ramadan_period` and `fasting_duration_hours`,
analyse those meals as a **separate stratum**, and do not pool them into
the headline result. Handled deliberately this is a novelty angle —
intermittent-fasting glucose prediction is understudied and directly
relevant to this population.

---

## 8. Ethics and participant safety

- **This is a research prototype, not a medical device.** No prediction is
  ever shown as clinical guidance.
- The UI disclaimer **"Research prototype. Bukan medical device."** is a
  hard requirement (`agents/pm/CLAUDE.md`).
- Participants continue their normal glucose monitoring and medication
  regime unchanged. Nothing in this study alters their care.
- Ethics clearance must be obtained before recruitment. Confirm the
  institutional requirements for human-subjects research before the pilot,
  not before the main collection.
- Raw photos, video, and text are identifiable data. Store with
  `participant_id` pseudonyms, keep the mapping separately, and state the
  retention period in the consent form.

---

## 9. Data retention — archive raw inputs forever

**Never store only the extracted features.**

The fusion model is trained on outputs of three track models. When the CV
team improves their model in Sprint 8 — and they will — every `carbs_g` in
the fusion training set becomes stale, and the fusion model's weights no
longer match the distribution it sees at inference. Nothing errors. The
numbers just quietly change for reasons nobody can trace.

Requirements:
- Archive **original JPEG, original video, original text**, keyed by `meal_id`.
- Stamp every feature dict with `cv_model_version`, `nlp_model_version`,
  `ppg_model_version`, `schema_version`.
- **Rule: any track model change requires full re-extraction** of the
  fusion training set. Because raw inputs are archived, this is a batch
  job rather than a re-collection.
- DVC is already in the project stack — use it for exactly this.
