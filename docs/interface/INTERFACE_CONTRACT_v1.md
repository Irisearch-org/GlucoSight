# GlucoSight Interface Contract v1.1

**Status:** DRAFT — to be finalized at Sprint 2 Interface Meeting
**Owner:** Project Manager
**Last updated:** Sprint 1 (post architecture review)
**Supersedes:** v1.0 draft

All tracks must comply with this contract from Sprint 3 onwards.
No track may change output variables without PM approval and
a new version of this document.

> **Read this with `docs/REVIEW_FINDINGS.md`.** Every change in v1.1
> traces to a numbered finding there. If you disagree with a change,
> argue against the finding, not the table.

---

## What changed in v1.1 (read this first)

| # | Change | Finding |
|---|--------|---------|
| 1 | `rppg_*` variables renamed to `ppg_*` — the track is **contact PPG**, not remote PPG | C6 |
| 2 | `ppg_glucose_estimate` **kept** but ring-fenced: separate split, never sole prediction, reported separately | E1 |
| 3 | Added `*_present` missingness masks for all three modalities | H1 |
| 4 | Degradation rule resolved: **down-weight, never exclude** | H1 |
| 5 | Added meal timing block: `meal_id`, `t0_timestamp`, `delta_t_minutes` | C1 |
| 6 | Added `portion_reported` (in-app question) and `carbs_source` | C5 |
| 7 | Normalization spec filled in — was "TBD in Sprint 3" | H7 |
| 8 | `gi_category` consumed as **one-hot**, not as integer 0/1/2 | H7 |
| 9 | Added `schema_version` and per-track model version stamps | H6 |
| 10 | Added shared capture protocol (camera lock, timestamps) | C7 |

**Naming note:** directory names (`rppg/`, `agents/rppg/`) are unchanged so
paths do not break. Only the *variables* are renamed. The track is referred
to as the **contact PPG track** in all prose from v1.1 onward.

---

## Contract Overview

```
CV Track     ──┐
NLP Track    ──┼──► Forecasting Track ──► Predicted glucose T+60, T+120
PPG Track    ──┘
```

---

## 0. Meal Envelope (NEW in v1.1)

**Producer:** Mobile app / data collection tooling
**Consumer:** All tracks
**Produced at:** Once per meal

Every record from every track is keyed to a `meal_id`. Without this block,
nothing downstream can be aligned or audited.

| Variable | Type | Range | Required | Notes |
|----------|------|-------|----------|-------|
| meal_id | str (uuid) | — | YES | Primary key across all tracks |
| participant_id | str | — | YES | Used for GroupKFold splits |
| t0_timestamp | ISO8601 | — | YES | **First bite**, logged by in-app button |
| photo_timestamp | ISO8601 | — | NO | May differ from t0 |
| ppg_timestamp | ISO8601 | — | NO | May differ from t0 |
| note_timestamp | ISO8601 | — | NO | May differ from t0 |
| schema_version | str | — | YES | "1.1" |

**T=0 is the first bite, not the photo.** See
`docs/protocol/DATA_COLLECTION_PROTOCOL.md` §1. The photo is typically
taken before eating; a slow-eaten dish (nasi padang) and a fast one (soto)
would otherwise carry different label offsets, and that offset correlates
with food type — a confound, not just noise.

---

## CV Track Output Schema

**Producer:** CV Sub-Team
**Consumer:** Forecasting Sub-Team
**Produced at:** Per meal (once per food photo)

| Variable | Type | Range | Required | Fallback if missing |
|----------|------|-------|----------|---------------------|
| carbs_g | float32 | 0–300 | YES | population_mean=45.0 |
| protein_g | float32 | 0–150 | YES | population_mean=18.0 |
| fat_g | float32 | 0–150 | YES | population_mean=12.0 |
| fiber_g | float32 | 0–50 | YES | population_mean=4.0 |
| gi_category | int | 0,1,2 | YES | 1 (medium) |
| food_class | str | — | NO | "unknown" |
| cv_confidence | float32 | 0.0–1.0 | YES | 0.0 |
| **cv_present** | int | 0,1 | YES | 0 |
| **carbs_source** | str | see below | YES | "population_mean" |
| **portion_reported** | int | 0,1,2 | YES | 1 (sedang) |
| **cv_model_version** | str | — | YES | — |

**Notes:**
- gi_category: 0=low (<55), 1=medium (55–70), 2=high (>70)
- **gi_category is consumed as one-hot (3 dims), never as the integer.**
  The integer encoding asserts that the low→medium gap equals the
  medium→high gap in glycemic response. It does not.
- cv_confidence < 0.3: Forecasting down-weights CV features (never excludes)
- **`carbs_source` ∈ {`weighed`, `cv_regressed`, `class_lookup`, `population_mean`}**
  Forecasting must know what it is actually consuming. In Sprint 3 the
  expected value is `class_lookup` — see the honesty note below.
- **`portion_reported`** comes from the in-app question
  ("porsi: kecil / sedang / besar" → 0/1/2). It is **not** a CV output;
  CV passes it through so Forecasting receives one meal-composition dict.

### Honesty note on `carbs_g` (finding C5)

The Indonesian food set is classification-labeled. Macros are derived by
`food_class` → DKBM lookup. That means **within-class variance in the
regression target is zero**: every nasi goreng photo carries the same
`carbs_g`. The regression head therefore learns
`carbs_g = f(food_class)` — a lookup table, not portion estimation.

Consequence: `carbs_g` is a **class-conditional population estimate**, and
`portion_reported` is the project's **primary portion signal**. Both the
CV track doc and the paper's Limitations must state this plainly.

Upgrade path if capacity appears later: weighed ground truth on a 200–300
photo subset, which is the only thing that makes the regression head learn
anything beyond the lookup. Not committed for this cycle.

---

## NLP Track Output Schema

**Producer:** NLP Sub-Team
**Consumer:** Forecasting Sub-Team
**Produced at:** Per meal (once per text note)

| Variable | Type | Range | Required | Fallback if missing |
|----------|------|-------|----------|---------------------|
| is_stressed | int | 0,1 | YES | 0 |
| is_poor_sleep | int | 0,1 | YES | 0 |
| is_high_activity | int | 0,1 | YES | 0 |
| is_fried_cooking | int | 0,1 | YES | 0 |
| is_large_portion | int | 0,1 | YES | 0 |
| nlp_confidence | float32 | 0.0–1.0 | YES | 0.0 |
| nlp_embedding | np.array(128,) | — | NO | np.zeros(128) |
| **nlp_present** | int | 0,1 | YES | 0 |
| **nlp_model_version** | str | — | YES | — |

**Notes:**
- **`nlp_present=0` means "no text was written."** All-zeros labels with
  `nlp_present=1` means "text was written and reported no stress, no poor
  sleep, etc." These are different states and must never share an encoding.
  `nlp_confidence` alone cannot carry this distinction (finding H1).
- `nlp_confidence` must be a **calibrated** probability (isotonic or Platt),
  not a raw sigmoid output. Forecasting uses it as a gating weight; an
  uncalibrated score makes that gate meaningless.
- Decision thresholds are calibrated on the **realistic (patient) label
  distribution**, not on the balanced 40%-positive annotation set.
- Label correlations to note: is_stressed and is_poor_sleep often co-occur.
- **`is_large_portion` and `is_fried_cooking` overlap with CV outputs**
  (`carbs_g`/`portion_reported` and `fat_g`). Forecasting must report the
  feature correlation matrix before interpreting ablation results (H2).

---

## Contact PPG Track Output Schema

**Producer:** PPG Sub-Team (directory: `rppg/`)
**Consumer:** Forecasting Sub-Team
**Produced at:** Per capture (30-sec finger video, per meal)

| Variable | Type | Range | Required | Fallback if missing |
|----------|------|-------|----------|---------------------|
| ppg_pulse_rate_bpm | float32 | 40–200 | YES | 75.0 (population mean) |
| ppg_hrv_rmssd | float32 | 0–200 | NO | 30.0 |
| ppg_signal_quality | float32 | 0.0–1.0 | YES | 0.0 |
| ppg_perfusion_index | float32 | 0.0–20.0 | NO | 2.0 |
| ppg_glucose_estimate | float32 | 70–400 | NO | None |
| ppg_embedding | np.array(128,) | — | NO | np.zeros(128) |
| **ppg_present** | int | 0,1 | YES | 0 |
| **ppg_clip_fraction** | float32 | 0.0–1.0 | YES | 1.0 |
| **ppg_n_windows** | int | 0–3 | YES | 0 |
| **ppg_model_version** | str | — | YES | — |

**Notes:**
- **`ppg_hrv_sdnn` is replaced by `ppg_hrv_rmssd`** (finding H3). SDNN needs
  5-minute windows; RMSSD is more stable over short recordings. Peak
  locations must use sub-sample (parabolic/cubic) interpolation — at 30 fps
  the raw inter-beat quantization is 33 ms, the same size as the effect.
- **`ppg_signal_quality < 0.3`: Forecasting DOWN-WEIGHTS. It does not
  exclude.** This resolves the v1.0 contradiction between this file and
  `agents/forecasting/CLAUDE.md` (finding H1). Excluding would change input
  dimensionality at inference time, which is strictly worse engineering.
- **`ppg_glucose_estimate` is ring-fenced** — see the section below.
- `ppg_perfusion_index` = AC/DC ratio. Cheap to compute, physiologically
  real, better motivated than a raw learned embedding.
- `ppg_clip_fraction` = fraction of frames where the red channel saturates.
  With the flash on, red frequently clips, which flattens the pulsatile
  component you are trying to measure. High clip fraction ⇒ low quality.
- `ppg_n_windows` = how many valid 10-second windows were extracted from
  the 30-second capture. Features are the **median across windows**.
- `ppg_embedding`: optional — include only if Forecasting requests.
- If video is not provided: all values at fallback, `ppg_present=0`,
  `ppg_signal_quality=0.0`.

### Ring-fence rules for `ppg_glucose_estimate` (finding E1)

Confirmed by PM: this output **stays in the contract**. It is the track's
novelty claim and it is worth pursuing. It is fenced so that it cannot
quietly contaminate the primary fusion result.

1. It is trained **only** on the PPG glucose dataset in
   `rppg/PPG_Dataset/`, split by **subject** (`GroupKFold`), never on the
   Indonesian fine-tuning set. This prevents leakage into the fusion model's
   own labels.
2. It is **never used as a sole prediction** and never surfaced to a user.
3. It is reported in the paper in **its own section**, with its own metrics
   and its own n. It does not appear in the headline fusion table.
4. Fusion ablations must be run **with and without** it, so its contribution
   is isolable.
5. Any claim about it is bounded by n=23 subjects. See the honesty note in
   `agents/rppg/CLAUDE.md`.

---

## Forecasting Track Output Schema

**Producer:** Forecasting Sub-Team
**Consumer:** Integration layer + Prototype
**Produced at:** Per meal (once per inference)

| Variable | Type | Range | Required | Notes |
|----------|------|-------|----------|-------|
| glucose_t60 | float32 | 70–400 | YES | mg/dL at T+60 min |
| glucose_t120 | float32 | 70–400 | YES | mg/dL at T+120 min |
| glucose_range | str | normal/borderline/high | YES | For UI traffic light |
| prediction_confidence | float32 | 0.0–1.0 | YES | Model confidence |
| **delta_t_minutes** | float32 | 0–240 | YES | Actual horizon predicted |
| **model_version** | str | — | YES | — |

**Glucose range thresholds:**
- normal: < 140 mg/dL
- borderline: 140–199 mg/dL
- high: ≥ 200 mg/dL

**On `delta_t_minutes` (finding C1):** the model predicts glucose at an
arbitrary Δt from t0, not at a magic 60/120. Ground-truth pricks will land
anywhere from T+45 to T+90 in real home use, and glucose moves 1–3 mg/dL
per minute during the rise — a 20-minute timing error is 20–60 mg/dL of
label noise, larger than any RMSE worth reporting. Passing Δt explicitly
turns a fatal bias into a controlled covariate. Evaluation then restricts
to Δt ∈ [50,70] and [110,130] and reports how many samples were excluded.

---

## Normalization Spec (was TBD — finding H7)

Applies at the Forecasting input boundary. Not negotiable per-track.

| Feature class | Treatment |
|---------------|-----------|
| Continuous (carbs_g, pulse_rate, …) | z-score, statistics from **training split only**, persisted to disk, applied unchanged to val/test |
| Binary flags (is_*, *_present) | left as 0/1, never z-scored |
| gi_category | one-hot (3 dims) |
| portion_reported | one-hot (3 dims) |
| Embeddings (128-dim) | **omitted from the Sprint 6 baseline**; if added, project to 8–16 dims first |

**Why embeddings are omitted by default:** two 128-dim embeddings against
~15 interpretable scalars means the embeddings dominate the gradient by
sheer dimensionality, and the clinically-motivated features the paper is
about get washed out. Add them as an explicit ablation, not as a default.

**Leakage rule:** normalization statistics computed over the full dataset
are a leak. The loader must compute them from the training split and
assert that val/test never contribute.

---

## Shared Capture Protocol (NEW in v1.1 — finding C7)

These are **contract-level requirements on the mobile app**, not
track-internal implementation details. The PPG track cannot enforce them
alone.

1. **Lock the camera before recording.** Android Camera2:
   `CONTROL_AE_MODE_OFF` with fixed `SENSOR_EXPOSURE_TIME` and
   `SENSOR_SENSITIVITY`; `CONTROL_AWB_MODE_OFF` with fixed
   `COLOR_CORRECTION_GAINS`; `CONTROL_AF_MODE_OFF`.
   *Rationale: auto-exposure exists to hold brightness constant. The
   pulsatile signal is a small periodic brightness variation. Leaving AE on
   means the camera actively cancels the signal being measured.*
2. **Record per-frame presentation timestamps.** Never assume 30 fps.
   Smartphone video is frequently variable-rate under the low light of a
   finger over the lens. Resample to a uniform grid before any FFT.
3. **Discard the first 2–3 seconds** of every capture (LED thermal settling).
4. **Tune exposure so the red channel sits near 70% of full scale** and log
   `ppg_clip_fraction`.
5. **Log `device_model`, `os_version`, `mean_fps`, `fps_jitter`** on every
   capture, so device-driven bias is detectable later.

---

## Shared Data Format

All cross-track data is passed as Python dictionaries.
Serialization format for saving: JSON (for logs), NumPy .npy (for embeddings).

```python
# Example full input to Forecasting at meal time
meal_input = {
    # Envelope
    "schema_version": "1.1",
    "meal_id": "a3f9…",
    "participant_id": "P042",
    "t0_timestamp": "2025-01-15 19:30:00",   # FIRST BITE
    "delta_t_minutes": 62.0,                 # actual horizon to the label

    # From CV
    "carbs_g": 65.2,
    "protein_g": 12.1,
    "fat_g": 8.3,
    "fiber_g": 2.1,
    "gi_category": 2,
    "portion_reported": 2,
    "carbs_source": "class_lookup",
    "cv_confidence": 0.87,
    "cv_present": 1,

    # From NLP
    "is_stressed": 1,
    "is_poor_sleep": 1,
    "is_high_activity": 0,
    "is_fried_cooking": 1,
    "is_large_portion": 0,
    "nlp_confidence": 0.91,
    "nlp_present": 1,

    # From contact PPG
    "ppg_pulse_rate_bpm": 82.3,
    "ppg_hrv_rmssd": 28.4,
    "ppg_perfusion_index": 3.1,
    "ppg_signal_quality": 0.74,
    "ppg_clip_fraction": 0.02,
    "ppg_n_windows": 3,
    "ppg_present": 1,

    # Glucose history (sparse, CAUSAL ONLY — see below)
    "glucose_history": [
        {"timestamp": "2025-01-15 07:30", "value": 118.0},
        {"timestamp": "2025-01-15 12:00", "value": 142.0}
    ],

    # Time features
    "time_since_last_meal_hours": 7.5
}
```

**Causality rule (finding C2):** every entry in `glucose_history` must
satisfy `timestamp < t0_timestamp`. The loader asserts this. Interpolation
happens *after* filtering, never before — otherwise the interpolated
sequence handed to the model is computed from the T+60 and T+120 readings
that are the prediction targets.

---

## Version History

| Version | Date | Changes | Approved by |
|---------|------|---------|-------------|
| v0.1 | Sprint 1 | Draft — proposals only | PM |
| v1.0 | Sprint 1 | Initial full schema | PM (draft) |
| v1.1 | Sprint 1 | Post-review: contact PPG rename, missingness masks, meal envelope, causality rule, normalization spec, capture protocol | PM (draft) |
| v2.0 | Sprint 2 | TBD after Interface Contract meeting | All tracks |

---

## Sign-off (to be completed at Sprint 2 meeting)

| Track | Member | Confirmed | Date |
|-------|--------|-----------|------|
| Contact PPG | | ☐ | |
| CV | | ☐ | |
| NLP | | ☐ | |
| Forecasting | | ☐ | |
| PM | | ☐ | |

**Open items requiring a decision at the Sprint 2 meeting:**
1. Primary endpoint and paper claim wording — deferred by PM, must be
   settled before fine-tuning begins.
2. Whether `ppg_embedding` / `nlp_embedding` are requested by Forecasting
   at all.
3. Capture window: 30 s is contracted. The PPG track may propose 60 s after
   the pilot if compliance data supports it.
