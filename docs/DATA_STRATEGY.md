# GlucoSight — Data Strategy

**Date:** 2026-08-20
**Status:** Decided by PM. Supersedes the primary-collection plan for the
Forecasting and NLP tracks.
**Affects:** Forecasting, NLP, the interface contract, and the paper's
population claim. CV and contact PPG are addressed in §6.

---

## 1. The decision

**Primary data collection is cancelled.** No participant recruitment, no
pilot, no team annotation. The Forecasting and NLP tracks run entirely on
public datasets, all of which are open access and downloadable today.

This was a capacity decision, not a scientific one. The consequences below
are stated so that nobody later mistakes a constraint for a finding.

---

## 2. What is available, and what is not

Every *pair* of (modality → glucose) exists in public data. **The triple
does not.** No public dataset carries a meal photograph, a text record, and
a PPG signal for the same meal with a glucose outcome.

| Dataset | Modalities paired with glucose | Cohort | Size | Licence |
|---|---|---|---|---|
| **CGMacros** | meal photos + weighed macros + CGM + Fitbit | 45 (15 healthy, 16 pre-D, **14 T2D**), 10 d | 627 MB | CC BY-NC-SA 4.0 |
| **ShanghaiT2DM** | dietary text records + CGM | **100 T2D**, Chinese, 3–14 d | small | Open (figshare) |
| **BIG IDEAs** | raw 64 Hz PPG/BVP + CGM + food log | 16 pre-diabetic, 8–10 d | 4.7 GB zip | ODC-BY 1.0 |
| `rppg/PPG_Dataset` | single-shot PPG + glucometer label | 23 subjects, 67 recordings | on disk | provided |
| ~~OhioT1DM~~ | CGM + insulin/carbs/HR | 12 T1D | — | **DUA required** |

Sources:
- CGMacros — <https://physionet.org/content/cgmacros/1.0.0/>
- BIG IDEAs — <https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/>
- ShanghaiT1DM / ShanghaiT2DM — figshare collection, *Scientific Data* (2023)

---

## 3. The fusion claim, revised

The research question in the root `CLAUDE.md` — *"Can fusion of CV + rPPG +
NLP achieve Clarke Zone A > 70%?"* — **cannot be answered on public data as
literally stated.** A tri-modal number would require stitching unrelated
cohorts together, and finding C3's baseline discipline is precisely what
would expose such a number as meaningless.

**Decision: anchor on CGMacros and claim bi-modal fusion.**

- **Headline claim becomes:** meal composition and meal context improve
  postprandial glucose prediction over B0/B1/B2 baselines and over
  glucose-history-only — measured on real paired data, on the same meals.
- **Contact PPG** reports its contribution separately, on BIG IDEAs and on
  the provided PPG dataset.
- **Rejected: the simulated join.** Sampling a PPG vector and a text note
  from unrelated cohorts onto CGMacros meals would preserve the original
  paper framing while measuring the join assumptions rather than
  physiology. It is not done, and no table in the paper should imply
  otherwise.

---

## 4. What this costs

State all of this in Limitations. None of it is hidden by the design; all
of it is hidden by *not writing it down*.

1. **The Indonesian population claim leaves the empirical results.** The
   evidence base is an American cohort (CGMacros) and a Chinese T2D cohort
   (ShanghaiT2DM). The project becomes a method validated on public
   cohorts, with a designed protocol for Indonesian validation as future
   work. `docs/protocol/DATA_COLLECTION_PROTOCOL.md` is not wasted — it
   becomes that future-work section, and is a contribution in itself.
2. **Simulated sparsity is missing-at-random; real fingerstick missingness
   is behavioural and missing-not-at-random** (finding M1, item 4). Under
   the old plan this was one domain gap among several. It is now a
   first-class limitation of the entire result, because no local data will
   correct it.
3. **Effective sample size is small and must be stated as participants.**
   Five-fold `GroupKFold` on CGMacros leaves ~9 held-out participants per
   fold; the T2D subgroup is 14 people, so a T2D-stratified claim rests on
   ~3 held-out T2D participants per fold (finding C4).
4. **The cohort's glucose distribution is lower than a medicated T2D
   cohort's**, so Clarke Zones C/D/E are even less evaluable than finding
   M3 anticipated. "Zone E: 0.0%" means *no data there*, not *safe there*.
5. **Licensing propagates.** CGMacros is CC BY-NC-SA 4.0 — non-commercial
   and share-alike apply to anything trained on it. Acceptable for the
   paper; a real constraint on the prototype.

---

## 5. What this buys

Worth stating alongside the costs, because the trade is not one-sided.

1. **The schedule stops depending on recruitment and ethics clearance.**
   Every dataset is on disk within a day.
2. **OhioT1DM's DUA and the T1D → T2D domain gap both disappear.**
   ShanghaiT2DM is 100 T2D patients with no agreement to negotiate.
   Pretraining transfer becomes a T2D → T2D question — a cleaner one.
3. **CV gets a real regression target.** CGMacros breakfasts and lunches
   have *weighed* macronutrients, so finding C5's "within-class variance is
   zero" problem does not apply on the anchor dataset.
4. **Label leakage through interpolation (finding C2) is structurally
   resolved for the targets.** Dense CGM means T+60 and T+120 are read
   directly off the trace; there is no interpolation across a gap. The
   causal loader and its assertions remain for the *input* history, where
   the failure mode moved to.
5. **A free methodological contribution:** CGMacros ships two CGMs (Libre
   Pro at 15 min, Dexcom G6 at 5 min) that disagree by roughly 15–20 mg/dL
   at postprandial levels — the same order as the effect being hunted.
   Measuring and publishing that disagreement gives the paper an empirical
   noise floor, which this literature rarely reports.

---

## 6. Scope of this change

**Updated:** `agents/forecasting/CLAUDE.md`, `agents/nlp/CLAUDE.md`,
`docs/interface/INTERFACE_CONTRACT_v1.md` (→ v1.2).

**Deliberately not updated:** `agents/cv/CLAUDE.md`,
`agents/rppg/CLAUDE.md`, and the root `CLAUDE.md`. Those tracks have work
in flight and their docs should not shift under them mid-sprint.

**Known inconsistencies left open for the PM to resolve:**
- The root `CLAUDE.md` still states the tri-modal research question and an
  Indonesian target population. §3 and §4 above supersede it in practice.
- `agents/cv/CLAUDE.md` still targets the Indonesian food dataset and
  treats `carbs_source = class_lookup` as the expected value. On CGMacros
  the macros are weighed, which is a **better** situation than that doc
  describes — CV should be told.
- `docs/protocol/DATA_COLLECTION_PROTOCOL.md` describes a collection that
  is no longer scheduled. Retain it; reframe it as future work.

---

## 7. First milestone — the independent-model week

Each track builds a model independently. **Success criterion is a
trustworthy pipeline producing one real number beside a baseline, not a
good model.**

- **Day 1** — download, inspect, record what is actually in the files. No
  modelling. Forecasting fixes the Dexcom-not-Libre decision and reports
  the ShanghaiT2DM dietary-record format to NLP by end of day.
- **Day 2** — baselines only. B0/B1/B2 for Forecasting; lexicon for NLP;
  predict-the-mean for CV and PPG. Nobody trains a model before their
  baseline number exists.
- **Days 3–4** — one model per track, scikit-learn tier. Ridge, random
  forest, frozen-backbone embeddings. No LSTM, no TFT, no fine-tuned
  EfficientNet, no fine-tuned IndoBERT.
- **Day 5** — report. Every number beside its baseline, grouped splits
  asserted in code, per-participant median and IQR, held-out participant
  count stated.

**The one rule that decides whether the week is worth anything:** no track
reports a model number that is not printed next to its baseline in the same
table. That is finding C3, and in a compressed week it is the first thing
that gets skipped.

**Interface contract sign-off is deliberately deferred until after this
week.** v1.1 was unsigned and this pivot rewrites it; week 1's findings are
exactly what should inform v1.2's final form. Sign it with real numbers in
hand.

---

## 8. Blocking item carried into week 1

**Finding M7 is no longer deferrable.** `utils/clarke_grid.py` is a
simplified reimplementation with hand-written boundaries, never validated
against a reference, and its `__main__` sanity check generates
`y_pred = y_true * U(0.85, 1.15)` — within ±20% *by construction*, a
self-test that cannot fail. The PM deferred it when no result depended on
it. Week 1's deliverable is a Clarke number, so it must be validated
against published boundary vectors first. Roughly half a day.
