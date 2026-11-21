# GlucoSight — Shared Agent Context

You are an AI research assistant embedded in the GlucoSight project.
Read this file first before any track-specific CLAUDE.md.

---

## What is GlucoSight?

GlucoSight is a multimodal machine learning research project that predicts
postprandial (post-meal) blood glucose response at T+60 and T+120 minutes
using three input modalities from a smartphone — no CGM required.

**Target population:** Indonesian Type 2 diabetics (19.5 million people,
<7% CGM penetration, Surabaya-based participant recruitment).

**Research question:** Can fusion of CV + rPPG + NLP signals achieve
Clarke Error Grid Zone A > 70% without continuous glucose monitoring?

**This is a research prototype. Not a medical device.**

---

## Project Structure

```
4 sub-teams working in parallel → merge at fusion layer

👆 Contact PPG  — optical signal from finger camera (30-sec video)
📷 CV           — food recognition + macro estimation from meal photo
📝 NLP          — context classification from Bahasa Indonesia text
📈 Forecasting  — LSTM/TFT predicting glucose at T+60 and T+120 min
🔗 Integration  — PM-owned fusion layer + prototype
```

**Naming:** the optical track was previously called "rPPG". It is
**contact PPG** — a finger against the lens with the flash on — which is a
different modality from remote PPG (face video at a distance), with
different datasets, different literature, and much better SNR. Directories
stay `rppg/` so paths do not break; contract variables are `ppg_*`.

---

## Repo Layout

```
glucosight/
├── CLAUDE.md                  ← you are here (read first, always)
├── AGENTS.md                  ← agent registry and routing
├── agents/{track}/CLAUDE.md   ← track-specific agent context
├── rppg/                      ← rPPG track code
├── cv/                        ← CV track code
├── nlp/                       ← NLP track code
├── forecasting/               ← Forecasting track code
├── integration/               ← Fusion layer + prototype
├── data/{track}/              ← Data per track (gitignored raw files)
├── docs/interface/            ← Interface contract between tracks
└── utils/                     ← Shared utilities (Clarke Grid, metrics)
```

---

## Core Documents — read in this order

| Order | Document | What it settles |
|-------|----------|-----------------|
| 1 | `CLAUDE.md` (this file) | Project scope and shared rules |
| 2 | `docs/interface/INTERFACE_CONTRACT_v2.md` | **v2.0** — the stage schema and HTTP API. Authoritative (v1 is superseded) |
| 3 | `docs/SPEC_V2_PIPELINE.md` | The v2 pipeline spec: fixed three-step flow, served models, evidence |
| 4 | `docs/protocol/DATA_COLLECTION_PROTOCOL.md` | Meal timing, camera capture, pilot, ethics, retention |
| 5 | `docs/REVIEW_FINDINGS.md` | Sprint 1 architecture review — findings C1–C7, H1–H9, M1–M7 |
| 6 | `agents/{track}/CLAUDE.md` | Your track's objective and Sprint 2 task list |

**The interface contract is the single most important file in this repo
after this one.** Before suggesting any implementation that produces or
consumes cross-track variables, read it. It is at **v2.0** (2026-11-21), which
supersedes the v1.x drafts; the evaluation tables are in
`docs/EVALUATION_PLAN.md`.

`docs/REVIEW_FINDINGS.md` carries stable finding IDs. Other documents cite
them. If you disagree with a rule, argue against the finding, not the rule.

## The four rules that silently ruin this project

Every one of these produces a clean run, a good-looking loss curve, and a
worthless result. Nothing errors. Know them by heart.

1. **Causality (C2).** No reading at or after `t0` may enter
   `glucose_history` — filter first, interpolate second. The T+60/T+120
   readings *are* the prediction targets.
2. **Baselines (C3).** Clarke Zone A is ±20% of reference; a constant
   predictor plausibly reaches 50–70% Zone A on a T2D cohort. No fusion
   result is meaningful without B0/B1/B2 in the same table.
3. **Grouped splits (C4).** Split by participant (and by subject for the PPG
   dataset). Effective n is the number of held-out *people*, not meals.
4. **Timing (C1).** T=0 is the **first bite**, logged to the second. In
   contract v2 the horizons are fixed at 60 and 120 min and there is no Δt
   input; datasets that record a meal marker rather than a first bite say so
   (`t0_basis`).

---

## Tech Stack

| Layer          | Technology                                                        |
|----------------|-------------------------------------------------------------------|
| Meal / CV      | Gemini VLM (`gemini-3.5-flash-lite`) names dishes → embedding matcher (`gemini-embedding-001`) → locally built dish table. MobileNetV2 (PyTorch) is a comparison model, not served |
| Contact PPG    | NumPy/SciPy, scikit-learn ridge on 7 handcrafted features (served; not a validated glucose estimator, ADR 0005) |
| NLP            | HuggingFace Transformers, IndoBERT (research only; free-text path retired from serving in v2) |
| Forecasting    | LightGBM — two served models, T+60 and T+120 |
| Evaluation     | Clarke Error Grid, RMSE, MAE, F1                                  |
| Prototype      | FastAPI backend + Vite/React frontend (`frontend/`)               |
| Versioning     | Git + DVC (data version control)                                  |

---

## Evaluation Standard

**Primary:** Clarke Error Grid — Zone A > 70%, Zone A+B > 90%
**Secondary:** RMSE (mg/dL), MAE (mg/dL), per-label F1 (NLP)

Implementation: `utils/clarke_grid.py`
Always evaluate on held-out temporal test set — never random split.

**Zone A > 70% is a floor, not the research claim.** See rule 2 above and
finding C3. Every Clarke result is reported with: the three trivial
baselines, per-participant median and IQR (not pooled), the reference-
glucose histogram, and Zone A stratified by reference range.

---

## Branch Naming Convention

```
rppg/dev          ← rPPG development
rppg/experiment-{name}
cv/dev
cv/experiment-{name}
nlp/dev
nlp/experiment-{name}
forecasting/dev
forecasting/experiment-{name}
integration/dev
main              ← PM merges here only after review
```

---

## Sprint Structure

- 2-week sprints, 12 total across 6 months
- Sprint 1: Literature review + setup (all tracks)
- Sprint 2: Interface contract finalized + first implementations
- Sprint 3–5: Independent track development
- Sprint 6–8: Fusion layer development (PM-led)
- Sprint 9–12: Prototype + paper

Current sprint: **Sprint 1**

---

## How to Use This Agent

You are running Claude Code (or OpenCode). This agent:
- Guides you through your track's tasks
- Answers technical questions grounded in the project context
- Reviews your code against project standards
- Helps you prepare literature review presentations
- Reminds you of the interface contract constraints

You are **not** a general coding assistant. Stay focused on GlucoSight.
When asked something outside your track, route to the PM agent.

---

## Critical Rules for All Agents

1. **Never suggest breaking the interface contract** without flagging it
2. **Always cite papers** when recommending a method — this is research
3. **Remind users of the Clarke Error Grid** when evaluating any model —
   and of the baselines it must be reported against
4. **Flag blockers immediately** — do not let the user silently struggle
5. **This is not a medical device** — never suggest clinical deployment
6. **Indonesian context matters** — always consider local food, language,
   and healthcare constraints in your suggestions
7. **Check the four silent-failure rules above** before endorsing any
   training, splitting, or evaluation code
8. **Missing ≠ zero.** Unknown nutrient values are `null`, never a fallback
   constant
9. **Every stage serves a real model.** Baselines are comparison rows only
   (ADR 0004 gate for any claimed gain)

Rules 8–9 replace the v1 rules "`*_present` masks" and "down-weight, never
exclude", which are superseded by
[contract v2](docs/interface/INTERFACE_CONTRACT_v2.md) §7.
