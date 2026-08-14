# GlucoSight — Agent Registry

This file describes all agents in the GlucoSight project,
their roles, scope, and how to invoke them.

---

## Agent Hierarchy

```
┌─────────────────────────────────────────────┐
│           PM Agent (root level)             │
│   Full repo access · Integration focus      │
│   Critical thinking · Cross-track review    │
└──────────┬──────────────────────────────────┘
           │
     ┌─────┴──────┐
     │            │
┌────┴───┐   ┌───┴───────────────────────────────┐
│ Track  │   │  Sub-Team Agents                  │
│ Agents │   │  Contact PPG · CV · NLP · Forecast│
└────────┘   └───────────────────────────────────┘
```

---

## PM Agent

**Location:** `agents/pm/CLAUDE.md`
**Run from:** repo root (`/`)
**Invoked by:** Project Manager only

**Responsibilities:**
- Cross-track integration and critical review
- Interface contract enforcement
- Sprint progress assessment
- Fusion layer architecture decisions
- Prototype and paper coordination
- Raising cross-track blockers

**Commands the PM agent understands:**
```
/status          — summarize current sprint progress across all tracks
/review {track}  — critically review a track's output or proposal
/contract        — load and reason about the interface contract
/blocker         — help diagnose and resolve a cross-track blocker
/fusion          — guide fusion layer architecture decisions
/paper           — assist with paper writing and structure
```

---

## Contact PPG Agent

**Location:** `agents/rppg/CLAUDE.md`
**Run from:** `rppg/` directory
**Invoked by:** contact PPG sub-team members

> Formerly the "rPPG agent". The track is **contact PPG** (finger against
> the lens with the flash on), not remote PPG (face video). Directory names
> are unchanged; contract variables are `ppg_*`. See finding C6.

**Responsibilities:**
- Signal processing guidance (RGB extraction, filtering, decimation)
- Contact PPG model architecture (classical features vs CNN/Transformer)
- **`rppg/PPG_Dataset/` handling** — the provided glucose set (23 subjects,
  10 s @ 2190 Hz, glucometer label per recording) and BUT PPG
  (`brno/`, 12 subjects, 10 s @ 30 Hz PPG + 1000 Hz ECG, CC-BY 4.0)
- Subject-level splitting and leakage prevention
- Output variable proposal and interface contract compliance
- Experiment tracking and ablation suggestions

**Commands:**
```
/help            — what can I ask this agent?
/papers          — list required reading for this sprint
/data            — summarize PPG_Dataset contents and label conversion status
/implement {task}— guide implementation of a specific task
/debug           — help debug signal processing issues
/output          — review output variable proposal
/experiment      — suggest next experiment based on current results
```

---

## CV Agent

**Location:** `agents/cv/CLAUDE.md`
**Run from:** `cv/` directory
**Invoked by:** CV sub-team members

**Responsibilities:**
- Food recognition model guidance (EfficientNet-B3 vs ViT)
- Indonesian food dataset handling and annotation
- Macro regression pipeline (carbs, protein, fat, fiber, GI)
- Transfer learning and fine-tuning strategy
- Output variable proposal and interface contract compliance

**Commands:**
```
/help            — what can I ask this agent?
/papers          — list required reading for this sprint
/implement {task}— guide implementation of a specific task
/debug           — help debug training issues
/output          — review output variable proposal
/dataset         — guidance on Indonesian food dataset collection
/experiment      — suggest next experiment
```

---

## NLP Agent

**Location:** `agents/nlp/CLAUDE.md`
**Run from:** `nlp/` directory
**Invoked by:** NLP sub-team members

**Responsibilities:**
- IndoBERT loading, tokenization, and fine-tuning
- Annotation methodology and label scheme design
- Multi-label classification pipeline
- Indonesian health language guidance
- Output variable proposal and interface contract compliance

**Commands:**
```
/help            — what can I ask this agent?
/papers          — list required reading for this sprint
/implement {task}— guide implementation of a specific task
/debug           — help debug fine-tuning issues
/annotate        — guidance on annotation task and label scheme
/output          — review output variable proposal
/experiment      — suggest next experiment
```

---

## Forecasting Agent

**Location:** `agents/forecasting/CLAUDE.md`
**Run from:** `forecasting/` directory
**Invoked by:** Forecasting sub-team members

**Responsibilities:**
- LSTM and TFT architecture guidance
- OhioT1DM and DiaTrend dataset handling
- Input variable requirements definition
- Clarke Error Grid implementation and evaluation
- Temporal train/test split strategy
- Interface contract compliance from consumer side

**Commands:**
```
/help            — what can I ask this agent?
/papers          — list required reading for this sprint
/implement {task}— guide implementation of a specific task
/debug           — help debug model training issues
/evaluate        — run Clarke Error Grid evaluation on predictions
/requirements    — review input variable requirements document
/experiment      — suggest next experiment
```

---

## Routing Rules

| Question type                          | Go to agent     |
|----------------------------------------|-----------------|
| PPG signal processing, noise, OpenCV   | Contact PPG agent|
| PPG_Dataset, label conversion, decimation | Contact PPG agent|
| Meal timing, capture protocol, pilot   | PM agent        |
| Data leakage / split questions         | PM agent        |
| Food recognition, EfficientNet, macros | CV agent        |
| IndoBERT, annotation, multi-label      | NLP agent       |
| LSTM, TFT, glucose forecasting         | Forecasting agent|
| Cross-track variable compatibility     | PM agent        |
| Fusion architecture decisions          | PM agent        |
| Interface contract questions           | PM agent        |
| Sprint planning, task prioritization   | PM agent        |
| Paper writing, results interpretation  | PM agent        |
| Clarke Error Grid implementation       | Forecasting agent|
| General Python / PyTorch debugging     | Your track agent|

---

## How Agents Load Context

When Claude Code starts in a directory, it reads CLAUDE.md files
in this order (each adds context on top of the previous):

1. `/CLAUDE.md`               ← shared project context (always)
2. `/agents/{track}/CLAUDE.md` ← track-specific context
3. Any file you explicitly reference in the conversation

The PM agent reads:
1. `/CLAUDE.md`
2. `/agents/pm/CLAUDE.md`
3. All track CLAUDE.md files on demand

---

## Required Reading for Every Track (Sprint 1)

Beyond your own track file, everyone reads:

| Document | Why |
|----------|-----|
| `docs/interface/INTERFACE_CONTRACT_v1.md` | **v1.1** — the schema you produce or consume. Authoritative |
| `docs/protocol/DATA_COLLECTION_PROTOCOL.md` | Meal timing and capture requirements affect every track |
| `docs/REVIEW_FINDINGS.md` | Sprint 1 architecture review. Your track's rules trace to numbered findings here |

Each `agents/{track}/CLAUDE.md` now opens with a **Track Objective** and
carries a **Sprint 2 Task List** in execution order. Start there.
