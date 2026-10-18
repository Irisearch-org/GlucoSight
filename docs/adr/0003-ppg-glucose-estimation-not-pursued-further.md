# PPG glucose estimation is not pursued further without an HbA1c/calibration data change

**Status:** accepted (2026-10-19)

Two trained models (Ridge, Random Forest) already lost to the population-
mean baseline on this project's own 23-subject contact-PPG dataset
(`rppg/README.md`). Before a third modeling attempt, we researched whether
that's a fixable engineering gap or a settled negative result.

**It's settled, independently.** A peer-reviewed 2026 paper (Ramesh,
Neufeld, Küttner, Röddiger & Beigl, *"Reassessing the Feasibility of
PPG-Based Non-Invasive Blood Glucose Level Estimation,"* ACM ISWC '26, DOI
`10.1145/3830727.3834821`, arXiv:2608.01820, Karlsruhe Institute of
Technology) re-ran this exact experiment — five model architectures, correct
subject-held-out split — on **this project's own dataset** (MUST: 23
individuals, 67 sessions, 88–183 mg/dL, the same `rppg/data/labels.csv`
this project uses) and got the same null result: near-zero/negative R²,
Clarke Zone A+B above 90% for every model *including the mean baseline* —
the same reason this project's own `rppg/models/glucose_estimator.py`
already distrusts Zone A on this cohort (finding C3).

We decided: **no further PPG glucose-estimation modeling work, on this
dataset, with this hardware, until the data itself changes.** The
literature's genuine successes are load-bearing on things this project
doesn't capture — an auxiliary HbA1c/medication covariate (one source: R²
0.06 → 0.71 purely from adding HbA1c) or per-subject repeated-measures
calibration — not better signal processing or a bigger model. A smartphone
RGB camera also cannot reach the 850–940 nm near-infrared band the
literature flags as more glucose-sensitive; that is a hardware ceiling a
software change cannot cross.

## Considered options

| option | rejected because |
|---|---|
| try a third model architecture (e.g. gradient boosting, a small neural net) | the independent re-test already covered five architectures on this exact dataset and got the same null result; a sixth model is not a new question |
| improve the existing feature extraction (`rppg/features/extractor.py`) | P1–P7 already found the current perfusion index and HRV extraction are measurably broken, but the independent paper's result holds even with *correct* extraction — the ceiling is the signal/hardware, not this project's specific bugs |
| lift ring-fence E1.2 and ship the mean-equivalent estimate anyway | would ship a number mathematically identical to the population fallback, labeled as if it carries PPG information — the exact failure mode E1.2 exists to prevent |
| **stop, document the ceiling, name the one credible reopening condition** | *chosen* |

## Consequences

- **Ring-fence E1.2 stays as-is.** There is currently no evidence to build
  a case for amending it — see `agents/pm/CLAUDE.md` /
  `docs/interface/INTERFACE_CONTRACT_v1.md` for the amendment process if
  that ever changes.
- **The one credible next step, if ever revisited:** capture a
  self-reported HbA1c/medication-status field at the pilot protocol stage
  and re-run the existing Ridge pipeline with it added — a data-collection
  decision for `docs/protocol/DATA_COLLECTION_PROTOCOL.md`, not a modeling
  task. Nothing scheduled; recorded so it isn't silently retried from
  scratch later.
- **Model-level forecasting fusion (ADR 0004) proceeds without PPG as an
  input.** `g0` continues to come from a direct measurement (CGM,
  fingerstick) or the population fallback only.

Full findings, all 15 primary sources: `rppg/LITERATURE_REVIEW_GLUCOSE_ESTIMATION.md`.
