# Spec: GlucoSight paper (Indonesian, .docx)

**Status:** ready for agents (2026-11-22). Built on the user's draft `Paper_GlucoSight.docx` (Introduction + 11 references written; sections 2–6 empty).

## Problem Statement

The team has finished the GlucoSight experiments, which are recorded in tickets #5–#30, ADRs 0004–0005 and `docs/SPEC_V2_PIPELINE.md`. The paper itself is mostly empty.

The draft Introduction promises that food information **and PPG** improve the forecast. The results say otherwise:
- the meal information helps;
- open-vocabulary dish identification with abstention beats the closed classifier;
- PPG carries no glucose signal on two datasets.

The team needs a complete paper that reports exactly what was found, numbers included, and nothing more. Negative results must be reported as negative. Section writers must not invent or round numbers, and must not cite papers that nobody has checked.

## Solution

The paper is written in Bahasa Indonesia, following the draft's format:
- named authors (IRIS);
- English section headings "1. Introduction" … "6. Conclusion", "References";
- APA author-year citations (citeproc renders "dkk.").

Each section is one Markdown file, owned by one writer. A single build turns the sections, the verified bibliography and the figures into `GlucoSight_paper.docx`, using the draft's own Word styles (A4, 1-inch margins).

The story is **"what a smartphone can and cannot add to post-meal glucose forecasting without a CGM"**. It has three findings, each evaluated on people held out from training and compared against trivial baselines:

1. **Forecaster.** A fingerstick (or single CGM reading) plus meal macros beats persistence-plus-mean-excursion (B2) on held-out people. The meal macros account for about 1.7–2.0 mg/dL of that gain. Trained within a second cohort (100 Asian T2D patients, ShanghaiT2DM), the same method passes the gate. The served CGMacros model does not transfer to that cohort without macros.
2. **PPG.** Smartphone and contact PPG carry no usable glucose signal. This holds on two datasets, for handcrafted features and a PPG foundation model, and also within a person after calibration. Heart rate from the same camera signal works (3.76 bpm MAE vs ECG).
3. **Dish identification.** Open-vocabulary dish identification (VLM naming → embedding matcher → TKPI 2017 dish table), with a similarity + margin rule that sends uncertain items to manual entry, beats a closed MobileNetV2 classifier at the dish-family level. At the chosen rule it is right 89.6% of the time when it answers.

The draft's three research questions stay. The Discussion answers them from these results:
- **RQ1:** food info yes; PPG no.
- **RQ2:** how the design handles missing inputs.
- **RQ3:** how accurate dish identification needs to be. Realistic dish-ID errors change MAE by only +0.01/+0.14 mg/dL.

## User Stories

1. As a reviewer, I want every reported improvement to sit next to its baseline (B0/B1/B2 or the mean predictor), so that I can judge whether it matters.
2. As a reviewer, I want all evaluations to hold out whole people, so that I know the numbers aren't inflated by within-person leakage.
3. As a reviewer, I want 95% bootstrap CIs over participants, fold-win counts and permutation p-values, so that I can judge the uncertainty.
4. As a reviewer, I want the PPG null result stated plainly, with the mean baseline, r, and the within-person calibration test, so that I don't mistake a served component for a working one.
5. As a reviewer, I want to know the pre-specified forecaster design failed, and why the nested procedure replaced it, so that I can judge the researcher degrees of freedom.
6. As a reviewer, I want the external ShanghaiT2DM check to separate "the method generalizes" from "the served model transfers", so that the claim isn't overstated.
7. As a reviewer, I want the dish-ID evaluation to compare against the closed classifier on the overlapping classes, with paired bootstrap CIs at both family and exact-dish level, so that the comparison is fair.
8. As a reviewer, I want the operating rule (similarity ≥ 0.80, margin ≥ 0.05) to have been chosen on the dev split and reported on test, so that the threshold isn't tuned on test.
9. As a reviewer, I want the error-band coverage gaps (high g0, T2D) reported, so that I know where the system is least reliable.
10. As a reviewer, I want the excursion/monotone variant reported as "no gain", so that the negative ablations are visible.
11. As a reader in Indonesia, I want the motivation tied to Indonesian prevalence, SMBG affordability and Indonesian dishes, so that the problem is concrete.
12. As a reader, I want a pipeline figure that shows the three steps, and what each step serves and achieves, so that I understand the system in one look.
13. As a reader, I want the datasets in one table (n, signal, label, licence, use), so that I can see what each result rests on.
14. As a reader, I want the Discussion to answer the three research questions from the Introduction one by one.
15. As a reader, I want limitations stated: no single dataset with photo + PPG + glucose for the same people; small cohorts; simulated meter noise; dish serving weights assumed; the operating rule calibrated on the 35-row table; English-language CGMacros meals.
16. As a clinician-reader, I want a clear statement that this is a research prototype and not a medical device.
17. As an author, I want my Introduction kept as written, apart from the minimum corrections (TKPI edition, the PPG promise), so that my voice stays.
18. As an author, I want my named authors and IRIS affiliation, my headings and APA style kept.
19. As an author, I want every number in the paper traceable to `NUMBERS.md`, so that I can defend each one.
20. As an author, I want every citation fetched and verified from a DOI/arXiv source, with unverifiable ones flagged, so that nothing is hallucinated.
21. As an author, I want the paper to build with one command into a .docx I can open in Word and keep editing.
22. As an author, I want the figures' text in Indonesian and readable in grayscale.
23. As a section writer, I want a fixed list of bibliography keys and figure file names, so that I can write without waiting for other writers.
24. As a section writer, I want the build to tell me which number I got wrong, so that I can fix it without re-reading every ticket.

## Implementation Decisions

**Language and number style:**
- Bahasa Indonesia, formal academic register. English technical terms appear in *italics* at first use with the Indonesian gloss, as in the draft (e.g. *continuous glucose monitoring* (CGM)).
- Decimal comma: 28,78. Thousands with a dot only above 9.999, else none (1534).
- Confidence intervals as "[1,36; 4,42]". Units "mg/dL" and "bpm". Percentages as "61,5%".
- Signs on gains: "+2,90 mg/dL".

**Numbers:**
- Every decimal number comes verbatim from `NUMBERS.md`, which is a copy of the consolidated results; tickets #12–#30 are the source of truth.
- External numbers (IDF, den Braber, Sun, Rein, Nutrition5k, Ramesh, PPG-Bench, HbA1c cohort, GlucoFM-Bench) come from `CITATIONS.md`, which records the verified source sentence.
- No rounding beyond what `NUMBERS.md` shows. No new analyses.

**Citations:**
- Pandoc `[@key]` / `@key` syntax, using keys from the verified `references.bib` only.
- A claim without a verified key is rewritten without the claim, not cited to a placeholder.

**Section files** (concatenated in name order; each has one owner):

| File | Heading(s) | Content | Words |
|---|---|---|---|
| `00_abstract.md` | `# Abstrak` (unnumbered `{-}`) | problem, method, three findings with headline numbers, honesty statement | 220–260 |
| `01_introduction.md` | `# 1. Introduction` | draft text verbatim with the fixes below + a contributions paragraph | ≈ 900 |
| `02_related_work.md` | `# 2. Related Work`, `## 2.1`–`2.4` | personalised post-prandial response prediction; glucose forecasting and its CGM-history dependence (TSFMs, GlucoFM-Bench); food recognition and VLMs (Nutrition5k, Indonesian food datasets); PPG glucose claims and reassessments (Ramesh 2026, PPG-Bench, large-cohort HbA1c study, PaPaGei) | ≈ 900 |
| `03a_methodology.md` | `# 3. Methodology`, `## 3.1`–`3.4` | system overview + Figure 1; datasets + Table 1; CGMacros cleaning; evaluation protocol (grouped nested CV, B0/B1/B2, ADR 0004 gate, permutation, Clarke error grid) | ≈ 1000 |
| `03b_methodology.md` | `## 3.5`–`3.8` | forecaster (features, grid, ISO 15197 meter noise, PPG g0 simulation); dish identification (prompt, matcher, dish table, operating rule chosen on dev); PPG estimation (7 features, ridge, PaPaGei-S, within-person design); system implementation and missing-input handling (RQ2) | ≈ 1000 |
| `04a_results.md` | `# 4. Results`, `## 4.1`–`4.4` | forecaster vs baselines (Table 2, Figure 2); g0 source (meter vs PPG simulated); contribution of meal information (permutation null, selection frequency, carb-error sensitivity); ShanghaiT2DM (Table 3) | ≈ 900 |
| `04b_results.md` | `## 4.5`–`4.7` | dish identification (Table 4, Figure 3); PPG on both datasets + within-person + heart rate (Table 5, Figure 4); negative results and reliability (pre-specified design, excursion/monotone, error-band coverage) | ≈ 900 |
| `05_discussion.md` | `# 5. Discussion`, `## 5.1`–`5.3` | answers to RQ1–RQ3; limitations; ethics/safety (not a medical device; PPG shown as a range with a meter recommendation; licence of TKPI) | ≈ 800 |
| `06_conclusion.md` | `# 6. Conclusion` | one paragraph + future work (paired photo+PPG+glucose collection, conditional error bands, re-calibrating the rule on the full table) | ≈ 200 |

The References section is generated by citeproc under the heading "References".

**Introduction fixes** (the only edits to the user's text):
- TKPI cited as the 2017 edition.
- The PPG paragraph stops implying that PPG improves the forecast. It says studies report PPG glucose estimation, and that this paper tests that claim.
- The phrase "dan teks makanan" stays: manual entry goes through the same matcher.
- The three research questions are kept verbatim.
- A closing contributions paragraph is added.

**Figures** (PNG for the docx, PDF kept; Indonesian labels; produced by `figures/make_figures.py`):
- `fig_pipeline` = Figure 1
- `fig_forecast` = Figure 2
- `fig_dish_operating` = Figure 3
- `fig_ppg` = Figure 4

A figure is referenced as `![Gambar N. caption](figures/<name>.png){width=100%}`.

**Tables:** pandoc pipe tables. The caption goes on the line before the table, as "Tabel N. …", in bold. Numbering:

| Table | Content | File |
|---|---|---|
| 1 | Datasets | 03a |
| 2 | Forecaster | 04a |
| 3 | Shanghai | 04a |
| 4 | Dish ID | 04b |
| 5 | PPG | 04b |
| 6 | Hyperparameters (optional, only if it fits) | 03b |

**Terminology** (use consistently):
- *prakiraan* (forecast)
- *glukosa awal g0* (current glucose)
- *baseline* B0 (population mean), B1 (persistence), B2 (persistence + mean excursion)
- *identifikasi hidangan* (dish identification)
- *tabel hidangan* (dish table)
- *aturan abstain* (the similarity + margin rule)
- *partisipan yang tidak dilatih* / held-out participants
- *zona A Clarke*

## Testing Decisions

There is one seam: the build. It runs two checks, and a section is accepted only when both pass with that section included.
- **`check_numbers.py`** fails if any decimal number in a section is absent from `NUMBERS.md`/`CITATIONS.md`, or if a section contains `PLACEHOLDER`, `[CITATION NEEDED]` or `TODO`.
- **`build.sh`** fails on any pandoc/citeproc warning (an unknown citation key, a missing figure) and writes `GlucoSight_paper.docx`.

After the build passes, the integrator does three things:
- reads the full docx text once for continuity, duplicated content, and section-to-section contradictions;
- checks that each research question is answered;
- checks that every negative result in `NUMBERS.md` appears.

Prior art is the repo's other "honest gate" checks: ADR 0004 and the `check_*` scripts in `forecasting/evaluate`. Like them, the check tests the observable output, not how the text was written.

## Out of Scope

- New experiments or re-analysis, including:
  - conformal/conditional error bands;
  - pooled CGMacros+Shanghai training;
  - re-calibrating the abstain rule on the 710-row table.

  These appear as limitations or future work only.
- The NeurIPS LaTeX version (dropped by the user on 2026-11-22).
- Translating the paper to English.
- Choosing a venue.
- Any TKPI-derived per-dish nutrient values in the paper or repo. Counts such as "710 rows" are fine.
- Student ID numbers (NIM) in the byline. The draft has one incomplete NIM; the authors add these themselves.

## Further Notes

- **Sources:**
  - `NUMBERS.md`: results.
  - `CITATIONS.md` + `references.bib`: verified literature.
  - `docs/SPEC_V2_PIPELINE.md`: system behaviour.
  - `cv/experiments/dish_id_trial/vlm_run.py` on `wayfinder/dish-id-trial` (`PROMPT`): the VLM prompt.
  - `forecasting/evaluate/gbdt_forecaster.py`: grid and meter-noise model.
  - ADR 0004 / 0005: gate and the PPG serving decision.
- **Grid:**
  - offset ∈ {no, yes}, num_leaves ∈ {4, 8}, min_child_samples ∈ {20, 60}, n_estimators ∈ {100, 300};
  - learning rate 0.05;
  - feature sets g0_hour / g0_hour_macros / g0_hour_profile / full;
  - 3 meter-noise training draws per meal, 5 evaluation draws.
- **VLM prompt (English, asks for Indonesian names):** "List every distinct food dish visible in this image, as specifically as you can (e.g. 'rendang', 'nasi goreng', not just 'rice dish'). Use the dish's common Indonesian name if you recognise it, otherwise whatever name best identifies it. Return JSON only: {"items": [{"food_name": "...", "confidence": 0.0}]}. One entry per distinct dish — don't split one dish into its ingredients." Temperature 0.
- **No end-to-end accuracy number exists, and the paper must say why (ADR 0004):** there is no dataset with photo + PPG + glucose for the same people.
