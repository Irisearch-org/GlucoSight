# NLP Literature Review — dish identity, not translation

**Date:** 2026-10-18
**Scope:** five questions raised by `docs/NLP_PATH_A_GATE.md`,
`docs/NLP_DISH_IDENTITY_TEST.md`, `docs/NLP_INDONESIAN_BRIDGE.md`, and ADR
`docs/adr/0001-dish-density-bridge.md`. Every source below is a primary
source — the paper itself, its official repo, or the publisher's own page —
read directly, not a blog's summary of it. Where a claim could not be traced
to a primary source, that is stated instead of a citation.

**Carried-forward facts** (established in this repo, not re-derived here):
dish identity improves T+60 rise MAE by **0.648 mg/dL** over shipped meal
features on ShanghaiT2DM, surviving resplits and a permutation control
(DI1–DI2); shipped features (`total_grams`, two flags) add **no reliable
gain** over `g0` alone (DI3); the pipeline collapses **2,762** meal texts to
**661** feature states and **1,825** dish names to **one** carb density
(N2); `nlp_confidence` takes **4** values, not a real confidence (N3); the
Indonesian lookup is **10 rows** (`cv/data/tkpi_macro_lookup.csv`).

---

## Q1 — Is IndoBERT the right tool for dish-identity matching?

**Primary source — IndoNLU / IndoBERT itself.**

> Wilie, B., Vincentio, K., Winata, G. I., Cahyawijaya, S., Li, X., Lim,
> Z. Y., Soleman, S., Mahendra, R., Fung, P., Bahar, S., & Purwarianti, A.
> (2020). *IndoNLU: Benchmark and Resources for Evaluating Indonesian
> Natural Language Understanding.* Proceedings of AACL-IJCNLP 2020,
> 843–857. `arXiv:2009.05387`. https://arxiv.org/abs/2009.05387 ·
> https://aclanthology.org/2020.aacl-main.85.pdf · code/models:
> https://github.com/indobenchmark/indonlu

**What it actually benchmarks.** IndoNLU's 12 tasks (read from the paper's
own task list) are: emotion classification, a QA-factoid dataset, sentiment
analysis, aspect-based sentiment analysis, POS tagging, NER, span
extraction, and textual entailment — single-sentence classification and
sequence-pair/sequence-labeling tasks, each over a **fixed, exhaustively
labeled task schema**. None of the 12 is "match free text to an entry in an
external reference table whose rows have few or no training examples,
and that table grows over time." IndoBERT (trained on the 4B-word Indo4B
corpus) is the shared encoder underneath those 12 tasks, not evidence that
fine-tuning a classification head is the right shape for *this* track's
sub-problem.

**What this project's sub-problem actually is.** The step in question —
`(dish_name) → carbohydrate density row in a lookup table` — has the shape
of **entity linking** (a.k.a. entity disambiguation/normalisation/grounding):
map a text mention to an entry in a knowledge base, where the KB is small,
has sparse or zero labelled examples per entry, and is expected to grow.
Primary sources on this shape:

> Wu, L., Petroni, F., Josifoski, M., Riedel, S., & Zettlemoyer, L. (2020).
> *Scalable Zero-shot Entity Linking with Dense Entity Retrieval.*
> Proceedings of EMNLP 2020. `arXiv:1911.03814`.
> https://arxiv.org/abs/1911.03814 · https://aclanthology.org/2020.emnlp-main.519/
> · code: https://github.com/facebookresearch/BLINK

BLINK's own numbers: a bi-encoder (two independent BERT towers, mention and
entity text, scored by dot product) retrieves candidates over **5.9 million**
Wikipedia entities in **2 ms**; a cross-encoder reranks the top candidates
and reaches **94.5%** accuracy on TAC-KBP-2010, beating prior
feature-engineered systems. The key property for this project: the entity
side needs only a short text description per row, **not per-row training
examples** — the opposite of a fixed softmax classifier, which needs
labelled examples for every class it must ever predict.

**Food-domain entity linking against a small structured table — the closest
prior art to this exact problem.**

> Feher, D., Ibrahim, F., Cheng, Z., Schlegel, V., Maidment, T., Bagshaw, J.,
> & Batista-Navarro, R. (2023). *Few-shot entity linking of food names.*
> Information Processing & Management, 60(5), 103463.
> https://doi.org/10.1016/j.ipm.2023.103463

Builds the **E.Care KB**, 664 food concepts (1,130 synonyms, 212 of the 664
concepts having only ≈3.7 linked examples on average — a sparsity profile
close to TKPI's), and a 468-recipe dataset of ingredient mentions linked to
it. Compares string-similarity, SVM, and Siamese-network (sentence-
transformer embedding) linkers. **Best hybrid: 89.40% accuracy**, 0.24 s per
mention; the deep-learning sentence-transformer Siamese model alone:
**86.99%**, fastest at 0.004 s/mention. Framed and solved explicitly as
entity linking, not classification, because of the same sparsity problem
GlucoSight's 10-row (or future 100+-row) table has.

> Lamarine, M., Hager, J., Saris, W. H. M., Astrup, A., & Valsesia, A.
> (2018). *Fast and Accurate Approaches for Large-Scale, Automated Mapping
> of Food Diaries on Food Composition Tables.* Frontiers in Nutrition, 5:38.
> https://doi.org/10.3389/fnut.2018.00038

~170,000 food-diary records (4,200 unique food items) mapped onto six Food
Composition Tables by **fuzzy string matching alone** — no training data at
all. At a relaxed threshold: **99.49%** of items remapped at **88.75%**
precision; at a stricter threshold: **96.81%** precision with **>95%**
recall. A machine-learning (C5.0) layer added **no improvement** over fuzzy
matching for items with a clear match, though it helped recall by
**+7 to +20 points** on items with no close equivalent in the target table —
directly the long-tail problem N2/DI4 describe for the 1,825 Shanghai dish
names and would describe for a scaled TKPI.

> Lemay, D. G., Strohmeier, M. P., Stoker, R. B., Larke, J. A., & Wilson,
> S. M. G. (2025/2026). *Evaluation of Large Language Models for Mapping
> Dietary Data to Food Databases.* The Journal of Nutrition.
> https://doi.org/10.1016/j.tjnut.2025.09.002 (ahead-of-print at
> https://jn.nutrition.org/article/S0022-3166(26)00327-5/fulltext) · ground
> truth + code: https://github.com/dglemay/USDA-Food-Mapping

Directly compares fuzzy matching, TF-IDF, semantic-embedding retrieval, and
LLM prompting for mapping free-text dietary records to a food-composition
database. **Semantic-embedding retrieval alone beat fuzzy matching and
TF-IDF** on both benchmarks (87.8% accuracy on a 9,910-entry target DB;
48.0% on a harder 256-entry DB with 46.9% "no match" cases). A **hybrid** —
embedding retrieval narrows to top-K, then an LLM chooses among them or
returns "no match" — was best overall (90.7% / 65.4%). Full classification
over the target vocabulary was not the winning design in either benchmark.

**Answer to Q1.** No primary source among IndoNLU's 12 benchmarked tasks
evaluates "classify free text into one of a small set of structured-table
rows with few training examples per row." Every primary source that *does*
address this exact shape — in the food domain specifically — frames it as
retrieval/entity linking (embedding similarity or fuzzy string matching
against row text), not fixed-head classification, and reports it winning
or tying classification/LLM baselines at a fraction of the training-data
cost. This repo's own ADR (`0001-dish-density-bridge.md`, IB6) already
reached the compatible conclusion that IndoBERT belongs at the **parsing**
step (`free text → (dish, grams, method)` slots); this review's finding
is that the **next** step (`dish → density`) is a retrieval problem, not a
second classification problem, and the literature above is the evidence
for that, not just this project's own reasoning.

---

## Q2 — Published work mapping free-text meal descriptions to carb/glycemic-load estimates

**Primary source, most directly on-point.**

> Dhaliwal, M., Hua, A., Pullela, L., Burke, R., & Qin, Y. (2024/2025).
> *NutriBench: A Dataset for Evaluating Large Language Models in Nutrition
> Estimation from Meal Descriptions.* `arXiv:2407.12843`; NeurIPS/ICLR 2025
> Datasets & Benchmarks. https://arxiv.org/abs/2407.12843 ·
> https://doi.org/10.48550/arxiv.2407.12843 · dataset:
> https://huggingface.co/datasets/dongx1997/NutriBench · site:
> https://mehak126.github.io/nutribench.html

11,857 human-verified natural-language meal descriptions (5,000 in the
first arXiv version) generated from real-world dietary-intake data (USDA
"What We Eat in America" and FAO/WHO GIFT), annotated with carbohydrate,
protein, fat, and calorie labels across several countries. **No Indonesian
or Bahasa-Indonesia subset was identified** in what was read — this is
noted as a gap, not filled by inference.

Twelve LLMs evaluated on carbohydrate estimation from the description text,
with four prompting strategies including **Retrieval-Augmented Generation**
(nearest-neighbour semantic search against a nutrition DB, then generation
grounded in the retrieved facts — the same retrieval architecture as Q4).
**Best result: GPT-4o with Chain-of-Thought, 66.82% accuracy at an
Acc@7.5 g threshold** (tied to the standard 1-unit-insulin-per-15 g
carbohydrate ratio), 99.16% answer rate. A study with three professional
nutritionists on the same 90 queries found the LLMs **matched or beat**
the nutritionists on accuracy and were far faster. The paper goes one step
further than a feature-level metric: predicted carbohydrates were fed into
a **glucose simulator** for Type 1 diabetics, and outcomes measured in
Time-in-Range / Time-Below-Range — directly the "propagate the signal to a
clinical outcome metric" pattern this review's Q5 recommends.

**Secondary but relevant — the field's actual gold-standard baseline and
its reported correlation.**

> Zeevi, D., Korem, T., Zmora, N., Israeli, D., Rothschild, D., et al.,
> Elinav, E., & Segal, E. (2015). *Personalized Nutrition by Prediction of
> Glycemic Responses.* Cell, 163(5), 1079–1094.
> https://doi.org/10.1016/j.cell.2015.11.001

800-person cohort, 46,898 real-life meals, continuous glucose monitoring.
The paper's own **baseline** — "carbohydrate counting," the standard of
care — achieves **R = 0.38** (p < 10⁻¹⁰) against measured postprandial
glycemic response (iAUC). A full model integrating meal content, personal,
and microbiome features reaches **R = 0.68** on held-out people and
**R = 0.70** on an independent 100-person validation cohort — approaching
but not reaching the **R = 0.71–0.77** test-retest ceiling (how well the
*same* meal predicts its own repeat in the *same* person). This is a
glycemic-*response* prediction paper, not a text-to-carb mapping paper, but
it is the primary source for the claim (used in `nlp/docs/DOMAIN_KNOWLEDGE.md`
§1.5 and §3.1 via its PREDICT-1 successor) that carbohydrate content alone
is a modest predictor of glucose response, and it sets the field's
precedent for reporting a correlation against a baseline and a
measurement-noise ceiling, not a bare number.

**Answer to Q2.** The closest primary source is NutriBench — LLM-based
carbohydrate estimation directly from free-text meal descriptions, with a
reported accuracy (66.82% at a clinically-motivated ±7.5 g threshold for
the best model) and an explicit downstream glucose-risk evaluation. No
primary source specific to **diabetes self-management in Bahasa Indonesia**
was found; this gap is real, not an oversight, and is consistent with
`docs/NLP_INDONESIAN_BRIDGE.md` IB5's statement that no Indonesian
glucose-linked dataset is known to be public.

---

## Q3 — Scaling the dish-density lookup table: the official TKPI source

**Primary source — the book itself, as catalogued by its publisher.**

> Kementerian Kesehatan Republik Indonesia [Ministry of Health, Republic of
> Indonesia]. *Tabel Komposisi Pangan Indonesia* [Indonesian Food
> Composition Table]. Jakarta: Kementerian Kesehatan RI, 2020 revised
> printing, 140 pp. Editors: Dr. Mien K. Mahmud, MS; Dr. Hermana, M.Sc.
> ISBN 978-623-301-036-8. Catalogue record:
> https://repository.kemkes.go.id/book/668

This is the Ministry of Health's own repository entry for the TKPI book —
author, publisher, year, and ISBN all confirmed from the catalogue page
itself (PDF not fetched; the catalogue metadata is the primary record read
here).

**Primary source — the online, queryable version, and its lineage.**

> *Data Komposisi Pangan Indonesia* [Indonesian Food Composition Data].
> https://www.panganku.org/ (English: https://www.panganku.org/en-EN/).
> Supported by the Global Alliance for Improved Nutrition (GAIN).

The site's own "Beranda"/home page states its lineage directly: the data
is "the development of the Tabel Komposisi Pangan Indonesia (TKPI) tahun
2009," expanded via imputed and borrowed nutrient values and new entries
from the Food Security Agency / Ministry of Agriculture. The 2009 TKPI
itself traces (per the same page) to the Centre for Food and Nutrition
Research and Development at the Ministry of Health, compiling prior works
including Mahmud, Slamet, Apriyantono & Hermana (1990) and Karmini et al.
(2001), both titled *Komposisi Zat Gizi Pangan Indonesia*, Departemen
Kesehatan RI. A sample query (`cari_nutrisi` / `semua_nutrisi` endpoints)
returns rows with a food code, Indonesian name (with English gloss where
available), food group, and raw/processed type — the same shape as
`tkpi_macro_lookup.csv`'s columns, confirming this is a drop-in source to
scale from 10 rows toward the hundreds-to-thousands TKPI actually contains.

**What was not independently verified.** The task brief named PERSAGI
(Persatuan Ahli Gizi Indonesia) as a possible co-source. No primary PERSAGI
publication distinct from the Kemenkes book was found in this review — it
is commonly cited as professionally associated with Indonesian nutrition
standards, but that association is not confirmed here from a primary
PERSAGI source, and this review does not assert one.

**Answer to Q3.** The lineage is: Kemenkes RI (Ministry of Health) is the
publisher of record for TKPI, most recently the 2020 printing; GAIN's
`panganku.org` is the practically scrapeable/queryable online form of the
2017 expansion of TKPI 2009, with per-food rows in the same shape the
10-row lookup already uses. Scaling the lookup is a data-entry and
licensing question against these two sources, not a new data-collection
effort — consistent with `docs/NLP_INDONESIAN_BRIDGE.md` IB7's conclusion
that "the entries are the hard part, not the lexicon."

---

## Q4 — Is retrieval/entity-linking better suited than discrete buckets?

This question is answered jointly by the entity-linking sources already
cited under Q1 (BLINK; Feher et al. 2023; Lamarine et al. 2018; Lemay et
al. 2025), which is why it is not repeated at length here. The specific
point this question asks — open-vocabulary mention → nearest entry in a
**small** structured table — is the exact subject of all four:

| source | table size | method that won | result |
|---|---|---|---|
| Lamarine et al. 2018 | 6 FCTs, 4,200 unique items/diary side | fuzzy string matching, **no training** | 96.81% precision / >95% recall at strict threshold |
| Feher et al. 2023 | 664 food concepts, ≤4 examples/concept avg | sentence-transformer Siamese embedding retrieval + hybrid rerank | 89.40% accuracy, 0.24 s/mention |
| Lemay et al. 2025 | 256–9,910 entries | semantic embedding retrieval (+ LLM rerank for the hybrid) | 87.8–90.7% (large DB), 48.0–65.4% (small, ambiguous DB) |
| Wu et al. 2020 (BLINK) | 5.9M Wikipedia entities (general domain, not food) | bi-encoder dense retrieval + cross-encoder rerank | 94.5% accuracy, 2 ms/query retrieval |

The pattern across all four, independent of domain: **retrieval
(string-similarity or embedding) against the table's own text wins or ties
classification/generation baselines, needs no or few labelled examples per
table row, and degrades gracefully on the long tail** by returning the
nearest neighbour or "no match" rather than being forced into one of a
fixed set of output classes. A fixed classifier's output layer is sized to
the table at training time; growing TKPI from 10 to 200 rows means
retraining that layer, where a retrieval index just gets new rows embedded.

**Answer to Q4.** Yes — the literature supports retrieval/entity-linking
over discrete-bucket classification for this exact problem shape, and the
food-domain papers (Lamarine, Feher, Lemay) are closer prior art than the
general-domain one (BLINK), because they already contend with the same
small-table, sparse-labels, growing-vocabulary conditions TKPI has.

---

## Q5 — What should a "working NLP contribution" be measured against?

**Primary sources for the evaluation standard.**

> Clarke, W. L., Cox, D., Gonder-Frederick, L. A., Carter, W., & Pohl, S. L.
> (1987). *Evaluating Clinical Accuracy of Systems for Self-Monitoring of
> Blood Glucose.* Diabetes Care, 10(5), 622–628.
> https://doi.org/10.2337/diacare.10.5.622

The original Error Grid Analysis paper. Zone A is defined as within ±20%
of the reference value (for references ≥70 mg/dL; both <70 mg/dL also
counts). This is the metric already named in `CLAUDE.md` and implemented in
`utils/clarke_grid.py` — cited here as its primary source, not a new
recommendation.

> Zeevi et al. (2015), cited in full under Q2.

The relevant methodological precedent from this paper is not its topic
(microbiome-personalized nutrition) but its **evaluation discipline**:
every reported number is relative to (a) a named baseline — carbohydrate
counting, the actual standard of care, R = 0.38 — and (b) a
measurement-noise ceiling — same-meal test-retest correlation, R = 0.71–
0.77 — and the model's R = 0.68–0.70 is reported sitting between them, not
in isolation. This is the same discipline `CLAUDE.md` rule C3 already
states for this project ("No fusion result is meaningful without B0/B1/B2
in the same table") — Zeevi et al. is a primary-literature instance of the
same rule being load-bearing in a closely related prediction problem.

> NutriBench (Dhaliwal/Hua et al.), cited in full under Q2.

The relevant methodological precedent here is going **past** the
feature-level metric: having estimated carbohydrates from text, the paper
does not stop at an MAE/accuracy number — it feeds the estimate into a
glucose simulator and reports the clinically meaningful outcome (time in
range). The isolated metric (Acc@7.5g) and the downstream outcome
(simulated glucose risk) are both reported, and the paper's conclusion
rests on the second one.

**Answer to Q5, applied to this project.** NLP has no glucose model of its
own (`docs/METRICS.md` §8, "NLP has a glucose model" — it does not), and
DI1's 0.648 mg/dL is an isolated-feature result on a ridge regression, not
a fusion result. Grounded in Clarke (the project's own primary metric),
Zeevi (baseline-and-ceiling discipline matching this project's own C3), and
NutriBench (propagate to the downstream outcome, don't stop at the
feature-level number), the working evaluation target for NLP's
contribution should be:

**Wire the dish-identity feature into the Forecasting track's actual
fusion model, and report whether Clarke Zone A moves — against B0/B1/B2 in
the same table (C3), on subject-grouped held-out folds (C4) — comparing
fusion-with-dish-identity to fusion-without-dish-identity on the same
splits.** A feature-level MAE delta (DI1) is evidence the signal exists; it
is not evidence the fusion layer can use it, and per the project's own
rule, no Zone A number from this comparison is meaningful without the
trivial baselines reported alongside it in the same table.

---

## Recommendation

1. **IndoBERT classification is not the right tool for dish-identity
   matching, and this review's literature is additional evidence for a
   conclusion this repo's own ADR already reached by a different route.**
   `docs/adr/0001-dish-density-bridge.md` (IB6) scoped IndoBERT to
   **parsing** (free text → `(dish, grams, method)` slots) on reasoning
   internal to this project (no dish supervision exists to fine-tune
   toward). This review adds the external literature: IndoNLU's own 12
   benchmarked tasks (Q1) do not include "match text to a sparse,
   small, growing reference table," and every primary source that
   studies that exact shape — in the food domain specifically (Lamarine
   2018, Feher 2023, Lemay 2025) and in general-domain entity linking
   (Wu 2020/BLINK) — treats it as **retrieval/entity linking**, not
   classification, and reports retrieval winning or tying classification
   and LLM baselines at a fraction of the labelled-data cost. **The
   dish → density step should be built as embedding or fuzzy-string
   retrieval against TKPI's own row text (dish name, aliases), not as a
   second fine-tuned classifier.**

2. **Scaling TKPI is a data-entry problem against two named primary
   sources, not a new data-collection project.** Kemenkes RI's own TKPI
   book (2020 printing) and GAIN's `panganku.org` (the queryable 2017
   expansion of TKPI 2009) are both primary, both citable, and both in
   a row shape compatible with `cv/data/tkpi_macro_lookup.csv`. PERSAGI's
   specific role was not independently confirmed and should not be
   asserted without a primary PERSAGI source.

3. **No Bahasa-Indonesia-specific or Indonesian-population primary source
   exists for free-text-to-carbohydrate mapping.** NutriBench is the
   closest primary source for the general problem (free text → carb
   estimate, with a clinically grounded accuracy threshold and a
   downstream glucose-risk evaluation) but is not Indonesian. This gap
   is consistent with, and does not contradict, `docs/NLP_INDONESIAN_BRIDGE.md`
   IB5's statement that no public Indonesian glucose-linked dataset is
   known.

4. **The project's open question "does dish identity move Zone A" has a
   literature-grounded answer for how to measure it, not what the answer
   is.** Per C3/C4 and the Zeevi/NutriBench precedent above: evaluate
   dish-identity-in-fusion against the same B0/B1/B2 baselines, on the
   same subject-grouped folds, reporting Zone A with and without the
   dish-identity feature — not a second isolated-feature MAE number. This
   is the next experiment this review recommends, and it is currently
   unrun: `docs/NLP_DISH_IDENTITY_TEST.md` DI1 is a standalone ridge-
   regression result, not a fusion-layer result.

5. **Nothing above licenses skipping C3 or C4.** Every comparative number
   proposed in this review — entity-linking accuracy, Zone-A-with-vs-
   without dish identity — must be reported with its trivial baseline and
   on grouped/subject-level splits, per `CLAUDE.md`'s four rules.
