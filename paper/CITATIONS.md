# CITATIONS: verification log for `references.bib`

49 entries, 0 placeholders. Every BibTeX record was fetched programmatically (doi.org content negotiation `Accept: application/x-bibtex`, the arXiv DOI `10.48550/arXiv.<id>` through the same route, or the publisher's own BibTeX block). Nothing was typed from memory, and keys were then renamed to `firstauthorYEARword`.
Validation: `pandoc --citeproc --bibliography paper/references.bib` on a file with `nocite: "@*"` in YAML front matter and a `::: {#refs}` div renders all 49 entries (Kemenkes 2020 prints as `———` under Kemenkes 2017), with no warnings and UTF-8 diacritics intact (Brügger, Giosuè, Küttner, Röddiger, Maršánová). The `-M nocite='@*'` form from the brief does not work, because `-M` stores a literal string and does not parse `@*` as a citation. Use the YAML form.

Source abbreviations: **CR** CrossRef API, **EPMC** Europe PMC (abstract), **arXiv** arXiv API, **S2** Semantic Scholar, **DOI-bib** doi.org BibTeX, **page** official page fetched directly, **full text** PDF text read.

## Discrepancies with `local://paper-numbers.md` (fix in the paper)

| Item | paper-numbers says | Source says |
|---|---|---|
| Indonesia diabetes prevalence | 19.5M | IDF Atlas 11th ed. page: **20,426.4 thousand (20.4 M) adults 20–79 in 2024**, 5th highest worldwide, **73.2% undiagnosed (14,950.1 thousand)**, age-standardised prevalence 11.3%, projected 28.6 M in 2050. 19.5 M does not appear on that page. [INFERENCE] It may be a 10th-edition (2021) figure. |
| TKPI edition | "TKPI 2017" (repo) vs "2020" (draft) | **Both exist.** The 2017 book was printed 2018 (ISBN 978-602-416-407-2, Kemenkes repository book/777). A 2020 revised edition also exists (repository book/668, 140 pp.). The repo's `cv/experiments/dish_id_trial/tkpi_parse.py` parses the **2017** PDF, so cite `kemenkes2017tkpi`. `kemenkes2020tkpi` is in the bib only if the draft insists on 2020, and it would then be a different source from the one used. |
| Wibisono classes | 35 classes | Wibisono 2020 abstract: **1,644 images, 34 types**. The In-TFK 2023 abstract also says 34 types (1,644 professional-camera and 1,020 smartphone images). The 35 may be the repo's own table (35-row table) or a Kaggle version. Say "34 classes in the source article; 35-row table here" or check. |
| CGMacros strata | 18 normal / 13–14 prediabetic / 14 T2D (by HbA1c) | The data paper says **15 healthy, 16 pre-diabetes, 14 T2D**. The paper's own HbA1c banding differs; state which one is used. |
| Chu 2021 zone A with HbA1c | 94.2% | Abstract: **94.3%** (best no-medication model with quarterly HbA1c; MAE 8.9; RMSE 12.4). The R² values (0.06 and 0.71) are not in the abstract; they come from the repo's literature review reading Table 3, so they are unverified here. |
| GlucoFM-Bench "TSFMs need 12 h CGM context" | | The paper **chooses 12 h as the default context** (ablation in Appendix Fig. 6 "balances forecasting accuracy with participant-level data utilization"). It does not say TSFMs *need* 12 h. Reword. The **TimesFM 2.5 CGMacros zero-shot RMSE 16.28 mg/dL (30-min horizon, 12-h context)** is confirmed in Table 2 (full text). |
| Ramesh "5 architectures on MUST" | | Abstract: five PPG-BGL methods on **published datasets** (VitalDB and MUST). MUST is the small OOD test set (23 subjects, 67 sessions, 2175 Hz, 88–183 mg/dL, mean 115.01, SD 18.67). Models were trained on VitalDB. Near-zero or negative R² under participant-aware and LSPO splits is confirmed. Over 90% of predictions fall in Clarke A+B, including the baseline. |
| PPG-Bench "FMs ≈ random on BUT PPG glucose" | | Abstract is aggregate only (35.3% of 68 task-regime combos show no FM advantage over random or constant). The BUT-PPG glucose result is in **Table 1 (full text)**: MAE 1.22–1.31 across five FMs against 1.23 for the random control (normalised units). Supported. |
| MUST sampling rate | 2190 Hz | Ramesh and the Mendeley description use 2175 Hz. Repo files say 2190 Hz. Report the number the data actually has. |

## Draft references (the 11 cited by the user's Indonesian draft): claim check

| Key | Claim in draft | Verdict | Evidence |
|---|---|---|---|
| `allen2007photoplethysmography` | PPG is a low-cost optical technique that detects blood volume changes in the microvascular bed | **SUPPORTED** | EPMC abstract: "PPG is a simple and low-cost optical technique that can be used to detect blood volume changes in the microvascular bed of tissue." |
| `brugger2025predicting` | Post-prandial excursion prediction relies on CGM with or without manual meal logs | **SUPPORTED** | "predict the next-in-time PPG excursion, using continuous glucose monitoring (CGM) data or CGM data combined with manually-logged meals and glucose-lowering agent intake." 67 Chinese adults with T2D, F1 mean 75.88%. |
| `denbraber2024contribution` | In T2D patients at the glycaemic target, post-prandial glucose contributes ~73% of hyperglycaemia after breakfast | **SUPPORTED (with nuance)** | "patients with target GMI values had a higher PPG (73 ± 21%) than FPG after breakfast (27 ± 21%)". The 73% is relative to GMI on target (DIALECT-2, n=59). The wording "contributes ~73% of hyperglycaemia" is the draft's gloss; the abstract gives a relative contribution, not an absolute one. |
| `giosue2026postprandial` | Post-prandial profiles to a standard meal reflect heterogeneity in insulin secretion and sensitivity in T2D | **SUPPORTED** | Title, and "CGM-derived PPGR features could identify T2D subtypes ... distinct insulin secretion and sensitivity impairments"; 100 T2D, standardised meal eaten twice, 3 clusters (HOMA2-B 77.4 vs 105.0, HOMA-IR 7.94 vs 4.84). |
| `idf2025atlasindonesia` | Indonesia ~20.4 M adults with diabetes in 2024, 5th largest worldwide, 73.2% undiagnosed | **SUPPORTED** | Page fetched 2026-11-22: "2024 20.4 million"; table 20,426.4 (thousand); "fifth highest number of adults (20-79 years) with diabetes in the world"; "Proportion of people with undiagnosed diabetes 73.2%". |
| `kemenkes2017tkpi` / `kemenkes2020tkpi` | "TKPI 2020" | **PARTLY**: both editions exist | See discrepancies. The repo uses the 2017 edition. |
| `ramadaniati2024availability` | Public facilities don't provide SMBG devices; the lowest-paid government worker needs 5–7 days' wages for a glucometer | **SUPPORTED** | "public sector facilities did not supply devices for self-testing"; "It required 5-7 days' and 4-5 day's wages to purchase a meter and a month's supply of test strips" (LPGW = lowest paid unskilled government worker; 34 public facilities, 37 pharmacies). |
| `rein2022effects` | Personalised post-prandial diet lowered PPGR vs Mediterranean and HbA1c −0.39% at 6 months; predictor uses microbiome + CGM | **SUPPORTED, with two corrections** | Average PPGR −19.8 ± 16.3 mg/dl×h vs MED (crossover, n=23). HbA1c "−0.39 ± 0.48%, p<0.001" at 6 months, but this is a **pre/post change in the n=16 extension with no control arm**. The predictor "integrates **clinical and microbiome** features"; CGM is the outcome measure, and the abstract does **not** list CGM as a predictor input (NOT FOUND IN ABSTRACT for that part). |
| `sun2023time` | FPG + PPG + HbA1c estimate TIR with r = 0.79 in T2D; TIR = % time 70–180 mg/dL | **r SUPPORTED; TIR definition NOT FOUND IN ABSTRACT** | "random forest model ... stronger correlation coefficient 0.79 (0.79, 0.80)" (linear 0.73, HbA1c alone 0.62; 399 T2D). The 70–180 range is not in the abstract. It is the standard consensus definition but unchecked here. |
| `thames2021nutrition5k` | Carb MAE from a 2D image is 6.1 g vs 2.7 g when mass is known | **SUPPORTED (full text, not abstract)** | arXiv 2103.03375 Table 3: Carb MAE "2D Direct Prediction" **6.1 g** (31.9%); "2D Portion Independent Model" **2.7 g** (13.9%). The 2.7 g model predicts per-gram composition, i.e. it takes portion mass out of the problem, which is the draft's "mass known". Baseline 12.5 g. Abstract only says it "outperforms professional nutritionists". |
| `zhang2024vision` | VLMs learn image–text correlations from web-scale image–text pairs | **SUPPORTED** | "learns rich vision-language correlation from web-scale image-text pairs ... enables zero-shot predictions on various visual recognition tasks". |

## Keys by use

### Datasets
| Key | Verified by | Cite for |
|---|---|---|
| `das2025cgmacros` | CR, EPMC, DOI-bib (Sci Data 2025, 10.1038/s41597-025-05851-7) | CGMacros data paper: 45 participants, 10 days, CGM + macros + photos |
| `gutierrezosuna2025cgmacrosdata` | DOI-bib, PhysioNet page (v1.0.0, 28 Jan 2025, CC BY-NC-SA 4.0) | Dataset DOI 10.13026/3z8q-x658 and licence |
| `zhao2023chinese` | CR, EPMC, DOI-bib, Nature page | ShanghaiT1DM and ShanghaiT2DM (100 T2D) with dietary logs |
| `zhao2023shanghaidata` | DOI-bib, figshare DOI found on the Nature "Data Records" section | Figshare collection 10.6084/m9.figshare.c.6310860 (CC BY 4.0) |
| `kermani2023dataset` | DOI-bib (Mendeley v3, CC BY 4.0), repo lit review | MUST / PPG_Dataset contact-PPG glucose data. DOI metadata lists only Ali Kermani; the repo's review also names Esmaeili H, which is unverified. Ramesh 2026 describes the same data (see discrepancy table). |
| `nemcova2021brno` | CR, EPMC, DOI-bib (BioMed Res Int 2021, 10.1155/2021/3453007) | BUT PPG data paper: smartphone PPG, 12 subjects, ECG HR reference |
| `nemcova2024butppgv2` | DOI-bib, PhysioNet page (v2.0.0, 23 Aug 2024, CC BY 4.0) | BUT PPG v2.0.0 (smartphone PPG + glycaemia) |
| `nemcova2021butppgv1` | DOI-bib, PhysioNet page (v1.0.0, 18 Jan 2021, CC BY 4.0) | BUT PPG v1.0.0 (heart rate vs ECG) |
| `wibisono2020traditional` | CR, DOI-bib, abstract (1,644 images, 34 types) | Indonesian traditional food dataset |
| `mursanto2023intfk` | CR, EPMC abstract, DOI-bib | In-TFK follow-up: adds 1,020 smartphone images (the Kaggle/test split source) |
| `kemenkes2017tkpi` | Kemenkes repository book/777 and panganku.org page fetched; 2017 PDF cover read (ISBN 978-602-416-407-2) | Dish table nutrient values |
| `kemenkes2020tkpi` | Kemenkes repository book/668 fetched | Only if the draft insists on 2020 |
| `marling2020ohiot1dm` | CEUR-WS Vol-2675 table of contents fetched (paper11, pp. 71–74, CC BY 4.0) | OhioT1DM dataset |

### Methods and standards
| Key | Verified by | Cite for |
|---|---|---|
| `ke2017lightgbm` | NeurIPS proceedings page and its BibTeX file | LightGBM |
| `clarke1987evaluating` | CR, DOI-bib | Clarke error grid |
| `iso2013iso15197` | iso.org page fetched with `read` (Edition 2, 2013-05, 46 pp., confirmed 2018, to be revised 2024-12) | Meter accuracy standard used to simulate meter g0 noise. Hand-written `@misc` from that page. |
| `team2023gemini` | arXiv, DOI-bib | Gemini technical report (original) |
| `comanici2025gemini` | arXiv, DOI-bib | Gemini 2.5 report (closest report for the Flash-Lite tier; there is no report for `gemini-3.5-flash-lite` as far as found) |
| `lee2025geminiembedding` | arXiv, DOI-bib | `gemini-embedding-001` matcher |
| `das2024decoder` | arXiv, PMLR page (ICML 2024, PMLR 235:10148–10167) | TimesFM. GlucoFM-Bench used TimesFM 2.5, which has no separate paper. |
| `ansari2024chronos` | arXiv; S2 reports venue TMLR 2024 | Chronos (`note` field carries the TMLR venue; there is no separate TMLR DOI) |

### PPG
| Key | Verified by | Cite for |
|---|---|---|
| `allen2007photoplethysmography` | CR, EPMC, DOI-bib | What PPG is |
| `ramesh2026reassessing` | arXiv abstract, CR, DOI-bib (ISWC '26, 10.1145/3830727.3834821) | Closest prior work: leakage collapse, near-zero R², and zone metrics hide failure |
| `kulkarni2026ppgbench` | arXiv abstract + full text (Table 1) | PPG foundation models near random baseline; BUT-PPG glucose in Table 1 |
| `pillai2025papagei` | arXiv, ICLR 2025 proceedings page and BibTeX | PaPaGei-S embeddings |
| `saha2025pulseppg` | arXiv, CR, DOI-bib (PACM IMWUT 9(3), 10.1145/3749494), S2 | Pulse-PPG |
| `nie2025anyppg` | arXiv (preprint only) | AnyPPG |
| `chu2021accuracy` | CR, EPMC abstract, DOI-bib | Positive claim tested: "90% accuracy". n=2,538 gives zone A 60.6%. Cohort split plus HbA1c gives 94.3%, MAE 8.9. |
| `chu2025improving` | CR, EPMC abstract, DOI-bib | Positive claim tested: monthly-calibrated dual-channel wearable PPG, MARD 9.59–16.40%, Parkes grid |
| `raju2022dnn` | CR, S2 abstract, DOI-bib (ICCIT 2022) | Positive smartphone-PPG claim: fingertip video, 93 subjects, R² = 0.96, "can be used clinically". The one smartphone-camera positive claim found. |

### Forecasting
| Key | Verified by | Cite for |
|---|---|---|
| `lu2026glucofm` | arXiv abstract + full text | TSFMs for glucose: 15 datasets, 1,117 people, best zero-shot within 5% of full-shot supervised, LSTM beats TSFMs by 4–21% with abundant data; TimesFM 2.5 CGMacros RMSE 16.28 |
| `das2024decoder`, `ansari2024chronos` | see Methods | TSFM baselines |
| `li2020glunet` | CR, EPMC abstract | GluNet: forecast from glucose, **meal information**, insulin (T1D, 30–60 min) |
| `zhu2020dilated` | CR, EPMC abstract | DRNN, OhioT1DM RMSE 18.9 at 30 min (the abstract does not state which inputs) |
| `martinsson2020blood` | CR, EPMC abstract | Glucose-history-only RNN, OhioT1DM, 1 h horizon. Cited as the contrast with meal-conditioned models. |
| `brugger2025predicting` | CR, EPMC | T2D post-prandial excursion prediction with CGM ± meal logs |
| `zeevi2015personalized`, `rein2022effects` | see Motivation | Meal-conditioned glycaemic response prediction |

### Food
| Key | Verified by | Cite for |
|---|---|---|
| `thames2021nutrition5k` | CR, S2 abstract, arXiv 2103.03375 full text | Image-to-nutrient estimation, portion limits |
| `min2019survey` | CR, EPMC/CR abstract | Food computing survey |
| `bossard2014food101` | CR, DOI-bib | Food-101 benchmark |
| `zhang2024vision` | CR, EPMC | VLMs: web-scale image–text |
| `wang2026comparative` | CR, EPMC abstract (Curr Res Food Sci) | VLMs recognise main ingredients but struggle to quantify nutrients (Nutrition5k) |
| `marchetti2026benchmarking` | CR, EPMC abstract (Nutrients) | 26 open VLM configs: best F1 0.68 (open) / 0.77 (GPT-4o); portion estimation is the limit |
| `wibisono2020traditional`, `mursanto2023intfk` | see Datasets | Indonesian dish data |

### Motivation
| Key | Verified by | Cite for |
|---|---|---|
| `idf2025atlasindonesia` | page fetched | Indonesia burden (see draft table) |
| `duncan2025idf` | CR, DOI-bib (Nephrol Dial Transplant 41(1)) | Peer-reviewed summary of the 11th-edition global estimates (secondary to the Atlas) |
| `zeevi2015personalized` | CR, EPMC abstract, DOI-bib | 800 people, 46,898 meals, high response variability, ML prediction, blinded RCT |
| `rein2022effects` | see Draft table | Personalised diet by predicted PPGR |
| `denbraber2024contribution` | see Draft table | Post-prandial share of hyperglycaemia |
| `giosue2026postprandial` | see Draft table | Heterogeneity of post-prandial profiles |
| `sun2023time` | see Draft table | FPG/PPG/HbA1c → TIR where CGM is unavailable |
| `ramadaniati2024availability` | see Draft table | Cost of SMBG in Indonesia |

## Placeholders

None. Every entry was retrieved from DOI, arXiv, or a publisher or official page that was actually fetched. Items verified from a landing page rather than a DOI record, and so worth a glance: `iso2013iso15197`, `idf2025atlasindonesia`, `kemenkes2017tkpi`, `kemenkes2020tkpi`, `marling2020ohiot1dm`, `ke2017lightgbm`, `das2024decoder`, `pillai2025papagei` (these have no DOI, or their BibTeX came from the proceedings site).

## Not found or deliberately excluded
- No original journal paper for the MUST dataset exists. The Mendeley record is the only source. Kermani and Esmaeili 2025 (10.34172/bi.30589) is a method paper that uses the data, not the dataset paper.
- Roy et al. 2022 (IJBET, smartphone PPG random glucose): the abstract was not retrievable from CR, EPMC or S2, so it was not used.
- Semantic Scholar throttled (HTTP 429) intermittently, so existence was confirmed with CrossRef, arXiv and Europe PMC (two sources each where a DOI exists). S2 was used only for the venue checks (Chronos TMLR, Pulse-PPG IMWUT) and a few abstracts.

| `sandler2018mobilenetv2` | DOI-bib (10.1109/CVPR.2018.00474), fetched by integrator | MobileNetV2 closed-classifier comparator |
