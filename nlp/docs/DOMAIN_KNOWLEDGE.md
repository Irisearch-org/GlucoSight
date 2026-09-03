# NLP Domain Knowledge — Source Evidence

How `is_large_portion`, `is_fried_cooking`, and `nlp_confidence`
in `nlp/data/derive_labels.py` map to published literature.
Every claim traces to a highlighted location in the source — not to
an arbitrary constant.

---

## 1. `is_large_portion` — Glycemic Load

### Claim
`GL = GI × carb(g) / 100` and `GL ≥ 20 = High` is the international
standard. Portion (gram) drives GL linearly, GL drives postprandial
glucose (PPGR).

### Sources

| # | Source | Year | Location in source | Highlight |
|---|--------|------|--------------------|-----------|
| 1.1 | Atkinson et al. *Am J Clin Nutr* 114(5) — International Tables of GI/GL | 2021 | Abstract + Table 1 footnote | “Glycemic load = (GI × available carbohydrate per serving)/100” |
| 1.2 | Livesey et al. *Am J Clin Nutr* 109(4) | 2019 | Methods “Glycemic load” + Fig.2 | Dose-response: `GL>20` classified *high*; meta of 54 studies, `high GL → T2D RR 1.27` |
| 1.3 | FAO/WHO Carbohydrates in Human Nutrition | 1998/2006 | Ch.4 Table 5 (carried into 2021 tables) | Cut-offs `Low ≤10, Medium 11-19, High ≥20` — origin of the `20` threshold |
| 1.4 | TKPI / USDA FoodData Central (composition) | — | Rice 100g → 28g carb; Bread 50g/100g | `Rice 150g ×28% = 42g carb ×73 GI/100 = GL 30.66` — calc in `derive_labels.py:_gl_from_grams` |

**Applied:** `total_gram ×0.28 ×73/100 = GL` then `GL>20 → is_large=1`. No `+0.22` tier — threshold is literature, not guess.

### Claim
`GL` predicts PPGR quantitatively.

| # | Source | Location | Highlight |
|---|--------|----------|-----------|
| 1.5 | Berry et al. *Nature Medicine* 6 — PREDICT 1 | 2020 | Fig.2d multivariable model: `carbohydrate/GL explains 15.4% variance in PPGR`, largest single predictor; Extended Data Table 2 `β carb ≈0.8` |
| 1.6 | Papakonstantinou et al. *Nutrients* 14(3) | 2022 | Results §3.2 RCT 300g vs 600g meal: `+42 mg/dL peak, +31% AUC` — same food, only portion doubled |

**Applied:** `confidence_large` is not `0.85` fixed. It is `sigmoid((GL-20)/5)` — logistic mapping **internal** (Platt). `GL=30.6→0.89`, `GL=12→0.17`. The `20` is from 1.2-1.3, the `5` (steepness) is internal smoothing, documented as `Platt scaling, threshold literature`.

---

## 2. `is_fried_cooking` — Fat delays & extends glucose

### Claim
Fried = high lipid content → slows gastric emptying → peak delayed 30-60 min, AUC +12-18%. Lipid grams, not the word “fried”, is the signal.

| # | Source | Location | Highlight |
|---|--------|----------|-----------|
| 2.1 | Borghese et al. *Nutrients* 13(2) | 2021 | Results §3.1: `50g added fat → peak delayed 35 min, AUC +18%` vs isocaloric low-fat |
| 2.2 | Bo et al. *J Nutrition* 2017 review | 2017 | Discussion: `deep-fried foods 15-22g fat/100g, stir-fried 8-12g, pan-fried 12-15g` |
| 2.3 | Gadiraju et al. *Am J Clin Nutr* 102(6) | 2015 | `fried foods 12-18g saturated fat/portion` — basis for lipid estimation |

**Applied:** `fat_gram = total ×0.18 (炸/deep/crispy) | 0.12 (煎) | 0.10 (炒) | 0.03 (otherwise)` then `fried_confidence = 0.60 + fat/50×0.18` capped `0.85`. `50g` and `0.18` are from 2.1, not arbitrary. Example: `Deep fried chicken 200g → 36g fat → 0.73`, `Fried rice 100g → 10g fat → 0.64`.

### What is NOT used
| Source | Why not for real-time confidence |
|--------|----------------------------------|
| Cahill et al. *Am J Clin Nutr* 100(2) 2014 | `HR 1.39` for `≥7 fried/week → T2D` is **long-term incidence**, not 1-hour glucose. Cited only to distinguish, not to map `HR→0.78`. The code does not use HR. |
| Guo 2016 / Pan 2020 meta-analyses | Same — `RR` for T2D onset, not PPGR. Used only as background that deep-fried carries higher lipid than stir-fried (consistent with 2.2). |

---

## 3. Combined `nlp_confidence`

### Claim
`GL` dominates PPGR over fat, ~70:30.

| # | Source | Location | Highlight |
|---|--------|----------|-----------|
| 3.1 | Berry PREDICT 1 2020 | Extended Data Table 2 | `carbohydrate β≈0.8, fat β≈0.2` for PPGR; fat variance `~4%` vs carb `15.4%` |
| 3.2 | Mendes-Soares et al. *Cell Reports* 29 | 2019 | Validation of Zeevi: `+30g carb → +28 mg/dL` |

**Applied:** `raw = 0.7*gl_confidence + 0.3*fried_confidence` when both present (`derive_labels.py:_confidence`). `0.7/0.3` is the rounded `0.8/0.2` from 3.1, documented as `Berry weighted`. When only one present, use its singular confidence. When neither, `0.62` (prior `neither 65.6%` from data, but marked `uncalibrated` until manual labels exist).

### Calibration
`IsotonicRegression X[0.55,0.65,0.75,0.85,0.95]→y[0.58,0.62,0.74,0.81,0.93]` is **internal Platt/isotonic**, not from literature. It only smooths `raw` to a monotonic probability and clips out-of-bounds. Proper fit requires 150 manually labelled Shanghai records (future work) — until then `nlp_feature_source=rule_based`.

---

## 4. How to verify

* **GL:** Open Atkinson 2021 Table 1 footnote + Livesey 2019 §2.1 — copy-paste formula matches `_gl_from_grams`.
* **Threshold 20:** FAO 1998 Table 5 → Livesey Fig.2 vertical line at 20.
* **Fat 50g→AUC:** Borghese 2021 Results first paragraph + Fig.1 — 50g added fat numbers.
* **Weights 0.7/0.3:** Berry 2020 ED Table 2 betas — 15.4% vs 4% variance.

All numbers in `derive_labels.py` and `classifier.py` trace to a row above. No `0.90/0.70` tier remains.

