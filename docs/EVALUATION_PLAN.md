# GlucoSight v2: evaluation plan

**Status:** decided 2026-11-21 in [ticket #24](https://github.com/Irisearch-org/GlucoSight/issues/24)
(agent decision under the wayfinder map's delegated authority; the user may veto).
**Spec:** [`SPEC_V2_PIPELINE.md`](SPEC_V2_PIPELINE.md) §6. Gate: [ADR 0004](adr/0004-model-level-fusion-plan.md).
Research prototype, not a medical device.

This plan fixes the results tables and figures the paper reports. Per-stage evaluations are
settled (ADR 0004, per stage, no end-to-end number): forecaster
([#17](https://github.com/Irisearch-org/GlucoSight/issues/17)), dish identification
([#18](https://github.com/Irisearch-org/GlucoSight/issues/18)), PPG
([#16](https://github.com/Irisearch-org/GlucoSight/issues/16)), meal cleaning
([#12](https://github.com/Irisearch-org/GlucoSight/issues/12)).

## Standing rules for every table

- **n *people* is stated in each caption.**
- **95% CIs by bootstrap over people** (over images for dish identification), 2,000 resamples.
- **Every model row sits next to its baselines on identical folds** (B0/B1/B2, or the training-fold mean for PPG).
- **Clarke is reported as per-participant median/IQR**, never pooled only, plus stratified by reference range.
- **No end-to-end accuracy number** (ADR 0004). Each stage is evaluated on its own.
- **Every number is produced by a committed script with seed and data path.**
- A stage claims a gain only by the ADR 0004 gate: positive gain vs baseline on held-out people, most folds won, bootstrap CI excluding zero, permutation control. A stage that fails is still served, reported honestly, and shown with its error.

## Tables and figures

| # | Table / figure | Rows and columns | Source |
|---|---|---|---|
| T1 | Datasets and cohorts | CGMacros 45 (14 T2D), cleaning drops per rule; PPG_Dataset 23 subjects / 67 recordings; Brno 12; Wibisono dev 164 / test 317; ShanghaiT2DM 100; licences | #12, #16, #18 |
| T2 | **Forecaster main result** | B0, B1, B2, GBDT × T+60/T+120: MAE, RMSE, Clarke A (median/IQR) and A+B, gain vs B2 + CI, fold wins, permutation p. **1,000 permutations** in the final run (the trial used 200). Overall and T2D subgroup | #17 |
| F1 | Clarke grids | GBDT vs B2 per horizon, plus the reference-glucose histogram | #17 |
| T3 | Forecaster by g0 source | CGM / meter (ISO 15197 simulated) / PPG (simulated shrink map), labelled simulated | #17, #21 |
| T4 | Forecaster sensitivity | Atwater ±20/30/50%; carb perturbation from dish-ID errors; feature-set selection frequencies; the failed pre-specified design (negative result) | #12, #17 |
| T5 | **Dish identification** | dev and test: VLM + embedding (served), VLM + difflib (pre-registered), MobileNetV2 on its 10-class overlap; top-1 class and family, abstain, false match; paired bootstrap vs MobileNet | #18 |
| F2 | Manual-entry operating point | coverage vs false-match curve over similarity/margin, with the chosen 0.80/0.05 point and the old 0.75 point marked | #19 |
| T6 | **PPG glucose (negative result)** | mean vs ridge / RF / PaPaGei-S: MAE, gain, CI, perm p; Clarke stratified by range; prediction–truth correlation 0.045 | #16 |
| T7 | PPG heart rate | HR vs ECG on Brno: pooled MAE 3.76 bpm with the quality filter stated and the per-subject range 1.27–30.85 | `rppg/reports/brno_validation.md` |
| T8 | Text dish identity (separate finding) | ShanghaiT2DM, −0.555 ± 0.252 mg/dL over 5 resplits, permutation control | ADR 0001, [`NLP_DISH_IDENTITY_TEST.md`](NLP_DISH_IDENTITY_TEST.md) |

## Negative results, reported as such

- **PPG glucose:** no candidate beats the mean under the gate (served ridge 14.69 vs mean 14.55 mg/dL, CI [−0.52, 0.12], perm p 0.54). It is served per [ADR 0005](adr/0005-ppg-option-serves-a-real-glucose-estimate.md) with a ±25 mg/dL range and is not a validated glucose estimator.
- **The pre-specified GBDT design:** T+60 −0.20 [−2.80, 2.35], T+120 +0.40 [−3.17, 3.77]. Profile features overfit with 45 people.
- **Dish identification at exact-dish level:** −20.5 pp vs MobileNet on its overlap (embedding matcher, paired bootstrap CI −31.8 to −9.1). At MobileNet's own family granularity the embedding matcher is +6.8 pp (+1.1 to +13.6).
- **PPG-started forecasts:** gain +1.51 [−0.80, 4.05] at T+60 and +2.65 [−0.42, 5.65] at T+120 (simulated g0); no PPG benefit is claimed.

## Limitations (stated, not buried)

- 45 people, so feature-set selection is unstable; a fresh cohort is needed to confirm one set.
- CGMacros meal time is a marker, not a verified first bite.
- Dish photos are studio, single-dish; mixed plates are not evaluated.
- 27 of 35 dish serving weights are assumed; portion error is unmeasured ([#22](https://github.com/Irisearch-org/GlucoSight/issues/22)).
- The PPG model comes from a dedicated contact sensor (2190 Hz), not a phone camera, and carries almost no glucose signal.
- Meter and PPG g0 conditions are simulated.
- The population is US-based (CGMacros) while the target is Indonesian T2D.

## Retired numbers

These no longer appear in the paper or in [`METRICS.md`](METRICS.md) as live figures; the numbers above replace them.

| Retired | Why | Replaced by |
|---|---|---|
| CV: food-10 86.6% / macro-F1 0.858, and the 95.96% (VLM) vs 70.71% (MobileNet) on 99 images | Not reproducible from the saved checkpoint; the 99-image raw records are missing and the 70.71% matches no reconstruction ([#11](https://github.com/Irisearch-org/GlucoSight/issues/11)) | food-10 top-1 **88.1%** (1,051/1,193), top-3 98.0%, macro-F1 0.874; T5 |
| PPG: ridge 15.64 / 15.04; mean 14.54 | Superseded by the nested run ([#16](https://github.com/Irisearch-org/GlucoSight/issues/16)) | ridge **14.69** vs training-fold mean **14.55**; T6 |
| Forecasting: RF 58.7%, GBDT 57.5%, calibrated 62.5% Zone A | Different n (1,669 vs 1,534), no nested selection ([#17](https://github.com/Irisearch-org/GlucoSight/issues/17)) | nested-CV table, T2–T3 |
| The `experimental_fusion` / B2 37.46 demo copy | Demo path removed in v2 | T2 |
