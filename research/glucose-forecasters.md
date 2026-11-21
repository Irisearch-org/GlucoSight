# Research: public pretrained glucose forecasters vs GBDT (CGMacros, T+60 / T+120)

Ticket: #7 (wayfinder, parent map #5). Slug: `glucose-forecasters`.

## Question

Which public pretrained glucose forecasters could compete with the GBDT forecaster at T+60 and T+120 on CGMacros, given that the app only has a single fingerstick g0 at inference?

## Answer (gist)

No public pretrained glucose forecaster can replace the GBDT under single-g0 inference: every candidate needs a multi-hour CGM history. Keep the GBDT as the deployable baseline. Only TimesFM 2.5 and Chronos-2 (both Apache-2.0, public weights) are worth a benchmark arm, and only if the app can supply CGM history.

## Key constraint

Single-g0 inference means the input is one glucose value plus meal context. The pretrained models were built for the opposite regime:

- GlucoFM-Bench (arXiv 2606.06881) uses a **12 h context** as the default and sweeps 1–24 h for zero-shot. TSFMs are not defined for a context of one point.
- CGM-LSM (npj Health Systems, s44401-025-00039-y) takes the **previous 24 h** of CGM as input (288 readings) and forecasts 2 h.
- GluFormer (Nature, s41586-025-09925-9) is an autoregressive generator over a context window, with downstream tasks of HbA1c, outcomes, and diet response. It is not a T+60/T+120 forecaster.

Running a TSFM on a one-point context reduces it to near-persistence. Any claim that a TSFM beats the GBDT at single-g0 would be unobserved, so it is marked `[INFERENCE]` here.

## Candidates (ranked)

| Rank | Model | Public weights | Licence | Input needs | Evidence | Verdict |
|---|---|---|---|---|---|---|
| 0 | GBDT (repo, current) | in repo | repo | g0 + meal context | repo docs: CGMacros T+60 MAE 31.44 mg/dL, Zone A 57.5% (FC1/FC2) | Keep as the deployable baseline |
| 1 | TimesFM 2.5 (200M) | yes, `google/timesfm-2.5-200m-pytorch` | Apache-2.0 | CGM context (≥12 h benchmarked) | GlucoFM-Bench zero-shot, 12 h context, 30 min: CGMacros RMSE 16.28 vs persistence 19.44; best zero-shot overall (RMSE 18.75 vs baseline 22.25) | Benchmark arm only, needs history |
| 2 | Chronos-2 (120M) | yes, `amazon/chronos-2` | Apache-2.0 | CGM context | GlucoFM-Bench zero-shot CGMacros RMSE 16.85; best overall few-shot RMSE 18.95 | Benchmark arm only, needs history |
| 3 | CGM-LSM | not verified (no release found in the fetched page) | not verified | 24 h CGM context | OhioT1DM zero-shot (not trained on it): rMSE 15.90 at 1 h, 26.88 at 2 h; held-out T2D 2 h rMSE 23.49 | Not usable: no verified weights, 24 h context, and no CGMacros result |
| 4 | GluFormer | code Apache-2.0 (`github.com/Guylu/GluFormer`), no checkpoint in the repo tree | code Apache-2.0; data by request (HPP) | CGM context | Paper tasks are HbA1c and outcomes; no T+60/T+120 result | Not usable for this task. Its diet-token extension is the only related feature, and it is unverified |

Excluded:

- **Moirai 1.1-R** (`Salesforce/moirai-1.1-R-large`): licence CC-BY-NC-4.0, so no commercial use. This blocks a deployed product, though not research use.
- **Moirai 2.0**: it appears in GlucoFM-Bench but its licence was not checked in this ticket.
- **Timer**: did not improve on persistence overall (GlucoFM-Bench Table 2).

## What the benchmarks do and do not show

- GlucoFM-Bench reports the 30 min horizon in its main tables. Its horizons are 15, 30, 60 and 90 min, and the 60 min numbers for CGMacros are in its Appendix Table 11, which I did not fetch. The T+120 horizon is not covered.
- No benchmark in the fetched sources compares a pretrained model against a GBDT on CGMacros at T+60 or T+120. The only GBDT numbers available are from the repo.
- The metrics are not directly comparable. GlucoFM-Bench reports RMSE at 30 min and the repo reports MAE at T+60 for a GBDT on a different setup. A comparison requires the same split, horizon, and metric.
- GlucoFM-Bench finds a lightweight LSTM is strongest under full-shot training, outperforming TSFMs by 4–21%. This suggests a supervised model on the same data beats zero-shot TSFMs when data exist.

## Recommendation

1. Keep the GBDT as the single-g0 forecaster. Do not replace it with a pretrained model.
2. If the team wants a pretrained arm for the paper, run TimesFM 2.5 and Chronos-2 zero-shot on CGMacros at T+60 and T+120. Use a 12 h CGM context, which the app does not have at inference. Report MAE, RMSE, and the constant and persistence baselines, following `docs/METRICS.md` §8. Do not quote Zone A alone.
3. Adopt a pretrained model into the product only if it passes the offline gate in ADR 0004: positive improvement over the GBDT, wins in most folds, a bootstrap CI that excludes zero, and a permutation test.
4. Do not use CGM-LSM or GluFormer without first confirming released weights and a CGMacros-compatible input format.

## Sources

- GlucoFM-Bench, arXiv 2606.06881: https://arxiv.org/html/2606.06881v1 (code https://github.com/Augmented-Health-Lab/GlucoseML_benchmark, data https://huggingface.co/datasets/glucofmbench/GlucoFM-Bench)
- CGM-LSM, npj Health Systems, DOI 10.1038/s44401-025-00039-y: https://www.nature.com/articles/s44401-025-00039-y
- GluFormer, Nature, DOI 10.1038/s41586-025-09925-9: https://www.nature.com/articles/s41586-025-09925-9 ; code https://github.com/Guylu/GluFormer (Apache-2.0)
- Chronos-2 weights and licence: https://huggingface.co/api/models/amazon/chronos-2 (apache-2.0)
- TimesFM 2.5 weights and licence: https://huggingface.co/api/models/google/timesfm-2.5-200m-pytorch (apache-2.0)
- Moirai 1.1-R licence: https://huggingface.co/api/models/Salesforce/moirai-1.1-R-large (cc-by-nc-4.0)
- Repo: `docs/METRICS.md`, `docs/adr/0004-model-level-fusion-plan.md`, `forecasting/evaluate/baselines.py`, `integration/predictor.py`
