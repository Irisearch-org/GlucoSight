# PPG pretrained models for glucose: candidates, ranking, recommendation

Ticket: Irisearch-org/GlucoSight #6 (`ppg-pretrained-models`), part of map #5.
Question: which public pretrained PPG models could beat the mean baseline (MAE 14.54 mg/dL, GroupKFold by subject) on `rppg/PPG_Dataset`?

## Bottom line

- **Likelihood of beating the mean baseline: low [INFERENCE].** No public PPG foundation model (FM) has a published glucose result on a dataset resembling ours. On the one glucose row that exists (PPG-Bench, linear probe), all five open FMs are statistically indistinguishable from a random-weights encoder.
- **Recommendation:** if one bounded test is run at all, use inference-only embeddings from **PaPaGei-S** (BSD-3-Clause code, public weights, 10 s at 125 Hz, matches our window length), with **AnyPPG** as the second candidate. Pass it through the ADR 0004 gate: positive gain vs B2 on held-out people, most folds won, bootstrap CI excluding zero, permutation control. This is consistent with ADR 0003 (accepted 2026-10-19), which records PPG glucose as a settled negative and names an HbA1c/medication covariate as the reopening condition, not a better model.
- No model was trained or fine-tuned for this ticket. All claims are from published sources or repos.

## Ranked candidates

| Rank | Model | Weights and license | Input spec | Embedding extraction | Glucose / HbA1c evidence | Compute |
|---|---|---|---|---|---|---|
| 1 | **PaPaGei-S** (Pillai et al., ICLR'25; PPG-Bench "Pillai 2025") | Code BSD-3-Clause (github.com/Nokia-Bell-Labs/papagei-foundation-model). Weights `papagei_s.pt` on Zenodo 13983110. Weights license not stated in the repo; check the Zenodo record before use. | 10 s segments, resampled to 125 Hz, then `preprocess_one_ppg_signal`. Single channel. Pretrained on public clinical PPG (~57k h, 20M segments). Finger/wrist mix. | `ResNet1DMoE` (in_channels=1, n_block=18, n_experts=3), 512-d embedding. Load via `load_model_without_module_prefix`. Example in `example_papagei.ipynb`. | PPG-Bench BUTPPG glucose linear probe: MAE 1.24±0.08, Random 1.23±0.07 (no gain). No glucose result on our dataset. | Inference only, 67 recordings, CPU-feasible [INFERENCE]. |
| 2 | **AnyPPG** (Nie et al., arXiv 2511.01747; KDD 2026 per repo README) | Weights `load_anyppg/anyppg_ckpt.pth` in the repo. **No LICENSE file in the repo tree** (checked via GitHub API). Treat as unlicensed until the authors clarify. | 10 s at 125 Hz, z-score along time, input shape (B,1,1250). Pretrained with 0.5–8 Hz bandpass (paper). ECG-guided, >100k h. | `Net1D` encoder (`load_anyppg/resnet1d.py`), 512-d output. Frozen encoder plus linear head (README). | PPG-Bench BUTPPG glucose linear probe: MAE 1.31±0.08 (worst of the set, within CI of random). AnyPPG's own paper does not report glucose. | Inference only [INFERENCE]. |
| 3 | **Pulse-PPG** (Saha et al., IMWUT 2025, DOI 10.1145/3749494) | Code and weights: MIT (github.com/maxxu05/pulseppg). Zenodo record 17345536 (v4, 2025-09-03). The repo cites weights DOI 10.5281/zenodo.17270930; the two IDs differ, so confirm which record holds the checkpoint. | Pretrained on 4-minute windows (12,000 samples) at 50 Hz, z-normalized per person, no filtering. Temporal pooling accepts variable-length input (paper §4.2.1, §4.2.4). Wrist field PPG (MOODS, 100 Hz raw). **10 s input is not tested in the paper** [INFERENCE that it works]. | ResNet1D backbone, 28.5M params, temporal pooling. Embedding dimension not checked. | PPG-Bench BUTPPG glucose linear probe: MAE 1.26±0.10, no gain. | Inference only. |
| 4 | **SigmaPPG** (Guo et al., 2026 per PPG-Bench refs) | Source at github.com/ZonghengGuo/SigmaPPG (listed by PPG-Bench, which needs the authors' source for this encoder). Weights and license not verified. | Not verified. | Not verified. | PPG-Bench BUTPPG glucose linear probe: MAE 1.22±0.09, equal to random. | Not verified. |
| — | Geenjaar et al., ICML 2026 (arXiv 2606.07365, PMLR v306) | Code and weights not stated in the abstract. Not verified. | ICU PPG plus ECG/respiration for pretraining. | Not verified. | No glucose result found. | Not verified. |
| excluded | Abbaspourazad et al., 2024 (ICLR; "Large-scale training of foundation models for wearable biosignals") | Closed weights and private training data (per Pulse-PPG §2.1). Cannot be used. | — | — | — | — |

Name mapping, one check still open: PPG-Bench cites "Nie et al., 2026" and "Guo et al., 2026". The mapping of Nie 2026 to AnyPPG and Guo 2026 to SigmaPPG is **[INFERENCE]** from PPG-Bench's encoder list and repo links. Confirm before citing.

## Evidence on glucose

**PPG-Bench (arXiv 2610.13435, Oct 2026; code github.com/mannas-ai/ppg-bench, GPL-3.0 for their code only, model weights keep their own terms).** Benchmarks five open PPG FMs (AnyPPG, PaPaGei-S, PaPaGei-P, PulsePPG, SigmaPPG) against Random, PRV, and MOMENT on 9 datasets and 17 primary tasks.

- Headline: an FM beats both PRV and MOMENT in only 4 of 17 linear-probe tasks. 24 of 68 method–task combinations show no improvement over random.
- BUTPPG glucose, linear probe, MAE (lower is better), from the HTML excerpt I read: AnyPPG 1.31±.08, PaPaGei-S 1.24±.08, PaPaGei-P 1.23±.08, PulsePPG 1.26±.10, SigmaPPG 1.22±.09, Random 1.23±.07, PRV 1.23±.12, MOMENT 1.30±.10.
- **Caveat:** units and label definition for this column are not stated in the excerpt. The MAE of ~1.2 suggests the label is not mg/dL (perhaps mmol/L or a normalized value). Do not quote it as mg/dL. BUTPPG (PhysioNet `butppg/2.0.0`) is not our dataset.
- PPG-Bench also reports PPG-BP diabetes balanced accuracy of about 0.50–0.54 (chance) for all models.

**Ramesh et al., arXiv 2608.01820 ("Reassessing the feasibility of PPG-based non-invasive blood glucose level estimation").** Tested five task-specific models on the MUST dataset under LSPO. All were near-zero R². Mean baseline MAE was 26.39 under LSPO. This is the closest published glucose analogue, but it tests task-specific models, not FMs. ADR 0003 cites it.

**Local context (already read).**
- `rppg/README.md`: GroupKFold by subject. Baseline 14.54 (per-subject median 13.94). Ridge 15.64, RF 16.52 on handcrafted features, both worse than the baseline. Glucose range 88–183 mg/dL, mean 115.0, SD 18.7.
- `docs/adr/0003-...`: settled negative. Reopening condition is an HbA1c/medication covariate. Smartphone RGB cannot reach 850–940 nm NIR.
- `rppg/LITERATURE_REVIEW_GLUCOSE_ESTIMATION.md`, lines 1–300 read. Lines 300–486 not read in this pass.

## Domain-fit risks (apply to every candidate)

1. **Sensor domain.** Our signal is a contact-finger, 550 nm green LED plus APDS-9008 photodiode, with the red channel carrying the signal (`docs/PPG_CAMERA_CAPTURE_FINDINGS.md`). PaPaGei trains on public clinical PPG. Pulse-PPG trains on wrist field PPG. Neither matches a finger-contact red-channel signal.
2. **Sampling-rate mismatch.** `rppg/PPG_Dataset` is 10 s at 2190 Hz. The ADR and lit review say 2175 Hz, which differs. Confirm the true rate before resampling to 125 Hz (PaPaGei, AnyPPG) or 50 Hz (Pulse-PPG).
3. **Window length.** Pulse-PPG was pretrained on 4-minute windows. Our 10 s windows are short for it. PaPaGei and AnyPPG use 10 s, which matches.
4. **Label noise and sample size.** 23 subjects, 67 recordings, one glucometer reading per recording. A 14.54 baseline with per-subject spread leaves little room. Any gain must clear the ADR 0004 gate.

## Compute

All candidates are inference-only for embedding extraction. 67 recordings × 10 s is a small load. Fine-tuning is out of scope for this ticket and for the ADR gate. Wall-clock estimates are **[INFERENCE]**, not measured.

## Sources

- PPG-Bench: https://arxiv.org/abs/2610.13435 ; https://arxiv.org/html/2610.13435v1 ; https://github.com/mannas-ai/ppg-bench
- PaPaGei: https://arxiv.org/abs/2410.20542 ; https://github.com/Nokia-Bell-Labs/papagei-foundation-model ; weights https://zenodo.org/records/13983110
- Pulse-PPG: https://arxiv.org/abs/2502.01108 ; https://arxiv.org/html/2502.01108v2 ; https://github.com/maxxu05/pulseppg ; https://zenodo.org/records/17345536 ; DOI 10.1145/3749494
- AnyPPG: https://arxiv.org/html/2511.01747 ; https://github.com/PKUDigitalHealth/AnyPPG
- Ramesh et al.: https://arxiv.org/html/2608.01820v1 ; DOI 10.1145/3830727.3834821
- Geenjaar et al.: https://arxiv.org/abs/2606.07365 ; https://proceedings.mlr.press/v306/geenjaar26a.html
- Local: `docs/adr/0003-ppg-glucose-estimation-not-pursued-further.md`, `rppg/README.md`, `rppg/LITERATURE_REVIEW_GLUCOSE_ESTIMATION.md`

## Open items

- Confirm the PPG-Bench glucose label units.
- Confirm the Pulse-PPG weights Zenodo record.
- Confirm the AnyPPG license.
- Read `rppg/LITERATURE_REVIEW_GLUCOSE_ESTIMATION.md` lines 300–486.
- Locate the full Guo 2026 and Nie 2026 references.
- Confirm the true sampling rate of `rppg/PPG_Dataset` (2190 vs 2175 Hz).
