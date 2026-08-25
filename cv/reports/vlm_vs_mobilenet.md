# VLM (Gemini) vs MobileNetV2 - Comparison Report

## Recommendation

**Worth pursuing for an MVP demo as a food-classification assist, with
caching and a reduced evaluation/demo scope.** Gemini showed a large
classification gain, but its direct macro estimates were less accurate than
the existing TKPI lookup proxy and its latency and quota dependence make it
unsuitable as the sole production nutrition path.

## Methodology

- Dataset: deterministic 100-image sample from the held-out test split, seed 42.
- Fair comparison set: 99 images after excluding `pic_338` from both VLM and
	MobileNetV2 sides. `pic_338` had a Gemini HTTP 503 deadline failure and was
	excluded before calculating every comparison metric.
- No simulated VLM records are included in this report.
- Ground truth for classification is the held-out directory class.
- Macro metrics use the TKPI standard-serving lookup values. These are proxy
	errors for class-derived serving estimates, not measured food nutrition or
	portion estimation.

## Top-1 accuracy

| System | Correct | Evaluated | Top-1 accuracy |
|---|---:|---:|---:|
| Gemini VLM (`gemini-3.5-flash-lite`) | 95 | 99 | **95.96%** |
| MobileNetV2 (`cv-baseline-v0.1`) | 70 | 99 | **70.71%** |

## Macro MAE

The direct-macro run asked Gemini for numeric `carbs_g`, `protein_g`, `fat_g`,
and `fiber_g` estimates in grams, alongside the class and confidence. All 99
responses parsed successfully with numeric values for all four fields.

| Prediction source | Carbs MAE (g) | Protein MAE (g) | Fat MAE (g) | Fiber MAE (g) | Overall macro MAE |
|---|---:|---:|---:|---:|---:|
| VLM direct macro output | 8.58 | 6.95 | 4.54 | 1.36 | **5.36** |
| TKPI lookup from VLM class | 0.01 | 0.06 | 0.15 | 0.00 | **0.05** |
| MobileNetV2 feature output | 4.42 | 1.97 | 2.08 | 0.63 | **2.27** |

The TKPI-from-VLM-class row evaluates the VLM classification followed by the
existing lookup table. It is a class-level proxy, not a measurement of food
nutrition. The direct VLM row is a fresh API run using the macro prompt, with
all 99 responses parsed as numeric grams.

## Cost and latency

| System | Requests / images | Measured wall time | Cost |
|---|---:|---:|---|
| Gemini VLM classification | 100 requests, 99 successful | Approximately 83.8 minutes including pacing/retries | Not available from the local run; depends on Gemini billing/token usage |
| Gemini VLM direct macros | 99 fresh requests, 99 successful | Approximately 23.4 minutes including pacing/retries | Additional API/token cost; exact amount unavailable locally |
| MobileNetV2 | 99 images | Approximately 35.4 seconds for the comparable records | Local inference; no external API cost |

Gemini latency was measured from generated record timestamps and includes the
configured 12-second inter-request pacing and transient retries. The direct
macro run completed 99/99 requests; its wall-time span was approximately 23.4
minutes. The classification run had one excluded HTTP 503 failure. MobileNet
timing is the timestamp span in its existing feature output, not a controlled
benchmark.

## Interpretation and limitations

The VLM classification result is substantially higher on this 99-image sample,
but direct VLM macro MAE is worse than both MobileNetV2 and the class-based
TKPI proxy. The sample is small and the VLM API had quota, timeout, and HTTP
503 availability issues. The TKPI proxy remains limited by fixed serving-size
assumptions and class-level lookup mappings. A production-quality comparison
needs a controlled latency/cost benchmark and repeated held-out samples.

The real records are retained at
`cv/features/cv/vlm_v0.1_gas/features_vlm_gas.jsonl`.
Direct macro records are retained at
`cv/features/cv/vlm_v0.1_gas/features_vlm_gas_macros.jsonl`.

