# Dish naming VLM for Indonesian meal photos (issue #9)

Parent map: #5 (GlucoSight v2 pipeline spec). Status: research, no code changed.
Local date of research: 2026-11-21.

## 1. Question

Which hosted or open vision-language model (VLM) should name dishes in an Indonesian meal photo, including mixed plates? How do we get per-dish confidence that is usable? What are latency, cost, API terms for a research demo, and photo retention terms?

## 2. Recommendation

1. **Default: Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) on the paid tier**, called with `generateContent`, `responseMimeType: application/json`, and `temperature: 0` (as in `cv/experiments/vlm_vs_mobilenet/run_dish_list_gas_vlm.py`).
   - It is already the model behind the only local evidence: 95/99 top-1 (95.96%) on the 99-photo local set, against 70/99 (70.71%) for MobileNetV2 (`cv/reports/vlm_vs_mobilenet.md`). That result is for single-dish classification from a closed list, so it is a lower bound for mixed plates.
   - Cheapest of the candidates per call. Paid price $0.30 input / $2.50 output per 1M tokens (Source 1).
   - **Paid tier only.** The free tier is marked "Used to improve our products: Yes" (Source 1). Photos of identifiable people or homes must not go to the free tier.
   - Per-dish confidence: sample N calls (N = 3–5, `temperature` > 0 for the samples) and use agreement over dish names. Also request `responseLogprobs` with `logprobs` set to a small integer (the documented range is 0–20; Source 2). Treat the logprob output as a ranking signal, not a probability, until calibrated on local labels.
2. **Alternative (API): OpenAI GPT-5-mini or GPT-5.4-mini with `logprobs` / `top_logprobs`**. Logprobs are documented for Chat Completions with `top_logprobs` (Source 4). This is the only hosted candidate with a documented per-token logprob API on the same request, and it is the second choice if Gemini logprobs are weak on Indonesian names. No Indonesian-dish benchmark found for it.
3. **Fallback (self-hosted, no per-call API cost): Qwen3-VL-8B-Instruct or InternVL3.5-8B**, both Apache-2.0 on Hugging Face (Sources 5, 6). Needs a GPU host; latency and quality on Indonesian dishes are unmeasured here. Use only if a privacy or cost constraint rules out hosted APIs.
4. **Not recommended for logprob confidence: Claude (Haiku 5.5 / Sonnet 5.5).** Cheap vision, but the Messages API has no logprob field (Source 7). Confidence would have to come from sampling agreement or verbalized scores only.

Confidence method for all candidates: **sampling agreement across N calls** as the primary signal, with the logprob or verbalized score as a secondary signal. Verbalized confidence from LLMs is poorly calibrated (Xiong et al. 2023, Source 13); sampling-based uncertainty (semantic entropy, Kuhn et al. 2023, Source 12) is the better-supported option. Calibrate on local labels before using any threshold in `docs/SPEC_DISH_IDENTIFICATION.md` (`cv_confidence` = MIN across matched items).

## 3. Ranked candidates

| Rank | Candidate | Why | Main risk |
|---|---|---|---|
| 1 | Gemini 3.5 Flash-Lite, paid tier | Local evidence (95.96%), cheapest hosted output, logprobs documented, paid tier not used for training | No Indonesian-dish benchmark; quota/503 seen in local run |
| 2 | GPT-5-mini or GPT-5.4-mini, Chat Completions with logprobs | Documented `top_logprobs`; training opt-out by default | Indonesian quality unmeasured; ZDR needs approval; image CSAM retention exception |
| 3 | Qwen3-VL-8B / InternVL3.5-8B self-hosted | Apache-2.0; no photo leaves the host | GPU cost, latency, Indonesian accuracy unmeasured |
| 4 | Claude Haiku 5.5 / Sonnet 5.5 | Low price, strong vision | No logprobs; ZDR only by sales arrangement |

## 4. Evidence on Indonesian dish naming

- **Local (strongest in-repo evidence):** 95.96% top-1 on 99 Indonesian photos with a closed candidate list (`cv/reports/vlm_vs_mobilenet.md`). Records in `cv/features/cv/vlm_v0.1_gas/features_vlm_gas.jsonl`.
- **Local failure mode (ADR 0002):** open-vocabulary naming works, but matching the VLM label to the dish table fails. difflib mapped "nasi gudeg" to `nasi_goreng`. Gemini embeddings (`gemini-embedding-001`, RETRIEVAL_QUERY/DOCUMENT) do not separate garnishes such as kacang goreng from soto mie. The VLM reasoning over a short candidate list (top-K from the dish table) is the proposed fix in `docs/adr/0002-cv-is-dish-identification-not-macro-estimation.md`. This is a matching problem, not a naming problem, so the choice of VLM matters less than the matcher.
- **WorldCuisines (arXiv:2410.12705):** a multilingual VQA benchmark with Indonesian-language questions and an Indonesia country tag. GPT-4o is reported as the strongest model overall, and Qwen2-VL-Instruct and Llama 3.2 Vision are included. No per-country Indonesian dish accuracy was extracted from the PDF text, so treat this as directional only.
- **DiningBench (arXiv:2604.10425, ACL 2026):** dietary perception and reasoning benchmark with MAPE per component. It does not mention Indonesia in the text extracted from the PDF. Its per-model numbers were not verified here.
- **Thai analogue (github.com/werapatlpw/thai-food-vlm-benchmark, IEEE Access 2026 submitted):** GPT-4o, Gemini, Grok, and Llama 4 on Thai dishes, with raw per-image predictions and the exact prompt. It is the closest SE-Asian replication design. Accuracy numbers were not extracted here.
- **Gap:** no published Indonesian dish-naming benchmark was found. A small labeled set of Indonesian photos (including mixed plates) with the dish-table names is the only way to settle the choice. This is a gap to fill before the paper, not something a literature search can close.

## 5. Latency and cost per photo

Assumptions: one photo, one call; prompt ~200 tokens; output ~150 tokens (JSON list of 2–4 items); image sizes as below. Estimates are [INFERENCE] unless marked with a source.

| Candidate | Input price (per 1M) | Output price (per 1M) | Image tokens (per photo) | Approx. cost per call | Source |
|---|---|---|---|---|---|
| Gemini 3.5 Flash-Lite (paid) | $0.30 | $2.50 | 258 per tile; ≤384 px = 258 total; larger tiled into 768×768 tiles at 258 each (Source 8) | ~$0.0007 (1 tile) to ~$0.0013 (4 tiles) | Source 1, 8 |
| GPT-5-mini | $0.25 | $2.00 | patch-based, 32 px patches (Source 9); ~768 patches for 1024×768 before the model-specific budget [INFERENCE] | ~$0.0006 | Source 3, 9 |
| GPT-5.4-mini | $0.75 | $4.50 | same as above | ~$0.0017 | Source 3 |
| Claude Haiku 5.5 (≤100k prompt) | $0.10 | $0.50 | ⌈w/28⌉×⌈h/28⌉ visual tokens, longest edge ≤1,568 px (Source 10); ~1,040 for 1024×768 | ~$0.0002 | Source 7, 10 |
| Qwen3-VL-8B / InternVL3.5-8B (self-hosted) | GPU hours | GPU hours | n/a | not measured | Source 5, 6 |

Wall-clock latency: not measured here. The local run recorded ~83.8 min for 100 classification calls with 12 s pacing (`cv/reports/vlm_vs_mobilenet.md`), which is dominated by the rate limit and retries, not model time. Plan the demo around quota (Gemini daily request caps apply on the free tier; paid tier has the RPM/TPM limits in the Gemini rate-limit docs).

Price change to plan around: the Gemini 4 Carbon Standard tier rises from $2/$10 to $4/$20 per 1M on 2026-12-01 (Source 1). Gemini 3.5 Flash-Lite is not in that change in the fetched page.

## 6. API terms and photo retention (research demo)

Photos are personal data (`docs/protocol/DATA_COLLECTION_PROTOCOL.md` §8–9). Use pseudonymous `participant_id`, store raw JPEG keyed by `meal_id`, and state the retention period in the consent form. The table below is vendor terms as of the sources' fetch dates.

| Provider / tier | Training use | Abuse-monitoring retention | Zero data retention (ZDR) | Notes |
|---|---|---|---|---|
| Gemini API, paid tier | No (Source 1, 2) | 55 days (Source 11) | Not on Gemini API; ZDR requires Vertex AI (Source 2) | Logs kept for billing-enabled projects; default 55 days, selectable 7/14/28/55 (Source 2) |
| Gemini API, free tier | **Yes** (Source 1) | — | — | Do not send participant photos |
| Gemini Contributor tier | Required (Source 2) | — | — | Do not donate EEA/UK/CH or personal data |
| OpenAI API (default) | No since 2023-03-01 (Source 4) | up to 30 days (Source 4) | ZDR by approval, `/v1/chat/completions` and `/v1/responses` eligible (Source 4) | Image inputs are CSAM-scanned and may be retained for manual review even under ZDR; Files API and batches not ZDR-eligible |
| Anthropic API | Not without express permission (Source 7) | Covered Models: 30 days, not ZDR (Source 7); flagged content up to 2 years (Source 7) | Per org by sales arrangement; Messages API eligible; Files, Batches, code execution not eligible (Source 7) | Claude Console usage not ZDR-covered (Source 7) |

Implications for the demo:
- Use **Gemini paid tier** or **OpenAI API default** for participant photos. Do not use the free tier, and do not use Anthropic Covered Models if retention must stay at zero.
- Ask the vendor for ZDR only if the protocol requires it. Gemini ZDR needs Vertex AI, a different billing and setup path.
- Strip EXIF (GPS, device) before upload. The protocol's identifiability rule still applies to the photo content itself.
- Upload with the inline image `inline_data` / base64, not the Files API, if retention must be minimal (Files API is stored until deleted on Gemini; explicit caching too).

## 7. Open questions for the paper

1. Does Gemini `responseLogprobs` on `gemini-3.5-flash-lite` return usable per-token probabilities for dish-name tokens? Not tested; the field is documented (Source 2) but untested on this model.
2. Mixed plates: does the VLM return all components (rice, sambal, tempeh) or only the main dish? Needs a labeled set.
3. Does top-K candidate reasoning (VLM chooses from K dish-table entries) beat open naming plus embedding matching? ADR 0002 proposes it; not measured.
4. Latency of sampling N=5 per photo: not measured; rate limits may dominate.

## 8. Sources

1. Gemini API pricing, https://ai.google.dev/gemini-api/docs/pricing (Gemini 3.5 Flash-Lite and Gemini 3.8 Flash rows; "Used to improve our products" row; Gemini 3.8 Flash price change to $0.75/$3.75 through 2026-12-31).
2. Gemini API generate-content reference, https://ai.google.dev/api/generate-content (GenerationConfig `responseLogprobs`, `logprobs` 0–20; logprobsResult). Logs policy, https://ai.google.dev/gemini-api/docs/logs-policy. Zero data retention, https://ai.google.dev/gemini-api/docs/zdr.
3. OpenAI API pricing, https://developers.openai.com/api/docs/pricing (gpt-5-mini $0.25/$2.00; gpt-5.4-mini $0.75/$4.50; gpt-5.6-luna, gpt-6-luna rows also present).
4. OpenAI data controls, https://developers.openai.com/api/docs/guides/your-data; Chat Completions reference (`logprobs`, `top_logprobs`), https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create.
5. Qwen3-VL-8B-Instruct model card, https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct (license: apache-2.0).
6. InternVL3.5-8B model card, https://huggingface.co/OpenGVLab/InternVL3_5-8B (license: apache-2.0).
7. Anthropic pricing, https://platform.claude.com/docs/en/about-claude/pricing (Claude Haiku 5.5 $0.10/$0.50; Sonnet 5.5 $2/$10); API retention, https://platform.claude.com/docs/en/manage-claude/api-and-data-retention. Messages API has no logprob field (no `logprob` occurrences in https://platform.claude.com/docs/en/api/messages).
8. Gemini token counting, https://ai.google.dev/gemini-api/docs/tokens (images ≤384 px = 258 tokens; tiles 768×768 = 258 tokens).
9. OpenAI vision guide, https://developers.openai.com/api/docs/guides/images-vision (patch-based tokenization, 32 px patches, detail levels).
10. Anthropic vision guide, https://platform.claude.com/docs/en/build-with-claude/vision (1 visual token per 28×28 patch; 1,568 px standard max).
11. Gemini usage policies and abuse monitoring, https://ai.google.dev/gemini-api/docs/usage-policies (55-day retention for policy enforcement).
12. Kuhn, Gal, Farquhar 2023, "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation", ICLR 2023, https://arxiv.org/abs/2302.09664.
13. Xiong et al. 2023, "Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs", https://arxiv.org/abs/2306.13063 (verbalized confidence; sampling-based consistency).
14. Gao et al. 2021, "CLIP-Adapter: Better Vision-Language Models with Feature Adapters", https://arxiv.org/abs/2110.04544 (few-shot adapter fallback, not evaluated here).
15. WorldCuisines, https://arxiv.org/abs/2410.12705 (Indonesian-language split; GPT-4o leads).
16. DiningBench, https://arxiv.org/abs/2604.10425 and https://aclanthology.org/2026.acl-long.1630/ (ACL 2026).
17. Thai food VLM benchmark, https://github.com/werapatlpw/thai-food-vlm-benchmark (IEEE Access 2026 submitted).

Local references: `cv/reports/vlm_vs_mobilenet.md`, `cv/experiments/vlm_vs_mobilenet/run_dish_list_gas_vlm.py`, `cv/LITERATURE_REVIEW.md` §3 and §5, `docs/adr/0002-cv-is-dish-identification-not-macro-estimation.md`, `docs/SPEC_DISH_IDENTIFICATION.md`, `docs/protocol/DATA_COLLECTION_PROTOCOL.md`.

## 9. Verification status

- Verified from fetched primary sources: all prices in §5 table rows marked Source 1, 3, 7; retention rows; logprob availability (Gemini generation config, OpenAI Chat Completions, Anthropic Messages absence); open-model licenses.
- Not verified: Gemini logprobs on `gemini-3.5-flash-lite`; OpenAI and Gemini exact image token counts for a typical photo (estimates labelled [INFERENCE]); latency per call; any Indonesian-specific accuracy number for any hosted or open model.
- Secondary (not used as primary evidence): practitioner blog reporting $0.005–0.012 per scan and 2.6–8.3 s latency, cited in `cv/LITERATURE_REVIEW.md`; self-reported, not peer reviewed.
