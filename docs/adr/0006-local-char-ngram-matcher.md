# The matcher is local character n-grams, not Gemini embeddings

**Status:** accepted (2026-11-28). Supersedes the matcher half of the photo path in `docs/SPEC_V2_PIPELINE.md` §2 (ticket #19's 0.80 / 0.05 operating point); the VLM half is unchanged.

The Gemini embedding matcher could not run on the deployed app. Embedding the 710-row table takes 930 calls; on the project key's free tier the daily embedding quota (`EmbedContentRequestsPerDayPerProjectPerModel-FreeTier`) ran out before one pass finished, and on Heroku the first photo request did that embedding inline and hit the 30 s router timeout (H12). The quota can't be raised from the project side. So the matcher now runs locally: hashed character 2–4-gram vectors (`char_wb`, accents and punctuation folded, L2-normalised), cosine per candidate string, max per row. The photo path makes one API call (the VLM), not one plus one per dish name, and matching takes milliseconds.

The operating point was re-chosen with ticket #19's protocol (minimise dev false matches subject to dev coverage ≥ 60%) on the trial's cached VLM names, scored by the served code against the **served 710-row table** (`cv/experiments/charngram_matcher/evaluate.py`, report in `cv/reports/charngram_matcher/`): **similarity ≥ 0.85, margin ≥ 0** (dev coverage 60.4%, false match 6.7%). Test at that frozen point: **coverage 61.5%, 85.6% correct when answered, false match 8.8%**, scored strictly by row id.

## Considered options

- **Keep Gemini embeddings, ship a precomputed cache.** Still needs the 930-call pass inside one day's quota, and each photo still spends one embedding call per dish name on a free-tier daily cap.
- **A local neural embedding model (ONNX multilingual MiniLM / e5).** Adds a 100+ MB model and a runtime to a 512 MB dyno. Never measured against this baseline.
- **Character TF-IDF fitted on the table.** Same test numbers here, but scores would shift whenever rows are added. Hashing makes a score a function of the two strings alone.

## Consequences

- The Gemini matcher figures (CV15/CV16: 63.4% coverage, 89.6% correct when answered, 6.6% false match) describe a matcher that is no longer served, and they were measured on the 35-row evaluation table, not the 710-row one. They are not like-for-like with the numbers above. The paper's dish-identification section must say which matcher it reports.
- The margin rule is kept in code with `MARGIN_MIN = 0.0`, since dev chose it. A tie at the top is accepted.
- The VLM still needs `GEMINI_API_KEY`; its free-tier limits now bound photo traffic alone.
