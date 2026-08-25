VLM vs MobileNet evaluation

This folder contains scaffolding to run a VLM (Gemini) evaluation on the
same held-out test split used by the MobileNet baseline.

The real Gemini runner reads `GEMINI_API_KEY` from
`cv/experiments/vlm_vs_mobilenet/GEMINI_API_KEY.env`. A repo-root `.env` is not
required. The file is intentionally named `GEMINI_API_KEY.env` and is loaded by
explicit path.

Quick local checks:
 - Real Gemini run (100 held-out images):

```bash
python cv/experiments/vlm_vs_mobilenet/run_gas_vlm.py --sample-size 100
```

Real runs are resumable. Existing successful records for the selected sample
and current model are skipped, and each new successful response is checkpointed.

 - Dry run (write stub outputs):

```bash
python cv/experiments/vlm_vs_mobilenet/run_vlm.py --dry-run
```

 - Simulate VLM by copying MobileNet outputs (pipeline smoke test):

```bash
python cv/experiments/vlm_vs_mobilenet/run_vlm.py --simulate-from-mobile
```

Output path: `cv/features/cv/vlm_v0.1/features_vlm.jsonl`

Report: `cv/reports/vlm_vs_mobilenet.md` (created by the evaluation step).