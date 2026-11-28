# GlucoSight Integration Layer (v2)

The backend of the website: a FastAPI app (`integration/api.py`) that serves the fixed
three-step flow. **Research prototype, not a medical device.**

1. **Current glucose** (`stages/glucose.py`): glucometer entry, CGM reading, or a finger PPG scan scored by the served ridge model.
2. **Meal** (`stages/meal.py`, `dishes/`): photo (Gemini names the dishes, the local character n-gram matcher maps each name to a dish-table row; items below the confidence rule must be entered manually) or manual item-by-item entry.
3. **Forecast** (`stages/forecast.py`): glucose at T+60 and T+120 from two served LightGBM models, each with an error band.

Authoritative documents: [`docs/SPEC_V2_PIPELINE.md`](../docs/SPEC_V2_PIPELINE.md) (what and why),
[`docs/interface/INTERFACE_CONTRACT_v2.md`](../docs/interface/INTERFACE_CONTRACT_v2.md) (schema and HTTP API),
[`docs/EVALUATION_PLAN.md`](../docs/EVALUATION_PLAN.md) (which numbers the paper reports),
[`docs/METRICS.md`](../docs/METRICS.md) (which numbers are real). Read METRICS before quoting any figure.

## Run it locally

Prerequisites: the gitignored artefacts below are in place (the `served.json` paths), and the Gemini key file exists.

```bash
# backend, from the repo root: http://127.0.0.1:8000
pip install -r integration/requirements.txt
python -m uvicorn integration.api:app --reload

# frontend (second terminal): Vite proxies the API paths to :8000
cd frontend
npm install
npm run dev            # desktop browser
npm run dev:phone      # phone on the same LAN: HTTPS (self-signed), required for the camera
npx playwright install chromium webkit   # once, for the browser UI tests
npm run test:ui       # layout suite at phone/tablet/desktop widths (API mocked, no backend needed)
```

Phone browsers expose the camera (meal photo and finger scan) only on a secure context, which is why
`dev:phone` serves HTTPS on the LAN. Open the printed `https://<LAN-IP>:5173` URL on the phone and accept
the certificate warning.

`GET /health` shows what is being served (`served.json`) and the dish-table build hash; a missing artefact
makes the dependent endpoint return 503 rather than fall back to something else.

### Accounts database (`glucosight`)

Login and history are backed by a MariaDB/MySQL database named `glucosight` (users and server-side
sessions). Start MySQL (XAMPP works with its defaults), then:

```bash
python -m integration.db.init_db      # creates the database and tables, safe to re-run
```

Connection settings are `GLUCOSIGHT_DB_HOST|PORT|USER|PASSWORD|NAME` (defaults `127.0.0.1`, `3306`, `root`,
empty, `glucosight`). Set `GLUCOSIGHT_COOKIE_SECURE=1` when serving over HTTPS. Auth and user routes
(`POST /auth/register|login|logout`, `GET /auth/me`, `GET|PUT /user/profile`, `GET|POST /user/history`) are
unchanged from v1. History keeps dish names and macros only; photos are never stored.

## Build the dish table

The dish table is built locally from the TKPI 2017 PDF (Kemenkes). TKPI has no open licence and this repo
is public, so **the table values are never committed**; git holds only the parser, aliases, recipes and
class mapping.

```bash
python -m integration.dishes.build --tkpi-pdf /path/to/TKPI-2017.pdf --out data/dish_table/dish_table.csv
```

`data/dish_table/dish_table.csv` is covered by the root `.gitignore` (`/data/`). The API reads it through
`DishTable.load`, and `build_hash` (first 12 hex of the CSV's sha256) is returned in `/health`,
`/meal/photo` and `/predict` so any result can be tied to the build that produced it. One row is one food
as eaten (`kind` = `dish` | `ingredient`), with per-100 g nutrients, nullable `fibre_g`, `serving_g` and a
`confidence` of `exact` | `approximate` | `computed`. Ingredient serving weights cite Kemenkes
KepDirjen HK.02.02/B/576/2025; dish `serving_g` values are assumed defaults and labelled as such.

## Runtime config and gitignored artefacts

`integration/models/served.json` is committed; the paths inside it point at files that are **not**:

| What | Path (from `served.json`) | How to get it |
|---|---|---|
| PPG model (ridge, 7 handcrafted features) | `rppg/models/weights/` (`ppg_served_model.*`) | trained by `rppg/experiments/ppg_served_model` (branch `wayfinder/ppg-served-model`); `rppg/models/weights/` is gitignored |
| Forecasters, T+60 and T+120 | `forecasting/models/gbdt_cgmacros_T60.txt`, `..._T120.txt` (+ `.json`) | trained by `forecasting/evaluate/gbdt_forecaster.py` (branch `wayfinder/gbdt-forecaster`); `forecasting/models/` is gitignored |
| Dish table | `data/dish_table/dish_table.csv` | `python -m integration.dishes.build`, see above; `/data/` is gitignored |
| Gemini key | `cv/experiments/vlm_vs_mobilenet/GEMINI_API_KEY.env` | a file containing `GEMINI_API_KEY=...`, or the `GEMINI_API_KEY` env var; `*.env` is gitignored |

The Gemini key is read by the backend only and is never sent to the browser. Never commit it or print it.
Model used: `gemini-3.5-flash-lite` (dish naming, prompted in Indonesian), one call per photo. Matching is
local and needs no API ([ADR 0006](../docs/adr/0006-local-char-ngram-matcher.md)). Error bands in `served.json` are the 90th percentile of absolute held-out error per horizon and
g0 source from the forecaster trial.

## HTTP endpoints

Full schemas are in [`INTERFACE_CONTRACT_v2.md`](../docs/interface/INTERFACE_CONTRACT_v2.md) §8. Errors are
`{"detail": "<message in Bahasa Indonesia>"}` with 422 (invalid input), 409 (stale or invalid glucose
capture) or 503 (model, table or VLM unavailable).

| Method and path | Purpose |
|---|---|
| `GET /health` | `status`, `schema_version` ("2.0"), the served config, `dish_table_build`, `dish_matcher` |
| `GET /dishes?q=&limit=10` | dish-table autocomplete (prefix and alias, case-insensitive); nutrients per 100 g |
| `POST /glucose/ppg` | finger-scan capture (`samples`, `fs`, `torch_verified`, `clip_fraction`) → a `G0Reading`; invalid capture → 409 "Ulangi scan" |
| `POST /meal/photo` | `image_data_url` → items `{vlm_name, accepted, row, suggestions, similarity, margin}` |
| `POST /predict` | glucose + meal items + profile → `g0`, meal totals, `forecast.t60` / `forecast.t120` with `error_band_mgdl`, `outside_training`, disclaimer |
| `/auth/*`, `/user/*` | accounts and history, unchanged |

Confidence rule for photo items (one copy, in `stages/meal.py`): accept when top-1 similarity ≥ 0.85 and the
top-1 − top-2 margin ≥ 0 (operating point in `cv/reports/charngram_matcher/`); otherwise the item is resolved manually. VLM self-reported confidence is never
used.

## What is honest about the served models

- **PPG glucose is not a validated estimator.** Held-out MAE 14.69 vs 14.55 for the mean (fails the ADR 0004 gate); predictions barely track glucose and high values are under-read. It is served per [ADR 0005](../docs/adr/0005-ppg-option-serves-a-real-glucose-estimate.md) and always shown as a range (±25 mg/dL, 90% of held-out errors).
- **The forecaster beats B2** under nested cross-validation for CGM and meter g0 (T+60 28.78 vs 31.68 mg/dL; T+120 28.49 vs 32.05) but **not for PPG g0**, where it runs on a near-constant input and no benefit is claimed.
- **No end-to-end accuracy number** is claimed (ADR 0004). Numbers: [`docs/METRICS.md`](../docs/METRICS.md).

## Deploy to Heroku

App `glucosight-iris` (Python buildpack, JawsDB MySQL Kitefin, Basic dyno). The root `Procfile` runs
`integration.db.init_db` as the release phase and uvicorn as the web process; uvicorn serves the built
frontend from `frontend/dist`. Config vars: `GEMINI_API_KEY` (read before the key file),
`JAWSDB_URL` (set by the add-on; `GLUCOSIGHT_DB_*` still override it), `GLUCOSIGHT_COOKIE_SECURE=1`.

The gitignored artefacts (models, dish table) and `frontend/dist` reach Heroku through a throwaway local
commit pushed **only** to the `heroku` remote, never to `origin`:

```bash
cd frontend && npm run build && cd ..
git worktree add --detach /tmp/gs-deploy HEAD && cp -R frontend/dist /tmp/gs-deploy/frontend/
mkdir -p /tmp/gs-deploy/data/dish_table /tmp/gs-deploy/forecasting/models /tmp/gs-deploy/rppg/models/weights
cp data/dish_table/dish_table.csv /tmp/gs-deploy/data/dish_table/
cp forecasting/models/gbdt_cgmacros_T*.{txt,json} /tmp/gs-deploy/forecasting/models/
cp rppg/models/weights/ppg_served_model.joblib /tmp/gs-deploy/rppg/models/weights/
cd /tmp/gs-deploy && git add -f frontend/dist data forecasting/models rppg/models/weights \
  && git commit -m "deploy: artefacts for Heroku (never push to origin)" \
  && git push --force heroku HEAD:main
cd - && git worktree remove --force /tmp/gs-deploy
```

## Tests

```bash
python -m pytest -q integration/tests
```

## Rules enforced here rather than trusted

- **C2 causality, C3 baselines, C4 grouped splits** stand (root `CLAUDE.md`). C1 is met by fixed T+60 / T+120 horizons, with no Δt input.
- **Missing ≠ zero:** unknown nutrient values are `null`, never a fallback constant; meal `fibre_g` is `null` when any item's row lacks it.
- **Every stage serves a real model;** B0/B1/B2 and the PPG mean are comparison rows in evaluation only.
- Contract v1's `*_present` masks, modality down-weighting and ring-fence E1.2 for the PPG-glucose path are superseded by contract v2.

## Not a medical device

Research prototype. Not for any treatment decision.
