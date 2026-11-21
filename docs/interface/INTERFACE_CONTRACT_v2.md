# GlucoSight Interface Contract v2.0

**Status:** current (2026-11-21). Supersedes [v1.x](INTERFACE_CONTRACT_v1.md).
**Decided in:** [ticket #23](https://github.com/Irisearch-org/GlucoSight/issues/23)
(agent decision under the wayfinder map's delegated authority; the user may veto).
**Spec:** [`docs/SPEC_V2_PIPELINE.md`](../SPEC_V2_PIPELINE.md) §4.
**Not a medical device.** Research prototype for Indonesian T2D users.

This contract describes what the three stages of the fixed user flow exchange:

1. **Current glucose** (`integration/stages/glucose.py`): glucometer, CGM or finger PPG scan.
2. **Meal** (`integration/stages/meal.py`, `integration/dishes/`): photo (Gemini names dishes, embedding matcher, dish table; items below the confidence rule are entered manually) or manual item-by-item entry.
3. **Forecast** (`integration/stages/forecast.py`): glucose at T+60 and T+120 from two served LightGBM models.

`integration/contract.py` keeps the schema only. The stage logic lives in the stage modules.

> **Missing ≠ zero.** An unknown nutrient value is `null`, never a fallback constant.
> The GBDT forecaster handles `null` natively. In v2 `g0` and the meal are always
> present, so the v1 `*_present` masks and modality down-weighting are retired.

---

## 1 · Envelope

| Field | Rule |
|---|---|
| `schema_version` | `"2.0"` |
| `meal_id` | string |
| `user_id` (live) or `participant_id` (datasets) | `participant_id` is the C4 grouping key |
| `source_dataset` | string (datasets only) |
| `t0_timestamp` | ISO-8601 |
| `t0_basis` | `submission` in the app (the user taps *Lihat prediksi* just before eating); `dataset_meal_marker` for CGMacros, which records the meal event, not a verified first bite |

Horizons are fixed at 60 and 120 minutes, so there is **no `delta_t_minutes` input** (closes audit defect **F7**).

## 2 · `G0Reading` (step 1)

| Field | Type | Rule |
|---|---|---|
| `g0_mgdl` | float, 40–400 | required |
| `g0_source` | `meter` \| `cgm` \| `ppg` | required |
| `g0_timestamp`, `g0_age_minutes` | datetime, float | age ≤ 60 min, otherwise step 1 asks to re-measure. **No population fallback** (closes **F12**) |
| `g0_error_range_mgdl` | float, nullable | PPG: 25 (90th percentile of held-out error, [#16](https://github.com/Irisearch-org/GlucoSight/issues/16)); meter: ISO 15197 (±15 mg/dL below 100, ±15% above); CGM: null |
| `ppg_heart_rate_bpm` | float, nullable | PPG only |
| `ppg_capture_valid` | bool | the capture gate ([#14](https://github.com/Irisearch-org/GlucoSight/issues/14)); null for meter/CGM |
| `ppg_model_version` | str | PPG only |

CGM is the single latest reading, with no history. A PPG `g0` is the served ridge model's output ([ADR 0005](../adr/0005-ppg-option-serves-a-real-glucose-estimate.md)); ring-fence E1.2 is lifted for this path only.

## 3 · `Meal` (step 2)

- `items[]`:

  | Field | Rule |
  |---|---|
  | `row_id` | required at forecast time |
  | `name` | dish-table row name |
  | `kind` | `dish` \| `ingredient` |
  | `grams` | entered grams, or portion scale (½ / 1 / 1.5) × the row's `serving_g` |
  | `source` | `photo` \| `manual` |
  | `match_similarity`, `match_margin` | photo items only |

- `totals`: `carbs_g`, `protein_g`, `fat_g` (sums over items), `fibre_g` (**null when any item's row lacks fibre**: missing ≠ zero), `kcal` (display only).
- `dish_table_build` (hash of the locally built table), `vlm_model_version`.
- **No fallback macros anywhere** (closes **F6**).

Confidence rule for photo items (one copy, in `stages/meal.py`): accept when top-1 embedding similarity ≥ **0.80** and top-1 − top-2 margin ≥ **0.05**; otherwise the item is resolved manually. VLM self-reported confidence is never used.

## 4 · `Profile`

`age_years`, `sex`, `bmi` (from height and weight), `diabetes_status`: `normal` \| `prediabetes` \| `t2d` \| `outside_training` (Tipe 1 and Lainnya).

## 5 · `ForecastResult` (step 3)

Per horizon (t60, t120): `glucose_mgdl`, `glucose_range` (`normal` < 140, `borderline` 140–199, `high` ≥ 200), `error_band_mgdl` **per horizon and `g0_source`** (90th percentile of absolute held-out error from the forecaster trial; replaces the uncalibrated `prediction_confidence`), `model_version`. Plus the `outside_training` flag.

## 6 · Dropped from v1.x

- **CV:** `gi_category`, `food_class`, `cv_confidence`, `cv_present`, `carbs_source`, `portion_reported` (replaced by per-item grams).
- **NLP:** every field (`is_fried_cooking`, `is_large_portion`, `nlp_*`). The free-text `note_text` path is retired.
- **PPG:** `ppg_hrv_rmssd`, `ppg_perfusion_index`, `ppg_signal_quality`, `ppg_clip_fraction`, `ppg_n_windows`. `ppg_glucose_estimate` becomes `g0_mgdl` with `g0_source=ppg`.
- **Context:** `time_since_last_meal_hours`.
- `CONFIDENCE_FLOOR` and `MIN_MODALITY_WEIGHT`.

## 7 · Project rules

Superseded: root `CLAUDE.md` rules 8–9 (`*_present` masks; down-weight, never exclude) belonged to the v1 fusion design, where modalities were optional. They are replaced by:

- **Missing ≠ zero:** unknown nutrient values are `null`, never a fallback constant.
- **Every stage serves a real model;** baselines are comparison rows only (ADR 0004 gate for any claimed gain).

The causality (C2), baseline (C3), grouped-split (C4) and timing (C1) rules stand. Timing C1 is met by fixed horizons rather than a Δt input.

---

## 8 · HTTP API (FastAPI, `integration/api.py`)

All bodies are JSON. Errors are `{"detail": "<Bahasa Indonesia message>"}` with status:

| Status | Meaning |
|---|---|
| 422 | invalid input |
| 409 | stale or invalid glucose capture |
| 503 | model, dish table or VLM unavailable |

Auth and user routes (`/auth/*`, `/user/*`) are unchanged from v1.

### `GET /health`

```json
{"status": "ok", "schema_version": "2.0",
 "served": "<contents of integration/models/served.json>",
 "dish_table_build": "<12 hex chars> | null"}
```

### `GET /dishes?q=<text>&limit=10`

Autocomplete (prefix and alias match, case-insensitive). Returns `{"items": [DishRowJSON, ...]}`, where

```
DishRowJSON = {id, name, kind, serving_g|null, energy_kcal, carbs_g, protein_g, fat_g, fibre_g|null, confidence}
```

Nutrient values are per 100 g edible portion. `confidence` is `exact` \| `approximate` \| `computed`.

### `POST /glucose/ppg`

```json
{"capture": {"samples": [0.0], "fs": 30.0, "timestamps": [0.0],
             "torch_verified": true, "clip_fraction": 0.0}}
```

(`timestamps` optional.) Returns a `G0ReadingJSON`:

```json
{"value_mgdl": 112, "source": "ppg", "timestamp": "ISO", "age_minutes": 0,
 "error_range_mgdl": 25.0, "heart_rate_bpm": 72, "capture_valid": true,
 "ppg_model_version": "ppg-ridge-handcrafted-nested-v1"}
```

An invalid capture returns **409** with the reason in Indonesian, and the UI shows "Ulangi scan". Sample, fs, jitter and window validation rules are those of `integration/api.py::_run_ppg` (fs 10–240 Hz, jitter ≤ 5%, ≥ 1 × 10 s window after a 3 s settle; HR 40–180 bpm).

### `POST /meal/photo`

```json
{"image_data_url": "data:image/jpeg;base64,..."}
```

Returns

```json
{"items": [{"vlm_name": "...", "accepted": true, "row": "DishRowJSON | null",
            "suggestions": ["DishRowJSON"], "similarity": 0.91, "margin": 0.12}],
 "vlm_model_version": "...", "dish_table_build": "..."}
```

`row` is set only when `accepted`. Photos are not stored.

### `POST /predict`

```json
{
  "t0_timestamp": "ISO-8601",
  "glucose": {"source": "meter" | "cgm", "value_mgdl": 123, "timestamp": "ISO"}
           | {"source": "ppg", "reading": "<G0ReadingJSON from /glucose/ppg>"},
  "meal": {"items": [{"row_id": "AP005", "grams": 150, "portion": 1, "source": "photo",
                      "similarity": 0.91, "margin": 0.12}],
           "vlm_model_version": "gemini-3.5-flash-lite" | null},
  "profile": {"age_years": 52 | null, "sex": "M" | "F" | null,
              "height_cm": 160 | null, "weight_kg": 70 | null,
              "diabetes_type": "Tipe 2" | "Tipe 1" | "Prediabetes" | "Lainnya"}
}
```

Each meal item gives `grams`, or `portion` (0, 1, 2 → ½, 1, 1.5 × `serving_g`). A PPG reading is re-validated server-side by age only: its value is trusted no further than what `/glucose/ppg` returned within the last 60 minutes.

Response:

```json
{
  "schema_version": "2.0",
  "g0": "G0ReadingJSON",
  "meal": {"items": [{"row": "DishRowJSON", "grams": 150, "source": "photo",
                      "carbs_g": 0, "protein_g": 0, "fat_g": 0, "fibre_g": null, "kcal": 0}],
           "totals": {"carbs_g": 0, "protein_g": 0, "fat_g": 0, "fibre_g": null, "kcal": 0},
           "dish_table_build": "...", "vlm_model_version": "..."},
  "forecast": {
    "t60":  {"glucose_mgdl": 0, "range": "normal", "error_band_mgdl": 0, "model_version": "..."},
    "t120": {"glucose_mgdl": 0, "range": "normal", "error_band_mgdl": 0, "model_version": "..."}
  },
  "outside_training": false,
  "disclaimer": "Prototipe riset — bukan alat medis. ..."
}
```

`range`: < 140 `normal`, 140–199 `borderline`, ≥ 200 `high`. `outside_training` is true for Tipe 1 and Lainnya. `fibre_g` is `null` whenever any contributing row has no fibre value.

## 9 · Runtime config: `integration/models/served.json`

Committed. Paths point to gitignored artefacts built locally.

```json
{
  "ppg_model": {"path": "rppg/models/weights/ppg_served_model.pt",
                "version": "ppg-ridge-handcrafted-nested-v1", "error_range_mgdl": 25.0},
  "forecaster": {"t60": {"path": "forecasting/models/gbdt_cgmacros_T60.txt", "version": "..."},
                 "t120": {"path": "forecasting/models/gbdt_cgmacros_T120.txt", "version": "..."},
                 "error_bands_mgdl": {"t60": {"meter": 0, "cgm": 0, "ppg": 0}, "t120": {}}},
  "vlm": {"model": "gemini-3.5-flash-lite", "embedding_model": "gemini-embedding-001",
          "key_file": "cv/experiments/vlm_vs_mobilenet/GEMINI_API_KEY.env"},
  "dish_table": {"path": "data/dish_table/dish_table.csv"}
}
```

Error bands are the 90th percentile of absolute held-out error per horizon and g0 source from the GBDT trial ([#17](https://github.com/Irisearch-org/GlucoSight/issues/17)).

## 10 · Audit defects closed

| Defect | How v2 closes it |
|---|---|
| **F6** fallback macros | No fallback values: unknown is `null`; items without a dish-table row cannot be forecast |
| **F7** Δt input | Horizons are fixed at 60 / 120; no `delta_t_minutes` |
| **F12** stale `g0` | `g0_age_minutes` ≤ 60 or step 1 asks to re-measure; no population fallback |
