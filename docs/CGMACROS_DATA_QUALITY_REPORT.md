# CGMacros v1.0.0 — meal-field range violations (report to maintainers)

**Dataset:** CGMacros: a scientific dataset for personalized nutrition and diet
monitoring, v1.0.0. PhysioNet. DOI [10.13026/3z8q-x658](https://doi.org/10.13026/3z8q-x658).
**Send to:** the corresponding author contact listed on the PhysioNet project
page (physionet.org/content/cgmacros/1.0.0/ — login required to view), citing
this DOI and `DataDictionary_CGMacros-00X.csv`/`DataDictionary.pdf`.

## Issue

Reproducing the per-field ranges in `DataDictionary.pdf` / `DataDictionary_CGMacros-00X.csv`
against a 15/45 participant sample (564 meal events — the remaining 30
participants were not available to us when this was written; see
"How to reproduce") finds 136 distinct rows with at least one nutrient value
outside the documented minimum/maximum:

| Field | Dictionary range | Violations (15/45 participants) | Example |
|---|---|---|---|
| Protein | 3–176 g | 86 below minimum | 0 g, `CGMacros-001` 2020-05-01 20:48 |
| Calories | 30–1180 kcal | 13 below min, 9 above max | 2250 kcal, `CGMacros-004` 2023-09-19 18:19 |
| Carbs | 0–176 g | 8 above max | 382 g, `CGMacros-021` 2020-06-09 20:23 |
| Fiber | 0–176 g | 13 above max | 2300 g, `CGMacros-019` 2020-10-01 18:34 — ~13x the documented maximum |
| Amount Consumed | 0–100 (%) | 37 above max | 500, `CGMacros-033` 2022-04-16 15:12 — reads as 5x a single serving |

Extrapolated to all 45 participants this is consistent with the ~400-500 row
range we saw in an earlier, unreproduced pass over the full dataset.

We have not found a single systematic correction (e.g. a consistent decimal
or unit shift) that resolves all violating rows — the Protein violations look
like genuinely low-protein snacks against an overly strict 3 g floor, while
the Fiber and Amount Consumed violations look like a unit or scale error
(2300 g of fiber and "500%" consumed are not physically plausible for a
single meal). So we are not clipping, rescaling, or imputing these values.
We'd like to know:

1. Are `DataDictionary.pdf` ranges authoritative for v1.0.0, or were they
   written for a different unit/scale than the CSV columns?
2. Is there a known encoding issue (e.g. Fiber in mg for some rows, or a
   logging bug for specific participants) we should correct for, or should
   out-of-range rows be treated as erroneous and excluded?
3. Is a corrected/updated version of the dataset planned?

## How to reproduce

```bash
python -m forecasting.data.cgmacros <path-to-CGMacros-root> --output audit.json
```

`audit.json["range_violations"]` gives below/above counts per field; cross-
reference against each participant's raw CSV for exact row-level values.
The counts above are from 15 of the 45 participant CSVs (`CGMacros-001/002/
004/007/012/014/015/018/019/021/033/039/042/046/048`); re-run against all 45
to get the authoritative full-cohort counts.

## Why this matters for us

We're building a glucose-forecasting baseline on this dataset
(`docs/CGMACROS_PIPELINE_SPEC.md`). Until this is resolved, we treat
meal-composition features as blocked and only evaluate glucose-history-only
baselines — see `forecasting/evaluate/run_cgmacros.py`'s `meal_feature_gate`,
which is computed automatically from this same audit and fails closed.
