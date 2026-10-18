# ShanghaiT2DM — the response variable, rebuilt

**Date:** 2026-10-18
**Dataset:** ShanghaiT2DM v3 — 100 T2D patients, Shanghai, China.
Zhao et al., *Scientific Data* 10:35 (2023), `PMC9849330`.
figshare `10.6084/m9.figshare.20444397.v3`, **CC BY 4.0**, 3.4 MB.
**Question:** finding N6 — the Path A feature table carries glucose *at* the
meal and no outcome. Can the outcome be rebuilt from the raw workbooks?
**Status:** Rebuilt. 3,045 of 3,060 meals (99.51%) now have a T+60 rise.

Artifacts: `nlp/data/shanghai_schema.py`, `nlp/data/shanghai_response.py`,
`nlp/tests/test_shanghai_response.py` (20 tests),
`nlp/features/shanghai_nlp_response.csv`,
`nlp/features/shanghai_nlp_response.summary.json`.

---

## Why this was blocked

N6 concluded that Path A "cannot be evaluated for glucose contribution at all
— not because the data is missing, but because only the meal rows were kept."
The 15-minute series was in the raw workbooks the whole time. What was
actually missing was the download: `nlp/data/Shanghai_T2DM/` did not exist,
the raw files were never committed, and no copy was on disk.

The v3 release is 3.4 MB. It was never a size problem.

---

## R1 — The response exists, and coverage is 99.5%

The workbook is a dense event log: one row every 15 minutes, with dietary
text on meal rows. Across 109 workbooks:

| measure | value |
|---|---:|
| readings | 112,379 |
| 15-minute cadence | 112,168 of 112,371 gaps |
| meal rows | 3,802 |
| of which placeholders (`Data not available`) | 742 |
| **real meals** | **3,060** |
| subjects / CGM periods | 100 / 109 |
| date range | 2020-09-07 → 2022-01-29 |

`g0` is the last reading at or before `t0`, never a future one, and never
more than 15 minutes stale. The outcome is the reading nearest `t0 + h`,
accepted within 15 minutes. Everything rejected carries a named reason;
nothing is imputed or clipped.

| horizon | valid meals | coverage | excluded |
|---|---:|---:|---|
| T+60 | 3,045 | 99.51% | 7 `g0_out_of_range`, 7 `no_outcome`, 1 `outcome_out_of_range` |
| T+120 | 3,028 | 98.95% | 7 `g0_out_of_range`, 22 `no_outcome`, 3 `outcome_out_of_range` |

The 3,060 reconciles exactly with the legacy table, and the rebuild joins it
on `meal_id` with **100% exact text agreement**. No meal was gained or lost;
one column was added.

## R2 — The old loader was silently dropping CGM for two workbooks

The v3 release ships **five** header layouts:

| layout | files |
|---|---:|
| documented | 102 |
| `进食量` instead of `饮食` | 3 |
| trailing spaces (`CGM `, `CBG `, …) | 2 |
| `Novolin R  IU` (double space) | 1 |
| `胰岛素泵基础量` for the basal column | 1 |

`shanghai_diet_loader._find_col` compared header strings exactly, after a
`.strip()`. `"CGM "` therefore never matched `"CGM (mg / dl)"`, `idx_cgm`
stayed `None`, and **every reading in that workbook was read as missing**.

This is not theoretical. The legacy feature table has exactly **96 rows with
no CGM value, and they are exactly the two trailing-space workbooks** —
`2045_0_20201216.xls` (64 meals) and `2095_0_20201116.xls` (32 meals). That
is 3.1% of the cohort silently marked `cgm_available = 0`, visible only as a
hole in a column nobody had checked against the raw files.

The rebuild recovers `g0` for all 96. Header matching now goes through
`nlp/data/shanghai_schema.py::canonical_columns`, which normalises
punctuation and CJK rather than comparing literals.

## R3 — The rise is real, and it behaves like BIG IDEAs

A first look, to check the pipeline is measuring glucose rather than noise.
**This is a sanity check, not a result** — no meal features are in it.

| measure | value |
|---|---:|
| median `g0` | 127.8 mg/dL |
| median rise at T+60 | **+23.4 mg/dL** (IQR +1.8 to +52.2) |
| median rise at T+120 | +25.2 mg/dL |
| meals with a positive T+60 rise | 75.1% |
| subjects with a positive median rise | 96 of 100 |
| **corr(`g0`, T+60 rise)** | **−0.296** |

The negative `g0` correlation is the same structure BIG IDEAs showed
(−0.355, negative for all 15 participants): a high starting value means a
smaller rise. Two independent cohorts, same sign, same order of magnitude.

The median rise is larger than BIG IDEAs (+23.4 vs +14.0). That is the
expected direction — these are Type 2 patients, BIG IDEAs is pre-diabetic —
but it is not evidence of anything until the meal term is fitted.

## R4 — This cohort is medicated, and that is the largest structural difference

| flag | meals |
|---|---:|
| insulin bolus or s.c. dose within 30 min of `t0` | 373 |
| `g0` taken during a ≥4-reading identical run at the sensor floor | 6 |

BIG IDEAs is pre-diabetic and unmedicated. **12% of these meals have a
meal-timed insulin dose**, so part of any measured "meal response" is the
response to the bolus. Every result on this cohort must report the medicated
subset, and preferably fit it separately.

This is not a caveat to bolt on at the end. It is the reason a Shanghai result
and a BIG IDEAs result are not directly comparable.

## R5 — Meal overlap matters at T+120, barely at T+60

| horizon | meals with another meal inside the window |
|---|---:|
| T+60 | 23 (0.8%) |
| T+120 | 107 (3.5%) |

At T+60 the overlap is small enough to keep and flag. At T+120 a T+120 "rise"
after a meal 40 minutes later is two meals, so the T+120 column is weaker
evidence than the coverage number suggests.

## R6 — The split is still wrong, and this rebuild does not fix it

Unchanged from N5: `participant_id` is the workbook stem (`<subject>_<period>_<date>`),
so the 109 ids are 109 *periods* belonging to 100 *subjects*. Grouping on it
puts a subject's second admission on both sides of the split, leaking 18.5–21.5%
of test meals.

The response table now carries `subject_id` and `period_index` as separate
columns so the correct grouping is one `groupby` away. It is not fixed,
only made possible.

---

## What this changes

N6 said Path A had a predictor and no outcome. It now has both, at 99.5%
coverage, with the exclusions named. The track is no longer blocked on data.

What it does **not** change:

- **No model has been fitted.** N2 still holds: the shipped feature vector is
  `(total_grams, is_fried, is_large, nlp_confidence)` and carbohydrate is one
  global constant. Nothing here extracts a dish.
- **Path A still has no macro ground truth.** Shanghai gives glucose, not
  carbohydrate values. The response is the only supervision available, which
  means the dish table can only be validated *through glucose*.
- **The 0.41 break-even is untested on this cohort.** Whether dish identity
  beats grams here is Step 3, not this document.

## Reproduce

```bash
python3 -m nlp.data.shanghai_response \
  --data-dir ~/Downloads/ShanghaiT2DM \
  --output nlp/features/shanghai_nlp_response.csv \
  --summary nlp/features/shanghai_nlp_response.summary.json
```

3.8 seconds for all 109 workbooks. Requires `xlrd` for the 89 `.xls` files
and `openpyxl` for the 20 `.xlsx` ones. The raw directory stays outside the
repository; only the derived table is committed.

Download and unpack (the archive has one GBK-encoded folder name, which
macOS `unzip` cannot create — use Python's `zipfile` with
`metadata_encoding="gbk"`):

```python
import zipfile, os
z = zipfile.ZipFile("data.zip", metadata_encoding="gbk")
for info in z.infolist():
    if "/Shanghai_T2DM/" in info.filename and not os.path.basename(info.filename).startswith("~$"):
        with z.open(info) as src, open(os.path.basename(info.filename), "wb") as dst:
            dst.write(src.read())
```

Tests: `python3 -m pytest nlp/tests/test_shanghai_response.py -q` (20 tests,
synthetic workbooks, no raw data required).

## What would change this answer

1. **A denser CGM grid.** At 15 minutes the T+60 reading can be 7.5 minutes
   early or late; a 5-minute device would sharpen every rise by roughly the
   slope times the offset. Nothing here corrects for that.
2. **Insulin data joined properly.** 373 meals have a bolus nearby; this
   rebuild flags them but does not model the dose.
3. **A macro ground truth.** If a Chinese food composition table is joined to
   the 1,841 dish names, Path A gains a second, non-glucose target and can be
   validated without the response at all.
