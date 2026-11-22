# Excursion + monotone carb constraint vs current forecaster (ticket #30)

## Pre-registered design and decision rule (written and committed BEFORE any run)

Data, cleaning (#12), features (#13), noise models, outer folds (5 repeats x GroupKFold(5) by participant),
seeds, noise draws and inner folds are exactly those of `forecasting/evaluate/gbdt_forecaster.py`
(#17). Every arm is fitted on the same training rows and scored on the same test rows (paired).

Arms (all with nested selection of feature set x config in inner GroupKFold(4), training participants only):
- `cur`: the current procedure as served: full #17 grid (4 feature sets x 16 configs, direct OR excursion
  target, no constraints). Must reproduce the committed trial (T+60 CGM 28.78, T+120 28.49).
- `cur_direct`: current procedure restricted to the direct-target configs (the served models are direct).
  Sensitivity comparator.
- `exc`: excursion target only (prediction = g0 + f(x)), unconstrained. Diagnostic: separates the effect of
  the target from the effect of the constraint.
- `mono_c` (**the candidate**): excursion target, monotone +1 on `carbs_g` (4 feature sets x 8 offset configs).
- `mono_cf` (second arm, exploratory): excursion, +1 on `carbs_g`, -1 on `fibre_g`.

Decision rule (applied to `mono_c`, CGM rows = gate condition, paired against `cur`):
adopt only if, at BOTH horizons, (a) paired MAE gain vs `cur` > 0, (b) participant-cluster bootstrap 95% CI
(2,000 resamples) excludes 0, (c) candidate wins a majority of the 25 outer folds (>12), and (d) the candidate
passes the ADR 0004 gate vs B2 (positive gain, >=4/5 folds every repeat, bootstrap CI > 0, 200-permutation
macro-permutation p <= 0.05). Also adopted if (a)-(d) hold at one horizon and the paired gain at the other is
>= 0 (point estimate). Otherwise the current model stays. `mono_cf` is exploratory: if it alone satisfies the
rule it is reported as a lead for a confirmatory cohort, not adopted. Gain vs `cur_direct` is reported as a
sensitivity; if its verdict differs from the verdict vs `cur` this is flagged.
Permutations: 200 for every arm gated (cur, exc, mono_c, mono_cf); 1,000 only for the final recommended arm.

## Verdict

**Not adopted: keep the current model.** The excursion target with a monotone +1 constraint on carbs
(`mono_c`) gains +0.09 mg/dL (T+60) and +0.20 mg/dL (T+120) over the current procedure on identical folds.
Both bootstrap CIs include zero (criterion (b) fails at both horizons). The candidate does pass the ADR 0004 gate vs B2 at both
horizons, but so does the current procedure, by about the same margin. `integration/models/served.json` is unchanged and no
`_excursion` models were saved ([NOT RUN]: only done on adoption).

Run: `forecasting/evaluate/excursion_monotone.py` (commands in its docstring; 3 workers, 1 thread each, 200
permutations per arm). Wall time about 50 min under heavy machine load. Raw predictions (about 12 MB) stay in `/tmp/em_raw`, outside git. Sanity check: the `cur` arm reproduces the committed #17
trial exactly (T+60 CGM 28.78, meter 29.69, gain +2.90 [1.36, 4.42], p 0.005; T+120 28.49 / 29.24, +3.56 [1.62, 5.51]) and
the served error bands (T+60 60.8 / 62.8, T+120 62.6 / 64.0 for CGM / meter).

## Main table (45 participants, CGM g0 rows = gate condition; MAE mg/dL; gain = MAE(cur) - MAE(arm), paired, participant bootstrap 95% CI, 2,000 resamples)

| Horizon | Arm | MAE CGM | MAE meter | Gain vs `cur` [CI] | Folds won vs `cur` (of 25) | Gain vs B2 [CI] | Perm p (200) | Gate vs B2 |
|---|---|---|---|---|---|---|---|---|
| T+60 | `cur` (current procedure) | 28.78 | 29.69 | - | - | +2.90 [1.36, 4.42] | 0.005 | PASS |
| T+60 | `cur_direct` | 28.31 | 29.24 | +0.47 [-0.15, 1.10] | 12 | +3.37 [2.08, 4.74] | 0.005 | PASS |
| T+60 | `exc` | 28.80 | 29.70 | -0.02 [-0.06, 0.01] | 0 | +2.88 [1.49, 4.38] | 0.005 | PASS |
| T+60 | **`mono_c` (candidate)** | **28.69** | 29.60 | **+0.09 [-0.07, 0.27]** | 16 | +2.99 [1.56, 4.49] | 0.005 | PASS |
| T+60 | `mono_cf` | 28.64 | 29.55 | +0.14 [-0.05, 0.34] | 15 | +3.04 [1.54, 4.62] | 0.005 | PASS |
| T+120 | `cur` | 28.49 | 29.24 | - | - | +3.56 [1.62, 5.51] | 0.005 | PASS |
| T+120 | `cur_direct` | 28.63 | 29.38 | -0.14 [-0.90, 0.62] | 9 | +3.42 [1.50, 5.38] | 0.005 | FAIL (fold wins [5,4,5,3,5]) |
| T+120 | `exc` | 28.35 | 29.11 | +0.14 [-0.10, 0.38] | 4 | +3.70 [1.70, 5.63] | 0.005 | PASS |
| T+120 | **`mono_c` (candidate)** | **28.29** | 29.11 | **+0.20 [-0.06, 0.53]** | 14 | +3.76 [1.85, 5.56] | 0.005 | PASS |
| T+120 | `mono_cf` | 28.43 | 29.19 | +0.06 [-0.22, 0.33] | 15 | +3.62 [1.69, 5.56] | 0.005 | PASS |

Baselines on the same folds (CGM): B0 38.72 / 36.93, B1 37.91 / 36.50, B2 31.68 / 32.05 (T+60 / T+120). Meter rows give the same picture
(`mono_c` vs `cur`: +0.09 [-0.06, 0.27] at T+60, +0.13 [-0.14, 0.42] at T+120).
Pre-registered criteria for `mono_c`: gain > 0 (met at both), CI excludes 0 (**not met at either**), majority of folds (met: 16 and 14 of 25), gate vs B2 (met). Verdict: do not adopt.

Sensitivity vs `cur_direct` (the served models are direct): `mono_c` -0.38 [-1.02, 0.25] at T+60, +0.34 [-0.44, 1.14] at T+120. Same verdict (no significant gain). `mono_cf` and `exc` do not clear the rule either, so there is no adoptable lead.
"Folds won" for `exc` is low because `exc` and `cur` pick the same model in most folds and tie.

What the pieces do: the target change alone (`exc` vs `cur`) is worth -0.02 / +0.14; the carb constraint adds about +0.1 on top at both horizons.
Both effects are far smaller than the sampling noise (CI half-widths of 0.2-0.5 mg/dL; the gap to B2 is about 3 mg/dL).

## Selection stability (25 outer folds; feature set chosen in the inner CV)

| Arm | T+60 | T+120 |
|---|---|---|
| `cur` | macros 17 / full 8 | full 16 / macros 9 |
| `cur_direct` | macros 25 / full 0 | macros 16 / full 9 |
| `exc` | macros 17 / full 8 | full 17 / macros 8 |
| `mono_c` | macros 17 / full 8 | full 17 / macros 8 |
| `mono_cf` | macros 17 / full 8 | full 18 / macros 7 |

The constraint does not stabilise the feature-set choice: `mono_c` picks the same sets as the unconstrained procedure (17/8 and 17/8).
The config choice is no more stable either (`mono_c` T+60: 11 folds `macros|nl8|mc60|n100`, 6 `macros|nl8|mc20|n100`, the rest spread over `full`
configs; full counts per fold are in `T60.json` / `T120.json`). At T+60 the direct-only grid is stable (`g0_hour_macros` in 25/25 folds), see caveat below.

## Error bands (90th percentile of absolute error, CGM / meter, mg/dL; pooled over test meals and noise draws)

| Horizon | `cur` (served) | `cur_direct` | `exc` | `mono_c` | `mono_cf` |
|---|---|---|---|---|---|
| T+60 | 60.8 / 62.8 | 60.2 / 62.4 | 60.9 / 62.8 | 60.5 / 62.7 | 60.1 / 62.5 |
| T+120 | 62.6 / 64.0 | 62.3 / 63.7 | 62.5 / 63.5 | 61.8 / 63.3 | 62.6 / 63.8 |

Each band is the empirical 90th percentile, so overall coverage is 90.0% by construction. The calibration problem is in the subgroups. A single band is too narrow where error is larger. Coverage of the `mono_c` band, CGM rows:

| Subgroup | T+60 | T+120 |
|---|---|---|
| g0 < 100 / 100-140 / 140-180 / >= 180 mg/dL | 95.2 / 92.0 / 82.2 / 79.6% | 94.7 / 92.4 / 83.2 / 73.9% |
| Normal / prediabetic / T2D (HbA1c) | 95.2 / 88.9 / 84.2% | 95.7 / 92.7 / 78.4% |
| Participants with coverage < 80% | 11 of 45 (lowest 58%) | 9 of 45 (lowest 57%) |

The other arms look the same (see JSON); the constraint does not change calibration. The bands are honest on average but a high-g0 or T2D user gets about 75-85% coverage, not 90%.
This applies to the currently served bands too (`cur`: T2D 84% / 78%, g0 >= 180 80% / 72%).

## Caveats

- **Which comparator is "the current procedure".** The served models are direct-target, but the #17 procedure also searched excursion configs, and in the
  outer folds it picked them in most folds at T+60. The decision rule used `cur` (the procedure that was gated and produced the served models); `cur_direct` is a sensitivity and gives the same verdict.
- **Side observation (post hoc, not adoptable here).** The direct-only grid (`cur_direct`) is the best arm at T+60 (28.31, `g0_hour_macros` in 25/25 folds, gate passes with 5/5 folds in every repeat), but it is not significantly better than `cur` (+0.47 [-0.11, 1.07]) and at T+120 it is worse and fails the fold-win check (one repeat won 3 of 5 folds). It was not a pre-registered candidate. Treat it, and `full` vs `g0_hour_macros`, as the confirmatory question already flagged in #17.
- Arms share inner and outer folds, so selection noise is paired, but five arms were compared; one was pre-registered as the decision arm (`mono_c`). None clears the rule, so multiplicity does not change the outcome.
- Monotone constraints only apply to feature sets that contain the constrained column. For `g0_hour` / `g0_hour_profile` the constrained and unconstrained fits are identical, so inner scores were reused.
- Same assumptions as #17: ISO 15197 noise as zero-mean normal, 3 training / 5 evaluation draws, no PPG condition, 200 permutations (p floor 0.005; 1,000 permutations [NOT RUN]: no arm was recommended for adoption).
- Final models on all 45 participants [NOT RUN]: only done if adopted.
