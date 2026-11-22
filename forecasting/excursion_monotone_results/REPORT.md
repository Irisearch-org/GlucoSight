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
