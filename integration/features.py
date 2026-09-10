"""Feature tiers and the normalization spec at the Forecasting boundary.

Two tiers exist because the Sprint 2 forecasting notebook reported a single
"Multimodal" column whose features were:

    weighed macros straight off CGMacros  (not a CV model output)
  + clinical labs: HbA1c, insulin, triglycerides, HDL, fasting glucose
  + a gut-health panel: butyrate production pathways, digestive efficiency
  + BMI, age, sex

Nothing after `carbs_g` in that list can be obtained from a phone at
inference time, and the gut panel is a CGMacros-specific commercial assay.
A number built on it is not a number the prototype can reproduce for a user,
so it cannot be the headline. It is still worth having: it bounds what any
amount of meal-composition modelling could achieve on this cohort.

So:

  TIER_DEPLOYABLE — everything the contract says a phone produces. This is
      the headline. It is what `integration/api.py` can actually serve.
  TIER_ORACLE     — TIER_DEPLOYABLE plus ground-truth macros and the
      clinical/gut panel. An upper bound, reported as one, never as fusion.

The gap between the two tiers is itself a result: it says how much of the
notebook's improvement over B2 came from meal composition (which CV can
approximate) versus from participant physiology (which it cannot).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from integration import contract

# ---------------------------------------------------------------------
# Tier 1 — what a phone can actually produce
# ---------------------------------------------------------------------

# Continuous features: z-scored using training-split statistics only.
DEPLOYABLE_CONTINUOUS: Tuple[str, ...] = (
    # CV
    "carbs_g", "protein_g", "fat_g", "fiber_g",
    # PPG
    "ppg_pulse_rate_bpm", "ppg_hrv_rmssd", "ppg_perfusion_index",
    "ppg_signal_quality", "ppg_clip_fraction",
    # Context / envelope
    "delta_t_minutes", "time_since_last_meal_hours",
    # Glucose history, derived causally in fusion.py
    "g0", "g0_age_minutes", "glucose_slope_30min", "glucose_mean_6h",
    "hour_of_day_sin", "hour_of_day_cos",
)

# Binary flags: left as 0/1, never z-scored (contract, Normalization Spec).
DEPLOYABLE_BINARY: Tuple[str, ...] = (
    "is_fried_cooking", "is_large_portion",
    "cv_present", "nlp_present", "ppg_present",
)

# One-hot expansions. gi_category as an integer would assert that the
# low->medium gap equals the medium->high gap in glycemic response (H7).
DEPLOYABLE_ONEHOT: Dict[str, int] = {
    "gi_category": 3,
    "portion_reported": 3,
}

# Integer counts kept as-is.
DEPLOYABLE_COUNT: Tuple[str, ...] = ("ppg_n_windows",)

# ---------------------------------------------------------------------
# Tier 2 — oracle upper bound. NOT available at inference.
# ---------------------------------------------------------------------
ORACLE_ONLY_CONTINUOUS: Tuple[str, ...] = (
    "calories",                      # dataset-provided, not CV-derived
    "Age", "BMI",
    "Fasting GLU - PDL (Lab)", "A1c PDL (Lab)", "Insulin",
    "Triglycerides", "Cholesterol", "HDL", "Cho/HDL Ratio",
    "Metabolic Fitness", "Gut Lining Health", "Inflammatory Activity",
    "Digestive Efficiency", "Butyrate Production Pathways",
)
ORACLE_ONLY_BINARY: Tuple[str, ...] = ("Gender_M",)

# Why each oracle feature cannot be served by the prototype. Printed in the
# report so the exclusion is argued, not just asserted.
ORACLE_JUSTIFICATION: Dict[str, str] = {
    "calories": "dataset-provided weighed value; CV emits macros, not calories",
    "Age": "obtainable by asking the user — promote to Tier 1 if the app asks",
    "BMI": "obtainable by asking the user — promote to Tier 1 if the app asks",
    "Gender_M": "obtainable by asking the user — promote to Tier 1 if the app asks",
    "Fasting GLU - PDL (Lab)": "requires a fasting venous lab draw",
    "A1c PDL (Lab)": "requires a venous lab draw",
    "Insulin": "requires a venous lab draw",
    "Triglycerides": "requires a venous lab draw",
    "Cholesterol": "requires a venous lab draw",
    "HDL": "requires a venous lab draw",
    "Cho/HDL Ratio": "requires a venous lab draw",
    "Metabolic Fitness": "commercial gut-microbiome assay, CGMacros-specific",
    "Gut Lining Health": "commercial gut-microbiome assay, CGMacros-specific",
    "Inflammatory Activity": "commercial gut-microbiome assay, CGMacros-specific",
    "Digestive Efficiency": "commercial gut-microbiome assay, CGMacros-specific",
    "Butyrate Production Pathways": "commercial gut-microbiome assay, CGMacros-specific",
}

# Three of the oracle features (age, BMI, sex) are a product decision away
# from Tier 1. Kept separate so the report can quote a third tier cheaply.
ASKABLE = ("Age", "BMI", "Gender_M")

TIER_DEPLOYABLE = "deployable"
TIER_ORACLE = "oracle"
TIER_ASKABLE = "deployable_plus_askable"
TIERS = (TIER_DEPLOYABLE, TIER_ASKABLE, TIER_ORACLE)


def tier_spec(tier: str) -> Dict[str, Any]:
    """Return the feature spec for a tier."""
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}; expected one of {TIERS}")

    continuous = list(DEPLOYABLE_CONTINUOUS)
    binary = list(DEPLOYABLE_BINARY)

    if tier == TIER_ASKABLE:
        continuous += [f for f in ASKABLE if f not in ORACLE_ONLY_BINARY]
        binary += [f for f in ASKABLE if f in ORACLE_ONLY_BINARY]
    elif tier == TIER_ORACLE:
        continuous += list(ORACLE_ONLY_CONTINUOUS)
        binary += list(ORACLE_ONLY_BINARY)

    return {
        "tier": tier,
        "continuous": tuple(continuous),
        "binary": tuple(binary),
        "onehot": dict(DEPLOYABLE_ONEHOT),
        "count": tuple(DEPLOYABLE_COUNT),
    }


def feature_names(tier: str, include_ringfenced: bool = False) -> List[str]:
    """Ordered feature names for a tier.

    `ppg_glucose_estimate` is excluded from every tier by default. It is
    ring-fenced (E1): it may never be a sole prediction and must be isolable
    in ablations, so entering the default feature set is exactly what it is
    fenced against. An ablation opts in explicitly.
    """
    spec = tier_spec(tier)
    names: List[str] = []
    for f in spec["continuous"]:
        if contract.is_ringfenced(f):
            continue
        names.append(f)
    for f in spec["binary"]:
        names.append(f)
    for f in spec["count"]:
        names.append(f)
    for f, width in spec["onehot"].items():
        names.extend(f"{f}__{i}" for i in range(width))
    if include_ringfenced:
        names.append("ppg_glucose_estimate")
    return names


def _onehot(value: Any, width: int) -> List[float]:
    vec = [0.0] * width
    if value is None:
        return vec
    idx = int(value)
    if 0 <= idx < width:
        vec[idx] = 1.0
    return vec


def vectorize(
    records: Sequence[Dict[str, Any]],
    tier: str = TIER_DEPLOYABLE,
    include_ringfenced: bool = False,
    apply_modality_weights: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Turn contract records into a numeric matrix.

    Down-weighting (H1) is applied here, at the boundary, so every consumer
    gets it and no track has to remember. A CV block with cv_confidence 0.1
    is scaled toward the population fallback rather than dropped; dropping
    would change input dimensionality at inference time.
    """
    spec = tier_spec(tier)
    names = feature_names(tier, include_ringfenced=include_ringfenced)

    modality_of = {}
    for modality, fields in contract.MODALITY_FIELDS.items():
        for f in fields:
            modality_of[f.name] = modality

    rows = []
    for rec in records:
        weights = {
            m: contract.modality_weight(rec, m)
            for m in contract.MODALITY_FIELDS
        } if apply_modality_weights else {m: 1.0 for m in contract.MODALITY_FIELDS}

        row: List[float] = []
        for f in spec["continuous"]:
            if contract.is_ringfenced(f):
                continue
            value = rec.get(f)
            value = np.nan if value is None else float(value)
            m = modality_of.get(f)
            # Weight only the modality's own measurements, never its mask or
            # its own quality score — scaling a quality score by itself is
            # circular and makes low quality look like a low pulse rate.
            if m and not f.endswith(("_present", "_confidence", "_signal_quality",
                                     "_clip_fraction", "_n_windows")):
                row.append(value * weights[m])
            else:
                row.append(value)
        for f in spec["binary"]:
            v = rec.get(f)
            row.append(np.nan if v is None else float(v))
        for f in spec["count"]:
            v = rec.get(f)
            row.append(np.nan if v is None else float(v))
        for f, width in spec["onehot"].items():
            row.extend(_onehot(rec.get(f), width))
        if include_ringfenced:
            v = rec.get("ppg_glucose_estimate")
            row.append(np.nan if v is None else float(v))
        rows.append(row)

    matrix = np.asarray(rows, dtype=float)
    if matrix.shape[1] != len(names):
        raise AssertionError(
            f"vectorize built {matrix.shape[1]} columns for {len(names)} names"
        )
    return matrix, names


class TrainOnlyScaler:
    """z-score with statistics from the training split only.

    Contract, Normalization Spec: "normalization statistics computed over the
    full dataset are a leak. The loader must compute them from the training
    split and assert that val/test never contribute."

    The assert is the point. `fit` records a fingerprint of the rows it saw;
    `transform` refuses to run if asked to transform rows that were part of
    a different fit, and `fit` refuses to run twice on the same instance.
    Binary and one-hot columns are passed through untouched.
    """

    def __init__(self, names: Sequence[str], skip_prefixes=("is_", "cv_present",
                                                            "nlp_present", "ppg_present")):
        self.names = list(names)
        self._skip = set()
        for i, n in enumerate(self.names):
            if (n.startswith(skip_prefixes) or "__" in n
                    or n in {"cv_present", "nlp_present", "ppg_present"}):
                self._skip.add(i)
        self.mean_ = None
        self.scale_ = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "TrainOnlyScaler":
        if self._fitted:
            raise RuntimeError(
                "TrainOnlyScaler.fit called twice. Refit on a new fold means a "
                "new scaler; reusing one silently mixes fold statistics."
            )
        X = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        self.scale_[~np.isfinite(self.scale_)] = 1.0
        self.mean_[~np.isfinite(self.mean_)] = 0.0
        for i in self._skip:
            self.mean_[i] = 0.0
            self.scale_[i] = 1.0
        self._fitted = True
        self.n_fit_rows_ = int(X.shape[0])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("transform before fit — that is the leak this class exists to stop")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != len(self.names):
            raise ValueError(
                f"expected {len(self.names)} columns, got {X.shape[1]}"
            )
        return (X - self.mean_) / self.scale_

    def to_dict(self) -> Dict[str, Any]:
        """Persist to disk, per the contract's normalization spec."""
        return {
            "names": self.names,
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
            "n_fit_rows": getattr(self, "n_fit_rows_", None),
        }

    @classmethod
    def from_dict(cls, blob: Dict[str, Any]) -> "TrainOnlyScaler":
        obj = cls(blob["names"])
        obj.mean_ = np.asarray(blob["mean"], dtype=float)
        obj.scale_ = np.asarray(blob["scale"], dtype=float)
        obj._fitted = True
        obj.n_fit_rows_ = blob.get("n_fit_rows")
        return obj
