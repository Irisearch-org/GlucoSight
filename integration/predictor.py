"""The prediction step of the fusion layer.

READ THIS BEFORE QUOTING ANY NUMBER THIS MODULE PRODUCES.

**There is no trained forecasting model in this repository.** The Sprint 2
work exists as a Kaggle notebook (`forecasting/notebooks/`) whose fitted
estimators were never serialised or committed. So the default predictor
here is *not* a model: it is baseline B2 — last causal reading plus the
cohort mean excursion — the same B2 that every real model must be reported
against under finding C3.

That is a deliberate choice. The alternative is a prototype that returns a
number a reviewer would read as a model output when no model exists, which
is the single most damaging thing an MVP demo can do to a research project.
Every response carries `model_version="baseline-b2-v1"` and
`is_trained_model=False` so the distinction cannot be lost between here and
a screenshot.

When Forecasting serialises a real estimator, implement `Predictor` around
it, and the API picks it up with no other change.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Protocol

from integration import contract

# Cohort mean postprandial excursion, mg/dL above the pre-meal reading.
#
# PROVENANCE: measured on CGMacros in the Sprint 2 forecasting notebook
# (`forecasting/notebooks/glucosight_forecasting_sprint2.ipynb`, cell 1
# output): 1669 meal events across 45 participants, Dexcom G6 trace,
# mean delta T+60 = 29.83 mg/dL (SD 40.83), T+120 = 25.82 (SD 43.55).
#
# The SD is larger than the mean at both horizons. That is not a footnote:
# it means the cohort mean excursion explains very little of any individual
# meal, and it is why B2 is a baseline rather than a product.
#
# This is an American mixed cohort (15 healthy / 16 pre-diabetic / 14 T2D).
# It is not an Indonesian T2D cohort and must be refitted before any claim
# about the target population (DATA_STRATEGY §4.1).
COHORT_MEAN_EXCURSION = {60: 29.83, 120: 25.82}
COHORT_EXCURSION_SD = {60: 40.83, 120: 43.55}
COHORT_PROVENANCE = (
    "CGMacros, n=1669 meals / 45 participants, Dexcom G6, Sprint 2 notebook"
)

# Contract range for a plausible glucose value.
_MIN_MGDL, _MAX_MGDL = 70.0, 400.0

# Fallback pre-meal glucose when no causal reading exists. Deliberately the
# CGMacros-cohort figure, not a healthy-population 90: predicting 90 for a
# T2D user with no history would read as reassuring and be wrong.
FALLBACK_G0 = 120.0


class Predictor(Protocol):
    model_version: str
    is_trained_model: bool

    def predict(self, meal: Dict[str, Any]) -> Dict[str, Any]: ...


class BaselineB2Predictor:
    """Persistence + cohort mean excursion. A baseline, not a model.

    prediction_confidence is built from what the input actually contained,
    not from anything the predictor knows about glucose:

      * no causal pre-meal reading -> capped hard, because the prediction is
        then a cohort constant with no personalisation at all;
      * modality weights (H1) pull it down as CV/NLP/PPG degrade;
      * a delta_t far from the horizon pulls it down, because the excursion
        constant is calibrated at 60 and 120 minutes (C1).

    It is never allowed above 0.45. A baseline that reports high confidence
    is worse than one that reports none.
    """

    model_version = "baseline-b2-v1"
    is_trained_model = False
    CONFIDENCE_CEILING = 0.45

    def __init__(
        self,
        mean_excursion: Optional[Dict[int, float]] = None,
        provenance: str = COHORT_PROVENANCE,
    ):
        self.mean_excursion = dict(mean_excursion or COHORT_MEAN_EXCURSION)
        self.provenance = provenance

    def _confidence(self, meal: Dict[str, Any], has_g0: bool) -> float:
        conf = self.CONFIDENCE_CEILING
        if not has_g0:
            conf = min(conf, 0.15)

        weights = [
            contract.modality_weight(meal, m) for m in contract.MODALITY_FIELDS
        ]
        conf *= sum(weights) / len(weights)

        delta_t = float(meal.get("delta_t_minutes") or 60.0)
        nearest = min(self.mean_excursion, key=lambda h: abs(h - delta_t))
        drift = abs(delta_t - nearest)
        if drift > 10:
            # Contract evaluation restricts to [50,70] and [110,130]; outside
            # that the constant is being extrapolated.
            conf *= max(0.3, 1.0 - (drift - 10) / 60.0)

        return round(max(0.0, min(1.0, conf)), 3)

    def predict(self, meal: Dict[str, Any]) -> Dict[str, Any]:
        g0 = meal.get("g0")
        has_g0 = g0 is not None and math.isfinite(float(g0))
        base = float(g0) if has_g0 else FALLBACK_G0

        t60 = base + self.mean_excursion[60]
        t120 = base + self.mean_excursion[120]
        t60 = float(min(_MAX_MGDL, max(_MIN_MGDL, t60)))
        t120 = float(min(_MAX_MGDL, max(_MIN_MGDL, t120)))

        delta_t = float(meal.get("delta_t_minutes") or 60.0)

        out = {
            "glucose_t60": round(t60, 1),
            "glucose_t120": round(t120, 1),
            "glucose_range": contract.glucose_range(t60),
            "prediction_confidence": self._confidence(meal, has_g0),
            "delta_t_minutes": delta_t,
            "model_version": self.model_version,
        }

        errors = contract.validate_forecast_output(out)
        if errors:
            raise contract.ContractViolation(
                "predictor produced a non-compliant record: " + "; ".join(errors)
            )

        out["is_trained_model"] = self.is_trained_model
        out["basis"] = (
            f"B2 baseline: last causal reading ({base:.0f} mg/dL"
            f"{'' if has_g0 else ', FALLBACK — no causal reading available'}) "
            f"+ cohort mean excursion. Provenance: {self.provenance}. "
            f"Cohort excursion SD ({COHORT_EXCURSION_SD[60]:.0f} mg/dL at T+60) "
            f"exceeds its mean, so this is a population constant, not a "
            f"meal-specific prediction."
        )
        out["used_fallback_g0"] = not has_g0
        out["modality_weights"] = {
            m: round(contract.modality_weight(meal, m), 3)
            for m in contract.MODALITY_FIELDS
        }
        return out


def default_predictor() -> Predictor:
    """The predictor the API serves.

    Returns BaselineB2Predictor until a trained estimator is committed. Do
    not change this to something that looks like a model without a
    serialised, grouped-split-validated estimator behind it and a results
    table with B0/B1/B2 beside it (C3).
    """
    return BaselineB2Predictor()
