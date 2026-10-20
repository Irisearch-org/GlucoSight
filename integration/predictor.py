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

import json
import math
import os
from typing import Any, Dict, Optional, Protocol

from integration import contract
from integration import glucose_source as gsrc

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

        # Scale by how the pre-meal value was obtained. A fingerstick and a
        # population constant must not produce the same confidence.
        conf *= float(meal.get("g0_trust", gsrc.SOURCE_TRUST[
            gsrc.SOURCE_POPULATION_FALLBACK]))

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
        # `g0_mgdl` always carries a usable number — the resolved
        # measurement, or the population fallback. `g0` is None precisely
        # when no measurement was available, which is what has_g0 records.
        base = float(meal.get("g0_mgdl", FALLBACK_G0)) if not has_g0 else float(g0)

        t60 = base + self.mean_excursion[60]
        t120 = base + self.mean_excursion[120]
        t60 = float(min(_MAX_MGDL, max(_MIN_MGDL, t60)))
        t120 = float(min(_MAX_MGDL, max(_MIN_MGDL, t120)))

        requested_horizon = float(meal.get("delta_t_minutes") or 60.0)
        if requested_horizon not in self.mean_excursion:
            raise ValueError(
                f"baseline B2 supports only fixed horizons {sorted(self.mean_excursion)} minutes"
            )

        selected_glucose = t60 if requested_horizon == 60.0 else t120
        out = {
            "glucose_t60": round(t60, 1),
            "glucose_t120": round(t120, 1),
            "glucose_range": contract.glucose_range(selected_glucose),
            "glucose_range_t60": contract.glucose_range(t60),
            "glucose_range_t120": contract.glucose_range(t120),
            "prediction_confidence": self._confidence(meal, has_g0),
            "delta_t_minutes": requested_horizon,
            "target_horizons_minutes": {"glucose_t60": 60, "glucose_t120": 120},
            "requested_horizon_minutes": requested_horizon,
            "model_version": self.model_version,
        }

        errors = contract.validate_forecast_output({
            **out,
            "glucose_t60": selected_glucose,
            "glucose_range": contract.glucose_range(selected_glucose),
        })
        if errors:
            raise contract.ContractViolation(
                "predictor produced a non-compliant record: " + "; ".join(errors)
            )

        out["is_trained_model"] = self.is_trained_model
        out["basis"] = (
            f"B2 baseline: pre-meal glucose {base:.0f} mg/dL "
            f"({meal.get('g0_detail', 'source unrecorded')}) "
            f"+ cohort mean excursion. Provenance: {self.provenance}. "
            f"Cohort excursion SD ({COHORT_EXCURSION_SD[60]:.0f} mg/dL at T+60) "
            f"exceeds its mean, so this is a population constant, not a "
            f"meal-specific prediction."
        )
        out["g0_source"] = meal.get("g0_source")
        out["g0_is_direct_measurement"] = meal.get("g0_is_direct_measurement")
        out["used_fallback_g0"] = not has_g0
        # So api.py can report what actually drove this number rather than
        # the pre-predictor resolution -- the two differ exactly when a
        # predictor (e.g. ExperimentalFusionPredictor) overrides g0 itself.
        out["g0_mgdl"] = meal.get("g0_mgdl")
        out["g0_trust"] = meal.get("g0_trust")
        out["g0_detail"] = meal.get("g0_detail")
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


EXPERIMENTAL_FUSION_ARTIFACT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "experimental_fusion.json"
)


class ExperimentalFusionPredictor:
    """CV + NLP + PPG, each actually affecting the forecast -- by explicit
    product decision (2026-10-20 grill), NOT by the project's evidence gate.

    READ THIS BEFORE QUOTING ANY NUMBER THIS CLASS PRODUCES. It deliberately
    skips the 4-criterion gate `rppg.models.glucose_estimator` and
    `nlp.models.dish_identity_ridge` use. It is not a single model trained
    on CV+NLP+PPG together -- no dataset pairs all three with glucose
    (ADR 0004) -- it is two small ridge fits, each on the one real dataset
    that actually supports it
    (`forecasting/evaluate/fit_experimental_fusion.py`):

        cv adjustment       CGMacros ground-truth macros -> T+60 glucose
        nlp/ppg adjustment  BIG IDEAs food-log text (+ wrist BVP once
                            downloaded) -> T+60 excursion

    blended onto the B2 baseline, each weighted by that modality's own
    confidence (contract.modality_weight, H1 down-weight-never-exclude) so
    an absent or low-confidence modality pulls the prediction only a
    little, never zero and never catastrophically.

    Reachable only via `prediction_mode="experimental_fusion"` -- it never
    becomes `default_predictor()`.
    """

    model_version = "experimental-fusion-v0"
    is_trained_model = True
    label = "experimental_ungated"

    def __init__(self, blob: Dict[str, Any]):
        self._blob = blob
        self._cgmacros = blob["cgmacros_meal"]
        self._bigideas = blob["bigideas_meal"]
        self._ppg = blob.get("ppg_absolute") if blob.get("ppg_absolute", {}).get("available") else None
        self._b2 = BaselineB2Predictor()

    @classmethod
    def load(cls, path: str = EXPERIMENTAL_FUSION_ARTIFACT) -> "ExperimentalFusionPredictor":
        with open(path) as fh:
            return cls(json.load(fh))

    # Any one feature's z-score is clipped before the dot product. A ridge
    # fit on a handful of rows (the PPG block here is n=13) has near-zero
    # training variance on some features; an unremarkable-looking input can
    # still z-score into the thousands and extrapolate to a wild output
    # (found while testing this class: perfusion_index=95.7 against a
    # training scale of 0.0075 -> glucose_t60=312). Same failure shape as
    # finding F9. Clipping bounds each feature's contribution to
    # 5*|coefficient| regardless of how far out of distribution the input is.
    _MAX_Z = 5.0

    @classmethod
    def _predict_block(cls, block: Dict[str, Any], values: Dict[str, Optional[float]]) -> float:
        x = []
        for name, median in zip(block["feature_names"], block["median"]):
            v = values.get(name)
            v = float(v) if v is not None and math.isfinite(float(v)) else median
            x.append(v)
        scaled = [
            max(-cls._MAX_Z, min(cls._MAX_Z, (v - m) / s))
            for v, m, s in zip(x, block["mean"], block["scale"])
        ]
        return float(block["intercept"] + sum(c * s for c, s in zip(block["coef"], scaled)))

    def predict(self, meal: Dict[str, Any]) -> Dict[str, Any]:
        # "Current glucose" (g0): a fingerstick or CGM reading wins if either
        # is present (DIRECT_MEASUREMENT_SOURCES). Only when neither exists
        # does a PPG capture get to estimate the SAME quantity -- this is
        # g0, the thing the rest of the pipeline already builds on, not a
        # separate number blended into the T+60 output. Scoped to this mode
        # only (never touches glucose_source.py's global registry, so
        # baseline_demo's g0 resolution is untouched). Known to fail the
        # evidence gate (ADR 0003) -- used anyway only because this mode is
        # explicitly ungated.
        working_meal = dict(meal)
        if (not meal.get("g0_is_direct_measurement") and self._ppg is not None
                and meal.get("ppg_present")):
            ppg_g0 = self._predict_block(self._ppg, {
                "pulse_rate_bpm": meal.get("ppg_pulse_rate_bpm"),
                "hrv_rmssd_ms": meal.get("ppg_hrv_rmssd"),
                "perfusion_index": meal.get("ppg_perfusion_index"),
                "signal_quality": meal.get("ppg_signal_quality"),
                "rise_time_ms": None, "pulse_width_half_ms": None, "n_beats_detected": None,
            })
            from integration import glucose_source as gsrc
            working_meal["g0"] = ppg_g0
            working_meal["g0_mgdl"] = ppg_g0
            working_meal["g0_source"] = gsrc.SOURCE_PPG_ESTIMATE
            working_meal["g0_trust"] = gsrc.SOURCE_TRUST[gsrc.SOURCE_PPG_ESTIMATE]
            working_meal["g0_detail"] = (
                f"PPG estimate (ungated, {self._ppg['known_result']})"
            )

        b2_out = self._b2.predict(working_meal)
        g0 = float(working_meal.get("g0_mgdl", FALLBACK_G0))

        t60_cv = self._predict_block(self._cgmacros, {
            "g0_mgdl": g0,
            "g0_age_minutes": meal.get("g0_age_minutes"),
            "history_n": len(meal.get("glucose_history") or []),
            "hour_of_day_sin": meal.get("hour_of_day_sin"),
            "hour_of_day_cos": meal.get("hour_of_day_cos"),
            "carbs_g": meal.get("carbs_g"),
            "protein_g": meal.get("protein_g"),
            "fat_g": meal.get("fat_g"),
            "fiber_g": meal.get("fiber_g"),
            "amount_consumed_pct": None,  # not producible live; median-imputed
        })
        delta_nlp = self._predict_block(self._bigideas, {
            "g0_mgdl": g0,
            "carbs_g": meal.get("carbs_g"),
            "is_fried_cooking": meal.get("is_fried_cooking"),
        })
        t60_nlp = g0 + delta_nlp

        from integration import contract
        w_cv = contract.modality_weight(meal, "cv")
        w_nlp = contract.modality_weight(meal, "nlp")

        # PPG's effect already happened above, in g0 -- it is not a second,
        # separate pull on the output. Double-counting it here (once via g0,
        # once as its own output blend) would silently double its weight.
        t60 = float(b2_out["glucose_t60"])
        t60 = t60 * (1 - w_cv) + t60_cv * w_cv
        t60 = t60 * (1 - w_nlp) + t60_nlp * w_nlp
        t60 = float(min(_MAX_MGDL, max(_MIN_MGDL, t60)))

        requested_horizon = b2_out["requested_horizon_minutes"]
        # No T+120 arm was fit for either adjustment (same limitation as
        # nlp_dish_identity_forecast) -- T+120 is the untouched B2 figure.
        selected_glucose = t60 if requested_horizon == 60.0 else b2_out["glucose_t120"]

        out = dict(b2_out)
        out["glucose_t60"] = round(t60, 1)
        out["glucose_range_t60"] = contract.glucose_range(t60)
        out["glucose_range"] = contract.glucose_range(selected_glucose)
        out["model_version"] = self.model_version
        out["is_trained_model"] = True
        out["experimental_label"] = self.label
        ppg_clause = (
            f"g0 came from a PPG estimate instead of the population fallback "
            f"(rppg/PPG_Dataset, {self._ppg['known_result']})"
            if working_meal.get("g0_source") == "ppg_estimate" else
            "PPG did not drive g0 this time (a direct measurement was available, "
            "or no PPG capture/model exists)"
        )
        out["basis"] = (
            f"EXPERIMENTAL, UNGATED (not the shipped default; "
            f"prediction_mode=experimental_fusion only). g0={g0:.1f} mg/dL "
            f"({working_meal.get('g0_detail', 'source unrecorded')}); {ppg_clause}. "
            f"T+60 = that g0, blended with a CGMacros carbs/macro adjustment "
            f"(weight {w_cv:.2f}, held-out MAE "
            f"{self._cgmacros['held_out_mae_mgdl_grouped_by_participant']}) and a "
            f"BIG IDEAs is_fried_cooking adjustment (weight {w_nlp:.2f}). None of "
            f"these has cleared the project's evidence gate; see "
            f"docs/INTEGRATION_AUDIT.md and the 2026-10-20 grill. T+120: no T+120 "
            f"arm exists for any adjustment -- served by B2's cohort mean "
            f"excursion instead."
        )

        errors = contract.validate_forecast_output({
            **out, "glucose_t60": selected_glucose,
            "glucose_range": contract.glucose_range(selected_glucose),
        })
        if errors:
            raise contract.ContractViolation(
                "predictor produced a non-compliant record: " + "; ".join(errors)
            )
        return out


def autoload_experimental_fusion_predictor() -> Optional[Predictor]:
    """Load the experimental_fusion artifact if one has been fit.

    Unlike the PPG/NLP autoloaders this one is NOT evidence-gated -- the
    artifact is written unconditionally by
    forecasting/evaluate/fit_experimental_fusion.py (2026-10-20 product
    decision to bypass the gate for this mode only). Absence just means
    nobody has run the fitter yet.
    """
    try:
        return ExperimentalFusionPredictor.load()
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return None


def autoload_nlp_dish_identity_predictor() -> Optional[Predictor]:
    """Load nlp.models.dish_identity_ridge's gated artifact if one exists.

    Mirrors integration.glucose_source.autoload_ppg_g0_estimator(): the
    trainer (nlp/models/dish_identity_ridge.py) writes its artifact only
    when the same 4-criterion gate rppg's PPG estimator uses has passed
    (ADR 0004), so the file's presence is itself the evidence. Absence is
    the expected state until someone has run the trainer and it earned
    the artifact — returns None, never raises, so api.py can treat this
    exactly like the PPG g0 path: a mode that is simply unavailable.

    This does NOT change default_predictor() — the mode this feeds
    (`nlp_dish_identity_forecast`) is explicit opt-in, never the default.
    """
    try:
        from nlp.models.dish_identity_ridge import NLPDishIdentityPredictor
        return NLPDishIdentityPredictor.load()
    except (ImportError, FileNotFoundError, KeyError, ValueError):
        return None
