"""Where the pre-meal glucose value `g0` comes from, and how much to trust it.

Motivation
----------
The root `CLAUDE.md` promises prediction "no CGM required". The Sprint 2
result does not deliver that: its `g0` is read off the Dexcom G6 trace,
sampled every 5 minutes, so every prediction stands on a CGM reading taken
minutes before the first bite. See `docs/INTEGRATION_AUDIT.md` F13.

This module makes the source of `g0` explicit and pluggable, so a user with
a cheap glucometer and a user with neither are handled as the different
cases they are, and so the paper can report accuracy stratified by source.

It is the same discipline the contract already applies to `carbs_source`
and `nlp_feature_source`: what the model consumed travels downstream rather
than disappearing into a number.

Resolution order
----------------
1. Any DIRECT measurement (CGM trace or a user-entered fingerstick),
   freshest first. Both are real glucose measurements; recency decides.
2. A PPG-derived estimate — **only** if explicitly enabled AND a real
   estimator is registered. Disabled by default, see below.
3. The population fallback.

Why the PPG path is disabled by default
---------------------------------------
Two independent reasons, both of which must be cleared before it ships:

**Contract.** Ring-fence rule E1.2 states `ppg_glucose_estimate` is "never
used as a sole prediction and never surfaced to a user". Using it as `g0`
in `forecast = g0 + excursion` makes it the sole driver of a user-visible
number. That is a contract violation, not an implementation detail. It
needs a v2.0 amendment — the Interface Contract meeting is the place.

**Evidence.** The only committed PPG-glucose model
(`rppg/models/baseline_mean.py`) predicts the training-fold population mean
for every subject: MAE 14.51 +/- 2.00 mg/dL over 23 subjects. Predicting one
constant for everyone is, in information terms, identical to the population
fallback this module already has. Registering it would change the label on
the number, not the number. `G0Estimator.provides_information` records that
distinction so the code cannot lose it.

When the PPG track has an estimator that beats its own mean baseline on
held-out subjects, register it here and the path lights up.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Protocol, Sequence

from integration import contract

# --- source identifiers -----------------------------------------------
SOURCE_CGM = "cgm"
SOURCE_FINGERSTICK = "fingerstick"
SOURCE_PPG_ESTIMATE = "ppg_estimate"
SOURCE_POPULATION_FALLBACK = "population_fallback"

VALID_G0_SOURCES = frozenset({
    SOURCE_CGM, SOURCE_FINGERSTICK, SOURCE_PPG_ESTIMATE,
    SOURCE_POPULATION_FALLBACK,
})

DIRECT_MEASUREMENT_SOURCES = frozenset({SOURCE_CGM, SOURCE_FINGERSTICK})

# Plausible range for a user-entered glucometer reading, mg/dL. Wider than
# the contract's 70-400 prediction range on purpose: a real hypo of 45 or a
# real hyper of 500 must be accepted, because rejecting a true extreme is
# worse than accepting a typo. Outside this, treat it as a typo.
FINGERSTICK_MIN_MGDL = 20.0
FINGERSTICK_MAX_MGDL = 600.0

# Trust multipliers applied to prediction confidence, by source.
#
# fingerstick 1.00 — capillary blood is the reference a glucometer is
#   calibrated against. ISO 15197:2013 requires 95% of readings within
#   +/-15 mg/dL (below 100) or +/-15% (at or above 100).
# cgm 0.95 — interstitial fluid lags blood by roughly 5-10 minutes, and
#   DATA_STRATEGY §5.5 records that CGMacros' two CGMs disagree by 15-20
#   mg/dL at postprandial levels. That is this project's own measured noise
#   floor and it is the same order as the effect being hunted.
# ppg_estimate 0.30 — placeholder. Not defensible until a registered
#   estimator has held-out numbers; the gate below stops it being used
#   before then.
# population_fallback 0.10 — no personalisation at all.
SOURCE_TRUST = {
    SOURCE_FINGERSTICK: 1.00,
    SOURCE_CGM: 0.95,
    SOURCE_PPG_ESTIMATE: 0.30,
    SOURCE_POPULATION_FALLBACK: 0.10,
}


class G0Estimator(Protocol):
    """A model that estimates current glucose from PPG features."""

    model_version: str
    #: False when the estimator returns a constant (a mean baseline). Such
    #: an estimator carries no more information than the population
    #: fallback, and `resolve_g0` refuses to prefer it over that fallback.
    provides_information: bool

    def estimate(self, ppg_block: Dict[str, Any]) -> Optional[float]: ...


_REGISTERED_ESTIMATOR: Optional[G0Estimator] = None


def register_ppg_g0_estimator(estimator: Optional[G0Estimator]) -> None:
    """Install the PPG-derived g0 estimator. Pass None to remove it.

    Registering is necessary but not sufficient: `resolve_g0` also requires
    `allow_ppg_estimate=True` from the caller, so enabling this path is
    always a deliberate, visible decision rather than a config default.
    """
    global _REGISTERED_ESTIMATOR
    if estimator is not None:
        for attr in ("model_version", "provides_information", "estimate"):
            if not hasattr(estimator, attr):
                raise TypeError(
                    f"g0 estimator is missing {attr!r}; it must satisfy the "
                    f"G0Estimator protocol"
                )
    _REGISTERED_ESTIMATOR = estimator


def registered_ppg_g0_estimator() -> Optional[G0Estimator]:
    return _REGISTERED_ESTIMATOR


@dataclass(frozen=True)
class G0Resolution:
    """The resolved pre-meal glucose value and everything about its origin."""

    value_mgdl: float
    source: str
    age_minutes: Optional[float]
    is_direct_measurement: bool
    trust: float
    detail: str
    rejected: tuple = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "g0_mgdl": self.value_mgdl,
            "g0_source": self.source,
            "g0_age_minutes": self.age_minutes,
            "g0_is_direct_measurement": self.is_direct_measurement,
            "g0_trust": self.trust,
            "g0_detail": self.detail,
            "g0_rejected_candidates": list(self.rejected),
        }


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def validate_fingerstick(
    value_mgdl: Any,
    timestamp: Any,
    t0_timestamp: Any,
) -> Optional[Dict[str, Any]]:
    """Validate a user-entered glucometer reading.

    Returns a candidate dict, or raises ContractViolation with a message
    written for a person to read, because a person typed this value.
    """
    try:
        value = float(value_mgdl)
    except (TypeError, ValueError):
        raise contract.ContractViolation(
            f"fingerstick reading {value_mgdl!r} is not a number"
        )

    if not (FINGERSTICK_MIN_MGDL <= value <= FINGERSTICK_MAX_MGDL):
        raise contract.ContractViolation(
            f"fingerstick reading {value:g} mg/dL is outside the plausible "
            f"range [{FINGERSTICK_MIN_MGDL:.0f}, {FINGERSTICK_MAX_MGDL:.0f}]. "
            f"If the meter reads in mmol/L, multiply by 18."
        )

    t0 = _naive(contract._parse_ts(t0_timestamp))
    ts = _naive(contract._parse_ts(timestamp))

    if ts >= t0:
        # Same rule as C2, and for the same reason: a reading taken after
        # the first bite is a postprandial value, not a pre-meal one.
        raise contract.ContractViolation(
            f"C2: fingerstick timestamp {ts} is at or after t0={t0}. A reading "
            f"taken after the first bite is a prediction target, not an input."
        )

    return {
        "value": value,
        "timestamp": ts,
        "age_minutes": (t0 - ts).total_seconds() / 60.0,
        "source": SOURCE_FINGERSTICK,
    }


def resolve_g0(
    *,
    t0_timestamp,
    causal_history: Optional[Sequence[Dict[str, Any]]] = None,
    fingerstick_mgdl: Optional[float] = None,
    fingerstick_timestamp: Optional[Any] = None,
    ppg_block: Optional[Dict[str, Any]] = None,
    allow_ppg_estimate: bool = False,
    fallback_mgdl: float = 120.0,
    max_age_minutes: float = contract.MAX_G0_AGE_MINUTES,
) -> G0Resolution:
    """Pick the pre-meal glucose value and record where it came from.

    Args:
        causal_history: readings already filtered to strictly before t0.
        fingerstick_mgdl / fingerstick_timestamp: a user-entered glucometer
            reading. Validated by `validate_fingerstick`.
        ppg_block: a contract PPG block, used only if the PPG path is both
            enabled and backed by a registered informative estimator.
        allow_ppg_estimate: must be explicitly True. Defaults to False
            because using a PPG estimate as g0 conflicts with ring-fence
            rule E1.2 until contract v2.0 says otherwise.
    """
    t0 = _naive(contract._parse_ts(t0_timestamp))
    candidates = []
    rejected = []

    # --- direct measurements -------------------------------------------
    for entry in causal_history or []:
        ts = _naive(contract._parse_ts(entry["timestamp"]))
        candidates.append({
            "value": float(entry["value"]),
            "timestamp": ts,
            "age_minutes": (t0 - ts).total_seconds() / 60.0,
            "source": entry.get("source", SOURCE_CGM),
        })

    if fingerstick_mgdl is not None:
        if fingerstick_timestamp is None:
            raise contract.ContractViolation(
                "a fingerstick reading needs a timestamp — without one its "
                "age cannot be checked, and a stale reading is not a "
                "pre-meal value"
            )
        candidates.append(
            validate_fingerstick(fingerstick_mgdl, fingerstick_timestamp, t0)
        )

    usable = []
    for c in candidates:
        if c["age_minutes"] > max_age_minutes:
            rejected.append(
                f"{c['source']} reading of {c['value']:.0f} mg/dL rejected: "
                f"{c['age_minutes']:.0f} min old, limit {max_age_minutes:.0f}"
            )
        else:
            usable.append(c)

    if usable:
        best = min(usable, key=lambda c: c["age_minutes"])
        return G0Resolution(
            value_mgdl=best["value"],
            source=best["source"],
            age_minutes=round(best["age_minutes"], 1),
            is_direct_measurement=True,
            trust=SOURCE_TRUST.get(best["source"], 0.5),
            detail=(
                f"{best['source']} reading of {best['value']:.0f} mg/dL taken "
                f"{best['age_minutes']:.0f} min before the first bite"
            ),
            rejected=tuple(rejected),
        )

    # --- PPG-derived estimate ------------------------------------------
    if allow_ppg_estimate and ppg_block and ppg_block.get("ppg_present"):
        estimator = _REGISTERED_ESTIMATOR
        if estimator is None:
            rejected.append(
                "PPG g0 path enabled but no estimator is registered — call "
                "register_ppg_g0_estimator() first"
            )
        elif not estimator.provides_information:
            # A mean baseline returns one constant for everyone. Preferring
            # it over the population fallback would label a fallback as a
            # measurement, which is the one thing this module exists to stop.
            rejected.append(
                f"PPG estimator {estimator.model_version!r} is a constant "
                f"predictor (provides_information=False); it carries no more "
                f"information than the population fallback, so the fallback "
                f"is used and labelled honestly"
            )
        else:
            estimate = estimator.estimate(ppg_block)
            if estimate is None:
                rejected.append(
                    f"PPG estimator {estimator.model_version!r} declined to "
                    f"produce an estimate for this capture"
                )
            else:
                return G0Resolution(
                    value_mgdl=float(estimate),
                    source=SOURCE_PPG_ESTIMATE,
                    age_minutes=0.0,
                    is_direct_measurement=False,
                    trust=SOURCE_TRUST[SOURCE_PPG_ESTIMATE],
                    detail=(
                        f"estimated from the contact-PPG capture by "
                        f"{estimator.model_version}. NOT a glucose "
                        f"measurement — an inference from a pulse waveform."
                    ),
                    rejected=tuple(rejected),
                )
    elif ppg_block and ppg_block.get("ppg_present") and not allow_ppg_estimate:
        rejected.append(
            "a PPG capture is available but the PPG g0 path is disabled "
            "(ring-fence E1.2 — see integration/glucose_source.py)"
        )

    # --- population fallback -------------------------------------------
    return G0Resolution(
        value_mgdl=float(fallback_mgdl),
        source=SOURCE_POPULATION_FALLBACK,
        age_minutes=None,
        is_direct_measurement=False,
        trust=SOURCE_TRUST[SOURCE_POPULATION_FALLBACK],
        detail=(
            f"no usable glucose measurement — cohort fallback of "
            f"{fallback_mgdl:.0f} mg/dL. This prediction is a population "
            f"constant and is not personalised."
        ),
        rejected=tuple(rejected),
    )


# ---------------------------------------------------------------------
# The estimator the PPG track has today, wrapped so it can be registered
# and so the rest of the system can see what it is.
# ---------------------------------------------------------------------

class MeanBaselineG0Estimator:
    """Wraps `rppg/models/baseline_mean.py` — predicts one constant.

    Present so the seam can be exercised end to end, and flagged with
    `provides_information = False` so `resolve_g0` will not prefer it over
    the population fallback. This is not a limitation to work around; it is
    the accurate description of a mean baseline.
    """

    model_version = "ppg-glucose-mean-baseline-v1"
    provides_information = False

    #: Training-fold means reported in rppg/models/reports/baseline_mean.md
    #: (113.2-116.3 mg/dL across five folds).
    COHORT_MEAN_MGDL = 115.0

    def estimate(self, ppg_block: Dict[str, Any]) -> Optional[float]:
        if not ppg_block.get("ppg_present"):
            return None
        return self.COHORT_MEAN_MGDL
