"""Boundary validation for utils/clarke_grid.py — closes finding M7.

The point of this file is that it CAN fail. The implementation it replaced
shipped a `__main__` sanity check that generated
`y_pred = y_true * U(0.85, 1.15)`, which is inside the ±20% Zone A band by
construction; it could not have detected a wrong boundary.

Every point in ZONE_TABLE is checked against the boundaries published in
Clarke et al. 1987. Points are chosen in pairs that straddle each dividing
line by ±1 mg/dL, so an off-by-one in any inequality fails a test.
"""

import numpy as np
import pytest

from utils.clarke_grid import (
    classify_zone,
    clarke_error_grid,
    per_participant_zone_a,
    zone_a_by_reference_range,
)

# (reference, prediction, expected_zone, why)
ZONE_TABLE = [
    # ---- Zone A: within ±20%, or both hypoglycaemic -------------------
    (100, 100, "A", "perfect prediction"),
    (100, 120, "A", "exactly +20% — inclusive upper edge"),
    (100, 80, "A", "exactly -20% — inclusive lower edge"),
    (250, 290, "A", "+16% at the high end is still A"),
    (70, 70, "A", "both at the hypo threshold"),
    (50, 60, "A", "both <= 70: the 20% band is not meaningful here"),
    (65, 72, "A", "+10.8%: A even though pred crosses 70 (old code said D)"),
    # ---- just outside Zone A ------------------------------------------
    (100, 121, "B", "+21% — one unit past the A edge"),
    (100, 79, "B", "-21% — one unit past the A edge"),

    # ---- Zone E: opposite treatment decisions -------------------------
    (200, 60, "E", "true hyper, predicted hypo"),
    (180, 70, "E", "on both E edges simultaneously"),
    (60, 200, "E", "true hypo, predicted hyper"),
    (70, 180, "E", "on both E edges from the other corner"),
    (179, 70, "B", "ref one unit below the E cut"),

    # ---- Zone C upper: pred >= ref + 110 ------------------------------
    (100, 210, "C", "exactly ref+110 — inclusive"),
    (100, 209, "B", "one unit inside — benign"),
    (290, 400, "C", "ref at the upper C limit"),
    (291, 401, "B", "ref past 290: published C region ends"),
    (70, 180, "E", "E is checked before C and wins this corner"),

    # ---- Zone C lower: the (130,0)-(180,70) line ----------------------
    (150, 28, "C", "exactly on (7/5)*150-182 = 28 — inclusive"),
    (150, 29, "B", "one unit above the line is NOT C (old code said C)"),
    (180, 70, "E", "E wins over lower-C at this corner"),
    (140, 60, "B", "the wedge the old rectangle wrongly called C"),
    (170, 50, "C", "below the lower-C line: (7/5)*170-182 = 56"),
    (170, 57, "B", "one unit above that line"),
    (131, 1, "C", "just inside the lower-C wedge near its (130,0) apex"),
    (131, 5, "B", "above the line at ref=131, where the wedge is 1.4 wide"),

    # ---- Zone D: failure to detect ------------------------------------
    (300, 150, "D", "true hyper, predicted normal"),
    (240, 70, "E", "ref>=180 and pred<=70: E is checked first and wins"),
    (240, 180, "D", "on the upper D edge"),
    (239, 179, "B", "one unit outside the high-D box"),
    (50, 100, "D", "true hypo missed, ref below 175/3"),
    (58, 71, "D", "ref just below 175/3 = 58.33"),
    (65, 79, "D", "58.33 <= ref <= 70 and pred >= 1.2*ref"),

    # ---- Zone B: benign errors ----------------------------------------
    (150, 200, "B", "+33%, not dangerous"),
    (200, 150, "B", "-25%, not dangerous"),
    (120, 90, "B", "-25% in the normal range"),
]


@pytest.mark.parametrize("ref,pred,expected,why", ZONE_TABLE)
def test_published_boundary_points(ref, pred, expected, why):
    got = classify_zone(ref, pred)
    assert got == expected, (
        f"ref={ref}, pred={pred}: expected Zone {expected} ({why}), got {got}"
    )


def test_zone_a_is_the_20_percent_band():
    """Zone A must be exactly ±20% above the hypo region."""
    rng = np.random.default_rng(0)
    refs = rng.uniform(71, 400, 3000)
    for ref in refs:
        assert classify_zone(ref, ref * 1.199) == "A"
        assert classify_zone(ref, ref * 0.801) == "A"
        assert classify_zone(ref, ref * 1.201) != "A"
        assert classify_zone(ref, ref * 0.799) != "A"


def test_every_point_gets_exactly_one_zone():
    for ref in range(20, 400, 7):
        for pred in range(20, 400, 7):
            assert classify_zone(ref, pred) in {"A", "B", "C", "D", "E"}


def test_perfect_prediction_is_100_percent_zone_a():
    y = np.array([80.0, 120.0, 200.0, 350.0, 65.0])
    zones, pct = clarke_error_grid(y, y)
    assert pct["A"] == 100.0
    assert zones["A"] == 5


def test_percentages_sum_to_100():
    rng = np.random.default_rng(7)
    y_true = rng.uniform(40, 400, 500)
    y_pred = rng.uniform(40, 400, 500)
    _, pct = clarke_error_grid(y_true, y_pred)
    assert abs(sum(pct.values()) - 100.0) < 0.05


# --------------------------------------------------------------------
# Regression pins. These numbers are quoted in the module docstring and in
# docs/INTEGRATION_AUDIT.md; if the implementation changes, they change,
# and both documents must be updated.
# --------------------------------------------------------------------

def _old_repo_zone(ref, pred):
    """The implementation this module replaced (pre-Sprint-2 utils)."""
    if (ref <= 70 and pred <= 70) or (ref >= 70 and abs(ref - pred) <= 0.20 * ref):
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if (ref >= 70 and ref <= 290 and pred >= ref + 110) or (
        ref >= 130 and ref <= 180 and pred <= 70
    ):
        return "C"
    if (ref >= 240 and pred >= 70 and pred <= 180) or (
        ref <= 70 and pred >= 70 and pred <= 180
    ):
        return "D"
    return "B"


def _notebook_zone(r, p):
    """`evaluate_clarke` from the Sprint 2 forecasting notebook."""
    if (p <= 1.2 * r and p >= 0.8 * r) or (r <= 70 and p <= 70):
        return "A"
    elif (r >= 180 and p <= 70) or (r <= 70 and p >= 180):
        return "E"
    elif (r <= 70 and p > 70 and p < 180) or (r >= 240 and p >= 70 and p <= 180):
        return "D"
    elif (r >= 70 and r <= 180 and p > 180) or (r >= 70 and r <= 180 and p < 70):
        return "C"
    else:
        return "B"


def _sweep(fn):
    refs = np.arange(20, 400, 1.0)
    preds = np.arange(20, 400, 1.0)
    disagree = 0
    zone_a_disagree = 0
    for r in refs:
        for p in preds:
            truth = classify_zone(r, p)
            other = fn(r, p)
            if truth != other:
                disagree += 1
            if (truth == "A") != (other == "A"):
                zone_a_disagree += 1
    return disagree, zone_a_disagree, len(refs) * len(preds)


def test_regression_against_old_implementation():
    disagree, zone_a_disagree, total = _sweep(_old_repo_zone)
    assert total == 144400
    assert disagree == 1734, "old-implementation delta changed — update the docstring"
    assert zone_a_disagree == 70


def test_notebook_zone_a_column_is_trustworthy():
    """The credibility claim in docs/INTEGRATION_AUDIT.md rests on this.

    The Sprint 2 notebook's Clarke function is wrong on 11% of the grid,
    but its Zone A membership is identical to the published grid on every
    cell. That is what lets the audit keep the notebook's Zone A column and
    reject its A+B / C / D / E columns.
    """
    disagree, zone_a_disagree, total = _sweep(_notebook_zone)
    assert total == 144400
    assert zone_a_disagree == 0, "notebook Zone A is NOT equivalent — audit claim is void"
    assert disagree == 15899, "notebook delta changed — update the audit"


# --------------------------------------------------------------------
# Reporting helpers required by finding C3
# --------------------------------------------------------------------

def test_per_participant_excludes_thin_participants():
    y_true = np.array([100.0] * 10 + [100.0, 100.0])
    y_pred = np.array([100.0] * 10 + [300.0, 300.0])
    pids = np.array(["P1"] * 10 + ["P2", "P2"])
    out = per_participant_zone_a(y_true, y_pred, pids, min_meals=5)
    assert out["n_participants"] == 1
    assert out["n_participants_excluded"] == 1
    assert out["median_zone_a"] == 100.0


def test_stratification_reports_empty_bins_as_none_not_zero():
    """"Zone A: 0.0%" and "no data here" must not share an encoding.

    DATA_STRATEGY §4.4 makes this point about Zone E; it applies to every
    empty stratum.
    """
    y_true = np.array([110.0, 120.0, 130.0])
    rows = zone_a_by_reference_range(y_true, y_true)
    by_range = {r["range"]: r for r in rows}
    assert by_range["[100, 140)"]["n"] == 3
    assert by_range["[100, 140)"]["zone_a_pct"] == 100.0
    assert by_range["[250, 10000)"]["n"] == 0
    assert by_range["[250, 10000)"]["zone_a_pct"] is None


def test_nan_input_is_rejected_not_dropped():
    y_true = np.array([100.0, 120.0, np.nan])
    y_pred = np.array([100.0, 120.0, 130.0])
    with pytest.raises(ValueError, match="NaN or inf"):
        clarke_error_grid(y_true, y_pred)


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        clarke_error_grid(np.array([1.0, 2.0]), np.array([1.0]))
