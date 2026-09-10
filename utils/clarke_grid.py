"""
Clarke Error Grid Analysis
--------------------------
Clinical evaluation standard for glucose prediction accuracy.
Used to evaluate all GlucoSight model outputs.

Zone A: Clinically accurate     — target > 70%
Zone B: Acceptable deviation    — target A+B > 90%
Zone C: Overcorrection          — dangerous
Zone D: Failure to detect       — dangerous
Zone E: Erroneous treatment     — most dangerous

Reference
---------
Clarke WL, Cox D, Gonder-Frederick LA, Carter W, Pohl SL.
"Evaluating clinical accuracy of systems for self-monitoring of blood
glucose." Diabetes Care. 1987;10(5):622-628.

Validation status (finding M7 — resolved)
-----------------------------------------
The zone boundaries below are the ones stated in the 1987 paper and used by
the reference implementations. They are pinned by
`utils/tests/test_clarke_grid.py`, which asserts a table of published
boundary points — including points that sit just inside and just outside
each dividing line. Do not edit a boundary without editing that table.

**What the previous implementation got wrong.** Both bugs were measured by
sweeping a 1 mg/dL grid over [20, 400]^2 and comparing zone assignments
(`utils/tests/test_clarke_grid.py::test_regression_against_old_implementation`
pins the counts):

1. Lower Zone C was the rectangle ``130 <= ref <= 180 and pred <= 70``.
   The published boundary is the *line* through (130, 0) and (180, 70),
   i.e. ``pred <= (7/5) * ref - 182``. The rectangle over-claimed C from B
   on 1664 grid cells — the old code reported dangerous overcorrection
   where the published grid says benign error.
2. Zone A used a ``ref >= 70`` guard on the ±20% band, so a reference in
   roughly [58, 70] with a prediction just above 70 fell through to D.
   70 grid cells, all A reported as D.

Net: 1734 of 144400 cells, 1.20% of the plotted area. Small, but it sits
entirely in the C/D reporting that the safety argument rests on.

**The notebook implementation is worse.** The `evaluate_clarke` in
`forecasting/notebooks/` (Sprint 2) disagrees with the published grid on
11.01% of the area: it calls any ``70 <= ref <= 180`` with ``pred > 180``
Zone C, which over-claims C from B on 9904 cells, and it has no
``pred >= ref + 110`` rule, which misses genuine Zone C on 5995 cells.
**Its Zone A column is nonetheless exactly correct** — Zone-A membership
is identical on all 144400 cells, because ±20% is the one boundary it got
right. So Sprint 2's Zone A numbers stand; its A+B, C, D and E numbers do
not and must be recomputed with this module.

Usage:
    from utils.clarke_grid import clarke_error_grid, plot_clarke_grid
    zones, percentages = clarke_error_grid(y_true, y_pred)
    plot_clarke_grid(y_true, y_pred, save_path="results/clarke.png")
"""

from typing import Dict, Tuple

import numpy as np

__all__ = [
    "clarke_error_grid",
    "classify_zone",
    "plot_clarke_grid",
    "evaluate_model",
    "per_participant_zone_a",
    "zone_a_by_reference_range",
]

# The three irrational-looking constants below are published boundary
# values, not tuning knobs.
_D_LOW_REF = 175.0 / 3.0      # ≈ 58.33 — where the lower-D wedge starts
_C_LOW_SLOPE = 7.0 / 5.0      # lower-C dividing line through (130,0)-(180,70)
_C_LOW_INTERCEPT = 182.0
_D_LOW_SLOPE = 6.0 / 5.0      # lower-D diagonal


def classify_zone(ref: float, pred: float) -> str:
    """Classify one (reference, prediction) pair into a Clarke zone.

    Order matters: the zones overlap as written in the paper, and the
    published convention resolves an overlap in favour of the more
    dangerous zone. A before E before C before D before B.
    """
    ref = float(ref)
    pred = float(pred)

    # Zone A — clinically accurate: within ±20%, or both in the
    # hypoglycaemic region where a 20% band is not meaningful.
    if (ref <= 70.0 and pred <= 70.0) or (0.8 * ref <= pred <= 1.2 * ref):
        return "A"

    # Zone E — erroneous treatment: the reading and the truth are on
    # opposite sides of the treatment decision.
    if (ref >= 180.0 and pred <= 70.0) or (ref <= 70.0 and pred >= 180.0):
        return "E"

    # Zone C — overcorrection.
    if (70.0 <= ref <= 290.0 and pred >= ref + 110.0) or (
        130.0 <= ref <= 180.0 and pred <= _C_LOW_SLOPE * ref - _C_LOW_INTERCEPT
    ):
        return "C"

    # Zone D — failure to detect.
    if (
        (ref >= 240.0 and 70.0 <= pred <= 180.0)
        or (ref <= _D_LOW_REF and 70.0 <= pred <= 180.0)
        or (_D_LOW_REF <= ref <= 70.0 and pred >= _D_LOW_SLOPE * ref)
    ):
        return "D"

    # Zone B — benign error: everything the dangerous zones did not claim.
    return "B"


# Kept as a private alias: older track code imports `_classify_zone`.
_classify_zone = classify_zone


def clarke_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Compute Clarke Error Grid zone counts and percentages.

    Args:
        y_true: Ground truth glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)

    Returns:
        zones: Dict with counts per zone {A, B, C, D, E}
        percentages: Dict with percentage per zone
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("cannot compute a Clarke grid over zero samples")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError(
            "y_true/y_pred contain NaN or inf — a Clarke grid computed over "
            "silently-dropped samples is not comparable to one that is not"
        )

    zones = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    for ref, pred in zip(y_true, y_pred):
        zones[classify_zone(ref, pred)] += 1

    total = len(y_true)
    percentages = {k: round(v / total * 100, 2) for k, v in zones.items()}
    return zones, percentages


def per_participant_zone_a(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    participant_ids: np.ndarray,
    min_meals: int = 5,
) -> Dict[str, float]:
    """Zone A per participant, summarised as median and IQR.

    Finding C3 requires this instead of the pooled figure: pooled Zone A is
    dominated by whoever contributed the most meals. Participants with fewer
    than `min_meals` meals are excluded and counted, because a Zone A
    computed over two meals only takes the values 0, 50 or 100 and widens
    the IQR for no reason.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    participant_ids = np.asarray(participant_ids).ravel()

    per_participant = []
    n_excluded = 0
    for pid in np.unique(participant_ids):
        mask = participant_ids == pid
        if mask.sum() < min_meals:
            n_excluded += 1
            continue
        _, pct = clarke_error_grid(y_true[mask], y_pred[mask])
        per_participant.append(pct["A"])

    if not per_participant:
        raise ValueError(
            f"no participant has at least {min_meals} meals — "
            "a per-participant summary is not defined here"
        )

    arr = np.asarray(per_participant, dtype=float)
    return {
        "median_zone_a": round(float(np.median(arr)), 2),
        "iqr_zone_a": round(
            float(np.percentile(arr, 75) - np.percentile(arr, 25)), 2
        ),
        "min_zone_a": round(float(arr.min()), 2),
        "max_zone_a": round(float(arr.max()), 2),
        "n_participants": int(len(arr)),
        "n_participants_excluded": int(n_excluded),
        "min_meals": int(min_meals),
    }


# Reference-glucose strata. Zone A is a ±20% band, so its width in mg/dL
# scales with the reference value: ±18 mg/dL at 90, ±50 at 250. Reporting a
# single pooled Zone A hides the fact that the metric is far harsher at the
# bottom of the range. Finding C3 and DATA_STRATEGY §4.4.
_DEFAULT_RANGES = ((0, 100), (100, 140), (140, 180), (180, 250), (250, 10_000))


def zone_a_by_reference_range(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ranges=_DEFAULT_RANGES,
) -> list:
    """Zone A stratified by reference glucose range."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    rows = []
    for low, high in ranges:
        mask = (y_true >= low) & (y_true < high)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {"range": f"[{low}, {high})", "n": 0,
                 "zone_a_pct": None, "zone_ab_pct": None}
            )
            continue
        _, pct = clarke_error_grid(y_true[mask], y_pred[mask])
        rows.append({
            "range": f"[{low}, {high})",
            "n": n,
            "zone_a_pct": pct["A"],
            "zone_ab_pct": round(pct["A"] + pct["B"], 2),
        })
    return rows


def plot_clarke_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None,
    title: str = "Clarke Error Grid",
    show: bool = False,
) -> None:
    """Plot the Clarke Error Grid with published zone boundaries."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    zones, percentages = clarke_error_grid(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)

    # Published boundaries. These mirror `classify_zone` exactly — if you
    # change one, change both, and update the boundary test.
    ax.plot([0, 400], [0, 400], "k:", linewidth=0.8)          # identity
    ax.plot([0, 175 / 3], [70, 70], "k-", linewidth=0.8)
    ax.plot([175 / 3, 400 / 1.2], [70, 400], "k-", linewidth=0.8)
    ax.plot([70, 70], [84, 400], "k-", linewidth=0.8)
    ax.plot([0, 70], [180, 180], "k-", linewidth=0.8)
    ax.plot([70, 290], [180, 400], "k-", linewidth=0.8)
    ax.plot([70, 70], [0, 56], "k-", linewidth=0.8)
    ax.plot([70, 400], [56, 320], "k-", linewidth=0.8)
    ax.plot([180, 180], [0, 70], "k-", linewidth=0.8)
    ax.plot([180, 400], [70, 70], "k-", linewidth=0.8)
    ax.plot([240, 240], [70, 180], "k-", linewidth=0.8)
    ax.plot([240, 400], [180, 180], "k-", linewidth=0.8)
    ax.plot([130, 180], [0, 70], "k-", linewidth=0.8)

    for x, y, label in [
        (30, 15, "A"), (370, 260, "A"), (30, 150, "B"), (370, 120, "B"),
        (160, 380, "C"), (160, 20, "C"), (30, 300, "D"), (340, 30, "D"),
        (30, 370, "E"), (370, 15, "E"),
    ]:
        ax.text(x, y, label, fontsize=13, fontweight="bold", color="#444")

    zone_colors = {
        "A": "#2e7d32", "B": "#1565c0", "C": "#ef6c00",
        "D": "#c62828", "E": "#6a1b1a",
    }
    point_zones = np.array([classify_zone(r, p) for r, p in zip(y_true, y_pred)])
    for zone, color in zone_colors.items():
        mask = point_zones == zone
        if mask.any():
            ax.scatter(y_true[mask], y_pred[mask], c=color, alpha=0.5, s=18)

    ax.legend(
        handles=[
            mpatches.Patch(color=v, label=f"Zone {k}: {percentages[k]:.1f}% (n={zones[k]})")
            for k, v in zone_colors.items()
        ],
        loc="upper left",
        fontsize=9,
    )

    pass_text = "PASS" if percentages["A"] >= 70 else "BELOW TARGET"
    pass_color = "#2e7d32" if percentages["A"] >= 70 else "#c62828"
    ax.set_title(
        f"{title}\nZone A: {percentages['A']:.1f}% — {pass_text} (target ≥ 70%)   n={len(y_true)}",
        fontsize=11,
        color=pass_color,
    )
    ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=11)
    ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Clarke Error Grid saved to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    horizon: str = "T+60",
    participant_ids: np.ndarray = None,
    verbose: bool = True,
) -> Dict:
    """Full evaluation report for a glucose prediction model.

    A Zone A figure without its baselines beside it is not a result
    (finding C3). This function evaluates one predictor; it is the caller's
    job to put B0/B1/B2 in the same table. `integration/report.py` does that.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    zones, percentages = clarke_error_grid(y_true, y_pred)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    report = {
        "model": model_name,
        "horizon": horizon,
        "n_samples": int(len(y_true)),
        "rmse_mgdl": round(rmse, 2),
        "mae_mgdl": round(mae, 2),
        "zone_A_pct": percentages["A"],
        "zone_B_pct": percentages["B"],
        "zone_C_pct": percentages["C"],
        "zone_D_pct": percentages["D"],
        "zone_E_pct": percentages["E"],
        "zone_AB_pct": round(percentages["A"] + percentages["B"], 2),
        "zone_A_pass": percentages["A"] >= 70.0,
        "zone_AB_pass": (percentages["A"] + percentages["B"]) >= 90.0,
    }
    report["overall_pass"] = report["zone_A_pass"] and report["zone_AB_pass"]

    if participant_ids is not None:
        report.update(per_participant_zone_a(y_true, y_pred, participant_ids))

    if verbose:
        print(f"\n{'=' * 56}")
        print(f"  {model_name} — {horizon}")
        print(f"{'=' * 56}")
        print(f"  Samples : {report['n_samples']}")
        print(f"  RMSE    : {report['rmse_mgdl']:.2f} mg/dL")
        print(f"  MAE     : {report['mae_mgdl']:.2f} mg/dL")
        print(f"  Zone A  : {report['zone_A_pct']:.1f}%  "
              f"{'PASS' if report['zone_A_pass'] else 'BELOW TARGET'} (>=70%)")
        print(f"  Zone A+B: {report['zone_AB_pct']:.1f}%  "
              f"{'PASS' if report['zone_AB_pass'] else 'BELOW TARGET'} (>=90%)")
        print(f"  Zone C/D/E: {report['zone_C_pct']:.1f}% / "
              f"{report['zone_D_pct']:.1f}% / {report['zone_E_pct']:.1f}%")
        if participant_ids is not None:
            print(f"  Per-participant Zone A: median {report['median_zone_a']:.1f}% "
                  f"(IQR {report['iqr_zone_a']:.1f}), n={report['n_participants']} participants")
        print("  NOTE: this number is not a result until B0/B1/B2 sit beside it (C3).")
        print(f"{'=' * 56}\n")

    return report


if __name__ == "__main__":
    # Deliberately NOT a self-test: the old __main__ generated
    # y_pred = y_true * U(0.85, 1.15), which is inside the ±20% Zone A band
    # by construction and therefore could not fail (finding M7).
    # Correctness is asserted in utils/tests/test_clarke_grid.py against
    # published boundary points. This block only demonstrates the API.
    rng = np.random.default_rng(42)
    y_true = rng.uniform(70, 300, 200)
    y_pred = y_true + rng.normal(0, 35, 200)   # a real error model
    evaluate_model(y_true, y_pred, "Demo (Gaussian error, sigma=35)", "T+60")
    print("Run `python -m pytest utils/tests/test_clarke_grid.py` to verify boundaries.")
