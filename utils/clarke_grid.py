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

Usage:
    from utils.clarke_grid import clarke_error_grid, plot_clarke_grid
    zones, percentages = clarke_error_grid(y_true, y_pred)
    plot_clarke_grid(y_true, y_pred, save_path="results/clarke.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Tuple, Dict


def clarke_error_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Compute Clarke Error Grid zone counts and percentages.

    Args:
        y_true: Ground truth glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)

    Returns:
        zones: Dict with counts per zone {A, B, C, D, E}
        percentages: Dict with percentage per zone
    """
    assert len(y_true) == len(y_pred), "Arrays must be same length"

    zones = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}

    for ref, pred in zip(y_true, y_pred):
        zone = _classify_zone(ref, pred)
        zones[zone] += 1

    total = len(y_true)
    percentages = {k: round(v / total * 100, 2) for k, v in zones.items()}

    return zones, percentages


def _classify_zone(ref: float, pred: float) -> str:
    """Classify a single (reference, prediction) pair into a Clarke zone."""

    # Zone A — clinically accurate
    if (ref <= 70 and pred <= 70) or \
       (ref >= 70 and abs(ref - pred) <= 0.20 * ref):
        return "A"

    # Zone E — erroneous treatment (most dangerous — check first)
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"

    # Zone C — overcorrection
    if (ref >= 70 and ref <= 290 and pred >= ref + 110) or \
       (ref >= 130 and ref <= 180 and pred <= 70):
        return "C"

    # Zone D — failure to detect
    if (ref >= 240 and pred >= 70 and pred <= 180) or \
       (ref <= 70 and pred >= 70 and pred <= 180):
        return "D"

    # Zone B — everything else within acceptable deviation
    return "B"


def plot_clarke_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None,
    title: str = "Clarke Error Grid"
) -> None:
    """
    Plot Clarke Error Grid with zone boundaries and scatter points.

    Args:
        y_true: Ground truth glucose values (mg/dL)
        y_pred: Predicted glucose values (mg/dL)
        save_path: If provided, save figure to this path
        title: Plot title
    """
    zones, percentages = clarke_error_grid(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)

    # Zone boundaries (simplified Clarke implementation)
    # Zone A boundaries
    ax.plot([0, 400], [0, 400], 'k-', linewidth=0.5)
    ax.plot([0, 175/3], [70, 70], 'k-', linewidth=0.5)
    ax.plot([175/3, 400], [70, 400], 'k-', linewidth=0.5)
    ax.plot([70, 70], [84, 400], 'k-', linewidth=0.5)
    ax.plot([0, 70], [180, 180], 'k-', linewidth=0.5)
    ax.plot([70, 290], [180, 400], 'k-', linewidth=0.5)
    ax.plot([70, 70], [0, 56], 'k-', linewidth=0.5)
    ax.plot([70, 400], [56, 320], 'k-', linewidth=0.5)
    ax.plot([180, 180], [0, 70], 'k-', linewidth=0.5)
    ax.plot([180, 400], [70, 70], 'k-', linewidth=0.5)
    ax.plot([240, 240], [70, 180], 'k-', linewidth=0.5)
    ax.plot([240, 400], [180, 180], 'k-', linewidth=0.5)
    ax.plot([130, 180], [0, 70], 'k-', linewidth=0.5)

    # Zone labels
    ax.text(30, 15, "A", fontsize=14, fontweight="bold", color="green")
    ax.text(370, 260, "A", fontsize=14, fontweight="bold", color="green")
    ax.text(30, 140, "B", fontsize=14, fontweight="bold", color="blue")
    ax.text(370, 120, "B", fontsize=14, fontweight="bold", color="blue")
    ax.text(160, 380, "C", fontsize=14, fontweight="bold", color="orange")
    ax.text(160, 20, "C", fontsize=14, fontweight="bold", color="orange")
    ax.text(30, 300, "D", fontsize=14, fontweight="bold", color="red")
    ax.text(340, 30, "D", fontsize=14, fontweight="bold", color="red")
    ax.text(30, 370, "E", fontsize=14, fontweight="bold", color="darkred")
    ax.text(370, 15, "E", fontsize=14, fontweight="bold", color="darkred")

    # Color scatter points by zone
    zone_colors = {
        "A": "green", "B": "blue", "C": "orange", "D": "red", "E": "darkred"
    }
    zone_labels_map = {}
    for ref, pred in zip(y_true, y_pred):
        zone = _classify_zone(ref, pred)
        color = zone_colors[zone]
        ax.scatter(ref, pred, c=color, alpha=0.5, s=20)
        zone_labels_map[zone] = color

    # Legend
    legend_patches = [
        mpatches.Patch(color=v, label=f"Zone {k}: {percentages[k]:.1f}%")
        for k, v in zone_colors.items()
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=10)

    # Threshold line
    pass_text = "PASS" if percentages["A"] >= 70 else "FAIL"
    pass_color = "green" if percentages["A"] >= 70 else "red"
    ax.set_title(
        f"{title}\nZone A: {percentages['A']:.1f}% — {pass_text} (target ≥ 70%)",
        fontsize=12,
        color=pass_color
    )
    ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=11)
    ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=11)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Clarke Error Grid saved to: {save_path}")

    plt.show()


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    horizon: str = "T+60"
) -> Dict:
    """
    Full evaluation report for a glucose prediction model.

    Returns dict with Clarke zones, RMSE, MAE, and pass/fail.
    """
    zones, percentages = clarke_error_grid(y_true, y_pred)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))

    zone_a_pass = percentages["A"] >= 70.0
    zone_ab_pass = (percentages["A"] + percentages["B"]) >= 90.0

    report = {
        "model": model_name,
        "horizon": horizon,
        "n_samples": len(y_true),
        "rmse_mgdl": round(rmse, 2),
        "mae_mgdl": round(mae, 2),
        "zone_A_pct": percentages["A"],
        "zone_B_pct": percentages["B"],
        "zone_C_pct": percentages["C"],
        "zone_D_pct": percentages["D"],
        "zone_E_pct": percentages["E"],
        "zone_AB_pct": round(percentages["A"] + percentages["B"], 2),
        "zone_A_pass": zone_a_pass,
        "zone_AB_pass": zone_ab_pass,
        "overall_pass": zone_a_pass and zone_ab_pass,
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — {horizon}")
    print(f"{'='*50}")
    print(f"  Samples : {report['n_samples']}")
    print(f"  RMSE    : {report['rmse_mgdl']:.2f} mg/dL")
    print(f"  MAE     : {report['mae_mgdl']:.2f} mg/dL")
    print(f"  Zone A  : {report['zone_A_pct']:.1f}%  {'✅ PASS' if zone_a_pass else '❌ FAIL'} (target ≥70%)")
    print(f"  Zone A+B: {report['zone_AB_pct']:.1f}%  {'✅ PASS' if zone_ab_pass else '❌ FAIL'} (target ≥90%)")
    print(f"  Zone C  : {report['zone_C_pct']:.1f}%")
    print(f"  Zone D  : {report['zone_D_pct']:.1f}%")
    print(f"  Zone E  : {report['zone_E_pct']:.1f}%")
    print(f"{'='*50}\n")

    return report


if __name__ == "__main__":
    # Quick sanity check
    np.random.seed(42)
    y_true = np.random.uniform(70, 300, 100)
    # Good predictions — within 20%
    y_pred = y_true * np.random.uniform(0.85, 1.15, 100)

    report = evaluate_model(y_true, y_pred, "Test Model", "T+60")
    plot_clarke_grid(y_true, y_pred, title="Test Clarke Grid")
