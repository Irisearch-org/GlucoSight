"""Reproduce the extractor- and contract-side findings in
docs/PPG_CAMERA_CAPTURE_FINDINGS.md.

Covers:
  P1  a near-zero DC manufactures a large perfusion index out of nothing
  P2  the quality-gate arithmetic that down-weighted the paper control to 0.493
  P3  `pulse_rate_bpm` is quantised to the FFT grid (1.5 bpm at 30.3 Hz / 10 s)
  P6  the contract's declared range for `ppg_perfusion_index` is in the wrong
      units, so an impossible value passes validation

These import the shipped code (`rppg.features.extractor`,
`integration.contract`) rather than restating it. The findings marked
*(observation)* in the document — the five captures and the device
capabilities — have no in-repo reproduction.

    python -m rppg.check_ppg_findings
"""
from __future__ import annotations

import sys

import numpy as np

from rppg.features.extractor import (
    compute_perfusion_index,
    estimate_pulse_rate_bpm,
)
from integration.contract import (
    CONFIDENCE_FLOOR,
    MIN_MODALITY_WEIGHT,
    PPG_FIELDS,
    modality_weight,
)

FS = 30.3          # observed sampling rate of the browser capture, Hz
WINDOW_S = 10.0    # API window length
GREEN_DC = 0.0021  # measured green DC, finger over a lit torch (fraction of full scale)
RED_DC = 0.5501    # measured red DC, same geometry

_failures: list[str] = []


def check(label: str, ok: bool, detail: str) -> None:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}: {detail}")
    if not ok:
        _failures.append(label)


def heading(title: str) -> None:
    print(f"\n{title}")


# --------------------------------------------------------------------------
# P1 — near-zero DC, large quotient
# --------------------------------------------------------------------------
def p1_dark_frame_inflates_the_ratio() -> None:
    heading("P1  a dark frame inflates AC/DC through the denominator")
    n = int(FS * WINDOW_S)
    rng = np.random.default_rng(0)

    # Green: DC at the quantisation floor (~0.002 of full scale, i.e. half a
    # count out of 255) with sub-count noise. This is the geometry the page
    # used to be hardcoded to.
    green_raw = GREEN_DC + rng.normal(0.0, 0.00020, n)
    ac = green_raw - green_raw.mean()
    green_pi = compute_perfusion_index(ac, green_raw)

    # Red: same finger, same PPG, but a DC with light behind it.
    red_pulse = 6.0e-3 * np.sin(2 * np.pi * 1.0 * np.arange(n) / FS)   # ~2.5% of DC
    red_raw = RED_DC + red_pulse + rng.normal(0.0, 0.0004, n)
    red_ac = red_raw - red_raw.mean()
    red_pi = compute_perfusion_index(red_ac, red_raw)

    print(f"  green DC {GREEN_DC:.4f}  ->  perfusion_index {green_pi:9.4f} %")
    print(f"  red   DC {RED_DC:.4f}  ->  perfusion_index {red_pi:9.4f} %")
    check("P1 green ratio is far above physiological",
          green_pi > 20.0, f"{green_pi:.1f} % vs blood's 1-3 %")
    check("P1 red ratio is physiological",
          1.0 <= red_pi <= 3.0, f"{red_pi:.2f} %")
    check("P1 the two differ by more than an order of magnitude",
          green_pi / red_pi > 10, f"x{green_pi / red_pi:.0f}")


# --------------------------------------------------------------------------
# P2 — the quality gate down-weights but never excludes
# --------------------------------------------------------------------------
def p2_paper_was_downweighted_not_dropped() -> None:
    heading("P2  the paper control was down-weighted, never excluded (H1)")
    quality = 0.1309   # measured ppg_signal_quality for a folded sheet of paper
    record = {"ppg_present": 1, "ppg_signal_quality": quality}
    weight = modality_weight(record, "ppg")
    expected = MIN_MODALITY_WEIGHT + (1.0 - MIN_MODALITY_WEIGHT) * (quality / CONFIDENCE_FLOOR)
    print(f"  quality {quality} < floor {CONFIDENCE_FLOOR}")
    print(f"  weight = {MIN_MODALITY_WEIGHT} + {1.0 - MIN_MODALITY_WEIGHT} "
          f"* ({quality}/{CONFIDENCE_FLOOR}) = {weight:.4f}")
    check("P2 weight matches the documented 0.493", abs(weight - 0.493) < 0.001,
          f"{weight:.3f}")
    check("P2 an inert sample is never dropped", weight > 0.0,
          "H1: a hard zero is indistinguishable from a real zero")


# --------------------------------------------------------------------------
# P3 — the pulse rate is quantised to the FFT grid
# --------------------------------------------------------------------------
def p3_pulse_rate_is_quantised() -> None:
    heading("P3  pulse_rate_bpm snaps to the FFT grid")
    n = int(FS * WINDOW_S)
    nfft = n * 4
    step_bpm = (FS / nfft) * 60.0
    print(f"  {n} samples @ {FS} Hz, nfft {nfft} -> bin {FS / nfft:.4f} Hz "
          f"= {step_bpm:.3f} bpm")
    t = np.arange(n) / FS
    rows = []
    for true_bpm in (55.0, 58.0, 59.0, 60.0, 61.0, 62.0, 75.0):
        f0 = true_bpm / 60.0
        signal = np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
        reported = estimate_pulse_rate_bpm(signal, FS)
        on_grid = abs(reported / step_bpm - round(reported / step_bpm)) < 1e-6
        rows.append((true_bpm, reported, on_grid))
        print(f"  true {true_bpm:5.1f} bpm  ->  reported {reported:8.4f} bpm")
    check("P3 every reported rate lies on the 1.5 bpm grid",
          all(r[2] for r in rows),
          f"grid step {step_bpm:.3f} bpm; no sub-bin interpolation")


# --------------------------------------------------------------------------
# P6 — the declared range is in different units from the emitted value
# --------------------------------------------------------------------------
def p6_units_disagree_between_producer_and_contract() -> None:
    heading("P6  ppg_perfusion_index: contract range vs emitted units")
    field = next(f for f in PPG_FIELDS if f.name == "ppg_perfusion_index")
    print(f"  contract: lo={field.lo} hi={field.hi} fallback={field.fallback}")
    print(f"  extractor.compute_perfusion_index returns percent "
          f"(`ac_pp / dc * 100`)")
    print(f"  integration/api.py:239 divides by 100 -> ratio")

    # The capture from P1 in the document, as the service emitted it.
    emitted = 0.5319
    problem = field.check(emitted)
    check("P6 an impossible perfusion index passes validation",
          problem is None, f"{emitted} -> {problem!r}")
    # hi=20.0 read as a ratio means an AC/DC of 2000 %.
    also_passes = field.check(20.0)
    check("P6 the declared ceiling permits a 2000 % AC/DC ratio",
          also_passes is None, "hi=20.0 is a percent bound applied to a ratio")
    # And the extractor-side value, in the units the extractor actually uses.
    n = int(FS * WINDOW_S)
    rng = np.random.default_rng(0)
    raw = GREEN_DC + rng.normal(0.0, 0.00020, n)
    extractor_value = compute_perfusion_index(raw - raw.mean(), raw)
    print(f"  same dark frame, extractor units: {extractor_value:.2f} (percent)")
    print(f"  same dark frame, service units:   {extractor_value / 100:.4f} (ratio)")


def main() -> int:
    print("PPG capture findings - reproduction")
    print("See docs/PPG_CAMERA_CAPTURE_FINDINGS.md")
    p1_dark_frame_inflates_the_ratio()
    p2_paper_was_downweighted_not_dropped()
    p3_pulse_rate_is_quantised()
    p6_units_disagree_between_producer_and_contract()
    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED: {', '.join(_failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
