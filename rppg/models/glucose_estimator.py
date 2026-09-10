"""PPG -> glucose estimator: train, evaluate against the baseline, and gate.

This is the thing that has to work before the app can offer a finger scan to
a user with no glucometer (`integration/glucose_source.py`).

Read the dataset-adequacy warning before running it
---------------------------------------------------
Predicting a single constant for every subject already achieves **Zone A
85.1%** on this dataset (n=67 recordings, 23 subjects, GroupKFold by
subject; computed from `reports/baseline_mean_predictions.csv`). The project
target is Zone A > 70%.

So on this dataset a Clarke Zone A figure **cannot distinguish a working
model from a constant**. The cause is the cohort, not the metric: glucose
runs 88-183 mg/dL with a median of 110 and an SD of 18.6, and Zone A is
+/-20%, which at 110 mg/dL is +/-22 mg/dL — wider than the entire label SD.

Consequences, both enforced below:

* **Zone A is reported but is not the gate.** The gate is MAE against the
  mean baseline on held-out subjects.
* **Beating the baseline is checked, not assumed.** Five folds over 23
  subjects is a small experiment; a lucky split looks exactly like success.
  So the gate requires a per-fold win count, a bootstrap CI on the MAE
  difference that excludes zero, and a label-permutation test.

If the gate fails, nothing is persisted and the finger-scan path stays off.
That is a valid, publishable outcome for this dataset — see the honesty
note in `agents/rppg/CLAUDE.md` bounding any claim to n=23 subjects.

Usage
-----
    python -m rppg.models.glucose_estimator --train

Requires `rppg/PPG_Dataset/RawData/*.mat`, which is gitignored (human-
subjects data, public repo, licence unconfirmed). The command fails with a
clear message if the signals are not in place.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(_HERE, "ppg_glucose_ridge.json")
REPORT_PATH = os.path.join(_HERE, "reports", "glucose_estimator.md")

# Features from rppg.features.extractor. Seven scalars for 67 recordings, so
# Ridge is the only defensible model class here — a forest on this many rows
# memorises the training subjects.
FEATURE_NAMES: Tuple[str, ...] = (
    "pulse_rate_bpm",
    "hrv_rmssd_ms",
    "perfusion_index",
    "signal_quality",
    "rise_time_ms",
    "pulse_width_half_ms",
    "n_beats_detected",
)

N_SPLITS = 5
RANDOM_STATE = 42
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 1000

# Gate thresholds. Deliberately strict: the cost of a false positive here is
# an app telling a diabetic their glucose based on a pulse waveform.
MIN_FOLD_WINS = 4          # of N_SPLITS
MAX_PERMUTATION_P = 0.05
REQUIRE_BOOTSTRAP_CI_EXCLUDES_ZERO = True


class DatasetUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------

def build_feature_table() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Extract features for every recording. Returns (X, y, groups, dropped)."""
    try:
        from rppg.data.loader import load_all_recordings
        from rppg.features.extractor import extract_features
    except ImportError as exc:
        raise DatasetUnavailable(f"rppg package not importable: {exc}") from exc

    try:
        records = load_all_recordings()
    except FileNotFoundError as exc:
        raise DatasetUnavailable(
            f"PPG raw signals not found: {exc}\n"
            f"rppg/PPG_Dataset/RawData/ is gitignored (human-subjects data in a "
            f"public repo, licence unconfirmed). Restore it out-of-band, then "
            f"re-run."
        ) from exc

    if not records:
        raise DatasetUnavailable(
            "no recordings loaded — rppg/PPG_Dataset/RawData/ is empty"
        )

    rows, ys, groups, dropped = [], [], [], []
    for rec in records:
        feats = extract_features(rec["signal"], rec["fs"])
        vec = [feats.get(name, np.nan) for name in FEATURE_NAMES]
        if not np.isfinite(vec).all():
            # A recording whose features could not be computed is not a
            # recording with zero heart rate. Drop it and say so.
            dropped.append(
                f"subject {rec['subject_id']} rec {rec['recording_id']}: "
                f"non-finite features"
            )
            continue
        rows.append(vec)
        ys.append(float(rec["glucose_mgdl"]))
        groups.append(rec["subject_id"])

    return (np.asarray(rows, dtype=float), np.asarray(ys, dtype=float),
            np.asarray(groups), dropped)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    n_test: int
    n_test_subjects: int
    baseline_mae: float
    model_mae: float

    @property
    def improvement(self) -> float:
        return self.baseline_mae - self.model_mae

    @property
    def model_wins(self) -> bool:
        return self.model_mae < self.baseline_mae


@dataclass
class Evaluation:
    folds: List[FoldResult] = field(default_factory=list)
    y_true: np.ndarray = None
    y_pred: np.ndarray = None
    y_baseline: np.ndarray = None
    groups: np.ndarray = None
    permutation_p: float = None
    bootstrap_ci: Tuple[float, float] = None
    dropped: List[str] = field(default_factory=list)

    @property
    def fold_wins(self) -> int:
        return sum(f.model_wins for f in self.folds)

    @property
    def baseline_mae(self) -> float:
        return float(np.mean(np.abs(self.y_true - self.y_baseline)))

    @property
    def model_mae(self) -> float:
        return float(np.mean(np.abs(self.y_true - self.y_pred)))

    @property
    def improvement(self) -> float:
        return self.baseline_mae - self.model_mae

    def passes_gate(self) -> Tuple[bool, List[str]]:
        """The evidence gate. Returns (passed, reasons_for_each_criterion)."""
        reasons = []
        ok = True

        if self.improvement <= 0:
            ok = False
            reasons.append(
                f"FAIL: model MAE {self.model_mae:.2f} is not better than the "
                f"mean baseline {self.baseline_mae:.2f} mg/dL"
            )
        else:
            reasons.append(
                f"pass: model MAE {self.model_mae:.2f} beats baseline "
                f"{self.baseline_mae:.2f} by {self.improvement:.2f} mg/dL"
            )

        if self.fold_wins < MIN_FOLD_WINS:
            ok = False
            reasons.append(
                f"FAIL: model beat the baseline in only {self.fold_wins}/"
                f"{len(self.folds)} folds (need {MIN_FOLD_WINS})"
            )
        else:
            reasons.append(
                f"pass: model beat the baseline in {self.fold_wins}/"
                f"{len(self.folds)} folds"
            )

        if self.bootstrap_ci is not None:
            lo, hi = self.bootstrap_ci
            if REQUIRE_BOOTSTRAP_CI_EXCLUDES_ZERO and lo <= 0 <= hi:
                ok = False
                reasons.append(
                    f"FAIL: bootstrap 95% CI on the MAE improvement "
                    f"[{lo:.2f}, {hi:.2f}] includes zero"
                )
            else:
                reasons.append(
                    f"pass: bootstrap 95% CI [{lo:.2f}, {hi:.2f}] excludes zero"
                )

        if self.permutation_p is not None:
            if self.permutation_p > MAX_PERMUTATION_P:
                ok = False
                reasons.append(
                    f"FAIL: label-permutation p = {self.permutation_p:.3f} "
                    f"(need <= {MAX_PERMUTATION_P}) — this improvement is "
                    f"within what shuffled labels produce"
                )
            else:
                reasons.append(
                    f"pass: label-permutation p = {self.permutation_p:.3f}"
                )

        return ok, reasons


def _fit_predict_fold(X_tr, y_tr, X_te, alpha):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_tr)          # training fold only
    model = Ridge(alpha=alpha).fit(scaler.transform(X_tr), y_tr)
    return model.predict(scaler.transform(X_te)), scaler, model


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float = 10.0,
    n_splits: int = N_SPLITS,
    run_permutation: bool = True,
    run_bootstrap: bool = True,
    dropped: Optional[List[str]] = None,
) -> Evaluation:
    """GroupKFold by subject, Ridge vs mean baseline, with a proper gate."""
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=n_splits)
    ev = Evaluation(dropped=list(dropped or []))
    yt, yp, yb, gp = [], [], [], []

    for i, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        # C4: held-out subjects must be genuinely unseen.
        assert not (set(groups[tr]) & set(groups[te])), "subject leak across folds"

        pred, _, _ = _fit_predict_fold(X[tr], y[tr], X[te], alpha)
        base = np.full(len(te), y[tr].mean())

        ev.folds.append(FoldResult(
            fold=i, n_test=len(te), n_test_subjects=len(set(groups[te])),
            baseline_mae=float(np.mean(np.abs(y[te] - base))),
            model_mae=float(np.mean(np.abs(y[te] - pred))),
        ))
        yt.extend(y[te]); yp.extend(pred); yb.extend(base); gp.extend(groups[te])

    ev.y_true = np.asarray(yt); ev.y_pred = np.asarray(yp)
    ev.y_baseline = np.asarray(yb); ev.groups = np.asarray(gp)

    # Bootstrap the MAE improvement, resampling SUBJECTS not recordings —
    # recordings from one subject are not independent. Skipped inside the
    # permutation loop, where only `improvement` is read.
    if run_bootstrap:
      rng = np.random.default_rng(RANDOM_STATE)
      subjects = np.unique(ev.groups)
      by_subject = {s: np.flatnonzero(ev.groups == s) for s in subjects}
      diffs = []
      for _ in range(N_BOOTSTRAP):
          picked = rng.choice(subjects, size=len(subjects), replace=True)
          mask = np.concatenate([by_subject[s] for s in picked])
          diffs.append(
              np.mean(np.abs(ev.y_true[mask] - ev.y_baseline[mask]))
              - np.mean(np.abs(ev.y_true[mask] - ev.y_pred[mask]))
          )
      ev.bootstrap_ci = (float(np.percentile(diffs, 2.5)),
                         float(np.percentile(diffs, 97.5)))

    if run_permutation:
        # Shuffle glucose labels BETWEEN subjects, keeping each subject's
        # recordings together, then re-run the whole thing. This is the
        # distribution of "improvement" under no real signal.
        observed = ev.improvement
        rng = np.random.default_rng(RANDOM_STATE + 1)
        subj_labels = {s: y[groups == s] for s in np.unique(groups)}
        keys = list(subj_labels)
        count = 0
        for _ in range(N_PERMUTATIONS):
            order = rng.permutation(len(keys))
            y_perm = np.empty_like(y)
            for src, dst in zip(order, range(len(keys))):
                idx = np.flatnonzero(groups == keys[dst])
                donor = subj_labels[keys[src]]
                y_perm[idx] = rng.choice(donor, size=len(idx), replace=True)
            perm_ev = evaluate(X, y_perm, groups, alpha=alpha,
                               n_splits=n_splits, run_permutation=False,
                               run_bootstrap=False)
            if perm_ev.improvement >= observed:
                count += 1
        ev.permutation_p = (count + 1) / (N_PERMUTATIONS + 1)

    return ev


# ---------------------------------------------------------------------
# The estimator that plugs into integration/glucose_source.py
# ---------------------------------------------------------------------

class PPGGlucoseEstimator:
    """Satisfies `integration.glucose_source.G0Estimator`.

    `provides_information` is read from the persisted artifact, which is
    only written when `evaluate().passes_gate()` returned True. It is never
    set by hand.
    """

    def __init__(self, blob: Dict[str, Any]):
        self._blob = blob
        self.model_version = blob["model_version"]
        self.provides_information = bool(blob["provides_information"])
        self._mean = np.asarray(blob["scaler_mean"], dtype=float)
        self._scale = np.asarray(blob["scaler_scale"], dtype=float)
        self._coef = np.asarray(blob["coef"], dtype=float)
        self._intercept = float(blob["intercept"])
        self._min_quality = float(blob.get("min_signal_quality", 0.3))

    @classmethod
    def load(cls, path: str = ARTIFACT_PATH) -> "PPGGlucoseEstimator":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no PPG glucose model at {path}. Train one with "
                f"`python -m rppg.models.glucose_estimator --train`. If the "
                f"evidence gate failed, no artifact is written — which means "
                f"the finger-scan path stays off, by design."
            )
        with open(path) as fh:
            return cls(json.load(fh))

    def estimate(self, ppg_block: Dict[str, Any]) -> Optional[float]:
        if not ppg_block.get("ppg_present"):
            return None
        quality = float(ppg_block.get("ppg_signal_quality") or 0.0)
        if quality < self._min_quality:
            # Refuse rather than return a number from a bad capture.
            return None

        # The contract block uses different names from the extractor.
        mapping = {
            "pulse_rate_bpm": "ppg_pulse_rate_bpm",
            "hrv_rmssd_ms": "ppg_hrv_rmssd",
            "perfusion_index": "ppg_perfusion_index",
            "signal_quality": "ppg_signal_quality",
        }
        vec = []
        for name in FEATURE_NAMES:
            key = mapping.get(name)
            value = ppg_block.get(key) if key else ppg_block.get(name)
            if value is None or not np.isfinite(float(value)):
                return None
            vec.append(float(value))

        z = (np.asarray(vec) - self._mean) / self._scale
        return float(np.clip(z @ self._coef + self._intercept, 40.0, 400.0))


# ---------------------------------------------------------------------
# Train + gate + persist
# ---------------------------------------------------------------------

def train_and_persist(alpha: float = 10.0, artifact_path: str = ARTIFACT_PATH,
                      report_path: str = REPORT_PATH) -> Evaluation:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X, y, groups, dropped = build_feature_table()
    ev = evaluate(X, y, groups, alpha=alpha, dropped=dropped)
    passed, reasons = ev.passes_gate()

    _write_report(ev, passed, reasons, report_path, alpha)

    if not passed:
        if os.path.exists(artifact_path):
            os.remove(artifact_path)
        print("\n".join(reasons))
        print(f"\nGATE FAILED — no artifact written. The finger-scan path "
              f"stays off. Report: {report_path}")
        return ev

    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=alpha).fit(scaler.transform(X), y)
    blob = {
        "model_version": f"ppg-glucose-ridge-a{alpha:g}-v1",
        "provides_information": True,
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "min_signal_quality": 0.3,
        "n_recordings": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
        "heldout_mae": round(ev.model_mae, 3),
        "baseline_mae": round(ev.baseline_mae, 3),
        "bootstrap_ci": [round(c, 3) for c in ev.bootstrap_ci],
        "permutation_p": ev.permutation_p,
    }
    with open(artifact_path, "w") as fh:
        json.dump(blob, fh, indent=2)
    print("\n".join(reasons))
    print(f"\nGATE PASSED — artifact written to {artifact_path}")
    print("NOTE: any claim from this model is bounded by n=%d subjects."
          % blob["n_subjects"])
    return ev


def _write_report(ev, passed, reasons, path, alpha):
    import sys
    sys.path.insert(0, os.path.join(_HERE, "..", ".."))
    from utils.clarke_grid import clarke_error_grid

    os.makedirs(os.path.dirname(path), exist_ok=True)
    _, m_pct = clarke_error_grid(ev.y_true, ev.y_pred)
    _, b_pct = clarke_error_grid(ev.y_true, ev.y_baseline)

    lines = [
        "# PPG Glucose Estimator — held-out evaluation",
        "",
        f"**Gate: {'PASSED' if passed else 'FAILED'}**",
        "",
        f"Ridge(alpha={alpha:g}) on {len(FEATURE_NAMES)} extracted features, "
        f"{N_SPLITS}-fold GroupKFold by subject.",
        f"n = {len(ev.y_true)} recordings, {len(np.unique(ev.groups))} subjects.",
        "",
        "## Why Zone A is reported but is not the gate",
        "",
        f"The mean baseline — one constant for everyone — reaches "
        f"**Zone A {b_pct['A']:.1f}%** on this dataset, against a project "
        f"target of 70%. Zone A is +/-20%, which at this cohort's median "
        f"glucose is wider than the label SD, so it cannot separate a working "
        f"model from a constant here. The gate is MAE against the baseline on "
        f"held-out subjects.",
        "",
        "## Results",
        "",
        "| Predictor | MAE | RMSE | Zone A | Zone A+B |",
        "|---|---|---|---|---|",
        f"| Mean baseline | {ev.baseline_mae:.2f} | "
        f"{np.sqrt(np.mean((ev.y_true-ev.y_baseline)**2)):.2f} | "
        f"{b_pct['A']:.1f}% | {b_pct['A']+b_pct['B']:.1f}% |",
        f"| Ridge | {ev.model_mae:.2f} | "
        f"{np.sqrt(np.mean((ev.y_true-ev.y_pred)**2)):.2f} | "
        f"{m_pct['A']:.1f}% | {m_pct['A']+m_pct['B']:.1f}% |",
        "",
        "## Per fold",
        "",
        "| Fold | n | subjects | Baseline MAE | Model MAE | Improvement |",
        "|---|---|---|---|---|---|",
    ]
    for f in ev.folds:
        lines.append(
            f"| {f.fold} | {f.n_test} | {f.n_test_subjects} | "
            f"{f.baseline_mae:.2f} | {f.model_mae:.2f} | {f.improvement:+.2f} |"
        )
    lines += ["", "## Gate criteria", ""]
    lines += [f"- {r}" for r in reasons]
    if ev.dropped:
        lines += ["", "## Dropped recordings", ""]
        lines += [f"- {d}" for d in ev.dropped]
    lines += [
        "", "## Bound on any claim", "",
        f"n = {len(np.unique(ev.groups))} subjects. Per "
        "`agents/rppg/CLAUDE.md` and contract ring-fence rule E1.5, every "
        "claim about this model is bounded by that number.",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true", help="train, gate, persist")
    ap.add_argument("--alpha", type=float, default=10.0)
    args = ap.parse_args()
    if not args.train:
        ap.print_help()
        return
    try:
        train_and_persist(alpha=args.alpha)
    except DatasetUnavailable as exc:
        raise SystemExit(f"cannot train: {exc}")


if __name__ == "__main__":
    main()
