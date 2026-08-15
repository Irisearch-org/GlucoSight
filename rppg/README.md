# Contact PPG Pipeline

Reusable signal-processing/feature-extraction pipeline for the PPG glucose
feasibility study, plus the notebook that trains and evaluates a glucose
regressor on it. This README documents what is actually implemented and how
to run it. For the track's fuller objective, guardrails, and Sprint 2 task
list, see [`agents/rppg/CLAUDE.md`](../agents/rppg/CLAUDE.md); for the
cross-track output schema, see
[`docs/interface/INTERFACE_CONTRACT_v1.md`](../docs/interface/INTERFACE_CONTRACT_v1.md).
For full implementation detail and the design decisions behind every
function here, see the plan this was built from:
[`docs/superpowers/plans/2026-08-15-ppg-glucose-pipeline.md`](../docs/superpowers/plans/2026-08-15-ppg-glucose-pipeline.md).

**This is a research prototype, not a medical device. Nothing here replaces
a glucometer.**

---

## Layout

```
rppg/
├── PPG_Dataset/          ← provided data (do not modify in place)
│   ├── RawData/             signal_XX_YYYY.mat, 10s @ 2190Hz, 23 subjects, 67 recordings
│   ├── Labels/               label_XX_YYYY.mat, MATLAB MCOS table objects
│   └── brno/                 BUT PPG dataset (30Hz PPG + 1000Hz ECG, WFDB) — out of scope here
├── data/
│   ├── convert_labels.py    MCOS table → labels.csv (via mat-io)
│   ├── labels.csv            committed, derived: subject_id, recording_id, glucose_mgdl
│   ├── loader.py              load a signal + join with labels.csv
│   └── preprocess.py          bandpass filter, exact-factor 30Hz decimation
├── features/
│   └── extractor.py          per-recording feature vector + contract-schema packaging
├── evaluate/
│   └── metrics.py             subject-leakage guard, per-subject MAE
├── notebooks/
│   └── ppg_glucose_starter.ipynb   the ML work: load → features → split → train → report
└── tests/                     pytest suite for everything above (27 tests)
```

## Setup

A venv already exists at `rppg/PPG_Dataset/.venv` with everything installed
(numpy, scipy, pandas, scikit-learn, `mat-io`, `wfdb`, pytest, Jupyter). It
registers a Jupyter kernel named **`glucosight-ppg`**.

**Always invoke it as `python3 -m pip ...` / `-m pytest ...` / `-m jupyter
...`, never the bare `pip`/`pytest`/`jupyter` scripts.** Those console
scripts have a shebang baked in at venv-creation time pointing at the
dataset folder's original (since-renamed) absolute path, so invoking them
directly silently no-ops or installs into the wrong place. `python3 -m X`
bypasses the broken shebang.

```bash
# from repo root
rppg/PPG_Dataset/.venv/bin/python3 -m pytest rppg/tests/ -v
```

## Running the notebook

```bash
cd rppg/notebooks
../PPG_Dataset/.venv/bin/python3 -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=glucosight-ppg ppg_glucose_starter.ipynb
```

or open it interactively in Jupyter/VS Code using the `glucosight-ppg`
kernel. It imports `rppg.data.*`, `rppg.features.*`, `rppg.evaluate.*`
directly — no signal-processing logic is duplicated into cells.

## Why labels needed `mat-io`, not Octave

The label files are MATLAB MCOS `table` objects. `scipy.io.loadmat` and
`pymatreader` both return an opaque, undecoded struct. GNU Octave — even
with the `tablicious` package, which adds a `table` class to Octave —
*also* fails: it produces an empty placeholder table rather than parsing
the real MATLAB-authored MCOS binary layout. The PyPI package **`mat-io`**
(`from matio import load_from_mat`) is purpose-built to decode
MCOS/classdef objects including `table`, and returns a real pandas
DataFrame. Verified against all 67 label files: consistent schema (`ID,
Gender, Age, Glucose, Height, Weight`), no filename/table-ID mismatches,
`glucose_mgdl` range 88–183 mg/dL (mean 115.0, std 18.7).

Run once (already committed as `rppg/data/labels.csv`, so this is only
needed to regenerate it):

```bash
rppg/PPG_Dataset/.venv/bin/python3 rppg/data/convert_labels.py
```

## Feature extraction design

- **Pulse rate** (`ppg_pulse_rate_bpm`) is the dominant frequency in a Welch
  PSD, restricted to a resting-plausible 0.7–2.0 Hz (42–120 bpm) band — not
  time-domain peak counting. A naive bandpass + `find_peaks` approach was
  tried first and gave 150 bpm on the very first recording; it was
  double-counting the systolic peak and the dicrotic notch as two beats
  because a wide passband leaves the 2nd cardiac harmonic mostly
  unattenuated. The PSD approach is robust across all 67 recordings (60–120
  bpm) and survives 30 Hz decimation almost exactly.
- **HRV (`ppg_hrv_rmssd`) and morphology** (rise time, pulse width at half
  amplitude — internal features, not contract fields) need actual beat
  positions. The detector narrows the bandpass around the PSD-estimated
  fundamental and applies median-based IBI outlier rejection before
  computing RMSSD. At native rate all six features are available on all
  67/67 recordings; at 30 Hz, `hrv_rmssd_ms` drops to 13/67 (19%) while
  everything else stays at 67/67 — HRV specifically needs the beat-timing
  resolution decimation removes.
- **Perfusion index** (`ppg_perfusion_index`) is AC peak-to-peak over DC
  mean, as a percentage.
- **Signal quality** (`ppg_signal_quality`) is an autocorrelation-based
  periodicity index (Elgendi-style SQI): the strongest autocorrelation peak
  within the physiological IBI lag range, normalized by zero-lag energy —
  always in `[0, 1]`.
- All bandpass filtering uses **second-order sections** (`butter(...,
  output='sos')` + `sosfiltfilt`), not transfer-function `b`/`a`
  coefficients + `filtfilt`. A unit test with a realistic asymmetric
  synthetic pulse caught the `b`/`a` form blowing up to ~1e10 on narrow
  relative bandwidths — numerically unstable, and it was silently degrading
  real beat detection before the fix (coverage was ~67–73% instead of
  100%). This is a standard, well-known remedy for exactly this failure
  mode; see `rppg/features/extractor.py::_bandpass`'s docstring.

## Results (native rate, GroupKFold by subject, 5 folds)

| Model | Pooled MAE | Per-subject median [IQR] |
|---|---|---|
| Baseline (population mean) | 14.57 mg/dL | 13.94 [10.99, 15.96] |
| Ridge | 15.64 mg/dL | 13.96 [12.67, 17.05] |
| Random Forest | 16.52 mg/dL | 16.47 [10.77, 19.19] |

**The population-mean baseline matches or beats both trained models.** This
is an honest result, not a failure to hide — it is exactly what the track's
own honest-expectation framing predicted (`agents/rppg/CLAUDE.md`: "any
signal you find is more likely autonomic and vascular than
optically-glucose"). No evidence yet of an optical glucose signal beyond
what a population mean already captures, at n=23 subjects.

## Guardrails this pipeline follows

- Every split is `GroupKFold` by `subject_id`, asserted disjoint in code
  (`rppg/evaluate/metrics.py::assert_group_disjoint`) — never a random
  split.
- Every MAE table reports a trivial baseline (`DummyRegressor(strategy=
  'mean')`) alongside trained models, and per-subject median/IQR alongside
  pooled error.
- `ppg_glucose_estimate` is trained only on `rppg/PPG_Dataset/`, split by
  subject, and every claim about it is bounded by n=23 subjects. It is
  never a sole prediction and never shown to a user.
- Output dicts are validated against Interface Contract v1.1's PPG schema
  (`rppg/features/extractor.py::validate_contract_output`) before being
  treated as done.
