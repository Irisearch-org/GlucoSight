# PPG Glucose Feasibility Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unblock the MATLAB-table glucose labels, build a reusable PPG signal-processing/feature-extraction pipeline as importable modules, and use those modules inside `rppg/PPG_Dataset/ppg_glucose_starter.ipynb` to train and honestly evaluate a glucose regressor at both native (2190 Hz) and smartphone-deployment (30 Hz) sampling rates, with subject-disjoint splitting throughout.

**Architecture:** Three small, pure-function modules (`rppg/data/loader.py`, `rppg/data/preprocess.py`, `rppg/features/extractor.py`) plus a thin evaluation helper (`rppg/evaluate/metrics.py`) do all signal processing and are unit-tested independently of the notebook. The notebook imports these modules and does only data assembly, model training, and reporting — no signal-processing logic lives in notebook cells. Labels are produced once by a standalone script (`rppg/data/convert_labels.py`) into a committed `rppg/data/labels.csv`.

**Tech Stack:** Python 3.13 venv at `rppg/PPG_Dataset/.venv` (already provisioned), numpy, scipy, pandas, scikit-learn, `mat-io` (MATLAB MCOS/table decoder), pytest, Jupyter (kernel `glucosight-ppg` already registered).

**Spec:** The user's task instructions (this conversation) plus `agents/rppg/CLAUDE.md` (track objective, contract obligations, guardrails) and `docs/interface/INTERFACE_CONTRACT_v1.md` §"Contact PPG Track Output Schema" (v1.1, authoritative schema). Executors should skim both before Task 8.

## Global Constraints

- **Environment already set up — do not redo:** the venv at `rppg/PPG_Dataset/.venv` has `mat-io`, `scikit-learn`, `pytest` installed and a registered Jupyter kernel named `glucosight-ppg`. **Always invoke it as `rppg/PPG_Dataset/.venv/bin/python3 -m pip ...` / `-m pytest ...`, never the bare `pip`/`pytest` scripts** — those scripts have a shebang baked in at creation time pointing at the dataset's original (now-renamed) folder path and will silently install into the wrong environment or fail. `python3 -m X` bypasses the broken shebang correctly.
- **Labels come from `mat-io`, not Octave/MATLAB.** `scipy.io.loadmat` and `pymatreader` both return an undecoded opaque MCOS struct for these files — confirmed by testing, not assumption. Octave (even with the `tablicious` package) also fails — it produces an empty placeholder table, not the real data. `mat-io`'s `load_from_mat` correctly decodes the table to a pandas DataFrame with columns `ID, Gender, Age, Glucose, Height, Weight` — verified against all 67 label files (consistent schema, no exceptions, `ID` matches the filename-encoded subject id in every case, `Glucose` range 88–183 mg/dL, mean 115.0, std 18.6).
- **Grouped splits, non-negotiable (finding C4 / track guardrail):** every train/test split in this plan is `sklearn.model_selection.GroupKFold` grouped by `subject_id`. Every split must assert `set(train_subjects) & set(test_subjects) == set()` and print the fold membership.
- **Baseline required (finding C3 spirit):** every MAE table includes a `DummyRegressor(strategy='mean')` baseline computed with the same grouped, leakage-free CV — never compare a trained model's error to nothing.
- **Report per-subject, not just pooled (finding C4):** every result includes per-subject mean absolute error, summarized as median and IQR across subjects, in addition to the pooled number.
- **`ppg_glucose_estimate` ring-fence (finding E1 / contract §"Ring-fence rules"):** trained only on `rppg/PPG_Dataset/`, split by subject; never treated as more than an experimental value; every claim about it is bounded by n=23 subjects.
- **No clinical framing anywhere** — code comments, print statements, and markdown cells must never suggest this replaces a glucometer or is fit for clinical use.
- **Default output format for the ML work is the notebook** (`rppg/PPG_Dataset/ppg_glucose_starter.ipynb`, extended in place, not a new file) — signal-processing logic belongs in the `rppg/` modules, not copy-pasted into cells.

---

## Task 1: Label conversion — `rppg/data/convert_labels.py` → `rppg/data/labels.csv`

**Files:**
- Create: `rppg/data/__init__.py` (empty)
- Create: `rppg/data/convert_labels.py`
- Create: `rppg/tests/__init__.py` (empty)
- Create: `rppg/tests/test_convert_labels.py`
- Produces (generated, also commit it): `rppg/data/labels.csv`

**Interfaces:**
- Produces: `convert_all(labels_dir=LABELS_DIR, output_csv=OUTPUT_CSV) -> pd.DataFrame` with columns `subject_id:int, recording_id:int, glucose_mgdl:float`, sorted by `(subject_id, recording_id)`.
- Produces: `parse_id(filename: str) -> tuple[int, int]` — `(subject_id, recording_id)` parsed from a `label_XX_YYYY.mat` filename.
- Produces: `print_distribution(df: pd.DataFrame) -> None` — prints n, min/max/mean/std, and a text histogram of `glucose_mgdl`.
- Later tasks (`rppg/data/loader.py`) consume `rppg/data/labels.csv` with exactly the three columns above.

- [ ] **Step 1: Write the failing test**

Create `rppg/tests/test_convert_labels.py`:

```python
import os

import numpy as np
import pandas as pd

from rppg.data import convert_labels


def test_parse_id():
    assert convert_labels.parse_id('label_01_0001.mat') == (1, 1)
    assert convert_labels.parse_id('/some/dir/label_23_0007.mat') == (23, 7)


def test_convert_all_produces_expected_schema_and_row_count():
    df = convert_labels.convert_all()
    assert list(df.columns) == ['subject_id', 'recording_id', 'glucose_mgdl']
    assert len(df) == 67
    assert df['subject_id'].nunique() == 23
    assert df.duplicated(['subject_id', 'recording_id']).sum() == 0


def test_convert_all_glucose_values_in_plausible_range():
    df = convert_labels.convert_all()
    assert df['glucose_mgdl'].min() > 40
    assert df['glucose_mgdl'].max() < 500
    # exact values verified by hand against mat-io output during plan validation
    row = df[(df.subject_id == 1) & (df.recording_id == 1)].iloc[0]
    assert row['glucose_mgdl'] == 99.0


def test_labels_csv_written_to_disk():
    convert_labels.convert_all()
    assert os.path.exists(convert_labels.OUTPUT_CSV)
    on_disk = pd.read_csv(convert_labels.OUTPUT_CSV)
    assert len(on_disk) == 67
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/auliamnaufal/Downloads/glucosight && rppg/PPG_Dataset/.venv/bin/python3 -m pytest rppg/tests/test_convert_labels.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'rppg.data.convert_labels'` (or `rppg.data`) since neither exists yet.

- [ ] **Step 3: Write minimal implementation**

Create `rppg/data/__init__.py` (empty file).

Create `rppg/data/convert_labels.py`:

```python
"""Convert MATLAB MCOS `table` label files (label_XX_YYYY.mat) to labels.csv.

scipy.io.loadmat and pymatreader both return an opaque, undecoded struct for
these files (confirmed by testing) because MATLAB's `table` class uses the
proprietary MCOS object serialization format. GNU Octave, even with the
`tablicious` package, also fails — it produces an empty placeholder table
rather than the real data. `mat-io` (PyPI: mat-io, import name `matio`) is
purpose-built to decode MCOS/classdef objects including `table`, and returns
a real pandas DataFrame with the original column names and values.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from matio import load_from_mat

_HERE = os.path.dirname(os.path.abspath(__file__))
LABELS_DIR = os.path.join(_HERE, '..', 'PPG_Dataset', 'Labels')
OUTPUT_CSV = os.path.join(_HERE, 'labels.csv')


def parse_id(filename):
    """Extract (subject_id, recording_id) from a label_XX_YYYY.mat filename."""
    m = re.match(r'label_(\d+)_(\d+)\.mat', os.path.basename(filename))
    if m is None:
        raise ValueError(f'unexpected label filename: {filename}')
    return int(m.group(1)), int(m.group(2))


def convert_all(labels_dir=LABELS_DIR, output_csv=OUTPUT_CSV):
    files = sorted(glob.glob(os.path.join(labels_dir, 'label_*.mat')))
    if not files:
        raise FileNotFoundError(f'no label_*.mat files found in {labels_dir}')

    rows = []
    for f in files:
        subject_id, recording_id = parse_id(f)
        decoded = load_from_mat(f)
        table = decoded['T_temp']
        glucose = float(np.asarray(table['Glucose'].iloc[0]).reshape(-1)[0])
        table_subject_id = int(np.asarray(table['ID'].iloc[0]).reshape(-1)[0])
        if table_subject_id != subject_id:
            raise ValueError(
                f'{f}: filename says subject {subject_id}, table ID says '
                f'{table_subject_id} — do not silently trust the filename'
            )
        rows.append({
            'subject_id': subject_id,
            'recording_id': recording_id,
            'glucose_mgdl': glucose,
        })

    out = pd.DataFrame(rows).sort_values(['subject_id', 'recording_id']).reset_index(drop=True)
    out.to_csv(output_csv, index=False)
    return out


def print_distribution(df):
    g = df['glucose_mgdl']
    print(f'n recordings   : {len(df)}')
    print(f'n subjects     : {df["subject_id"].nunique()}')
    print(f'glucose_mgdl   : min={g.min():.1f}  max={g.max():.1f}  '
          f'mean={g.mean():.1f}  std={g.std():.1f}')
    counts, edges = np.histogram(g, bins=10)
    print('histogram:')
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        print(f'  [{lo:6.1f}, {hi:6.1f}) : {"#" * count} ({count})')


if __name__ == '__main__':
    labels_df = convert_all()
    print_distribution(labels_df)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/auliamnaufal/Downloads/glucosight && rppg/PPG_Dataset/.venv/bin/python3 -m pytest rppg/tests/test_convert_labels.py -v`
Expected: 4 passed.

Then generate the committed CSV and inspect the printed distribution:

Run: `cd /Users/auliamnaufal/Downloads/glucosight && rppg/PPG_Dataset/.venv/bin/python3 rppg/data/convert_labels.py`
Expected output: `n recordings   : 67`, `n subjects     : 23`, `glucose_mgdl   : min=88.0  max=183.0  mean=115.0  std=18.6`, plus a 10-bin histogram.

- [ ] **Step 5: Commit**

```bash
cd /Users/auliamnaufal/Downloads/glucosight
git add rppg/data/__init__.py rppg/data/convert_labels.py rppg/data/labels.csv rppg/tests/__init__.py rppg/tests/test_convert_labels.py
git commit -m "feat(rppg): convert MCOS table glucose labels to labels.csv via mat-io"
```

---

## Task 2: `rppg/data/loader.py` — load signals and join with labels

**Files:**
- Create: `rppg/data/loader.py`
- Create: `rppg/tests/test_loader.py`

**Interfaces:**
- Consumes: `rppg/data/labels.csv` (from Task 1) with columns `subject_id, recording_id, glucose_mgdl`.
- Produces: `FS_NATIVE = 2190`, `DURATION_S = 10` (module constants; consumed by Task 4 tests and the notebook).
- Produces: `parse_signal_id(filename: str) -> tuple[int, int]`.
- Produces: `load_signal(subject_id: int, recording_id: int, raw_dir=RAW_DIR) -> np.ndarray` — shape `(21900,)`, dtype `float64`.
- Produces: `load_labels(labels_csv=LABELS_CSV) -> pd.DataFrame`.
- Produces: `load_all_recordings(raw_dir=RAW_DIR, labels_csv=LABELS_CSV) -> list[dict]`, each dict has keys `subject_id, recording_id, signal, glucose_mgdl`. Consumed directly by the notebook (Task 6) and by Task 4's extractor tests.

- [ ] **Step 1: Write the failing test**

Create `rppg/tests/test_loader.py`:

```python
import numpy as np
import pytest

from rppg.data import loader


def test_parse_signal_id():
    assert loader.parse_signal_id('signal_01_0003.mat') == (1, 3)
    assert loader.parse_signal_id('/x/y/signal_23_0007.mat') == (23, 7)


def test_load_signal_shape_dtype_and_range():
    sig = loader.load_signal(1, 1)
    assert sig.shape == (21900,)
    assert sig.dtype == np.float64
    # raw ADC values are small positive integers on this sensor
    assert 0 < sig.min() < sig.max() < 65536


def test_load_labels_schema():
    df = loader.load_labels()
    assert list(df.columns) == ['subject_id', 'recording_id', 'glucose_mgdl']
    assert len(df) == 67


def test_load_all_recordings_joins_signal_and_label():
    records = loader.load_all_recordings()
    assert len(records) == 67
    r = records[0]
    assert set(r.keys()) == {'subject_id', 'recording_id', 'signal', 'glucose_mgdl'}
    assert r['signal'].shape == (21900,)
    assert 40 < r['glucose_mgdl'] < 500
    subject_ids = {r['subject_id'] for r in records}
    assert len(subject_ids) == 23


def test_load_all_recordings_matches_labels_csv_exactly():
    records = loader.load_all_recordings()
    labels = loader.load_labels()
    record_keys = {(r['subject_id'], r['recording_id']) for r in records}
    label_keys = {(row.subject_id, row.recording_id) for row in labels.itertuples()}
    assert record_keys == label_keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/auliamnaufal/Downloads/glucosight && rppg/PPG_Dataset/.venv/bin/python3 -m pytest rppg/tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rppg.data.loader'`.

- [ ] **Step 3: Write minimal implementation**

Create `rppg/data/loader.py`:

```python
"""Load PPG glucose-set signals (RawData/signal_XX_YYYY.mat) and join with labels.csv."""
import glob
import os
import re

import numpy as np
import pandas as pd
import scipy.io as sio

FS_NATIVE = 2190
DURATION_S = 10

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(_HERE, '..', 'PPG_Dataset')
RAW_DIR = os.path.join(DATASET_ROOT, 'RawData')
LABELS_CSV = os.path.join(_HERE, 'labels.csv')


def parse_signal_id(filename):
    m = re.match(r'signal_(\d+)_(\d+)\.mat', os.path.basename(filename))
    if m is None:
        raise ValueError(f'unexpected signal filename: {filename}')
    return int(m.group(1)), int(m.group(2))


def load_signal(subject_id, recording_id, raw_dir=RAW_DIR):
    path = os.path.join(raw_dir, f'signal_{subject_id:02d}_{recording_id:04d}.mat')
    mat = sio.loadmat(path)
    return mat['signal'].flatten().astype(np.float64)


def load_labels(labels_csv=LABELS_CSV):
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(
            f'{labels_csv} not found — run '
            '`rppg/PPG_Dataset/.venv/bin/python3 rppg/data/convert_labels.py` first'
        )
    return pd.read_csv(labels_csv)


def load_all_recordings(raw_dir=RAW_DIR, labels_csv=LABELS_CSV):
    labels = load_labels(labels_csv)
    signal_files = sorted(glob.glob(os.path.join(raw_dir, 'signal_*.mat')))
    records = []
    for f in signal_files:
        subject_id, recording_id = parse_signal_id(f)
        row = labels[(labels['subject_id'] == subject_id) &
                      (labels['recording_id'] == recording_id)]
        if row.empty:
            continue
        records.append({
            'subject_id': subject_id,
            'recording_id': recording_id,
            'signal': load_signal(subject_id, recording_id, raw_dir),
            'glucose_mgdl': float(row['glucose_mgdl'].iloc[0]),
        })
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/auliamnaufal/Downloads/glucosight && rppg/PPG_Dataset/.venv/bin/python3 -m pytest rppg/tests/test_loader.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/auliamnaufal/Downloads/glucosight
git add rppg/data/loader.py rppg/tests/test_loader.py
git commit -m "feat(rppg): add signal+label loader for the PPG glucose dataset"
```

---

## Task 3: `rppg/data/preprocess.py` — bandpass filter and 30 Hz decimation

**Files:**
- Create: `rppg/data/preprocess.py`
- Create: `rppg/tests/test_preprocess.py`

**Interfaces:**
- Produces: `FS_DEPLOY = 30` (module constant).
- Produces: `bandpass_filter(signal: np.ndarray, fs: float, low_hz=0.5, high_hz=8.0, order=3) -> np.ndarray`.
- Produces: `decimate_to_30hz(signal: np.ndarray, fs_in=loader.FS_NATIVE, fs_out=FS_DEPLOY) -> tuple[np.ndarray, int]` — `(decimated_signal, fs_out)`. Raises `ValueError` if `fs_in` is not an integer multiple of `fs_out`.
- Consumed by: Task 4 (`extractor.py` uses `bandpass_filter` internally) and the notebook (Task 7 calls `decimate_to_30hz` directly).

- [ ] **Step 1: Write the failing test**

Create `rppg/tests/test_preprocess.py`:

```python
import numpy as np
import pytest

from rppg.data import preprocess


def test_bandpass_filter_removes_dc_and_keeps_pulsatile_component():
    fs = 2190
    t = np.arange(fs * 10) / fs
    sig = 1000 + 5 * np.sin(2 * np.pi * 1.2 * t)  # ~72 bpm tone on a large DC offset
    filt = preprocess.bandpass_filter(sig, fs)
    assert abs(filt.mean()) < 1.0
    assert filt.std() > 1.0


def test_decimate_to_30hz_exact_length():
    fs = 2190
    sig = np.random.randn(fs * 10)
    dec, fs_out = preprocess.decimate_to_30hz(sig, fs_in=fs)
    assert fs_out == 30
    assert dec.shape[0] == 300


def test_decimate_to_30hz_rejects_non_integer_ratio():
    sig = np.random.randn(100)
    with pytest.raises(ValueError):
        preprocess.decimate_to_30hz(sig, fs_in=100, fs_out=30)


def test_decimate_preserves_low_frequency_tone():
    fs = 2190
    t = np.arange(fs * 10) / fs
    sig = np.sin(2 * np.pi * 1.2 * t)  # 72 bpm, far below the 15 Hz Nyquist at 30 Hz
    dec, fs_out = preprocess.decimate_to_30hz(sig, fs_in=fs)
    t_dec = np.arange(len(dec)) / fs_out
    expected = np.sin(2 * np.pi * 1.2 * t_dec)
    corr = np.corrcoef(dec, expected)[0, 1]
    assert corr > 0.99
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/auliamnaufal/Downloads/glucosight && rppg/PPG_Dataset/.venv/bin/python3 -m pytest rppg/tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rppg.data.preprocess'`.

- [ ] **Step 3: Write minimal implementation**

Create `rppg/data/preprocess.py`:

```python
"""Signal conditioning: bandpass filtering and 30 Hz deployment-rate decimation.

2190 / 30 = 73 exactly, so decimate_to_30hz uses a single-stage
scipy.signal.decimate with an FIR anti-alias filter (ftype='fir') rather than
the default IIR — FIR is more numerically stable at large decimation factors
and this is a one-shot offline operation, so its extra cost is irrelevant.
Naive slicing (signal[::73]) would alias high-frequency content back into the
pulse band and is exactly what this function exists to avoid.
"""
import numpy as np
from scipy.signal import butter, decimate, sosfiltfilt

FS_DEPLOY = 30


def bandpass_filter(signal, fs, low_hz=0.5, high_hz=8.0, order=3):
    # SOS (not b/a) for numerical stability at narrow relative bandwidths —
    # see rppg/features/extractor.py's _bandpass docstring for why plain
    # butter+filtfilt can blow up.
    nyq = fs / 2.0
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, signal)


def decimate_to_30hz(signal, fs_in, fs_out=FS_DEPLOY):
    if fs_in % fs_out != 0:
        raise ValueError(f'fs_in={fs_in} must be an integer multiple of fs_out={fs_out}')
    q = fs_in // fs_out
    decimated = decimate(signal, q, ftype='fir', zero_phase=True)
    return decimated, fs_out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/auliamnaufal/Downloads/glucosight && rppg/PPG_Dataset/.venv/bin/python3 -m pytest rppg/tests/test_preprocess.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/auliamnaufal/Downloads/glucosight
git add rppg/data/preprocess.py rppg/tests/test_preprocess.py
git commit -m "feat(rppg): add bandpass filter and exact-factor 30 Hz decimation"
```

---

## Task 4: `rppg/features/extractor.py` — per-recording feature vector

**Files:**
- Create: `rppg/features/__init__.py` (empty)
- Create: `rppg/features/extractor.py`
- Create: `rppg/tests/test_extractor.py`

**Design notes (read before implementing — these came from empirically testing the naive approach against this exact dataset and finding it silently wrong):**

A naive "bandpass 0.5–8 Hz then find_peaks" approach was tested directly against `RawData/signal_01_0001.mat` and produced 150 bpm — implausible, and traced to double-counting the systolic peak and the dicrotic notch within the same beat, because the wide passband leaves the 2nd cardiac harmonic largely unattenuated. **Pulse rate is therefore estimated from the frequency domain, not by counting time-domain peaks**: a Welch PSD of the raw (mean-removed) signal, with the dominant frequency taken only from a resting-plausible 0.7–2.0 Hz (42–120 bpm) band. This was validated against all 67 native-rate recordings (mean 77.2 bpm, range 60–120 bpm, physiologically sane) and survives 30 Hz decimation almost exactly (mean absolute difference 0.2 bpm across 15 spot-checked recordings) — it is the robust, always-available feature.

Beat-level features (HRV, morphology) genuinely need time-domain beat positions, which are noisier on this low-amplitude ADC signal (dynamic range ~30 ADC counts on a ~510 baseline). The detector below narrows the bandpass around the PSD-estimated fundamental (suppressing the 2nd harmonic that caused the double-counting above) and applies IBI outlier rejection before computing RMSSD.

**Implementation note (found during Task 4 execution, not during planning):** the first implementation used `scipy.signal.butter(..., output='ba')` + `filtfilt`, which is numerically unstable for the narrow relative bandwidths used here — a unit test with a sharp synthetic pulse shape caught the filtered signal blowing up to ~1e10, and it was silently degrading real-data beat detection too (coverage was ~67-73% before the fix, artificially low). The fix is `output='sos'` + `sosfiltfilt` (applied in both `extractor._bandpass` and `preprocess.bandpass_filter`), which is the standard numerically-stable representation for exactly this situation. After the fix, on the real dataset at **native rate all six features are available on all 67/67 recordings** (pulse_rate_bpm, hrv_rmssd_ms, perfusion_index, signal_quality, rise_time_ms, pulse_width_half_ms). At **30 Hz, hrv_rmssd_ms drops to 13/67 (19%)** while the other five stay at 67/67 — a clean, honest feature-survival finding: HRV specifically needs the timing resolution decimation removes, but pulse rate, perfusion, signal quality, and even the morphology features (rise time, pulse width) survive. **This is a genuine property of the dataset/signal, not a bug to hide** — the notebook (Task 7) must report the coverage rate, not silently impute it away.

**Interfaces:**
- Produces: `F0_BAND_HZ = (0.7, 2.0)`, `AC_BAND_HZ = (0.5, 8.0)`, `PPG_MODEL_VERSION = "ppg-baseline-ridge-v0.1"` (module constants).
- Produces: `estimate_pulse_rate_bpm(signal: np.ndarray, fs: float, band_hz=F0_BAND_HZ) -> float`.
- Produces: `detect_beats(signal: np.ndarray, fs: float, pulse_rate_bpm: float) -> tuple[np.ndarray, np.ndarray]` — `(beat_positions_in_samples_float, narrowband_filtered_signal)`.
- Produces: `compute_hrv_rmssd(beat_positions: np.ndarray, fs: float) -> float` — NaN if fewer than 3 usable beats.
- Produces: `compute_perfusion_index(ac_signal: np.ndarray, raw_signal: np.ndarray) -> float`.
- Produces: `compute_signal_quality(ac_signal: np.ndarray, fs: float, min_bpm=40, max_bpm=200) -> float` — always in `[0, 1]`.
- Produces: `compute_morphology(beat_positions: np.ndarray, ac_signal: np.ndarray, fs: float) -> tuple[float, float]` — `(rise_time_ms, pulse_width_half_ms)`, NaN if fewer than 2 usable beats.
- Produces: `extract_features(signal: np.ndarray, fs: float) -> dict` — keys `pulse_rate_bpm, hrv_rmssd_ms, perfusion_index, signal_quality, rise_time_ms, pulse_width_half_ms, n_beats_detected`. **This is the function the notebook calls per recording** (Task 6, Task 7).
- Produces: `to_contract_dict(feature_dict: dict, glucose_estimate, present=1, n_windows=1, model_version=PPG_MODEL_VERSION, clip_fraction=0.0) -> dict` — keys match Interface Contract v1.1 exactly (`ppg_pulse_rate_bpm`, `ppg_hrv_rmssd`, `ppg_signal_quality`, `ppg_perfusion_index`, `ppg_glucose_estimate`, `ppg_present`, `ppg_clip_fraction`, `ppg_n_windows`, `ppg_model_version`), with contract fallback values (`75.0`, `30.0`, `0.0`, `2.0`, `None`) substituted for any NaN/None input. Consumed by the notebook's Task 8 packaging cell.
- Produces: `CONTRACT_SPEC: dict` and `validate_contract_output(d: dict) -> list[str]` — empty list means valid; consumed by the notebook's Task 8 validation cell.

- [ ] **Step 1: Write the failing test**

Create `rppg/tests/test_extractor.py`:

```python
import numpy as np
import pytest

from rppg.data import loader
from rppg.features import extractor


def _synthetic_pulse_train(fs, duration_s, bpm, noise_std=0.0, seed=0):
    """A repeating asymmetric bump (fast rise, slow fall) at a known bpm — a
    crude but adequate stand-in for a real PPG pulse waveform."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration_s)) / fs
    period = 60.0 / bpm
    phase = (t % period) / period
    # fast rise (0 -> 1 over first 30% of the cycle), slower decay after
    pulse = np.where(phase < 0.3, phase / 0.3, np.exp(-(phase - 0.3) * 4))
    baseline = 500.0
    sig = baseline + 20.0 * pulse
    if noise_std > 0:
        sig = sig + rng.normal(0, noise_std, size=sig.shape)
    return sig


def test_estimate_pulse_rate_bpm_on_synthetic_signal():
    fs = 2190
    bpm_true = 72.0
    sig = _synthetic_pulse_train(fs, 10, bpm_true)
    bpm_est = extractor.estimate_pulse_rate_bpm(sig, fs)
    assert abs(bpm_est - bpm_true) < 6.0  # within one Welch frequency bin


def test_estimate_pulse_rate_bpm_on_all_real_recordings_is_always_finite_and_plausible():
    records = loader.load_all_recordings()
    for r in records:
        bpm = extractor.estimate_pulse_rate_bpm(r['signal'], loader.FS_NATIVE)
        assert np.isfinite(bpm), f"non-finite bpm for subject {r['subject_id']} rec {r['recording_id']}"
        assert 40.0 <= bpm <= 130.0


def test_detect_beats_and_rmssd_on_clean_synthetic_signal():
    fs = 2190
    sig = _synthetic_pulse_train(fs, 10, 72.0, noise_std=0.05)
    bpm = extractor.estimate_pulse_rate_bpm(sig, fs)
    beats, _ = extractor.detect_beats(sig, fs, bpm)
    assert len(beats) >= 10  # ~12 beats expected in 10s at 72bpm
    rmssd = extractor.compute_hrv_rmssd(beats, fs)
    # a perfectly regular synthetic pulse train should have near-zero RMSSD
    assert rmssd < 15.0


def test_compute_hrv_rmssd_returns_nan_for_too_few_beats():
    assert np.isnan(extractor.compute_hrv_rmssd(np.array([10.0, 20.0]), 2190))
    assert np.isnan(extractor.compute_hrv_rmssd(np.array([]), 2190))


def test_compute_signal_quality_is_bounded():
    fs = 2190
    sig = _synthetic_pulse_train(fs, 10, 72.0)
    ac = extractor._bandpass(sig, fs, *extractor.AC_BAND_HZ)
    q = extractor.compute_signal_quality(ac, fs)
    assert 0.0 <= q <= 1.0
    noise = np.random.default_rng(0).normal(500, 1.0, size=fs * 10)
    ac_noise = extractor._bandpass(noise, fs, *extractor.AC_BAND_HZ)
    q_noise = extractor.compute_signal_quality(ac_noise, fs)
    assert q_noise < q  # clean periodic signal must score higher than pure noise


def test_extract_features_returns_all_expected_keys_on_real_data():
    records = loader.load_all_recordings()
    feats = extractor.extract_features(records[0]['signal'], loader.FS_NATIVE)
    expected_keys = {
        'pulse_rate_bpm', 'hrv_rmssd_ms', 'perfusion_index',
        'signal_quality', 'rise_time_ms', 'pulse_width_half_ms', 'n_beats_detected',
    }
    assert set(feats.keys()) == expected_keys
    assert np.isfinite(feats['pulse_rate_bpm'])
    assert 0.0 <= feats['signal_quality'] <= 1.0


def test_to_contract_dict_substitutes_fallbacks_for_missing_values():
    feats = {
        'pulse_rate_bpm': float('nan'), 'hrv_rmssd_ms': float('nan'),
        'perfusion_index': float('nan'), 'signal_quality': 0.0,
        'rise_time_ms': float('nan'), 'pulse_width_half_ms': float('nan'),
        'n_beats_detected': 0,
    }
    d = extractor.to_contract_dict(feats, glucose_estimate=float('nan'))
    assert d['ppg_pulse_rate_bpm'] == 75.0
    assert d['ppg_hrv_rmssd'] == 30.0
    assert d['ppg_perfusion_index'] == 2.0
    assert d['ppg_glucose_estimate'] is None


def test_to_contract_dict_passes_through_real_values():
    feats = {
        'pulse_rate_bpm': 82.3, 'hrv_rmssd_ms': 28.4, 'perfusion_index': 3.1,
        'signal_quality': 0.74, 'rise_time_ms': 150.0, 'pulse_width_half_ms': 110.0,
        'n_beats_detected': 12,
    }
    d = extractor.to_contract_dict(feats, glucose_estimate=118.5, n_windows=1)
    assert d['ppg_pulse_rate_bpm'] == 82.3
    assert d['ppg_glucose_estimate'] == 118.5
    assert d['ppg_n_windows'] == 1
    assert d['ppg_present'] == 1


def test_validate_contract_output_accepts_valid_dict():
    feats = {
        'pulse_rate_bpm': 82.3, 'hrv_rmssd_ms': 28.4, 'perfusion_index': 3.1,
        'signal_quality': 0.74, 'rise_time_ms': 150.0, 'pulse_width_half_ms': 110.0,
        'n_beats_detected': 12,
    }
    d = extractor.to_contract_dict(feats, glucose_estimate=118.5)
    problems = extractor.validate_contract_output(d)
    assert problems == []


def test_validate_contract_output_flags_out_of_range_value():
    feats = {
        'pulse_rate_bpm': 999.0, 'hrv_rmssd_ms': 28.4, 'perfusion_index': 3.1,
        'signal_quality': 0.74, 'rise_time_ms': 150.0, 'pulse_width_half_ms': 110.0,
        'n_beats_detected': 12,
    }
    d = extractor.to_contract_dict(feats, glucose_estimate=118.5)
    problems = extractor.validate_contract_output(d)
    assert any('ppg_pulse_rate_bpm' in p for p in problems)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/auliamnaufal/Downloads/glucosight && rppg/PPG_Dataset/.venv/bin/python3 -m pytest rppg/tests/test_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rppg.features'`.

- [ ] **Step 3: Write minimal implementation**

Create `rppg/features/extractor.py`:

```python
"""Per-recording PPG feature extraction.

See docs/superpowers/plans/2026-08-15-ppg-glucose-pipeline.md Task 4 for why
pulse rate is estimated from a Welch PSD peak rather than time-domain beat
counting, and why beat-level features (HRV, morphology) are best-effort.
"""
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

F0_BAND_HZ = (0.7, 2.0)    # 42-120 bpm: resting-plausible pulse-rate search band
AC_BAND_HZ = (0.5, 8.0)    # general pulsatile-component band (perfusion, quality)
MIN_BEATS_FOR_HRV = 3
MIN_BEATS_FOR_MORPHOLOGY = 2
IBI_OUTLIER_LOW = 0.7
IBI_OUTLIER_HIGH = 1.4
PPG_MODEL_VERSION = 'ppg-baseline-ridge-v0.1'


def _bandpass(signal, fs, low_hz, high_hz, order=3):
    """Second-order-sections Butterworth bandpass. SOS (not transfer-function
    b/a coefficients) is used because narrow relative bandwidths — as used
    here for beat detection around a specific fundamental frequency — make
    the b/a representation numerically unstable; filtfilt on b/a can blow up
    to absurd magnitudes on sharp/asymmetric waveforms even though the same
    filter is fine in SOS form. Caught during Task 4 by a unit test with a
    realistic asymmetric synthetic pulse: filt values reached ~1e10 with b/a."""
    nyq = fs / 2.0
    low_hz = max(low_hz, 1e-3)
    high_hz = min(high_hz, nyq * 0.99)
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, signal)


def estimate_pulse_rate_bpm(signal, fs, band_hz=F0_BAND_HZ):
    s = signal - signal.mean()
    freqs, psd = welch(s, fs=fs, nperseg=len(s), nfft=len(s) * 4, detrend='linear')
    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(mask):
        return float('nan')
    f0 = freqs[mask][np.argmax(psd[mask])]
    return float(f0 * 60.0)


def _parabolic_interp(y, i):
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return float(i)
    return i + 0.5 * (y0 - y2) / denom


def detect_beats(signal, fs, pulse_rate_bpm):
    """Narrowband-filter around the PSD-estimated fundamental (suppressing the
    2nd cardiac harmonic) then peak-pick with sub-sample parabolic refinement."""
    if not np.isfinite(pulse_rate_bpm) or pulse_rate_bpm <= 0:
        return np.array([]), np.zeros_like(signal)
    f0 = pulse_rate_bpm / 60.0
    lo = max(0.4, f0 * 0.5)
    hi = min(fs / 2.0 * 0.95, f0 * 1.8)
    filt = _bandpass(signal, fs, lo, hi)
    min_dist = max(1, int(fs * 0.5 / f0))
    prominence = 0.4 * np.std(filt)
    peaks, _ = find_peaks(filt, distance=min_dist, prominence=prominence)
    refined = np.array([_parabolic_interp(filt, p) for p in peaks])
    return refined, filt


def compute_hrv_rmssd(beat_positions, fs):
    if len(beat_positions) < MIN_BEATS_FOR_HRV:
        return float('nan')
    ibi_ms = np.diff(beat_positions) / fs * 1000.0
    median_ibi = np.median(ibi_ms)
    valid = (ibi_ms > median_ibi * IBI_OUTLIER_LOW) & (ibi_ms < median_ibi * IBI_OUTLIER_HIGH)
    diffs = [ibi_ms[i + 1] - ibi_ms[i]
             for i in range(len(ibi_ms) - 1) if valid[i] and valid[i + 1]]
    if len(diffs) < 1:
        return float('nan')
    return float(np.sqrt(np.mean(np.square(diffs))))


def compute_perfusion_index(ac_signal, raw_signal):
    dc = raw_signal.mean()
    if dc == 0:
        return float('nan')
    ac_pp = ac_signal.max() - ac_signal.min()
    return float(ac_pp / dc * 100.0)


def compute_signal_quality(ac_signal, fs, min_bpm=40, max_bpm=200):
    """Autocorrelation-based periodicity SQI (Elgendi-style): the strength of
    the strongest autocorrelation peak within the physiological IBI lag
    range, normalized by zero-lag energy. 1.0 = perfectly periodic, near 0 =
    aperiodic/noisy."""
    s = ac_signal - ac_signal.mean()
    autocorr = np.correlate(s, s, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    if autocorr[0] == 0:
        return 0.0
    autocorr = autocorr / autocorr[0]
    min_lag = int(fs * 60 / max_bpm)
    max_lag = min(int(fs * 60 / min_bpm), len(autocorr) - 1)
    if min_lag >= max_lag:
        return 0.0
    return float(np.clip(autocorr[min_lag:max_lag].max(), 0.0, 1.0))


def compute_morphology(beat_positions, ac_signal, fs, search_frac=0.3):
    """Rise time and pulse width at half amplitude, averaged across beats.

    Measured on the wideband AC signal (not the narrowband beat-detection
    signal) so the measured shape reflects the true waveform rather than the
    detector's own filter response.
    """
    if len(beat_positions) < MIN_BEATS_FOR_MORPHOLOGY:
        return float('nan'), float('nan')
    rise_times, widths = [], []
    for i in range(1, len(beat_positions)):
        prev_beat, this_beat = beat_positions[i - 1], beat_positions[i]
        ibi_samples = this_beat - prev_beat
        window_start = max(int(this_beat - search_frac * ibi_samples), int(prev_beat))
        peak_idx = int(round(this_beat))
        if window_start >= peak_idx or peak_idx >= len(ac_signal):
            continue
        window = ac_signal[window_start:peak_idx + 1]
        if len(window) < 2:
            continue
        foot_idx = window_start + int(np.argmin(window))
        foot_val, peak_val = ac_signal[foot_idx], ac_signal[peak_idx]
        if peak_val <= foot_val:
            continue
        rise_times.append((peak_idx - foot_idx) / fs * 1000.0)

        half_amp = foot_val + 0.5 * (peak_val - foot_val)
        left = peak_idx
        while left > foot_idx and ac_signal[left] > half_amp:
            left -= 1
        right = peak_idx
        max_right = min(peak_idx + (peak_idx - foot_idx), len(ac_signal) - 1)
        while right < max_right and ac_signal[right] > half_amp:
            right += 1
        if right > left:
            widths.append((right - left) / fs * 1000.0)

    rise = float(np.mean(rise_times)) if rise_times else float('nan')
    width = float(np.mean(widths)) if widths else float('nan')
    return rise, width


def extract_features(signal, fs):
    """The per-recording feature vector consumed by the notebook's regressor."""
    pulse_rate_bpm = estimate_pulse_rate_bpm(signal, fs)
    ac_signal = _bandpass(signal, fs, *AC_BAND_HZ)
    beats, _ = detect_beats(signal, fs, pulse_rate_bpm)
    hrv_rmssd = compute_hrv_rmssd(beats, fs)
    perfusion_index = compute_perfusion_index(ac_signal, signal)
    signal_quality = compute_signal_quality(ac_signal, fs)
    rise_time_ms, pulse_width_half_ms = compute_morphology(beats, ac_signal, fs)
    return {
        'pulse_rate_bpm': pulse_rate_bpm,
        'hrv_rmssd_ms': hrv_rmssd,
        'perfusion_index': perfusion_index,
        'signal_quality': signal_quality,
        'rise_time_ms': rise_time_ms,
        'pulse_width_half_ms': pulse_width_half_ms,
        'n_beats_detected': len(beats),
    }


def _fallback(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, (int, float)) and not np.isfinite(value):
        return fallback
    return float(value)


def to_contract_dict(feature_dict, glucose_estimate, present=1, n_windows=1,
                      model_version=PPG_MODEL_VERSION, clip_fraction=0.0):
    """Map an extract_features() output (+ a glucose estimate) onto Interface
    Contract v1.1's PPG schema, substituting the contract's documented
    fallback values for anything missing/non-finite."""
    quality = feature_dict.get('signal_quality', 0.0)
    return {
        'ppg_pulse_rate_bpm': _fallback(feature_dict.get('pulse_rate_bpm'), 75.0),
        'ppg_hrv_rmssd': _fallback(feature_dict.get('hrv_rmssd_ms'), 30.0),
        'ppg_signal_quality': float(np.clip(quality, 0.0, 1.0)) if np.isfinite(quality) else 0.0,
        'ppg_perfusion_index': _fallback(feature_dict.get('perfusion_index'), 2.0),
        'ppg_glucose_estimate': _fallback(glucose_estimate, None),
        'ppg_present': int(present),
        'ppg_clip_fraction': float(clip_fraction),
        'ppg_n_windows': int(n_windows),
        'ppg_model_version': model_version,
    }


# (type, low, high, required) — low/high are None for non-numeric or unbounded fields
CONTRACT_SPEC = {
    'ppg_pulse_rate_bpm': (float, 40.0, 200.0, True),
    'ppg_hrv_rmssd': (float, 0.0, 200.0, False),
    'ppg_signal_quality': (float, 0.0, 1.0, True),
    'ppg_perfusion_index': (float, 0.0, 20.0, False),
    'ppg_glucose_estimate': (float, 70.0, 400.0, False),
    'ppg_present': (int, 0, 1, True),
    'ppg_clip_fraction': (float, 0.0, 1.0, True),
    'ppg_n_windows': (int, 0, 3, True),
    'ppg_model_version': (str, None, None, True),
}


def validate_contract_output(d):
    """Check an output dict against Interface Contract v1.1's PPG schema.
    Returns a list of problem strings; empty list means valid."""
    problems = []
    for key, (expected_type, lo, hi, required) in CONTRACT_SPEC.items():
        if key not in d:
            problems.append(f'{key}: missing')
            continue
        value = d[key]
        if value is None:
            if required:
                problems.append(f'{key}: required but None')
            continue
        if expected_type in (float, int):
            if not isinstance(value, (int, float, np.integer, np.floating)):
                problems.append(f'{key}: expected {expected_type.__name__}, got {type(value).__name__}')
            elif lo is not None and not (lo <= value <= hi):
                problems.append(f'{key}: {value} outside [{lo}, {hi}]')
        elif expected_type is str:
            if not isinstance(value, str):
                problems.append(f'{key}: expected str, got {type(value).__name__}')
    return problems
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/auliamnaufal/Downloads/glucosight && rppg/PPG_Dataset/.venv/bin/python3 -m pytest rppg/tests/test_extractor.py -v`
Expected: 10 passed. (If `test_estimate_pulse_rate_bpm_on_all_real_recordings_is_always_finite_and_plausible` fails on a specific subject/recording, that is real information about the dataset — investigate rather than loosening the bound blindly; it passed against all 67 recordings during plan validation.)

- [ ] **Step 5: Commit**

```bash
cd /Users/auliamnaufal/Downloads/glucosight
git add rppg/features/__init__.py rppg/features/extractor.py rppg/tests/test_extractor.py
git commit -m "feat(rppg): add PSD-based feature extractor and contract packaging"
```

---

## Task 5: `rppg/evaluate/metrics.py` — grouped-split guard and error reporting

**Files:**
- Create: `rppg/evaluate/__init__.py` (empty)
- Create: `rppg/evaluate/metrics.py`
- Create: `rppg/tests/test_metrics.py`

**Interfaces:**
- Produces: `assert_group_disjoint(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None` — raises `AssertionError` on any subject overlap. This is the "subject-leakage assertion" the track's code standards mandate (`agents/rppg/CLAUDE.md` §Code Standards: "`rppg/tests/test_pipeline.py` must include the subject-leakage assertion").
- Produces: `per_subject_mae(y_true: np.ndarray, y_pred: np.ndarray, subject_ids: np.ndarray) -> pd.DataFrame` — index `subject_id`, column `mae`, plus `n_recordings`.
- Produces: `summarize_per_subject(per_subject_df: pd.DataFrame, column='mae') -> dict` — `{median, iqr_low, iqr_high}`.
- Consumed by: the notebook (Task 6, Task 7) for every model comparison table, and by `rppg/tests/test_pipeline.py` (this task) for the mandated leakage-guard test.

- [ ] **Step 1: Write the failing test**

Create `rppg/tests/test_metrics.py`:

```python
import numpy as np
import pandas as pd
import pytest

from rppg.evaluate import metrics


def test_assert_group_disjoint_passes_on_disjoint_groups():
    groups = np.array([1, 1, 2, 2, 3, 3])
    train_idx = np.array([0, 1, 2, 3])
    test_idx = np.array([4, 5])
    metrics.assert_group_disjoint(groups, train_idx, test_idx)  # must not raise


def test_assert_group_disjoint_raises_on_leaked_subject():
    groups = np.array([1, 1, 2, 2, 3, 3])
    train_idx = np.array([0, 1, 2, 4])  # subject 3 (idx 4) leaks into train
    test_idx = np.array([3, 5])         # subject 2 (idx 3) and subject 3 (idx 5)
    with pytest.raises(AssertionError):
        metrics.assert_group_disjoint(groups, train_idx, test_idx)


def test_per_subject_mae():
    y_true = np.array([100.0, 110.0, 200.0])
    y_pred = np.array([105.0, 100.0, 190.0])
    subject_ids = np.array([1, 1, 2])
    df = metrics.per_subject_mae(y_true, y_pred, subject_ids)
    assert df.loc[1, 'mae'] == pytest.approx((5.0 + 10.0) / 2)
    assert df.loc[2, 'mae'] == pytest.approx(10.0)
    assert df.loc[1, 'n_recordings'] == 2
    assert df.loc[2, 'n_recordings'] == 1


def test_summarize_per_subject():
    df = pd.DataFrame({'mae': [10.0, 20.0, 30.0, 40.0]})
    summary = metrics.summarize_per_subject(df)
    assert summary['median'] == pytest.approx(25.0)
    assert summary['iqr_low'] <= summary['median'] <= summary['iqr_high']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/auliamnaufal/Downloads/glucosight && rppg/PPG_Dataset/.venv/bin/python3 -m pytest rppg/tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rppg.evaluate'`.

- [ ] **Step 3: Write minimal implementation**

Create `rppg/evaluate/metrics.py`:

```python
"""Grouped-split leakage guard and per-subject error reporting.

Pooled error over-weights subjects with many recordings (this dataset ranges
1-7 recordings/subject) and a random split of PPG recordings almost
guarantees the same subject appears in both train and test, at which point
the model memorises per-subject baseline glucose instead of learning
anything optical. assert_group_disjoint makes that failure loud instead of
silent.
"""
import numpy as np
import pandas as pd


def assert_group_disjoint(groups, train_idx, test_idx):
    train_subjects = set(np.asarray(groups)[train_idx])
    test_subjects = set(np.asarray(groups)[test_idx])
    overlap = train_subjects & test_subjects
    assert overlap == set(), f'subject leakage across split: {sorted(overlap)}'


def per_subject_mae(y_true, y_pred, subject_ids):
    df = pd.DataFrame({
        'subject_id': subject_ids,
        'abs_err': np.abs(np.asarray(y_true) - np.asarray(y_pred)),
    })
    grouped = df.groupby('subject_id')['abs_err']
    out = grouped.mean().to_frame('mae')
    out['n_recordings'] = grouped.size()
    return out


def summarize_per_subject(per_subject_df, column='mae'):
    values = per_subject_df[column]
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return {'median': float(median), 'iqr_low': float(q1), 'iqr_high': float(q3)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/auliamnaufal/Downloads/glucosight && rppg/PPG_Dataset/.venv/bin/python3 -m pytest rppg/tests/test_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/auliamnaufal/Downloads/glucosight
git add rppg/evaluate/__init__.py rppg/evaluate/metrics.py rppg/tests/test_metrics.py
git commit -m "feat(rppg): add subject-leakage guard and per-subject MAE reporting"
```

---

## Task 6: Notebook Part A — data loading, native-rate features, GroupKFold, baseline model

**Files:**
- Modify: `rppg/PPG_Dataset/ppg_glucose_starter.ipynb` (use the `NotebookEdit` tool for all cell insertions/edits, not raw JSON editing)

**Context:** The notebook currently has 18 cells and stops at Cell 9 (index 9, the label-loading blocker — it prints the MCOS opacity warning and stops). Cells 10–12 describe/attempt the now-obsolete MATLAB/CSV path; cells 13–17 are unrelated exploration (single-signal plot, cross-subject plot, a BUT-PPG/brno load) that are out of scope for this plan and must be left alone. This task **replaces cells 9–12** (the blocked label loading and the obsolete MATLAB script) with working cells that use the Task 1–5 modules, and **inserts new cells after cell 8** (leave cells 0–8 — the dataset intro, imports, file scanning, and signal/label inspection — untouched, they already work).

**Interfaces:**
- Consumes: `rppg.data.loader.load_all_recordings`, `rppg.data.loader.FS_NATIVE`, `rppg.features.extractor.extract_features`, `rppg.evaluate.metrics.assert_group_disjoint`, `rppg.evaluate.metrics.per_subject_mae`, `rppg.evaluate.metrics.summarize_per_subject`.
- Produces (notebook-local names used by Task 7 and Task 8, which append further cells): `df_native` (DataFrame with `subject_id, recording_id, glucose_mgdl` + one column per feature from `extract_features`), `FEATURE_COLS` (list of the 6 feature column names), `records` (the raw list from `load_all_recordings()`), `gkf` (the fitted `GroupKFold` instance), `results_native` (DataFrame with per-recording predictions/errors for baseline/ridge/rf at native rate).

- [ ] **Step 1: Replace the blocked label cells with a working label-loading + feature-extraction cell**

Using `NotebookEdit`, delete the markdown/code cells that currently correspond to "Cell 4 — Inspect a Single Label File" through "Cell 6 — Load Labels" (the ones containing the `⚠️ Label is stored as a MATLAB Table` warning and the MATLAB `writetable` script), and insert a new markdown cell followed by a new code cell in their place:

Markdown cell:

```markdown
## Cell 4 — Load Labels and Build the Native-Rate Feature Table

Labels are MATLAB MCOS `table` objects. `scipy.io.loadmat` and `pymatreader`
both return them undecoded; GNU Octave (even with `tablicious`) also fails.
`mat-io` decodes them correctly — see `rppg/data/convert_labels.py`. Run it
once (already committed as `rppg/data/labels.csv`) and load the reusable
pipeline modules.
```

Code cell:

```python
import sys, os
sys.path.insert(0, os.path.abspath('../..'))  # repo root, so `import rppg...` works

from rppg.data import loader, convert_labels
from rppg.features import extractor
from rppg.evaluate import metrics

# labels.csv is committed; regenerate only if missing
if not os.path.exists(loader.LABELS_CSV):
    convert_labels.convert_all()

labels_df = loader.load_labels()
convert_labels.print_distribution(labels_df)

records = loader.load_all_recordings()
print(f'\nLoaded {len(records)} recordings from {len(set(r["subject_id"] for r in records))} subjects')
```

Expected output: distribution block matching Task 1's Step 4 output (`n recordings   : 67`, etc.), then `Loaded 67 recordings from 23 subjects`.

- [ ] **Step 2: Add native-rate feature extraction cell**

Insert a new code cell:

```python
FEATURE_COLS = [
    'pulse_rate_bpm', 'hrv_rmssd_ms', 'perfusion_index',
    'signal_quality', 'rise_time_ms', 'pulse_width_half_ms',
]

rows = []
for r in records:
    feats = extractor.extract_features(r['signal'], loader.FS_NATIVE)
    row = {'subject_id': r['subject_id'], 'recording_id': r['recording_id'],
           'glucose_mgdl': r['glucose_mgdl']}
    row.update(feats)
    rows.append(row)

df_native = pd.DataFrame(rows)
print(f'Feature table: {df_native.shape[0]} recordings x {len(FEATURE_COLS)} features')
print('\nFeature availability (native 2190 Hz) — fraction non-NaN:')
print(df_native[FEATURE_COLS].notna().mean().round(3))
df_native.head()
```

Expected: a table printed. All six columns should be at 1.0 at native rate (verified against the real dataset during Task 4 execution, after fixing a filter-stability bug — see the Task 4 implementation note).

- [ ] **Step 3: Add subject-disjoint GroupKFold split cell with visible assertion**

Insert a new markdown cell:

```markdown
## Cell 5 — Subject-Disjoint Split (GroupKFold) — Native Rate
```

followed by a code cell:

```python
from sklearn.model_selection import GroupKFold

X_native = df_native[FEATURE_COLS].values
y_native = df_native['glucose_mgdl'].values
groups_native = df_native['subject_id'].values

N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)

print(f'Verifying {N_SPLITS}-fold GroupKFold is subject-disjoint (23 subjects, ~{23//N_SPLITS} held out per fold):')
for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X_native, y_native, groups_native)):
    metrics.assert_group_disjoint(groups_native, train_idx, test_idx)  # raises on any leakage
    train_subjects = sorted(set(groups_native[train_idx]))
    test_subjects = sorted(set(groups_native[test_idx]))
    print(f'  fold {fold_i}: {len(train_subjects)} train subjects, '
          f'{len(test_subjects)} test subjects, overlap={set(train_subjects) & set(test_subjects)}')
print('PASS: all folds are subject-disjoint (no participant appears in both train and test of any fold).')
```

Expected: 5 fold lines each showing `overlap=set()`, then the PASS line. If any fold raised `AssertionError`, stop — do not proceed to Step 4 until this passes, since every downstream metric is meaningless under subject leakage.

- [ ] **Step 4: Train baseline, Ridge, and Random Forest with out-of-fold predictions**

Insert a new markdown cell:

```markdown
## Cell 6 — Native-Rate Models: Population-Mean Baseline vs Ridge vs Random Forest

Per finding C3, no regression result is meaningful without a trivial
baseline in the same table. `DummyRegressor(strategy='mean')` run through
the same grouped, leakage-free CV is that baseline here.
```

followed by a code cell:

```python
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

baseline_pipe = make_pipeline(SimpleImputer(strategy='median'), DummyRegressor(strategy='mean'))
ridge_pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=1.0))
rf_pipe = make_pipeline(SimpleImputer(strategy='median'),
                         RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42))

pred_baseline = cross_val_predict(baseline_pipe, X_native, y_native, groups=groups_native, cv=gkf)
pred_ridge = cross_val_predict(ridge_pipe, X_native, y_native, groups=groups_native, cv=gkf)
pred_rf = cross_val_predict(rf_pipe, X_native, y_native, groups=groups_native, cv=gkf)

results_native = df_native[['subject_id', 'recording_id', 'glucose_mgdl']].copy()
results_native['pred_baseline'] = pred_baseline
results_native['pred_ridge'] = pred_ridge
results_native['pred_rf'] = pred_rf
for col in ['baseline', 'ridge', 'rf']:
    results_native[f'abs_err_{col}'] = (results_native[f'pred_{col}'] - results_native['glucose_mgdl']).abs()

print('Pooled MAE (native rate, out-of-fold):')
for col in ['baseline', 'ridge', 'rf']:
    print(f'  {col:10s}: {results_native[f"abs_err_{col}"].mean():.2f} mg/dL')
```

Expected: three MAE lines. Ridge/RF are not required to beat baseline — report whatever the numbers are, do not tune toward a nicer-looking result.

- [ ] **Step 5: Per-subject error table**

Insert a new code cell:

```python
print('Per-subject MAE (native rate):\n')
for col in ['baseline', 'ridge', 'rf']:
    per_subj = metrics.per_subject_mae(
        results_native['glucose_mgdl'], results_native[f'pred_{col}'], results_native['subject_id'])
    summary = metrics.summarize_per_subject(per_subj)
    print(f'{col}: median={summary["median"]:.2f}  '
          f'IQR=[{summary["iqr_low"]:.2f}, {summary["iqr_high"]:.2f}]  '
          f'(n={len(per_subj)} subjects, recording counts {per_subj["n_recordings"].min()}-{per_subj["n_recordings"].max()})')

print('\nFull per-subject table (ridge):')
metrics.per_subject_mae(results_native['glucose_mgdl'], results_native['pred_ridge'], results_native['subject_id'])
```

Expected: three summary lines plus a printed per-subject DataFrame (23 rows).

- [ ] **Step 6: Execute the notebook end-to-end and verify no errors**

Run: `cd /Users/auliamnaufal/Downloads/glucosight/rppg/PPG_Dataset && .venv/bin/python3 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=glucosight-ppg ppg_glucose_starter.ipynb`
Expected: exits 0, no `Traceback` in the output, and re-opening the notebook shows the `PASS: all folds are subject-disjoint` line and the three MAE tables from Steps 4–5 with real numbers (not placeholders).

- [ ] **Step 7: Commit**

```bash
cd /Users/auliamnaufal/Downloads/glucosight
git add rppg/PPG_Dataset/ppg_glucose_starter.ipynb
git commit -m "feat(rppg): notebook — native-rate features, GroupKFold, baseline vs Ridge vs RF"
```

---

## Task 7: Notebook Part B — 30 Hz decimation experiment and feature-survival report

**Files:**
- Modify: `rppg/PPG_Dataset/ppg_glucose_starter.ipynb`

**Interfaces:**
- Consumes: `records`, `FEATURE_COLS`, `gkf`, `groups_native` from Task 6; `rppg.data.preprocess.decimate_to_30hz`.
- Produces: `df_30hz` (same schema as `df_native`), `results_30hz` (same schema as `results_native`). These row orders must match `df_native`/`records` exactly so `gkf`'s fold assignment (which only depends on `groups`, unshuffled) is identical between native and decimated runs — this is what makes the comparison paired/fair.

- [ ] **Step 1: Add decimation + feature extraction cell**

Insert a markdown cell after Task 6's last cell:

```markdown
## Cell 7 — Decimate to 30 Hz (Smartphone Deployment Rate) and Re-Extract Features

`preprocess.decimate_to_30hz` uses `scipy.signal.decimate` with an FIR
anti-alias filter — not naive slicing, which would alias high-frequency
content into the pulse band. 2190 / 30 = 73 exactly, so this is a
single-stage, exact-factor decimation.
```

followed by a code cell:

```python
from rppg.data import preprocess

rows_30hz = []
for r in records:
    decimated, fs_30 = preprocess.decimate_to_30hz(r['signal'], fs_in=loader.FS_NATIVE)
    feats = extractor.extract_features(decimated, fs_30)
    row = {'subject_id': r['subject_id'], 'recording_id': r['recording_id'],
           'glucose_mgdl': r['glucose_mgdl']}
    row.update(feats)
    rows_30hz.append(row)

df_30hz = pd.DataFrame(rows_30hz)
assert (df_30hz['subject_id'].values == df_native['subject_id'].values).all(), \
    'row order must match df_native for the GroupKFold folds to be comparable'

print('Feature availability comparison — fraction non-NaN:')
availability = pd.DataFrame({
    'native_2190hz': df_native[FEATURE_COLS].notna().mean(),
    'decimated_30hz': df_30hz[FEATURE_COLS].notna().mean(),
})
availability['delta'] = availability['decimated_30hz'] - availability['native_2190hz']
print(availability.round(3))
```

Expected: a 6-row table. Verified during Task 4 execution: `pulse_rate_bpm`, `perfusion_index`, `signal_quality`, `rise_time_ms`, and `pulse_width_half_ms` stay at 1.0 (67/67) at both rates; `hrv_rmssd_ms` drops from 1.0 at native rate to ~0.19 (13/67) at 30 Hz — HRV specifically needs the beat-timing resolution decimation removes, everything else survives. Report whatever the notebook run actually prints, this table is itself a contribution per the track's Task 2 guidance ("report which features survive; that table is a genuine contribution").

- [ ] **Step 2: Retrain on the same folds, same model configs, at 30 Hz**

Insert a code cell:

```python
X_30hz = df_30hz[FEATURE_COLS].values
y_30hz = df_30hz['glucose_mgdl'].values
groups_30hz = df_30hz['subject_id'].values
assert (groups_30hz == groups_native).all()

pred_baseline_30 = cross_val_predict(baseline_pipe, X_30hz, y_30hz, groups=groups_30hz, cv=gkf)
pred_ridge_30 = cross_val_predict(ridge_pipe, X_30hz, y_30hz, groups=groups_30hz, cv=gkf)
pred_rf_30 = cross_val_predict(rf_pipe, X_30hz, y_30hz, groups=groups_30hz, cv=gkf)

results_30hz = df_30hz[['subject_id', 'recording_id', 'glucose_mgdl']].copy()
results_30hz['pred_baseline'] = pred_baseline_30
results_30hz['pred_ridge'] = pred_ridge_30
results_30hz['pred_rf'] = pred_rf_30
for col in ['baseline', 'ridge', 'rf']:
    results_30hz[f'abs_err_{col}'] = (results_30hz[f'pred_{col}'] - results_30hz['glucose_mgdl']).abs()

print('Pooled MAE comparison — native 2190 Hz vs decimated 30 Hz (out-of-fold):')
comparison = pd.DataFrame({
    'native_2190hz': {c: results_native[f'abs_err_{c}'].mean() for c in ['baseline', 'ridge', 'rf']},
    'decimated_30hz': {c: results_30hz[f'abs_err_{c}'].mean() for c in ['baseline', 'ridge', 'rf']},
})
print(comparison.round(2))

print('\nPer-subject MAE (median [IQR]) — native vs 30 Hz:')
for col in ['baseline', 'ridge', 'rf']:
    s_native = metrics.summarize_per_subject(
        metrics.per_subject_mae(results_native['glucose_mgdl'], results_native[f'pred_{col}'], results_native['subject_id']))
    s_30 = metrics.summarize_per_subject(
        metrics.per_subject_mae(results_30hz['glucose_mgdl'], results_30hz[f'pred_{col}'], results_30hz['subject_id']))
    print(f'  {col:10s} native: {s_native["median"]:.2f} [{s_native["iqr_low"]:.2f}, {s_native["iqr_high"]:.2f}]   '
          f'30hz: {s_30["median"]:.2f} [{s_30["iqr_low"]:.2f}, {s_30["iqr_high"]:.2f}]')
```

Expected: a comparison table and per-subject median[IQR] lines for both rates. Do not editorialize the numbers in code — the markdown cell in Step 3 is where interpretation belongs.

- [ ] **Step 3: Add the feasibility-study markdown disclaimer**

Insert a markdown cell:

```markdown
## Feasibility Study, n = 23 Subjects — Read Before Interpreting Any Number Above

These results describe whether a glucose-relevant (or, more likely,
autonomic/vascular) signal survives smartphone-grade 30 Hz sampling on
`rppg/PPG_Dataset/` — 23 subjects, 67 recordings, a lab pulse sensor, not the
target Indonesian population and not a smartphone. They are:

- **A feasibility signal, not a validated accuracy claim.** With 23 subjects
  and a 5-fold GroupKFold, each fold holds out only ~4-5 people —
  underpowered to detect small effect sizes, per finding C4.
- **Bounded by n=23 subjects everywhere**, per the ring-fence rules for
  `ppg_glucose_estimate` in `docs/interface/INTERFACE_CONTRACT_v1.md`.
- **Not evidence this replaces a glucometer.** GlucoSight is a research
  prototype, not a medical device, and this signal — whatever its accuracy
  above — is a trend signal at most, never a sole prediction, never shown
  directly to a user.
- Honest about degradation: the feature-availability and per-subject tables
  above report real coverage and real errors, including wherever 30 Hz
  loses ground on morphology/HRV features relative to native rate — that
  degradation, if present, is itself the reportable finding for this sprint
  (see `agents/rppg/CLAUDE.md` Task 2), not something to paper over.
```

- [ ] **Step 4: Execute the notebook end-to-end and verify no errors**

Run: `cd /Users/auliamnaufal/Downloads/glucosight/rppg/PPG_Dataset && .venv/bin/python3 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=glucosight-ppg ppg_glucose_starter.ipynb`
Expected: exits 0, no `Traceback`, and the feature-availability and MAE-comparison tables from Steps 1–2 show real numbers.

- [ ] **Step 5: Commit**

```bash
cd /Users/auliamnaufal/Downloads/glucosight
git add rppg/PPG_Dataset/ppg_glucose_starter.ipynb
git commit -m "feat(rppg): notebook — 30Hz decimation experiment and feature-survival report"
```

---

## Task 8: Notebook Part C — contract-schema packaging and validation

**Files:**
- Modify: `rppg/PPG_Dataset/ppg_glucose_starter.ipynb`

**Interfaces:**
- Consumes: `results_native`, `df_native`, `extractor.to_contract_dict`, `extractor.validate_contract_output`, `extractor.PPG_MODEL_VERSION`.

- [ ] **Step 1: Add the contract-packaging example cell**

Insert a markdown cell:

```markdown
## Cell 8 — Package One Recording as a Contract-v1.1 Output Dict

`ppg_glucose_estimate` here is the out-of-fold Ridge prediction from Cell 6
— i.e. produced by a model that never saw this recording's subject during
training. Per the contract's ring-fence rules, this value is experimental,
bounded by n=23 subjects, and is never a sole prediction or shown to a user.
```

followed by a code cell:

```python
example_idx = 0
example_row = df_native.iloc[example_idx]
example_feats = {col: example_row[col] for col in
                  ['pulse_rate_bpm', 'hrv_rmssd_ms', 'perfusion_index',
                   'signal_quality', 'rise_time_ms', 'pulse_width_half_ms', 'n_beats_detected']}

example_glucose_estimate = results_native.loc[
    (results_native.subject_id == example_row.subject_id) &
    (results_native.recording_id == example_row.recording_id), 'pred_ridge'
].iloc[0]

contract_output = extractor.to_contract_dict(
    example_feats, glucose_estimate=example_glucose_estimate,
    present=1, n_windows=1, model_version=extractor.PPG_MODEL_VERSION,
)

print(f'Example: subject {example_row.subject_id}, recording {example_row.recording_id}')
print(f'True glucose (glucometer): {example_row.glucose_mgdl:.1f} mg/dL')
print()
for k, v in contract_output.items():
    print(f'  {k:24s}: {v}')
```

Expected: a printed dict with all 9 keys and real (non-placeholder) values.

- [ ] **Step 2: Add the validation cell**

Insert a code cell:

```python
problems = extractor.validate_contract_output(contract_output)
if problems:
    print('FAIL — contract violations:')
    for p in problems:
        print(f'  - {p}')
else:
    print('PASS — output matches Interface Contract v1.1 PPG schema '
          '(types, ranges, required fields all satisfied).')
```

Expected: `PASS — output matches Interface Contract v1.1 PPG schema ...`. If it prints FAIL, fix `to_contract_dict` or `CONTRACT_SPEC` in `rppg/features/extractor.py` (whichever is wrong) — do not edit the notebook to hide the failure.

- [ ] **Step 3: Execute the full notebook end-to-end one final time**

Run: `cd /Users/auliamnaufal/Downloads/glucosight/rppg/PPG_Dataset && .venv/bin/python3 -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=glucosight-ppg ppg_glucose_starter.ipynb`
Expected: exits 0, zero `Traceback` occurrences anywhere in the executed notebook, and the final cell prints the PASS line from Step 2.

Then also run the full test suite one more time to confirm nothing in the modules regressed while iterating on the notebook:

Run: `cd /Users/auliamnaufal/Downloads/glucosight && rppg/PPG_Dataset/.venv/bin/python3 -m pytest rppg/tests/ -v`
Expected: all tests from Tasks 1-5 pass (27 tests total: 4 + 5 + 4 + 10 + 4).

- [ ] **Step 4: Commit**

```bash
cd /Users/auliamnaufal/Downloads/glucosight
git add rppg/PPG_Dataset/ppg_glucose_starter.ipynb
git commit -m "feat(rppg): notebook — package and validate one example against contract v1.1"
```

---

## Self-Review Notes (completed during plan authoring)

**Spec coverage:**
- "Unblock the labels" → Task 1 (via `mat-io`, empirically verified against all 67 files; the user's suggested `pymatreader` and Octave fallbacks were both tried first and confirmed non-working, documented in Task 1's module docstring and Global Constraints).
- "loader.py", "preprocess.py", "extractor.py" as importable modules → Tasks 2, 3, 4.
- "Load all 67 + extract native-rate features + GroupKFold + assert disjoint + print it" → Task 6, Steps 1-3.
- "Train baseline regressor, report MAE + per-subject error" → Task 6, Steps 4-5 (three models: dummy baseline, Ridge, RF — user said "start with Ridge or RandomForestRegressor," this plan does both since the comparison is cheap and more informative than picking one).
- "Repeat after decimating to 30Hz, report which features survive" → Task 7.
- "Markdown cell stating feasibility study, n=23" → Task 7, Step 3.
- "Package output to contract schema, validate against documented types/ranges" → Task 8.
- Hard constraints (glucose_estimate trained only on this dataset split by subject; never claims glucometer replacement; explicit about insufficient-fold-size rather than switching to random split; no clinical framing) → encoded in Global Constraints and reinforced in Task 7 Step 3's markdown and Task 8's docstring comment.

**Placeholder scan:** no TBD/TODO/"add appropriate handling" in any task; every code step above is complete, runnable code with concrete file paths and verified-plausible expected output values (pulse-rate ranges, glucose distribution, RMSSD bound, etc. were all measured against the real dataset during plan authoring, not guessed).

**Type consistency:** `extract_features()`'s return keys (Task 4) match `FEATURE_COLS` used in Task 6/7 exactly (`pulse_rate_bpm, hrv_rmssd_ms, perfusion_index, signal_quality, rise_time_ms, pulse_width_half_ms` — `n_beats_detected` is returned but intentionally excluded from `FEATURE_COLS`, it's diagnostic not a regression input). `to_contract_dict`'s output keys match `CONTRACT_SPEC`'s keys exactly (both defined together in Task 4). `loader.FS_NATIVE` (Task 2) is the single source of truth for 2190, used by Task 4's tests, Task 6, and Task 7 — no hardcoded `2190` duplicated elsewhere except inside test fixtures where it names a local variable for clarity.
