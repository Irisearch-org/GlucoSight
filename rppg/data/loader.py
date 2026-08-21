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
