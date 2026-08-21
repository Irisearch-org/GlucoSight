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
