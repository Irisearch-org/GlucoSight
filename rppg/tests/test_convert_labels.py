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
