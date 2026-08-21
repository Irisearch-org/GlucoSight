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
