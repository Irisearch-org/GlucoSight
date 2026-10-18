#!/usr/bin/env python3
"""Tests for the ShanghaiT2DM response rebuild (finding N6).

The raw ShanghaiT2DM workbooks are not in the repository (they are 9 MB of
Excel, extracted to a directory outside it), so every test here writes a
synthetic workbook with the same shape: one row per 15-minute CGM reading,
dietary text only on meal rows.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nlp.data.shanghai_response import (  # noqa: E402
    ResponseConfig,
    build_response_table,
    compute_responses,
    meal_mask,
    repeat_run_lengths,
)
from nlp.data.shanghai_schema import (  # noqa: E402
    canonical_columns,
    read_sheet,
    workbook_paths,
)

HEADERS = [
    "Date",
    "CGM (mg / dl)",
    "CBG (mg / dl)",
    "Blood Ketone (mmol / L)",
    "Dietary intake",
    "饮食",
    "Insulin dose - s.c.",
    "Non-insulin hypoglycemic agents",
    "CSII - bolus insulin (Novolin R, IU)",
    "CSII - basal insulin (Novolin R, IU / H)",
    "Insulin dose - i.v.",
]


def grid(start="2021-03-01 08:00", steps=24, step_min=15, base=100.0, slope=2.0):
    stamps = pd.date_range(start, periods=steps, freq=f"{step_min}min")
    values = [base + slope * i for i in range(steps)]
    return stamps, values


def write_sheet(
    directory: Path,
    name: str = "2000_0_20210301.xlsx",
    headers=None,
    meals=None,
    timestamps=None,
    glucose=None,
    extra=None,
):
    """Write a synthetic participant workbook and return its path.

    A meal whose timestamp is not on the CGM grid is appended as a diet-only
    row with no reading, which is what the real workbooks look like when a
    meal is logged before the sensor period starts.
    """
    headers = headers or HEADERS
    if timestamps is None or glucose is None:
        timestamps, glucose = grid()
    frame = pd.DataFrame({headers[0]: timestamps})
    frame[headers[1]] = glucose
    for header in headers[2:]:
        frame[header] = np.nan
    frame[headers[4]] = ""

    on_grid = set(pd.to_datetime(timestamps))
    for stamp, text in (meals or {}).items():
        stamp = pd.Timestamp(stamp)
        if stamp in on_grid:
            frame.loc[pd.to_datetime(frame[headers[0]]) == stamp, headers[4]] = text
        else:
            row = {header: np.nan for header in headers}
            row[headers[0]] = stamp
            row[headers[4]] = text
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)

    if extra:
        for header, values in extra.items():
            frame[header] = values

    frame = frame.sort_values(headers[0], kind="stable").reset_index(drop=True)
    path = directory / name
    frame.to_excel(path, index=False)
    return path


class HeaderVariantsTests(unittest.TestCase):
    """Five header layouts ship in the v3 release; all must resolve."""

    def test_documented_layout(self):
        resolved = canonical_columns(HEADERS)
        self.assertIn("cgm_mgdl", resolved)
        self.assertIn("dietary_intake_cn", resolved)
        self.assertIn("insulin_bolus_iu", resolved)

    def test_trailing_space_layout_keeps_cgm(self):
        headers = [
            "Date", "CGM ", "CBG ", "Blood Ketone ",
            "Dietary intake", "饮食", "Insulin dose - s.c.",
            "Non-insulin hypoglycemic agents",
            "CSII - bolus insulin ", "CSII - basal insulin ",
            "Insulin dose - i.v.",
        ]
        resolved = canonical_columns(headers)
        self.assertEqual(resolved["cgm_mgdl"], "CGM ")
        self.assertEqual(resolved["cbg_mgdl"], "CBG ")

    def test_cjk_diet_header_variant(self):
        headers = list(HEADERS)
        headers[5] = "进食量"
        resolved = canonical_columns(headers)
        self.assertEqual(resolved["dietary_intake_cn"], "进食量")

    def test_double_space_bolus_and_cjk_basal(self):
        headers = list(HEADERS)
        headers[8] = "CSII - bolus insulin (Novolin R  IU)"
        headers[9] = "胰岛素泵基础量 (Novolin R, IU / H)"
        resolved = canonical_columns(headers)
        self.assertEqual(resolved["insulin_bolus_iu"], headers[8])
        self.assertEqual(resolved["insulin_basal_iu_h"], headers[9])

    def test_read_sheet_reads_cgm_despite_trailing_space(self):
        headers = list(HEADERS)
        headers[1] = "CGM "
        with tempfile.TemporaryDirectory() as tmp:
            path = write_sheet(Path(tmp), headers=headers)
            sheet = read_sheet(path)
        self.assertEqual(sheet["cgm_mgdl"].notna().sum(), len(sheet))

    def test_missing_required_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_sheet(Path(tmp), headers=["Date", "Notes", "x", "y", "z", "w"])
            with self.assertRaises(ValueError):
                read_sheet(path)

    def test_workbook_paths_skips_lock_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_sheet(Path(tmp))
            (Path(tmp) / "~$2000_0_20210301.xlsx").write_bytes(b"lock")
            self.assertEqual(len(workbook_paths(Path(tmp))), 1)


class ResponseTests(unittest.TestCase):
    def _sheet(self, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = write_sheet(Path(tmp.name), **kwargs)
        return read_sheet(path)

    def test_g0_is_last_reading_at_or_before_meal(self):
        stamps, values = grid(start="2021-03-01 08:00", steps=12, slope=2.0)
        sheet = self._sheet(
            timestamps=stamps, glucose=values, meals={stamps[4]: "Rice 100 g"}
        )
        row = compute_responses(sheet).iloc[0]
        self.assertEqual(row["g0_mgdl"], 108.0)  # index 4 -> 100 + 2*4
        self.assertEqual(row["g0_staleness_min"], 0.0)

    def test_g0_never_reads_a_future_reading(self):
        stamps, values = grid(start="2021-03-01 08:00", steps=12)
        sheet = self._sheet(
            timestamps=stamps, glucose=values, meals={"2021-03-01 07:30": "Rice 100 g"}
        )
        row = compute_responses(sheet).iloc[0]
        self.assertTrue(pd.isna(row["g0_mgdl"]))
        self.assertEqual(row["reason_t60"], "no_g0")

    def test_outcome_is_nearest_reading_to_horizon(self):
        stamps, values = grid(start="2021-03-01 08:00", steps=24, slope=1.0)
        sheet = self._sheet(
            timestamps=stamps, glucose=values, meals={stamps[0]: "Rice 100 g"}
        )
        row = compute_responses(sheet).iloc[0]
        self.assertEqual(row["g0_mgdl"], 100.0)
        self.assertEqual(row["glucose_t60"], 104.0)  # 4 steps * 15 min
        self.assertEqual(row["rise_t60"], 4.0)
        self.assertEqual(int(row["valid_t60"]), 1)
        self.assertEqual(row["reason_t60"], "")

    def test_stale_g0_is_excluded(self):
        stamps, values = grid(start="2021-03-01 08:00", steps=24, step_min=30)
        sheet = self._sheet(
            timestamps=stamps, glucose=values, meals={"2021-03-01 08:20": "Rice 100 g"}
        )
        row = compute_responses(sheet).iloc[0]
        self.assertEqual(row["g0_staleness_min"], 20.0)
        self.assertEqual(int(row["valid_t60"]), 0)
        self.assertEqual(row["reason_t60"], "g0_stale")

    def test_out_of_range_glucose_is_excluded(self):
        stamps, values = grid(start="2021-03-01 08:00", steps=24, base=20.0, slope=0.0)
        sheet = self._sheet(
            timestamps=stamps, glucose=values, meals={stamps[0]: "Rice 100 g"}
        )
        row = compute_responses(sheet).iloc[0]
        self.assertEqual(row["reason_t60"], "g0_out_of_range")

    def test_series_gap_is_excluded(self):
        stamps = [pd.Timestamp("2021-03-01 08:00")] + list(
            pd.date_range("2021-03-01 09:00", periods=8, freq="15min")
        )
        values = [100.0 + i for i in range(len(stamps))]
        sheet = self._sheet(
            timestamps=stamps, glucose=values, meals={stamps[0]: "Rice 100 g"}
        )
        row = compute_responses(sheet).iloc[0]
        self.assertEqual(int(row["valid_t60"]), 0)
        self.assertEqual(row["reason_t60"], "series_gap")

    def test_placeholder_text_is_not_a_meal(self):
        stamps, values = grid(steps=8)
        sheet = self._sheet(
            timestamps=stamps,
            glucose=values,
            meals={stamps[0]: "Data not available", stamps[2]: "未进食", stamps[4]: "Rice 100 g"},
        )
        self.assertEqual(int(meal_mask(sheet).sum()), 1)

    def test_meal_id_matches_legacy_scheme(self):
        stamps, values = grid(steps=8)
        sheet = self._sheet(
            timestamps=stamps, glucose=values, meals={stamps[1]: "Rice 100 g"}
        )
        row = compute_responses(sheet).iloc[0]
        self.assertEqual(row["meal_id"], "SH-2000_0_20210301-202103010815-0003")

    def test_overlap_with_next_meal_is_flagged(self):
        stamps, values = grid(steps=16)
        sheet = self._sheet(
            timestamps=stamps,
            glucose=values,
            meals={stamps[0]: "Rice 100 g", stamps[2]: "Apple 80 g"},
        )
        frame = compute_responses(sheet)
        self.assertEqual(frame["next_meal_min"].iloc[0], 30.0)
        self.assertEqual(int(frame["overlaps_next_meal_t60"].iloc[0]), 1)

    def test_insulin_near_meal_is_flagged(self):
        stamps, values = grid(steps=16)
        header = "CSII - bolus insulin (Novolin R, IU)"
        extra = {header: [np.nan] * len(stamps)}
        extra[header][1] = 6.0
        sheet = self._sheet(
            timestamps=stamps, glucose=values, meals={stamps[0]: "Rice 100 g"}, extra=extra
        )
        row = compute_responses(sheet).iloc[0]
        self.assertEqual(int(row["insulin_near_meal"]), 1)

    def test_custom_horizon_config(self):
        stamps, values = grid(steps=16, slope=1.0)
        sheet = self._sheet(
            timestamps=stamps, glucose=values, meals={stamps[0]: "Rice 100 g"}
        )
        frame = compute_responses(sheet, ResponseConfig(horizons_minutes=(30,)))
        self.assertEqual(frame["glucose_t30"].iloc[0], 102.0)
        self.assertEqual(frame["rise_t30"].iloc[0], 2.0)
        self.assertNotIn("rise_t60", frame.columns)

    def test_repeat_run_lengths(self):
        self.assertEqual(
            list(repeat_run_lengths(np.array([1.0, 1.0, 1.0, 2.0, 3.0]))),
            [3, 3, 3, 1, 1],
        )

    def test_summary_counts_reasons(self):
        stamps, values = grid(steps=16)
        with tempfile.TemporaryDirectory() as tmp:
            write_sheet(
                Path(tmp),
                timestamps=stamps,
                glucose=values,
                meals={stamps[0]: "Rice 100 g", "2021-03-01 07:00": "Rice 50 g"},
            )
            frame, summary = build_response_table(Path(tmp))
        self.assertEqual(summary["meals"], 2)
        self.assertEqual(summary["subjects"], 1)
        self.assertEqual(summary["t60"]["valid_meals"], 1)
        self.assertEqual(summary["t60"]["excluded_by_reason"], {"no_g0": 1})

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                build_response_table(Path(tmp))


if __name__ == "__main__":
    unittest.main()
