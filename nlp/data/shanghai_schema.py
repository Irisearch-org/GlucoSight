"""Schema-tolerant reader for the ShanghaiT2DM participant workbooks.

The v3 figshare release (``10.6084/m9.figshare.20444397.v3``, CC BY 4.0)
ships 109 workbooks, one per CGM period, and **five** different header
layouts:

===================================  =====
layout                               files
===================================  =====
documented layout                      102
``进食量`` instead of ``饮食``              3
trailing spaces (``CGM ``, ``CBG ``)      2
``Novolin R  IU`` (no comma)              1
``胰岛素泵基础量`` for the basal column      1
===================================  =====

``shanghai_diet_loader._find_col`` compares header strings exactly, after a
``.strip()``.  For the two trailing-space files that means ``CGM `` never
matches ``"CGM (mg / dl)"``, so ``idx_cgm`` is ``None`` and the whole
workbook is silently read as *having no glucose*.  Nothing raises; the
period just contributes measurements with a missing outcome.

Every reader in this package therefore goes through
:func:`canonical_columns` instead of matching literal header strings.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

SOURCE_DATASET = "shanghai_t2dm"
SCHEMA_VERSION = "1.2"

#: Canonical field names, in output order.
CANONICAL_COLUMNS = (
    "timestamp",
    "cgm_mgdl",
    "cbg_mgdl",
    "ketone_mmol_l",
    "dietary_intake",
    "dietary_intake_cn",
    "insulin_sc_iu",
    "insulin_bolus_iu",
    "insulin_basal_iu_h",
    "insulin_iv_iu",
    "oral_agent",
)

_TEXT_COLUMNS = frozenset({"dietary_intake", "dietary_intake_cn", "oral_agent"})
_REQUIRED_COLUMNS = ("timestamp", "cgm_mgdl", "dietary_intake")

_CJK_DIET_HEADERS = frozenset({"饮食", "进食量"})
_CJK_BASAL_MARKERS = ("基础量",)

#: ``2000_0_20201230`` -> subject ``2000``, period ``0``, period date.
PARTICIPANT_FILE_RE = re.compile(r"^(\d{4})_(\d+)_(\d{8})$")


def normalise_header(name) -> str:
    """Fold a header cell to lowercase words, keeping CJK characters.

    Punctuation collapses to single spaces so that ``"CGM (mg / dl)"``,
    ``"CGM "`` and ``"CGM"`` all normalise identically.
    """
    text = "" if name is None else str(name)
    text = text.strip().lower()
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text).strip()


def canonical_columns(columns: Iterable) -> Dict[str, str]:
    """Map canonical field names to the workbook's actual header strings."""
    resolved: Dict[str, str] = {}
    for column in columns:
        name = normalise_header(column)
        if name == "date" and "timestamp" not in resolved:
            resolved["timestamp"] = column
        elif name.startswith("cgm") and "cgm_mgdl" not in resolved:
            resolved["cgm_mgdl"] = column
        elif name.startswith("cbg") and "cbg_mgdl" not in resolved:
            resolved["cbg_mgdl"] = column
        elif "ketone" in name and "ketone_mmol_l" not in resolved:
            resolved["ketone_mmol_l"] = column
        elif name == "dietary intake" and "dietary_intake" not in resolved:
            resolved["dietary_intake"] = column
        elif name in _CJK_DIET_HEADERS and "dietary_intake_cn" not in resolved:
            resolved["dietary_intake_cn"] = column
        elif name.startswith("insulin dose s c") and "insulin_sc_iu" not in resolved:
            resolved["insulin_sc_iu"] = column
        elif name.startswith("csii bolus") and "insulin_bolus_iu" not in resolved:
            resolved["insulin_bolus_iu"] = column
        elif (
            name.startswith("csii basal")
            or any(marker in name for marker in _CJK_BASAL_MARKERS)
        ) and "insulin_basal_iu_h" not in resolved:
            resolved["insulin_basal_iu_h"] = column
        elif name.startswith("insulin dose i v") and "insulin_iv_iu" not in resolved:
            resolved["insulin_iv_iu"] = column
        elif name.startswith("non insulin") and "oral_agent" not in resolved:
            resolved["oral_agent"] = column
    return resolved


def participant_file_ids(path: Path) -> Optional[Dict[str, object]]:
    """Return the participant/period metadata encoded in a workbook filename."""
    match = PARTICIPANT_FILE_RE.match(Path(path).stem)
    if not match:
        return None
    subject, period, _ = match.groups()
    return {"subject_id": subject, "period_index": int(period)}


def read_sheet(path: Path) -> pd.DataFrame:
    """Read one participant workbook with canonical column names.

    Returns one row per spreadsheet row, including the dense CGM grid.  Rows
    with an unparseable timestamp are kept (``timestamp`` is ``NaT``) so the
    caller can count them rather than lose them silently.
    """
    path = Path(path)
    raw = pd.read_excel(path)

    resolved = canonical_columns(raw.columns)
    missing = [name for name in _REQUIRED_COLUMNS if name not in resolved]
    if missing:
        raise ValueError(
            f"{path.name}: missing required columns {missing}; "
            f"headers={list(raw.columns)}"
        )

    out = pd.DataFrame(index=raw.index)
    out["timestamp"] = pd.to_datetime(
        raw[resolved["timestamp"]], errors="coerce"
    )
    for name in CANONICAL_COLUMNS:
        if name == "timestamp":
            continue
        if name in resolved:
            column = raw[resolved[name]]
            if name in _TEXT_COLUMNS:
                text = column.where(column.notna(), "").astype(str).str.strip()
                # Meal cells are multi-line in the workbooks ("Rice 150 g\n
                # Vegetable 100 g"). Flatten to a single line so the text
                # survives CSV round-trips and joins against the legacy
                # feature table, which already uses " | ".
                text = text.str.replace("\r\n", "\n", regex=False)
                # Meal cells are multi-line in the workbooks ("Rice 150 g\n
                # Vegetable 100 g"). Flatten to a single line so the text
                # survives CSV round-trips and joins against the legacy
                # feature table, which already uses " | ".  Replace the
                # newline only: the surrounding spaces are the author's.
                text = text.str.replace("\n", " | ", regex=False)
                out[name] = text
            else:
                out[name] = pd.to_numeric(column, errors="coerce")
        elif name in _TEXT_COLUMNS:
            out[name] = ""
        else:
            out[name] = np.nan

    out["source_row"] = np.arange(len(out), dtype=int) + 2
    out["participant_id"] = path.stem
    out["source_dataset"] = SOURCE_DATASET
    out["schema_version"] = SCHEMA_VERSION
    identity = participant_file_ids(path)
    out["subject_id"] = identity["subject_id"] if identity else path.stem
    out["period_index"] = identity["period_index"] if identity else 0
    return out


def workbook_paths(data_dir: Path) -> List[Path]:
    """List participant workbooks, ignoring Excel lock files."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"ShanghaiT2DM directory not found: {data_dir}")
    paths = sorted(
        path
        for path in data_dir.iterdir()
        if path.suffix.lower() in {".xls", ".xlsx"} and not path.name.startswith("~$")
    )
    if not paths:
        raise FileNotFoundError(f"No .xls/.xlsx workbooks in: {data_dir}")
    return paths
