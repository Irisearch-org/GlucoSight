from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .shanghai_response import meal_mask
from .shanghai_schema import (
    SCHEMA_VERSION,
    SOURCE_DATASET,
    read_sheet,
    workbook_paths,
)

_HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = _HERE / "Shanghai_T2DM"


def _meal_id(participant_id: str, file_name: str, row_index: int, t0: datetime) -> str:
    return f"SH-{str(participant_id).zfill(4)}-{t0.strftime('%Y%m%d%H%M')}-{row_index:04d}"


def load_shanghai_diet_records(data_dir: Optional[Path] = None, *, generate_meal_ids: bool = True) -> pd.DataFrame:
    """Load one row per meal from the ShanghaiT2DM workbooks.

    Header matching goes through
    :func:`nlp.data.shanghai_schema.canonical_columns`, not literal string
    comparison: two of the 109 workbooks spell the CGM column ``"CGM "`` with
    a trailing space, and the old exact match read those periods as having no
    glucose at all.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    records: List[Dict] = []
    for path in workbook_paths(Path(data_dir)):
        participant_id = path.stem
        parts = participant_id.split("_")
        visit_id = parts[1] if len(parts) > 1 else ""

        sheet = read_sheet(path)
        for _, row in sheet.loc[meal_mask(sheet)].iterrows():
            t0 = row["timestamp"]
            if pd.isna(t0):
                continue
            t0 = pd.Timestamp(t0)

            cgm_val = None if pd.isna(row["cgm_mgdl"]) else float(row["cgm_mgdl"])
            cbg_val = None if pd.isna(row["cbg_mgdl"]) else float(row["cbg_mgdl"])

            records.append({
                "meal_id": _meal_id(participant_id, path.name, int(row["source_row"]), t0) if generate_meal_ids else "",
                "participant_id": str(participant_id),
                "visit_id": str(visit_id),
                "file_name": path.name,
                "row_index": int(row["source_row"]),
                "t0_timestamp": t0,
                "t0_timestamp_iso": t0.isoformat(),
                "source_dataset": SOURCE_DATASET,
                "schema_version": SCHEMA_VERSION,
                "dietary_intake": row["dietary_intake"],
                "dietary_intake_cn": row["dietary_intake_cn"],
                "cgm_mgdl": cgm_val,
                "cbg_mgdl": cbg_val,
                "cgm_available": 1 if cgm_val is not None else 0,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df.sort_values(by=["participant_id", "t0_timestamp"], inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)
    return df


def inspect_shanghai_dataset(df: Optional[pd.DataFrame] = None, data_dir: Optional[Path] = None) -> Dict:
    if df is None:
        df = load_shanghai_diet_records(data_dir=data_dir)
    if df.empty:
        print("[inspect] No records found.")
        return {"n_records": 0, "has_variation": False}

    n_records = len(df)
    n_participants = df["participant_id"].nunique()
    n_files = df["file_name"].nunique()
    n_unique_en = df["dietary_intake"].nunique()
    n_unique_cn = df["dietary_intake_cn"].nunique()

    counter = Counter(df["dietary_intake"])
    top_en = counter.most_common(5)
    top_share = top_en[0][1] / n_records if top_en else 0

    en_texts = df["dietary_intake"].tolist()
    cn_texts = df["dietary_intake_cn"].tolist()

    en_methods = {kw: sum(1 for t in en_texts if kw.lower() in t.lower()) for kw in ["fried", "boiled", "steamed", "grilled", "baked", "stir"]}
    cn_methods = {ch: sum(1 for t in cn_texts if ch in t) for ch in ["炒", "煮", "蒸", "煎", "炸", "烤", "拌", "炖", "烧"]}
    avg_len = sum(len(t) for t in en_texts) / len(en_texts) if en_texts else 0
    has_variation = n_unique_en > 500 and avg_len > 20 and top_share < 0.10 and (sum(en_methods.values()) + sum(cn_methods.values()) > 100)

    print("=" * 70)
    print("SHANGHAIT2DM DIETARY RECORD INSPECTION")
    print("=" * 70)
    print(f"Files scanned          : {n_files}")
    print(f"Participants           : {n_participants}")
    print(f"Non-empty records      : {n_records}")
    print(f"Unique EN / CN         : {n_unique_en} / {n_unique_cn}")
    print(f"Avg EN length          : {avg_len:.1f}")
    print(f"Top EN share           : {top_share:.1%} ({top_en[0][0][:60]!r})" if top_en else "Top EN share: —")
    print(f"Top 5 EN:")
    for txt, cnt in top_en:
        print(f"  {cnt:4d}x  {txt.replace(chr(10), ' / ')[:80]!r}")
    print(f"EN methods             : {en_methods}")
    print(f"CN methods             : {cn_methods}")
    print("-" * 70)
    if has_variation:
        print("GATE: PASS — real variation, Path A is genuine extraction task.")
    else:
        print("GATE: COLLAPSE — uniform/thin, Path A -> lookup, focus Path B.")
    print("=" * 70)
    print(f"Contract: {SOURCE_DATASET} / {SCHEMA_VERSION}, missing excluded (H1).")
    return {
        "n_files": n_files, "n_participants": n_participants, "n_records": n_records,
        "n_unique_en": n_unique_en, "n_unique_cn": n_unique_cn,
        "top_en": top_en, "avg_len": avg_len, "has_variation": has_variation,
        "en_methods": en_methods, "cn_methods": cn_methods,
    }


if __name__ == "__main__":
    import argparse
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="ShanghaiT2DM dietary loader — v1.2")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--head", type=int, default=5)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    print(f"Scanning: {data_dir}")
    df = load_shanghai_diet_records(data_dir=data_dir)
    if df.empty:
        print("No records.")
    else:
        print(f"Loaded {len(df)} records from {df['file_name'].nunique()} files, {df['participant_id'].nunique()} participants.")
        print(df.head(args.head).to_string(index=False))
        if args.inspect:
            inspect_shanghai_dataset(df)
        if args.save:
            out = Path(args.save)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"Saved to {out}")
