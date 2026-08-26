from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from nlp.data.shanghai_diet_loader import DEFAULT_DATA_DIR, load_shanghai_diet_records

MODEL_VERSION = "nlp-rule-v0.1"
FEATURE_SOURCE = "rule_based"

FRIED_EN = ("fried", "crispy")
FRIED_CN = ("炒", "煎", "炸")

GRAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:g|ml)\b", re.I)
LARGE_THRESHOLD_G = 400


def _total_grams(text: str) -> float:
    return sum(float(m.group(1)) for m in GRAM_RE.finditer(text or ""))


def _is_fried(en: str, cn: str) -> int:
    en_l = (en or "").lower()
    if any(k in en_l for k in FRIED_EN):
        return 1
    combined = f"{en} {cn}"
    return 1 if any(c in combined for c in FRIED_CN) else 0


def _is_large(en: str, cn: str) -> int:
    total = _total_grams(en)
    if total == 0:
        total = _total_grams(cn)
    return 1 if total > LARGE_THRESHOLD_G else 0


def _confidence(is_fried: int, is_large: int) -> float:
    if is_fried and is_large:
        return 0.95
    if is_fried or is_large:
        return 0.85
    return 0.65


def derive_labels(df: Optional[pd.DataFrame] = None, data_dir: Optional[Path] = None) -> pd.DataFrame:
    if df is None:
        df = load_shanghai_diet_records(data_dir=data_dir)
    if df.empty:
        return pd.DataFrame(columns=[
            "meal_id", "participant_id", "t0_timestamp_iso",
            "source_dataset", "schema_version",
            "is_fried_cooking", "is_large_portion",
            "nlp_confidence", "nlp_present", "nlp_feature_source", "nlp_model_version",
        ])

    rows = []
    for _, r in df.iterrows():
        en = r.get("dietary_intake", "") or ""
        cn = r.get("dietary_intake_cn", "") or ""
        is_fried = _is_fried(en, cn)
        is_large = _is_large(en, cn)
        rows.append({
            "meal_id": r["meal_id"],
            "participant_id": r["participant_id"],
            "t0_timestamp_iso": r["t0_timestamp_iso"],
            "source_dataset": r["source_dataset"],
            "schema_version": r["schema_version"],
            "is_fried_cooking": is_fried,
            "is_large_portion": is_large,
            "nlp_confidence": _confidence(is_fried, is_large),
            "nlp_present": 1,
            "nlp_feature_source": FEATURE_SOURCE,
            "nlp_model_version": MODEL_VERSION,
            "dietary_intake": en,
            "total_grams": _total_grams(en) or _total_grams(cn),
        })

    out = pd.DataFrame(rows)
    out.sort_values(by=["participant_id", "t0_timestamp_iso"], inplace=True, kind="mergesort")
    out.reset_index(drop=True, inplace=True)
    return out


if __name__ == "__main__":
    import argparse
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Derive v1.2 NLP features from Shanghai dietary records")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--save", type=str, default=None, help="CSV output path")
    parser.add_argument("--head", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    df = derive_labels(data_dir=data_dir)
    print(f"Derived {len(df)} records, fried={df['is_fried_cooking'].sum()}, large={df['is_large_portion'].sum()}")
    if not df.empty:
        print(df.head(args.head).to_string(index=False))
        print(f"confidence mean {df['nlp_confidence'].mean():.3f}")
    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"Saved to {out}")
