from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from nlp.data.shanghai_diet_loader import DEFAULT_DATA_DIR, load_shanghai_diet_records

MODEL_VERSION = "nlp-rule-v3"
FEATURE_SOURCE = "rule_based"

FRIED_EN = ("fried", "crispy", "deep-fried")
FRIED_ID = ("digoreng", "goreng", "menggoreng", "gorengan")
FRIED_CN = ("炒", "煎", "炸")
FRIED_NEGATIONS = ("tidak digoreng", "bukan digoreng", "tidak goreng", "bukan goreng", "not fried", "not deep-fried")
LARGE_PORTION_PATTERNS = (
    re.compile(r"\b(?:jumbo|large|big)\b", re.I),
    re.compile(r"\b(?:porsi|portion)\s+(?:besar|large|jumbo|extra besar)\b", re.I),
    re.compile(r"\b(?:besar|large|jumbo)\s+(?:porsi|portion)\b", re.I),
)
SMALL_PORTION_PATTERNS = (
    re.compile(r"\b(?:porsi|portion)\s+(?:kecil|small)\b", re.I),
    re.compile(r"\b(?:kecil|small)\s+(?:porsi|portion)\b", re.I),
)
GRAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:g|ml)\b", re.I)
LARGE_THRESHOLD_G = 400

def _total_grams(text: str) -> float:
    return sum(float(m.group(1)) for m in GRAM_RE.finditer(text or ""))

def _is_fried(en: str, cn: str) -> int:
    combined = f"{en or ''} {cn or ''}".casefold()
    if any(negation in combined for negation in FRIED_NEGATIONS):
        return 0
    return int(
        any(term in combined for term in (*FRIED_EN, *FRIED_ID, *FRIED_CN))
    )

def _is_large(en: str, cn: str) -> int:
    combined = f"{en or ''} {cn or ''}"
    if any(pattern.search(combined) for pattern in LARGE_PORTION_PATTERNS):
        return 1
    if any(pattern.search(combined) for pattern in SMALL_PORTION_PATTERNS):
        return 0
    total = _total_grams(en) or _total_grams(cn)
    return int(total > LARGE_THRESHOLD_G)

def _clean(text: str) -> str:
    return str(text or "").replace("\r", "").replace("\n", " | ").strip()

def derive_labels(df: Optional[pd.DataFrame] = None, data_dir: Optional[Path] = None) -> pd.DataFrame:
    if df is None:
        df = load_shanghai_diet_records(data_dir=data_dir)
    if df.empty:
        return pd.DataFrame(columns=[
            "meal_id", "participant_id", "t0_timestamp_iso",
            "dietary_intake", "cgm_mgdl",
            "total_grams", "is_fried_cooking", "is_large_portion",
            "nlp_confidence", "nlp_calibration_status", "nlp_present", "nlp_feature_source", "nlp_model_version",
        ])
    rows = []
    for _, r in df.iterrows():
        en = r.get("dietary_intake", "") or ""
        cn = r.get("dietary_intake_cn", "") or ""
        is_fried = _is_fried(en, cn)
        is_large = _is_large(en, cn)
        total = _total_grams(en) or _total_grams(cn)
        pid = str(r.get("file_name", "")).replace(".xlsx", "").replace(".xls", "") if r.get("file_name") else str(r["participant_id"])
        rows.append({
            "meal_id": r["meal_id"],
            "participant_id": pid,
            "t0_timestamp_iso": r["t0_timestamp_iso"],
            "dietary_intake": _clean(en),
            "cgm_mgdl": r.get("cgm_mgdl"),
            "total_grams": total,
            "is_fried_cooking": is_fried,
            "is_large_portion": is_large,
            "nlp_confidence": 0.0,
            "nlp_calibration_status": "unavailable",
            "nlp_present": 1,
            "nlp_feature_source": FEATURE_SOURCE,
            "nlp_model_version": MODEL_VERSION,
        })
    out = pd.DataFrame(rows)
    out.sort_values(by=["participant_id", "t0_timestamp_iso"], inplace=True, kind="mergesort")
    out.reset_index(drop=True, inplace=True)
    cols = [
        "meal_id", "participant_id", "t0_timestamp_iso",
        "dietary_intake", "cgm_mgdl",
        "total_grams", "is_fried_cooking", "is_large_portion",
        "nlp_confidence", "nlp_calibration_status", "nlp_present", "nlp_feature_source", "nlp_model_version",
    ]
    return out[cols]

if __name__ == "__main__":
    import argparse
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Derive v1.2 NLP features")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
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
