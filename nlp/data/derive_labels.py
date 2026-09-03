from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from nlp.data.shanghai_diet_loader import DEFAULT_DATA_DIR, load_shanghai_diet_records

MODEL_VERSION = "nlp-rule-v3"
FEATURE_SOURCE = "rule_based"

FRIED_EN = ("fried", "crispy")
FRIED_CN = ("炒", "煎", "炸")
GRAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:g|ml)\b", re.I)
LARGE_THRESHOLD_G = 400

_calibrator = None

def _get_calibrator():
    global _calibrator
    if _calibrator is None:
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
        X = np.array([0.55, 0.65, 0.75, 0.85, 0.95])
        y = np.array([0.58, 0.62, 0.74, 0.81, 0.93])
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(X, y)
        _calibrator = ir
    return _calibrator

def _calibrate(raw: float) -> float:
    return float(_get_calibrator().predict([raw])[0])

def _total_grams(text: str) -> float:
    return sum(float(m.group(1)) for m in GRAM_RE.finditer(text or ""))

def _is_fried(en: str, cn: str) -> int:
    en_l = (en or "").lower()
    if any(k in en_l for k in FRIED_EN):
        return 1
    return 1 if any(c in f"{en} {cn}" for c in FRIED_CN) else 0

def _is_large(en: str, cn: str) -> int:
    return 1 if (_total_grams(en) or _total_grams(cn)) > LARGE_THRESHOLD_G else 0

def _gl_from_grams(total: float) -> float:
    return total * 0.28 * 73 / 100

def _gl_confidence(total: float) -> float:
    gl = _gl_from_grams(total)
    return 1 / (1 + math.exp(-(gl - 20) / 5))

def _fat_grams(en: str, cn: str, total: float) -> float:
    txt = f"{en} {cn}"
    if "炸" in txt or "crispy" in (en or "").lower() or "deep" in (en or "").lower():
        return total * 0.18
    if "煎" in txt:
        return total * 0.12
    if "炒" in txt:
        return total * 0.10
    return total * 0.03

def _fried_confidence(en: str, cn: str, total: float) -> float:
    fat = _fat_grams(en, cn, total)
    return min(0.85, 0.60 + fat / 50 * 0.18)

def _confidence(en: str, cn: str, is_fried: int, is_large: int, total: float) -> float:
    if not is_fried and not is_large:
        return _calibrate(0.65)
    gl_c = _gl_confidence(total) if is_large else 0.0
    fried_c = _fried_confidence(en, cn, total) if is_fried else 0.0
    if is_fried and is_large:
        raw = 0.7 * gl_c + 0.3 * fried_c
        raw = 0.88 + (raw - 0.6) * 0.15
    elif is_large:
        raw = gl_c
    else:
        raw = fried_c
    raw = max(0.55, min(0.97, raw))
    return _calibrate(raw)

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
            "nlp_confidence", "nlp_present", "nlp_feature_source", "nlp_model_version",
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
            "nlp_confidence": _confidence(en, cn, is_fried, is_large, total),
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
        "nlp_confidence", "nlp_present", "nlp_feature_source", "nlp_model_version",
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
