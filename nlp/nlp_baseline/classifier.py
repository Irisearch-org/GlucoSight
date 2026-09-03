from __future__ import annotations

import math
import re

import numpy as np
from sklearn.isotonic import IsotonicRegression

MODEL_VERSION = "nlp-rule-v3"
FEATURE_SOURCE = "rule_based"

LABELS = ["is_fried_cooking", "is_large_portion"]
GRAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:g|ml)\b", re.I)
LARGE_THRESHOLD_G = 400
NEGATIONS = ["tidak", "nggak", "gak", "ga", "bukan", "tanpa"]

_calibrator = None

def _get_calibrator():
    global _calibrator
    if _calibrator is None:
        X = np.array([0.55, 0.65, 0.75, 0.85, 0.95])
        y = np.array([0.58, 0.62, 0.74, 0.81, 0.93])
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(X, y)
        _calibrator = ir
    return _calibrator

def _calibrate(raw: float) -> float:
    return float(_get_calibrator().predict([raw])[0])

def preprocess_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())

def _total_grams(text: str) -> float:
    return sum(float(m.group(1)) for m in GRAM_RE.finditer(text or ""))

def _gl_confidence(total: float) -> float:
    gl = total * 0.28 * 73 / 100
    return 1 / (1 + math.exp(-(gl - 20) / 5))

def _fat_grams(en: str, cn: str, total: float) -> float:
    txt = f"{en} {cn}"
    if "炸" in txt or "crispy" in (en or "").lower():
        return total * 0.18
    if "煎" in txt:
        return total * 0.12
    if "炒" in txt:
        return total * 0.10
    return total * 0.03

def _fried_confidence(en: str, cn: str, total: float) -> float:
    fat = _fat_grams(en, cn, total)
    return min(0.85, 0.60 + fat / 50 * 0.18)

def classify_context(text: str, text_cn: str = "") -> dict:
    en = str(text or "").strip()
    cn = str(text_cn or "").strip()
    if not en and not cn:
        return {
            "is_fried_cooking": 0,
            "is_large_portion": 0,
            "nlp_confidence": 0.0,
            "nlp_present": 0,
            "nlp_feature_source": "unavailable",
            "nlp_model_version": MODEL_VERSION,
        }
    en_norm = preprocess_text(en)
    is_fried = 1 if any(k in en_norm for k in ["fried", "crispy"]) or any(c in f"{en}{cn}" for c in ["炒", "煎", "炸"]) else 0
    # negation check for fried
    for kw in ["fried", "crispy"]:
        if kw in en_norm and any(n in en_norm[max(0, en_norm.find(kw)-20):en_norm.find(kw)] for n in NEGATIONS):
            is_fried = 0
    total = _total_grams(en) or _total_grams(cn)
    is_large = 1 if total > LARGE_THRESHOLD_G else 0

    if not is_fried and not is_large:
        conf = _calibrate(0.65)
    elif is_fried and is_large:
        gl_c = _gl_confidence(total)
        fried_c = _fried_confidence(en, cn, total)
        raw = 0.7 * gl_c + 0.3 * fried_c
        raw = 0.88 + (raw - 0.6) * 0.15
        conf = _calibrate(max(0.55, min(0.97, raw)))
    elif is_large:
        conf = _calibrate(_gl_confidence(total))
    else:
        conf = _calibrate(_fried_confidence(en, cn, total))

    return {
        "is_fried_cooking": is_fried,
        "is_large_portion": is_large,
        "nlp_confidence": conf,
        "nlp_present": 1,
        "nlp_feature_source": FEATURE_SOURCE,
        "nlp_model_version": MODEL_VERSION,
    }

if __name__ == "__main__":
    for en, cn in [("Fried rice 100 g",""),("Rice 150 g","米饭150g"),("Cabbage 986 g",""),("", "炒饭100g"),("", "")]:
        print(en, cn, classify_context(en, cn))
