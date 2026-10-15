from __future__ import annotations

import re

MODEL_VERSION = "nlp-rule-v3"
FEATURE_SOURCE = "rule_based"

LABELS = ["is_fried_cooking", "is_large_portion"]
GRAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:g|ml)\b", re.I)
LARGE_THRESHOLD_G = 400
FRIED_TERMS = ("fried", "crispy", "digoreng", "goreng", "menggoreng", "gorengan", "炒", "煎", "炸")
FRIED_NEGATIONS = ("tidak digoreng", "bukan digoreng", "tidak goreng", "bukan goreng", "not fried", "not deep-fried")
LARGE_PATTERNS = (
    re.compile(r"\b(?:jumbo|large|big)\b", re.I),
    re.compile(r"\b(?:porsi|portion)\s+(?:besar|large|jumbo|extra besar)\b", re.I),
    re.compile(r"\b(?:besar|large|jumbo)\s+(?:porsi|portion)\b", re.I),
)
SMALL_PATTERNS = (
    re.compile(r"\b(?:porsi|portion)\s+(?:kecil|small)\b", re.I),
    re.compile(r"\b(?:kecil|small)\s+(?:porsi|portion)\b", re.I),
)

def preprocess_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().strip())

def _total_grams(text: str) -> float:
    return sum(float(m.group(1)) for m in GRAM_RE.finditer(text or ""))

def classify_context(text: str, text_cn: str = "") -> dict:
    en = str(text or "").strip()
    cn = str(text_cn or "").strip()
    if not en and not cn:
        return {
            "is_fried_cooking": 0,
            "is_large_portion": 0,
            "nlp_confidence": 0.0,
            "nlp_calibration_status": "unavailable",
            "nlp_present": 0,
            "nlp_feature_source": "unavailable",
            "nlp_model_version": MODEL_VERSION,
        }
    combined = f"{en} {cn}"
    combined_folded = combined.casefold()
    is_fried = int(
        not any(negation in combined_folded for negation in FRIED_NEGATIONS)
        and any(term in combined_folded for term in FRIED_TERMS)
    )
    if any(pattern.search(combined) for pattern in LARGE_PATTERNS):
        is_large = 1
    elif any(pattern.search(combined) for pattern in SMALL_PATTERNS):
        is_large = 0
    else:
        is_large = int((_total_grams(en) or _total_grams(cn)) > LARGE_THRESHOLD_G)

    return {
        "is_fried_cooking": is_fried,
        "is_large_portion": is_large,
        "nlp_confidence": 0.0,
        "nlp_calibration_status": "unavailable",
        "nlp_present": 1,
        "nlp_feature_source": FEATURE_SOURCE,
        "nlp_model_version": MODEL_VERSION,
    }

if __name__ == "__main__":
    for en, cn in [("Fried rice 100 g",""),("Rice 150 g","米饭150g"),("Cabbage 986 g",""),("", "炒饭100g"),("", "")]:
        print(en, cn, classify_context(en, cn))
