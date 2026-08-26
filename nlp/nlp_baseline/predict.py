from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from nlp.data.derive_labels import FEATURE_SOURCE, MODEL_VERSION, _is_fried, _is_large, _confidence

MODALITY = "meal_context"


def _meal_id(text: str, participant_id: str = "") -> str:
    base = f"{participant_id}|{text.strip()}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))


def _sample_id(text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"S{int(h[:8], 16) % 10000:04d}"


def predict(input_payload: dict) -> dict:
    if not isinstance(input_payload, dict):
        raise TypeError("input_payload must be dict")

    meal_id = input_payload.get("meal_id") or input_payload.get("sample_id")
    text_en = input_payload.get("dietary_intake") or input_payload.get("text") or ""
    text_cn = input_payload.get("dietary_intake_cn") or input_payload.get("text_cn") or ""
    text_en = str(text_en).strip()
    text_cn = str(text_cn).strip()
    has_text = bool(text_en or text_cn)

    participant_id = str(input_payload.get("participant_id") or "").strip()
    created_at = datetime.now(timezone.utc).isoformat()

    if not has_text:
        mid = meal_id or _sample_id("empty")
        return {
            "sample_id": mid,
            "meal_id": mid,
            "modality": MODALITY,
            "is_fried_cooking": 0,
            "is_large_portion": 0,
            "nlp_confidence": 0.0,
            "nlp_present": 0,
            "nlp_feature_source": "unavailable",
            "nlp_model_version": MODEL_VERSION,
            "feature_status": "missing_input",
            "created_at": created_at,
        }

    is_fried = _is_fried(text_en, text_cn)
    is_large = _is_large(text_en, text_cn)
    conf = _confidence(is_fried, is_large)

    if not meal_id:
        meal_id = _meal_id(text_en or text_cn, participant_id) if participant_id else _sample_id(text_en or text_cn)

    return {
        "sample_id": meal_id,
        "meal_id": meal_id,
        "modality": MODALITY,
        "is_fried_cooking": is_fried,
        "is_large_portion": is_large,
        "nlp_confidence": float(conf),
        "nlp_present": 1,
        "nlp_feature_source": FEATURE_SOURCE,
        "nlp_model_version": MODEL_VERSION,
        "feature_status": "valid",
        "created_at": created_at,
    }


if __name__ == "__main__":
    tests = [
        {"text": "Fried rice 100 g"},
        {"dietary_intake": "Rice 150 g\nVegetable 100 g", "dietary_intake_cn": "米饭150g\n青菜100g"},
        {"dietary_intake": "Cabbage 986 g\nChicken 20 g", "participant_id": "2000"},
        {"text": ""},
        {"text": "Milk 250 ml"},
    ]
    for p in tests:
        print(predict(p))
