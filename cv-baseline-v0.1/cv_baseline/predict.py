"""Kontrak utama: predict(input_payload: dict) -> dict.

Input:
    {"sample_id": "meal_0001", "meal_image_path": "data/raw/images/meal_0001.jpg"}

Output: lihat REQUIRED_FIELDS di schema.py.

Model dan tabel makro di-cache di level modul, jadi pemanggilan predict()
berulang tidak memuat ulang checkpoint dari disk.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from .classifier import MobileNetV2Classifier, ModelNotAvailable
from .config import CONFIDENCE_THRESHOLD, MODEL_VERSION, TIMEZONE_OFFSET_HOURS, TOP_K
from .macro_lookup import MacroLookup, MacroLookupError
from .preprocessing import ImageLoadError, preprocess

WIB = timezone(timedelta(hours=TIMEZONE_OFFSET_HOURS))

_classifier = None
_macro_lookup = None


def _now_wib() -> str:
    return datetime.now(WIB).isoformat()


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = MobileNetV2Classifier().load()
    return _classifier


def get_macro_lookup() -> MacroLookup:
    global _macro_lookup
    if _macro_lookup is None:
        _macro_lookup = MacroLookup()
    return _macro_lookup


def set_classifier(classifier):
    """Inject classifier lain (mis. MockClassifier untuk testing)."""
    global _classifier
    _classifier = classifier


def _error_record(sample_id: str, reason: str) -> dict:
    """Record error tetap memenuhi schema — field makro null, bukan 0.

    Nilai 0 akan terbaca downstream sebagai 'makanan tanpa kalori' dan diam-diam
    merusak agregasi. Null memaksa Forecasting/Fusion menangani gap secara sadar.
    """
    return {
        "sample_id": sample_id,
        "modality": "vision",
        "food_top1": None,
        "food_top3": [],
        "carbs_g": None,
        "protein_g": None,
        "fat_g": None,
        "fiber_g": None,
        "calories_kcal": None,
        "confidence": None,
        "feature_status": "error",
        "model_version": MODEL_VERSION,
        "created_at": _now_wib(),
        "error_reason": reason,
    }


def predict(input_payload: dict) -> dict:
    sample_id = str(input_payload.get("sample_id", "")).strip()
    image_path: Optional[str] = input_payload.get("meal_image_path")

    if not sample_id:
        return _error_record("UNKNOWN", "sample_id kosong atau tidak ada")
    if not image_path:
        return _error_record(sample_id, "meal_image_path tidak ada di payload")

    # 1. preprocessing
    try:
        image_array = preprocess(image_path)
    except ImageLoadError as exc:
        return _error_record(sample_id, str(exc))

    # 2. klasifikasi
    try:
        clf = get_classifier()
        topk = clf.predict_topk(image_array, k=TOP_K)
    except ModelNotAvailable as exc:
        return _error_record(sample_id, f"model tidak tersedia: {exc}")
    except Exception as exc:
        return _error_record(sample_id, f"inference gagal: {exc}")

    if not topk:
        return _error_record(sample_id, "classifier tidak mengembalikan prediksi")

    food_top1, confidence = topk[0]
    food_topk = [name for name, _ in topk]

    # 3. lookup makro
    try:
        macros = get_macro_lookup().get_macros(food_top1)
    except MacroLookupError as exc:
        return _error_record(sample_id, str(exc))

    # 4. tentukan feature_status
    if getattr(clf, "is_mock", False):
        feature_status = "mock"
    elif confidence < CONFIDENCE_THRESHOLD:
        feature_status = "low_confidence"
    else:
        feature_status = "ok"

    return {
        "sample_id": sample_id,
        "modality": "vision",
        "food_top1": food_top1,
        "food_top3": food_topk,
        "carbs_g": macros["carbs_g"],
        "protein_g": macros["protein_g"],
        "fat_g": macros["fat_g"],
        "fiber_g": macros["fiber_g"],
        "calories_kcal": macros["calories_kcal"],
        "confidence": round(float(confidence), 4),
        "feature_status": feature_status,
        "model_version": MODEL_VERSION,
        "created_at": _now_wib(),
        # --- field tambahan (di luar kontrak wajib, aman diabaikan downstream) ---
        "assumed_serving_g": macros["_serving_g"],
        "macro_mapping_confidence": macros["_mapping_confidence"],
        "tkpi_item_name": macros["_tkpi_item_name"],
        "fiber_source": "estimated_from_carb_ratio",
    }
