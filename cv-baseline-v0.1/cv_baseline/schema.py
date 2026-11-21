"""Schema validator untuk output contract cv-baseline-v0.1.

Catatan: tim lain (Forecasting/Fusion) belum menyediakan shared validator, jadi
validator ini dibangun mandiri dari spesifikasi di task brief. Kalau nanti
shared validator tersedia, modul ini harus diganti — bukan dipertahankan
berdampingan, agar tidak ada dua sumber kebenaran.
"""

from datetime import datetime
from typing import List

REQUIRED_FIELDS = {
    "sample_id": str,
    "modality": str,
    "food_top1": str,
    "food_top3": list,
    "carbs_g": (int, float),
    "protein_g": (int, float),
    "fat_g": (int, float),
    "fiber_g": (int, float),
    "calories_kcal": (int, float),
    "confidence": (int, float),
    "feature_status": str,
    "model_version": str,
    "created_at": str,
}

VALID_FEATURE_STATUS = {
    "ok",              # prediksi model terlatih, confidence di atas ambang
    "low_confidence",  # prediksi model terlatih, confidence di bawah ambang
    "mock",            # dihasilkan MockClassifier — TIDAK valid untuk downstream
    "error",           # gagal diproses; field makro bernilai null
}

NULLABLE_ON_ERROR = {
    "food_top1", "carbs_g", "protein_g", "fat_g",
    "fiber_g", "calories_kcal", "confidence",
}


def validate_output(record: dict) -> List[str]:
    """Kembalikan list pesan error. List kosong = valid."""
    errors: List[str] = []
    is_error_record = record.get("feature_status") == "error"

    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in record:
            errors.append(f"field wajib hilang: '{field}'")
            continue

        value = record[field]
        if value is None:
            if not (is_error_record and field in NULLABLE_ON_ERROR):
                errors.append(f"'{field}' tidak boleh null")
            continue

        if not isinstance(value, expected_type):
            errors.append(
                f"'{field}' bertipe {type(value).__name__}, "
                f"seharusnya {expected_type}"
            )

    if errors:
        return errors  # tipe belum benar, cek nilai di bawah tidak bermakna

    if record["modality"] != "vision":
        errors.append(f"'modality' harus 'vision', bukan '{record['modality']}'")

    if record["feature_status"] not in VALID_FEATURE_STATUS:
        errors.append(
            f"'feature_status' tidak dikenal: '{record['feature_status']}'. "
            f"Nilai valid: {sorted(VALID_FEATURE_STATUS)}"
        )

    if not is_error_record:
        if not 0.0 <= record["confidence"] <= 1.0:
            errors.append(f"'confidence' di luar rentang [0,1]: {record['confidence']}")

        for field in ("carbs_g", "protein_g", "fat_g", "fiber_g", "calories_kcal"):
            if record[field] < 0:
                errors.append(f"'{field}' negatif: {record[field]}")

        if record["fiber_g"] > record["carbs_g"]:
            errors.append("'fiber_g' melebihi 'carbs_g' — tidak konsisten")

        top3 = record["food_top3"]
        if not top3:
            errors.append("'food_top3' kosong")
        elif not all(isinstance(x, str) for x in top3):
            errors.append("'food_top3' harus berisi string")
        elif top3[0] != record["food_top1"]:
            errors.append("'food_top3'[0] harus sama dengan 'food_top1'")

    if not record["sample_id"].strip():
        errors.append("'sample_id' kosong")

    try:
        datetime.fromisoformat(record["created_at"])
    except (ValueError, TypeError):
        errors.append(f"'created_at' bukan ISO 8601 valid: {record['created_at']}")

    return errors


def is_valid(record: dict) -> bool:
    return not validate_output(record)
