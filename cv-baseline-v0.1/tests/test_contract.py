#!/usr/bin/env python3
"""Test kontrak cv-baseline-v0.1.

Jalan tanpa PyTorch — memakai MockClassifier. Yang diuji di sini adalah bentuk
kontrak, schema, dan penanganan error; BUKAN kualitas prediksi.

    python tests/test_contract.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_baseline.classifier import MockClassifier  # noqa: E402
from cv_baseline.macro_lookup import MacroLookup, MacroLookupError  # noqa: E402
from cv_baseline.predict import predict, set_classifier  # noqa: E402
from cv_baseline.preprocessing import preprocess  # noqa: E402
from cv_baseline.schema import REQUIRED_FIELDS, validate_output  # noqa: E402

PASSED, FAILED = 0, 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}  {detail}")


def make_image(path, size=(640, 480)):
    arr = (np.random.default_rng(0).random((size[1], size[0], 3)) * 255).astype("uint8")
    Image.fromarray(arr).save(path)


def main():
    lookup = MacroLookup()
    classes = sorted(lookup.table)
    set_classifier(MockClassifier({c: i for i, c in enumerate(classes)}))

    tmpdir = Path(tempfile.mkdtemp())
    img = tmpdir / "meal_0001.jpg"
    make_image(img)

    print("\n[1] Preprocessing")
    for size in [(640, 480), (480, 640), (224, 224), (1200, 300)]:
        p = tmpdir / f"t_{size[0]}x{size[1]}.jpg"
        make_image(p, size)
        arr = preprocess(p)
        check(f"shape untuk input {size}", arr.shape == (3, 224, 224), str(arr.shape))
    check("dtype float32", preprocess(img).dtype == np.float32)

    print("\n[2] Kontrak output")
    rec = predict({"sample_id": "meal_0001", "meal_image_path": str(img)})
    for field in REQUIRED_FIELDS:
        check(f"field '{field}' ada", field in rec)
    check("schema valid", not validate_output(rec), str(validate_output(rec)))
    check("modality = vision", rec["modality"] == "vision")
    check("top3 punya 3 kandidat", len(rec["food_top3"]) == 3, str(rec["food_top3"]))
    check("top3[0] == top1", rec["food_top3"][0] == rec["food_top1"])
    check("tidak ada kandidat duplikat", len(set(rec["food_top3"])) == 3)
    check("timestamp WIB (+07:00)", rec["created_at"].endswith("+07:00"), rec["created_at"])
    check("status mock terdeteksi", rec["feature_status"] == "mock")
    check("JSON-serializable", json.dumps(rec) is not None)

    print("\n[3] Determinisme")
    r2 = predict({"sample_id": "meal_0001", "meal_image_path": str(img)})
    check("prediksi sama untuk gambar sama", rec["food_top1"] == r2["food_top1"])

    print("\n[4] Penanganan error")
    cases = [
        ("file tidak ada", {"sample_id": "e1", "meal_image_path": str(tmpdir / "x.jpg")}),
        ("sample_id kosong", {"sample_id": "  ", "meal_image_path": str(img)}),
        ("path hilang", {"sample_id": "e3"}),
        ("payload kosong", {}),
    ]
    for name, payload in cases:
        r = predict(payload)
        check(f"{name}: status=error", r["feature_status"] == "error")
        check(f"{name}: tetap lolos schema", not validate_output(r))
        check(f"{name}: makro null bukan 0", r["calories_kcal"] is None)

    corrupt = tmpdir / "corrupt.jpg"
    corrupt.write_bytes(b"ini bukan gambar")
    r = predict({"sample_id": "e5", "meal_image_path": str(corrupt)})
    check("file korup: status=error", r["feature_status"] == "error")

    print("\n[5] Validator menolak record rusak")
    bad = [
        ("confidence > 1", {**rec, "confidence": 1.5}),
        ("confidence negatif", {**rec, "confidence": -0.1}),
        ("fiber > carbs", {**rec, "fiber_g": 9999.0}),
        ("kalori negatif", {**rec, "calories_kcal": -10}),
        ("modality salah", {**rec, "modality": "audio"}),
        ("top3[0] != top1", {**rec, "food_top1": "bakso", "food_top3": ["soto", "sate", "gudeg"]}),
        ("status tidak dikenal", {**rec, "feature_status": "selesai"}),
        ("created_at rusak", {**rec, "created_at": "kemarin"}),
        ("field hilang", {k: v for k, v in rec.items() if k != "calories_kcal"}),
        ("sample_id kosong", {**rec, "sample_id": "   "}),
    ]
    for name, r in bad:
        check(f"tertangkap: {name}", bool(validate_output(r)))

    print("\n[6] Macro lookup")
    check("10 kelas di tabel", len(lookup.table) == 10, str(len(lookup.table)))
    for c in classes:
        m = lookup.get_macros(c)
        check(f"{c}: makro non-negatif",
              all(m[k] >= 0 for k in ("carbs_g", "protein_g", "fat_g", "fiber_g", "calories_kcal")))
        check(f"{c}: fiber <= carbs", m["fiber_g"] <= m["carbs_g"])

    try:
        lookup.get_macros("pizza")
        check("kelas tak dikenal ditolak", False, "seharusnya raise")
    except MacroLookupError:
        check("kelas tak dikenal ditolak", True)

    print("\n[7] Konsistensi kelas classifier vs tabel makro")
    check("kelas tabel makro == kelas mock classifier",
          set(lookup.table) == set(classes))

    print(f"\n{'='*50}")
    print(f"PASS: {PASSED}   FAIL: {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
