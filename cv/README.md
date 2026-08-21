# cv-baseline-v0.1 — CV Food Macro Estimation

Mengubah foto makanan Indonesia menjadi fitur nutrisi terstruktur untuk pipeline
Forecasting/Fusion.

## Setup

```bash
pip install -r requirements.txt
cp /path/ke/mobilenetv2_food10.pt models/
```

## Pakai sebagai library

```python
from cv_baseline import predict

hasil = predict({
    "sample_id": "meal_0001",
    "meal_image_path": "data/raw/images/meal_0001.jpg",
})
```

## Pakai lewat CLI

```bash
# dari folder gambar (sample_id = nama file tanpa ekstensi)
python scripts/run_batch.py --image-dir data/raw/images/

# dari manifest JSON
python scripts/run_batch.py --manifest samples/manifest.json

# uji kontrak tanpa PyTorch — output ditandai feature_status="mock"
python scripts/run_batch.py --image-dir samples/ --mock
```

Output ditulis ke `features/cv/cv-baseline-v0.1/`:
- `features.jsonl` — satu record per baris, join by `sample_id`
- `run_summary.json` — hitungan valid/invalid dan sebaran `feature_status`

Exit code 1 kalau ada record yang gagal validasi schema.

## Struktur

```
cv_baseline/
  config.py         konstanta terpusat (path, threshold, preprocessing)
  preprocessing.py  Pillow + numpy, replika eval transform saat training
  classifier.py     MobileNetV2 (torch di-import lazy) + MockClassifier
  macro_lookup.py   kelas -> makro dari tabel TKPI
  schema.py         validator kontrak output
  predict.py        kontrak predict(input_payload) -> dict
scripts/run_batch.py
data/tkpi_macro_lookup.csv
models/             taruh mobilenetv2_food10.pt di sini
features/cv/cv-baseline-v0.1/
tests/test_contract.py
MODEL_CARD.md       metrik + keterbatasan (baca ini sebelum pakai angkanya)
```

## Test

```bash
python tests/test_contract.py
```

## Penting

`MODEL_CARD.md` memuat keterbatasan yang harus dibaca sebelum angka makro
dipakai. Ringkas: **tidak ada estimasi porsi**, `fiber_g` adalah heuristik, dan
nilai tabel TKPI masih perlu diverifikasi ulang terhadap dokumen resmi.
