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
python tests/test_mean_baseline.py
```

## B0 population-mean pada CGMacros

Baseline B0 adalah referensi non-informatif yang sengaja tidak membaca gambar.
Untuk setiap meal dengan foto awal yang valid, outputnya selalu
`carbs_g=45.0`, `protein_g=18.0`, `fat_g=12.0`, `fiber_g=4.0`, dan
`gi_category=1`.

Raw CGMacros disimpan di `data/cgmacros/raw/` pada root repo agar tidak pernah
masuk Git. Jalankan evaluator dengan menunjuk direktori yang langsung berisi
folder peserta `CGMacros-001`, `CGMacros-002`, dan seterusnya:

```powershell
$cgRoot = "D:\Project\GlucoSight\data\cgmacros\raw\cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0\cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0\CGMacros_dateshifted365\CGMacros"
python scripts/evaluate_mean_baseline.py --dataset-root $cgRoot
```

Output per-meal (`meal_manifest.jsonl`, `predictions.jsonl`, exclusion, dan
warning) berada di `data/cgmacros/results/b0_population_mean/` dan diabaikan
Git. Ringkasan yang boleh dilacak berada di
`reports/b0_population_mean/{metrics.json,report.md}`.

Parser hanya menjadikan baris dengan `Meal Type` sebagai awal meal. Foto pada
baris itu adalah `before_image`; foto sesudahnya disimpan sebagai metadata
`after_image_paths` dan tidak pernah menjadi input B0. Makro CSV tidak dikali
lagi dengan `Amount Consumed`, dan provenance dicatat sebagai
`cgmacros_reported_estimate` karena data dictionary lokal menyebutnya sebagai
estimasi, bukan secara eksplisit weighed ground truth. Nilai makro di luar
rentang 0–176 g yang didokumentasikan CGMacros dikeluarkan sebagai error data
dan dilaporkan dalam audit; nilainya tidak diperbaiki atau dihapus diam-diam.

## Penting

`MODEL_CARD.md` memuat keterbatasan yang harus dibaca sebelum angka makro
dipakai. Ringkas: **tidak ada estimasi porsi**, `fiber_g` adalah heuristik, dan
nilai tabel TKPI masih perlu diverifikasi ulang terhadap dokumen resmi.
