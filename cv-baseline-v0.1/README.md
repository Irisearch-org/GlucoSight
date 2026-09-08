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

## Notebook evaluasi OOD pada CGMacros

Buka `notebooks/cgmacros_ood_evaluation.ipynb` di VS Code/Jupyter dengan
kernel Python 3.10–3.12. Jalankan sel dari atas; notebook mencari checkout repo
dan raw CGMacros secara otomatis, atau gunakan override path pada konfigurasi.

```powershell
Set-Location D:\Project\GlucoSight
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r cv-baseline-v0.1/requirements-notebook.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name glucosight-cv --display-name "GlucoSight CV (NumPy 1.26)"
.\.venv\Scripts\python.exe -m jupyter lab cv-baseline-v0.1/notebooks/cgmacros_ood_evaluation.ipynb
```

Default `MAX_MEALS=32` adalah uji pipeline dengan subset deterministik.
Untuk hasil penuh, ubah menjadi `None`, restart kernel, lalu **Run All**.
Notebook memakai model sungguhan, menyertakan prediksi low-confidence,
mengaudit error, dan menghitung ulang B0 pada meal valid yang persis sama.
Output meliputi MAE/RMSE per makro, median/IQR per peserta, histogram
confidence, laporan Markdown, dan draft bagian model card.

Prediksi/detail peserta disimpan di `data/cgmacros/results/` pada root repo
(gitignored). Ringkasan full run disimpan di `cv/reports/indonesian_classifier_ood/`
pada root repo; hasil trial tetap di direktori data. Model card tidak diubah
otomatis. Bersihkan output notebook sebelum commit.

Jika audit berisi `inference gagal: Numpy is not available`, periksa pasangan
versi NumPy/PyTorch. Untuk PyTorch 2.2.x yang dipakai baseline, jalankan
`%pip install "numpy>=1.26.4,<2"` di sel notebook pada kernel environment CV,
lalu **Restart Kernel** dan
**Run All**. Kernel yang masih hidup dapat tetap memakai NumPy versi lama
walaupun pip sudah selesai. Notebook mengecek konversi NumPy ↔ tensor sebelum
menjalankan batch agar error kompatibilitas ini terlihat lebih awal.
Pilih kernel **GlucoSight CV (NumPy 1.26)**; jangan memasang NumPy 1.x ke
Python global yang juga dipakai paket lain yang membutuhkan NumPy 2.

## Penting

`MODEL_CARD.md` memuat keterbatasan yang harus dibaca sebelum angka makro
dipakai. Ringkas: **tidak ada estimasi porsi**, `fiber_g` adalah heuristik, dan
nilai tabel TKPI masih perlu diverifikasi ulang terhadap dokumen resmi.
