# Model Card — cv-baseline-v0.1

## Ringkasan

Baseline computer vision untuk mengubah foto makanan menjadi fitur nutrisi
terstruktur, dikonsumsi tim Forecasting/Fusion lewat join `sample_id`.

Arsitektur dua tahap:

```
gambar → MobileNetV2 (klasifikasi 10 kelas) → lookup makro TKPI → fitur nutrisi
```

| | |
|---|---|
| Versi | `cv-baseline-v0.1` |
| Modality | `vision` |
| Backbone | MobileNetV2, pretrained ImageNet, fine-tuned penuh |
| Jumlah kelas | 10 |
| Ukuran input | 224 × 224 RGB |
| Ukuran checkpoint | 9.2 MB |
| Tanggal training | 2026-08-01 |

---

## Data

**Sumber gambar:** Indonesian Food Image Dataset (Mendeley), 10 kelas.

| Split | Jumlah |
|---|---|
| Train | 2.293 |
| Val | 573 (split 20% dari train, seed 42) |
| Test | 1.193 (held-out, tidak dipakai memilih checkpoint) |

Distribusi train per kelas berkisar 173–293 gambar. Kelas paling sedikit:
`gudeg` (173), `bebek_betutu` (186), `rendang` (189).

**Sumber nutrisi:** TKPI (Tabel Komposisi Pangan Indonesia, Kemenkes),
dipetakan manual ke 10 kelas → `data/tkpi_macro_lookup.csv`.

---

## Konfigurasi Training

```
optimizer      Adam
lr             1e-4
epochs         10
batch_size     32
loss           CrossEntropyLoss
freeze         tidak (seluruh backbone ikut ter-update)
augmentasi     RandomResizedCrop(0.7–1.0), HorizontalFlip, ColorJitter(0.2)
seleksi        checkpoint dengan val accuracy tertinggi
```

---

## Hasil

| Metrik | Nilai |
|---|---|
| Best val accuracy | 0.867 (epoch 10) |
| **Test accuracy** | **0.866** |
| Macro F1 (test) | 0.858 |

### Per kelas (test set)

| Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bakso | 0.917 | 0.854 | 0.884 | 130 |
| bebek_betutu | 0.767 | 0.814 | 0.790 | 97 |
| gado_gado | 0.853 | 0.823 | 0.838 | 113 |
| gudeg | 0.780 | 0.780 | 0.780 | 82 |
| nasi_goreng | 0.952 | 0.903 | 0.927 | 155 |
| pempek | 0.852 | 0.935 | 0.891 | 123 |
| rawon | 0.852 | 0.885 | 0.868 | 104 |
| rendang | 0.872 | 0.798 | 0.833 | 94 |
| sate | 0.940 | 0.909 | 0.924 | 154 |
| soto | 0.810 | 0.879 | 0.844 | 141 |

**Terbaik:** `nasi_goreng` dan `sate` (F1 > 0.92) — visualnya paling khas.

**Terlemah:** `gudeg` (0.780) dan `bebek_betutu` (0.790). Keduanya kelas dengan
data paling sedikit, dan secara visual mudah tertukar dengan hidangan berkuah
santan / daging berbumbu gelap lainnya.

Kesenjangan train accuracy (0.992) vs test (0.866) menunjukkan overfitting
sedang. Val loss berhenti turun sekitar epoch 4 sementara train loss terus
turun — menambah epoch tanpa regularisasi tambahan kemungkinan tidak menolong.

---

## Keterbatasan

Bagian ini sengaja ditulis eksplisit. Angka makro dari model ini **tidak boleh
diperlakukan sebagai pengukuran nutrisi**.

### 1. Tidak ada estimasi porsi — sumber error terbesar

Model ini classifier, bukan estimator volume/massa. Setiap kelas dipetakan ke
satu asumsi porsi tetap (`assumed_serving_g` di output). Sepiring nasi goreng
porsi kecil dan porsi jumbo menghasilkan angka makro identik.

Ini bukan error kecil — variasi porsi nyata bisa ±50% atau lebih. Akurasi
klasifikasi 0.866 **tidak berarti** akurasi makro 0.866.

### 2. `fiber_g` adalah heuristik, bukan data TKPI

TKPI tidak memiliki kolom serat. Nilai `fiber_g` diestimasi sebagai proporsi
dari karbohidrat per kelas (`fiber_ratio_of_carbs`). Ditandai di output lewat
field `fiber_source: "estimated_from_carb_ratio"`. Ini yang paling lemah di
antara kelima makro.

### 3. Dua mapping TKPI berkepercayaan rendah

| Kelas | Masalah |
|---|---|
| `sate` | TKPI tidak punya entri sate generik; nilai adalah rata-rata beberapa jenis daging. Sate ayam vs sate kambing berbeda signifikan di lemak. |
| `bebek_betutu` | Tidak ada entri langsung; diturunkan dari daging bebek olahan. |

Ditandai di output lewat `macro_mapping_confidence`. Filter
`macro_mapping_confidence == "low"` untuk mengecualikan sample ini dari analisis
sensitif.

### 4. Nilai tabel makro perlu diverifikasi ulang

`data/tkpi_macro_lookup.csv` di repo ini adalah rekonstruksi, bukan ekstraksi
langsung dari file TKPI resmi. Nilai-nilainya masuk akal secara besaran, tapi
**harus diverifikasi terhadap dokumen TKPI Kemenkes asli sebelum dipakai
produksi.** Kolom `source_note` mencatat asal tiap baris.

### 5. Hanya 10 kelas — tidak ada penanganan out-of-distribution

Model selalu mengembalikan salah satu dari 10 kelas. Foto pizza, foto non-makanan,
atau makanan Indonesia di luar 10 kelas tetap dapat prediksi, kadang dengan
confidence tinggi. Belum ada deteksi OOD.

### 6. Satu gambar = satu makanan

Tidak ada deteksi multi-item. Foto nasi campur dengan beberapa lauk akan
diklasifikasi sebagai satu kelas saja.

### 7. Domain gambar

Dataset Mendeley didominasi foto rapi/well-lit. Foto meal real-world (cahaya
redup, sudut miring, sebagian termakan) kemungkinan berperforma lebih rendah
dari angka test di atas.

---

## Kontrak Output

```json
{
  "sample_id": "meal_0001",
  "modality": "vision",
  "food_top1": "nasi_goreng",
  "food_top3": ["nasi_goreng", "gado_gado", "soto"],
  "carbs_g": 67.5,
  "protein_g": 11.3,
  "fat_g": 16.3,
  "fiber_g": 2.0,
  "calories_kcal": 465,
  "confidence": 0.9123,
  "feature_status": "ok",
  "model_version": "cv-baseline-v0.1",
  "created_at": "2026-08-01T18:30:00+07:00",

  "assumed_serving_g": 250.0,
  "macro_mapping_confidence": "high",
  "tkpi_item_name": "Nasi goreng",
  "fiber_source": "estimated_from_carb_ratio"
}
```

Field di bawah garis kosong adalah tambahan di luar kontrak wajib — aman
diabaikan downstream, tapi berguna untuk audit.

### Nilai `feature_status`

| Nilai | Arti | Aksi downstream |
|---|---|---|
| `ok` | Prediksi model terlatih, confidence ≥ 0.60 | Pakai normal |
| `low_confidence` | Prediksi model terlatih, confidence < 0.60 | Pertimbangkan down-weight |
| `mock` | Dihasilkan MockClassifier saat testing | **Jangan pakai** |
| `error` | Gagal diproses; field makro bernilai `null` | Tangani sebagai missing |

Pada record `error`, field makro sengaja `null` — bukan `0`. Nilai `0` akan
terbaca sebagai "makanan tanpa kalori" dan diam-diam merusak agregasi.

---

## Penggunaan yang Tidak Sesuai

Model ini **tidak boleh** dipakai untuk:

- Perhitungan dosis insulin atau keputusan medis apa pun
- Klaim nutrisi kepada pengguna akhir tanpa disclaimer
- Makanan di luar 10 kelas yang dilatih

---

## Rencana Perbaikan

| Prioritas | Item |
|---|---|
| Tinggi | Estimasi porsi (referensi objek / segmentasi) — sumber error terbesar |
| Tinggi | Verifikasi ulang tabel makro terhadap TKPI resmi |
| Sedang | Deteksi OOD (threshold entropi / kelas "unknown") |
| Sedang | Tambah data untuk `gudeg` dan `bebek_betutu` |
| Rendah | Regresi makro langsung menggantikan lookup (butuh dataset berlabel massa) |
| Rendah | Deteksi multi-item untuk piring campur |
