# Model Card — cv-baseline-v0.1

## Ringkasan

Baseline computer vision untuk mengubah foto makanan menjadi fitur nutrisi
terstruktur, dikonsumsi tim Forecasting/Fusion lewat join `sample_id`.

Arsitektur dua tahap:

```
gambar → MobileNetV2 (klasifikasi 10 kelas) → lookup makro TKPI → fitur nutrisi
```

> **Status (2026-11-21): MobileNetV2 tidak lagi dipakai di jalur serving.**
> Identifikasi hidangan di aplikasi kini memakai VLM (Gemini
> `gemini-3.5-flash-lite`) + embedding matcher (`gemini-embedding-001`) ke tabel
> hidangan, dengan aturan penerimaan kemiripan ≥ 0,80 dan margin ≥ 0,05
> ([#18](https://github.com/Irisearch-org/GlucoSight/issues/18),
> [#19](https://github.com/Irisearch-org/GlucoSight/issues/19),
> [`docs/SPEC_V2_PIPELINE.md`](../docs/SPEC_V2_PIPELINE.md) §2). Pada granularitas
> label MobileNet sendiri (n = 88 foto overlap), VLM + embedding unggul +6,8 pp
> (+1,1 sampai +13,6) tingkat keluarga, tetapi −20,5 pp (−31,8 sampai −9,1) pada
> tingkat hidangan eksak; MobileNet juga tidak dapat menamai ~75% hidangan di
> luar 10 kelasnya. Kartu ini dipertahankan sebagai catatan model pembanding.
> Bagian "Kontrak Output" dan lookup makro TKPI di bawah menggambarkan kontrak
> v1 yang sudah digantikan ([contract v2](../docs/interface/INTERFACE_CONTRACT_v2.md)).

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

Angka di bawah dihitung ulang dari checkpoint tersimpan
`models/mobilenetv2_food10.pt` (sha256 `394c8869…`) pada seluruh split test
food-10 Mendeley (n = 1.193) lewat jalur aplikasi (`cv/cv_baseline`). Sumber:
[`cv_accuracy_reconciliation.md`](https://github.com/Irisearch-org/GlucoSight/blob/wayfinder/cv-accuracy-reconciliation/cv/reports/cv_accuracy_reconciliation.md)
(branch `wayfinder/cv-accuracy-reconciliation`, skrip `cv/scripts/eval_food10_reconcile.py`),
[tiket #11](https://github.com/Irisearch-org/GlucoSight/issues/11).

| Metrik (food-10 test, n = 1.193) | Nilai |
|---|---|
| **Top-1 accuracy (jalur aplikasi)** | **88,1%** (1.051/1.193) |
| Top-1 accuracy (transform training) | 88,0% (1.050/1.193) — selisih 7 gambar berasal dari preprocessing |
| Top-3 accuracy | 98,0% |
| Macro F1 | 0,874 |

**Angka lama dihapus.** 0,8659 / 86,6% dan macro-F1 0,858 (notebook dan versi
kartu ini sebelumnya), serta rentang "86,6–88,0%", tidak dapat direproduksi dari
checkpoint tersimpan [INFERENCE: berasal dari bobot lain]; tabel per kelas dari
run itu ikut dihapus. Nilai 0,8801 di `training_metadata.json` memang nyata
(= 88,0% pada transform training), tetapi `train_mobilenet.py` tidak menghitung
test accuracy, jadi angka itu berasal dari evaluasi di luar repo. Klaim "Gemini
95,96% vs MobileNet 70,71% pada 99 gambar" juga dihapus: rekaman mentah VLM dan
fitur MobileNet tidak ada, dan skor MobileNet hasil rekonstruksi adalah 89/99
(89,9%).

**Bug kunci stem.** 130 stem nama file (310 gambar) muncul di lebih dari satu
folder kelas. Skrip yang mengindeks hasil dengan stem saja menghasilkan skor
tidak valid; indeks selalu dengan path lengkap.

**Hanya 10 kelas.** Angka ini hanya berlaku untuk 10 kelas food-10 dengan foto
rapi satu hidangan, dan bukan akurasi identifikasi hidangan secara umum.

Kesenjangan antara akurasi train dan test menunjukkan overfitting sedang.

---

## Evaluasi out-of-domain: CGMacros — Sprint 2 Task 2

Full run 8 September 2026 mengevaluasi 1.616 meal dari 45 peserta, dengan
coverage 100% dan tanpa kegagalan inference. Semua prediksi low-confidence
tetap dinilai, dan B0 memakai meal yang persis sama. B0 adalah konstanta
karbohidrat/protein/lemak/serat 45/18/12/4 g.

| Makro | MAE classifier (g) | MAE B0 (g) | Selisih classifier − B0 (g) |
|---|---:|---:|---:|
| Karbohidrat | 35.019 | 27.342 | +7.677 |
| Protein | 20.696 | 19.699 | +0.997 |
| Lemak | 13.099 | 12.509 | +0.590 |
| Serat | 4.277 | 4.178 | +0.098 |

Classifier tidak mengalahkan B0 pada MAE agregat keempat makro; selisih
terbesar terdapat pada karbohidrat. Median confidence adalah 0.622;
758 meal (46.9%) berstatus low-confidence dan 282 meal (17.5%) memiliki
confidence ≥0.9. Confidence ini berupa softmax mentah, belum dikalibrasi.

**Perbedaan dataset dan ruang label:** model dilatih pada 10 kelas makanan
Indonesia dari Mendeley, sedangkan evaluasi menggunakan foto makanan CGMacros
(domain Amerika). Untuk makanan di luar 10 kelas tersebut, model tetap
memilih salah satu label Indonesia yang tersedia sehingga label prediksi
dapat berbeda dari identitas makanan aslinya; misalnya, 531/1.616 foto
(32.9%) diprediksi sebagai `pempek`. Angka ini adalah distribusi prediksi,
bukan jumlah salah klasifikasi yang sudah diverifikasi: evaluasi ini tidak
memiliki anotasi kelas CGMacros yang sepadan untuk menghitung accuracy.

Estimasi makro berasal dari label prediksi melalui lookup TKPI dengan porsi
tetap (`carbs_source=class_lookup`), termasuk serat heuristik. Error mencakup
ketidakcocokan kelas/domain, lookup, dan asumsi porsi; selisih terhadap B0
tidak mengisolasi efek domain shift dan tidak menilai akurasi model pada
test set makanan Indonesia. Label acuan memakai provenance
`cgmacros_reported_estimate` sesuai data dictionary lokal, sehingga tidak
diklaim sebagai weighed ground truth. Hasil OOD ini dilaporkan terpisah
dari tabel utama evaluasi fusion.

[Laporan lengkap dan distribusi confidence](../cv/reports/indonesian_classifier_ood/20260908T053051_450315Z/report.md)
memuat hash manifest, checkpoint, dan tabel lookup, serta median/IQR MAE
antar peserta. Run terbaru ini menjadi acuan hasil Sprint 2 Task 2.

---

## Keterbatasan

Bagian ini sengaja ditulis eksplisit. Angka makro dari model ini **tidak boleh
diperlakukan sebagai pengukuran nutrisi**.

### 1. Tidak ada estimasi porsi — sumber error terbesar

Model ini classifier, bukan estimator volume/massa. Setiap kelas dipetakan ke
satu asumsi porsi tetap (`assumed_serving_g` di output). Sepiring nasi goreng
porsi kecil dan porsi jumbo menghasilkan angka makro identik.

Ini bukan error kecil — variasi porsi nyata bisa ±50% atau lebih. Akurasi
klasifikasi 0.881 **tidak berarti** akurasi makro 0.881.

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
