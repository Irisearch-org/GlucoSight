# 3. Methodology

## 3.1 Gambaran Sistem

GlucoSight adalah prototipe penelitian, bukan perangkat medis. Alurnya terdiri dari tiga langkah tetap. Setiap langkah melayani model yang dipakai prototipe. Komponen yang gagal evaluasinya tetap dipakai, tetapi ditampilkan bersama galatnya.

Langkah pertama memperoleh glukosa awal (g0) dengan salah satu dari tiga cara: memasukkan hasil glukometer, membaca nilai terakhir dari *continuous glucose monitoring* (CGM), yaitu pemantauan glukosa kontinu, atau memindai jari dengan kamera ponsel. Pemindaian jari memakai *photoplethysmography* (PPG), yaitu fotopletismografi, sebuah teknik optik berbiaya rendah yang mendeteksi perubahan volume darah pada pembuluh mikro jaringan [@allen2007photoplethysmography]. Nilai glukosa dari PPG diperkirakan oleh regresi *ridge* pada tujuh fitur buatan tangan dan hanya ditampilkan sebagai rentang ±25 mg/dL, yang dibulatkan dari galat absolut persentil ke-90 sebesar 24,7 mg/dL. Pembacaan yang berumur lebih dari 60 menit harus diukur ulang.

Langkah kedua mencatat makanan. Pengguna dapat memotret makanan atau memasukkan item satu per satu, berupa hidangan atau bahan tunggal, dengan gram atau porsi. Jalur foto memakai *vision-language model* (VLM), yaitu model visual-bahasa, yang mempelajari korelasi citra–teks dari pasangan citra–teks berskala web [@zhang2024vision]. VLM Gemini diminta menyebut daftar hidangan dalam bahasa Indonesia. Setiap nama kemudian dicocokkan dengan satu baris tabel hidangan oleh pencocok berbasis *embedding* (representasi vektor) terhadap nama dan alias. Sebuah item diterima hanya jika kemiripan teratasnya ≥ 0,80 dan selisih antara kandidat teratas dan kedua ≥ 0,05. Aturan ini disebut *aturan abstain*. Item lain harus diisi manual. Kepercayaan diri yang dilaporkan VLM tidak dipakai karena tidak terkalibrasi. Entri manual melewati pencocok dan tabel yang sama. Gram tiap item diambil dari nilai yang dimasukkan, atau dari skala porsi (setengah, satu, dan satu setengah porsi) dikali berat sajian. Berat sajian hidangan merupakan nilai asumsi, sehingga galat porsi tidak diukur. Prakiraan baru ditampilkan setelah seluruh item terselesaikan.

Langkah ketiga adalah *prakiraan* (*forecast*) glukosa pada T+60 dan T+120 menit setelah makan. Prakiraan memakai dua model *gradient boosting decision tree* (GBDT) yang diimplementasikan sebagai LightGBM [@ke2017lightgbm]. Fiturnya terdiri dari g0 dan sumbernya; gram karbohidrat, protein, lemak, dan serat, dengan nilai yang hilang tetap hilang; jam makan dalam bentuk sin dan cos; usia; jenis kelamin; indeks massa tubuh (IMT); dan status diabetes. Profil tipe 1 dan lainnya ditandai berada di luar data latih, dan layar hasil menyebutkannya. Setiap horizon ditampilkan dengan pita galat untuk tiap sumber g0. Pita galat dihitung dari residu pada partisipan yang tidak dilibatkan dalam pelatihan (*held-out*).

![Gambar 1. Alur GlucoSight dan temuan utama pada setiap langkah. (1) Glukosa saat ini dari glukometer, CGM, atau pindai PPG kamera; (2) foto makanan dikenali VLM lalu dicocokkan ke tabel hidangan, item yang ragu diisi manual; (3) dua model LightGBM memprakirakan glukosa T+60 dan T+120 beserta pita galat.](figures/fig_pipeline.png){width=100%}

## 3.2 Dataset

Tidak ada dataset dalam Tabel 1 yang memuat foto makanan, PPG, dan glukosa dari orang yang sama. CGMacros memuat CGM, makro hidangan, dan foto makanan [@das2025cgmacros; @gutierrezosuna2025cgmacrosdata], tetapi dalam penelitian ini tidak ada sinyal PPG yang dipasangkan dengannya. Dataset PPG dalam Tabel 1 hanya memuat sinyal dengan label glukosa atau detak jantung. Karena itu setiap komponen divalidasi pada datasetnya sendiri, dan penelitian ini tidak melaporkan angka akurasi ujung-ke-ujung. Angka tiap komponen dilaporkan terpisah pada Bagian 4.

**Tabel 1.** Dataset yang dipakai, komponen yang divalidasi pada masing-masing dataset, dan lisensinya.

| Dataset | Komponen | Subjek/data | Label | Lisensi |
|---|---|---|---|---|
| CGMacros [@das2025cgmacros; @gutierrezosuna2025cgmacrosdata] | Forecaster (pelatihan dan evaluasi) | 45 partisipan (15 sehat, 16 prediabetes, 14 T2D); 1706 hidangan, menjadi 1534 (T+60) dan 1531 (T+120) setelah pembersihan | Glukosa CGM; makro hidangan | CC BY-NC-SA 4.0 |
| ShanghaiT2DM [@zhao2023chinese; @zhao2023shanghaidata] | Pemeriksaan eksternal forecaster | 100 T2D; 3045 dan 3028 hidangan | Glukosa CGM | CC BY 4.0 |
| PPG_Dataset / MUST [@kermani2023dataset] | Glukosa dari PPG jari | 23 subjek; 67 rekaman; PPG kontak jari | Glukosa glukometer, 88–183 mg/dL | CC BY 4.0 (Mendeley Data) |
| BUT PPG v2 [@nemcova2024butppgv2] | Glukosa dari PPG kamera ponsel | Kamera ponsel 30 Hz; subset jari 18 subjek, 1246 rekaman, 35 sesi; 17 subjek dengan dua sesi | Glikemia | CC BY 4.0 |
| BUT PPG v1 [@nemcova2021brno; @nemcova2021butppgv1] | Detak jantung dari PPG kamera ponsel | 12 subjek; 48 rekaman | Detak jantung, referensi EKG | CC BY 4.0 |
| Makanan tradisional Indonesia [@wibisono2020traditional; @mursanto2023intfk] | Identifikasi hidangan | 35 kelas pada rilis yang digunakan; 164 foto dev dan 317 foto uji yang dinilai | Kelas hidangan | CC BY 4.0 (artikel); CC BY-SA 4.0 (Kaggle) |
| TKPI 2017 [@kemenkes2017tkpi] | Tabel hidangan | 710 baris (568 hidangan, 142 bahan) | Nilai gizi per 100 g dan berat sajian | Tidak berlisensi terbuka; dikutip, tidak didistribusikan |

Nilai TKPI tidak didistribusikan ulang. Prototipe menampilkan baris tabel hidangan satu per satu dengan kutipan sumbernya. Setiap baris menyatakan satu makanan sebagaimana dimakan.

## 3.3 Pembersihan Data CGMacros

Pembersihan diterapkan pada CGMacros untuk data latih dan data evaluasi forecaster. Hidangan dibuang jika rasio Atwater, yaitu (4 × karbohidrat + 4 × protein + 9 × lemak) dibagi kalori, berada di luar rentang 0,7–1,3. Dari 1706 hidangan, 85 dibuang. Jumlah yang dibuang bergantung pada toleransi: 107 hidangan pada ±20% dan 58 pada ±50%. Aturan ini memeriksa konsistensi antara makro dan kalori. Gerbang berbasis rentang nilai pada kamus data akan menandai 404 hidangan (23,7%) dan tidak dipakai.

Jika serat lebih besar daripada karbohidrat, nilai serat dianggap hilang. Hal ini terjadi pada 33 hidangan dari 5 partisipan. Kolom *Amount Consumed* tidak dipakai, karena pada 8 partisipan kolom ini dikodekan sebagai angka hitung. Penanda hidangan yang seluruh nilainya nol, sebanyak 44, dibuang. Tidak ada pemotongan nilai ekstrem maupun imputasi; nilai yang hilang dibiarkan hilang. Setelah pembersihan, tersisa 1534 hidangan untuk horizon T+60 dan 1531 untuk T+120.

## 3.4 Protokol Evaluasi

Unit evaluasi adalah partisipan. Target adalah pembacaan CGM tepat 60 dan 120 menit setelah makan. g0 adalah pembacaan CGM terakhir sebelum makan. Metrik utama adalah *mean absolute error* (MAE) dalam mg/dL.

Evaluasi memakai *nested cross-validation* (validasi silang bersarang) yang dikelompokkan menurut partisipan. Lipatan luar berjumlah lima dan diulang lima kali, sehingga partisipan tidak pernah muncul di data latih dan data uji pada lipatan yang sama. Di dalam setiap lipatan luar, pemilihan set fitur dan hiperparameter dilakukan hanya dengan partisipan latih.

Tiga baseline dibandingkan pada setiap lipatan. B0 adalah rata-rata populasi. B1 adalah *persistence* (persistensi), yaitu prakiraan yang sama dengan g0. B2 adalah persistensi ditambah rata-rata *excursion* (ekskursi, yaitu perubahan glukosa terhadap g0) dari partisipan latih.

Sebuah komponen hanya diklaim memberi peningkatan jika lulus empat pemeriksaan sekaligus: (i) selisih MAE terhadap B2 bernilai positif; (ii) GBDT mengalahkan B2 pada minimal 4 dari 5 lipatan di setiap ulangan; (iii) interval kepercayaan 95% dari selisih MAE, dihitung dengan *bootstrap* berdasarkan partisipan, tidak memuat nol; dan (iv) *permutation test* (uji permutasi), yaitu makro hidangan diacak di dalam partisipan yang sama sebanyak 200 kali dan selisih MAE dihitung ulang, menghasilkan p ≤ 0,05. Dengan 200 permutasi, nilai p terkecil yang dapat dicapai adalah 0,005.

Komponen yang gagal satu pemeriksaan tidak diklaim memberi peningkatan. Komponen tersebut tetap dipakai, dan hasilnya dilaporkan apa adanya.

Galat klinis dilaporkan dengan *Clarke error grid* [@clarke1987evaluating], sebagai median per partisipan dari persentase prakiraan yang jatuh di zona A Clarke. Evaluasi PPG memakai pengelompokan partisipan yang sama, dengan *mean predictor* (rata-rata) sebagai baseline. Protokol tiap komponen lain dijelaskan pada Subbagian 3.5–3.8.
