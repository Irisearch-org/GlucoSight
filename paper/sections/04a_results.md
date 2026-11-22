# 4. Results

## 4.1 Prakiraan Glukosa pada Partisipan Baru

Bagian ini membandingkan prakiraan *gradient boosting decision tree* (GBDT, diimplementasikan dengan LightGBM [@ke2017lightgbm]) dengan baseline B0 (rata-rata populasi), B1 (persistensi) dan B2 (persistensi + rata-rata ekskursi, *mean excursion*) pada 45 partisipan CGMacros [@das2025cgmacros]. Glukosa awal g0 diambil dari *continuous glucose monitoring* (CGM) sebelum makan. Setiap prakiraan dievaluasi pada partisipan yang tidak dilatih (*held-out*): dalam setiap lipatan (*fold*), model dilatih pada partisipan lain dan diuji pada partisipan yang disisihkan. Pembagian ini diulang lima kali, sehingga setiap angka pada Tabel 2 adalah hasil atas 25 lipatan. Setelah pembersihan, tersisa 1534 makanan untuk T+60 dan 1531 makanan untuk T+120.

**Tabel 2.** *Mean absolute error* (MAE, mg/dL) pada 45 partisipan CGMacros dengan g0 dari CGM. Interval kepercayaan 95% dihitung dengan *bootstrap* atas partisipan; p berasal dari *permutation test* dengan 200 pengacakan.

| Horizon | B0 | B1 | B2 | GBDT | Gain vs B2 [CI 95%] | Lipatan menang per ulangan | p permutasi |
|---|---|---|---|---|---|---|---|
| T+60 | 38,72 | 37,91 | 31,68 | 28,78 | +2,90 [1,36; 4,42] | 5/5/4/5/4 | 0,005 |
| T+120 | 36,93 | 36,50 | 32,05 | 28,49 | +3,56 [1,62; 5,51] | 5/4/5/4/5 | 0,005 |

Pada kedua horizon, GBDT melampaui B2 dengan interval kepercayaan yang seluruhnya di atas nol, dan lulus gerbang evaluasi: selisih positif, sebagian besar lipatan dimenangkan, interval di atas nol, dan p permutasi 0,005. Selisih terhadap B1 dan B0 juga positif, dan lebih besar: +9,13 [4,51; 13,53] dan +8,01 [5,06; 11,14] terhadap B1, serta +9,94 [6,55; 14,12] dan +8,44 [4,82; 11,91] terhadap B0, masing-masing untuk T+60 dan T+120.

Gambar 2b memakai prediksi yang dirata-ratakan atas lima ulangan; tanpa perataan, zona A Clarke untuk seluruh prediksi adalah 61,6% [@clarke1987evaluating]. Median persentase titik pada zona A Clarke per partisipan adalah 66,4% (*interquartile range*, IQR 20,4) untuk GBDT dan 58,7% (IQR 19,7) untuk B2 pada T+60, serta 62,6% (IQR 18,3) dan 56,5% (IQR 23,3) pada T+120. GBDT lebih baik daripada B2 pada 35 dari 45 partisipan (T+60) dan 36 dari 45 (T+120), dengan median selisih per partisipan +2,50 mg/dL dan +4,37 mg/dL (Gambar 2a). Rata-rata tak berbobot MAE per partisipan adalah 29,40 vs 32,64 (T+60) dan 29,41 vs 33,43 (T+120). Angka ini berbeda dari Tabel 2 karena Tabel 2 menggabungkan seluruh makanan, sedangkan rata-rata per partisipan memberi bobot sama pada setiap orang.

Pada subkelompok diabetes tipe 2 (14 partisipan), MAE GBDT adalah 34,44 dibanding 41,15 untuk B2 pada T+60, dan 41,51 dibanding 45,81 pada T+120. Interval kepercayaan untuk subkelompok ini tidak tersedia dalam hasil ini, sehingga angka tersebut bersifat deskriptif.

![Gambar 2. (a) Selisih MAE per partisipan, B2 dikurangi GBDT (positif berarti GBDT lebih baik; 45 partisipan, g0 dari CGM). (b) Clarke error grid GBDT pada T+60 (1534 makanan, prediksi dirata-rata atas 5 ulangan validasi silang); zona A 62,6% dan A+B 98,0%.](figures/fig_forecast.png){width=100%}

## 4.2 Sumber Glukosa Awal

Tiga sumber g0 diuji dengan protokol yang sama: CGM sebagai acuan, glukometer yang disimulasikan, dan PPG yang disimulasikan. Galat glukometer mengikuti standar ISO 15197:2013 [@iso2013iso15197]. g0 dari PPG disimulasikan menurut perilaku model PPG yang disajikan: g0 = 113,35 + 0,0101 × glukosa sebenarnya, dengan simpangan baku galat 4,20 mg/dL. Kemiringan 0,0101 berarti nilai ini hampir tidak mengikuti glukosa sebenarnya; pada evaluasi model PPG (Bagian 4.6), simpangan baku prediksi hanya 4,2 mg/dL, dibanding 18,7 mg/dL untuk glukosa sebenarnya. Model yang disajikan dilatih hanya dengan g0 dari glukometer dan CGM.

Dengan g0 dari glukometer, GBDT tetap lulus gerbang pada kedua horizon: MAE 29,69 dibanding 32,60 pada T+60 (selisih +2,92 [1,49; 4,30]) dan 29,24 dibanding 33,04 pada T+120 (+3,80 [1,79; 5,64]). Dengan g0 dari PPG, GBDT gagal: MAE 36,02 dibanding 37,52 pada T+60 (+1,51 [−0,80; 4,05]) dan 32,86 dibanding 35,51 pada T+120 (+2,65 [−0,42; 5,65]). Titik estimasi selisihnya positif, tetapi interval kepercayaan melewati nol pada kedua horizon. Keunggulan atas B2 pada skenario PPG karena itu tidak dapat dibedakan dari nol pada data ini.

Lebar *error band* (persentil ke-90 galat absolut) untuk T+60 adalah 62,8 mg/dL (glukometer), 60,8 mg/dL (CGM) dan 77,9 mg/dL (PPG). Untuk T+120, lebarnya 64,0, 62,6 dan 69,1 mg/dL. Pita PPG paling lebar pada kedua horizon, sejalan dengan galat g0 PPG yang lebih besar.

## 4.3 Kontribusi Informasi Makanan

Pertanyaan di bagian ini adalah seberapa besar keunggulan atas B2 yang berasal dari informasi makanan. Uji permutasi mengacak makronutrien (karbohidrat, protein, lemak dan serat) di dalam setiap partisipan, lalu mengevaluasi ulang prakiraan sebanyak 200 kali. Rata-rata selisih terhadap B2 dengan makronutrien teracak hanya +1,24 mg/dL pada T+60 (persentil ke-95: 1,38) dan +1,56 mg/dL pada T+120 (persentil ke-95: 1,68). Selisih teramati adalah +2,90 dan +3,56 mg/dL, dengan p = 0,005 pada kedua horizon. Sekitar 1,7 mg/dL pada T+60 dan 2,0 mg/dL pada T+120 dari keunggulan atas B2 karena itu berasal dari makronutrien. Seleksi fitur dalam tidak pernah memilih himpunan fitur tanpa makronutrien (0 dari 25 lipatan pada kedua horizon).

Pemilihan himpunan fitur tidak stabil antarlipatan. Pada T+60, himpunan dengan g0, jam dan makronutrien dipilih pada 17 dari 25 lipatan, dan himpunan penuh pada 8 dari 25. Pada T+120, himpunan penuh dipilih pada 16 dari 25 lipatan. Karena itu, gerbang menguji prosedur seleksi, bukan satu himpunan fitur tetap.

Untuk menguji sensitivitas terhadap galat identifikasi hidangan, kesalahan karbohidrat pada data uji diambil ulang secara acak (*resampling*) dan dimasukkan sebagai masukan. Galat ini hanya menaikkan MAE sebesar +0,01 mg/dL pada T+60 dan +0,14 mg/dL pada T+120, dan selisih terhadap B2 tetap signifikan. Galat porsi tidak diukur, dan 27 dari 35 bobot saji diasumsikan, sehingga angka ini hanya mencakup galat identifikasi.

## 4.4 Validasi Eksternal pada ShanghaiT2DM

ShanghaiT2DM berisi 100 partisipan diabetes tipe 2 beserta log makanan [@zhao2023chinese; @zhao2023shanghaidata]. Median g0 adalah 127,8 mg/dL, dan median kenaikan glukosa 60 menit setelah makan adalah 23,4 mg/dL. Terdapat 3045 makanan untuk T+60 dan 3028 untuk T+120. Log makanan dalam kohort ini hanya memuat nama hidangan dalam bentuk teks, tanpa makronutrien.

**Tabel 3.** Selisih MAE terhadap B2 pada ShanghaiT2DM (100 partisipan diabetes tipe 2) dengan interval kepercayaan 95%. Gerbang sama dengan Tabel 2.

| Model | Gain vs B2 T+60 [CI 95%] | Gain vs B2 T+120 [CI 95%] | Gerbang |
|---|---|---|---|
| Model yang disajikan tanpa makronutrien | +0,06 [−1,44; 1,55] | −5,89 [−9,22; −2,58] | gagal |
| Model CGMacros dengan g0 dan jam | +1,34 [0,02; 2,75] | +1,74 [0,36; 3,13] | gagal (lipatan menang 19/25 dan 21/25) |
| GBDT dilatih dalam ShanghaiT2DM | +2,55 [1,52; 3,55] | +3,56 [2,20; 4,94] | lulus (25/25 lipatan, p 0,005) |

Hasil ini memisahkan dua klaim. Pertama, metode prakiraan dapat digeneralisasi: GBDT yang dilatih dalam ShanghaiT2DM, tanpa teks hidangan, lulus gerbang pada kedua horizon, dengan MAE 29,81 dibanding 32,36 pada T+60 dan 35,92 dibanding 39,48 pada T+120. Kedua, model yang disajikan tidak dapat langsung dipindahkan ke kohort ini tanpa makronutrien. Pada T+60 selisihnya tidak berbeda dari B2 (+0,06 mg/dL), dan pada T+120 selisihnya negatif dengan interval kepercayaan yang seluruhnya di bawah nol (−5,89 mg/dL). Model CGMacros dengan g0 dan jam tanpa makronutrien memiliki selisih positif dengan interval yang tidak melewati nol pada kedua horizon, tetapi gagal pada syarat jumlah lipatan yang dimenangkan.

Menambahkan skor identitas hidangan dari teks tidak mengubah hasil: selisih tambahannya +0,00 mg/dL pada T+60 dan +0,06 mg/dL pada T+120. Dengan demikian, metode ini tergeneralisasi ke kohort lain, tetapi model yang disajikan tidak dapat dipindahkan begitu saja ketika makronutrien tidak tersedia.
