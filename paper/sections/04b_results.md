## 4.5 Identifikasi Hidangan

Pada 317 foto uji, akurasi *top-1* (kelas teratas benar) adalah 61,5% [56,1; 66,7], dan akurasi pada tingkat famili hidangan adalah 69,1%. Identifikasi dilakukan dalam tiga langkah. Model bahasa-visi (*vision-language model*, VLM) menamai hidangan dalam bahasa Indonesia tanpa daftar kelas tertutup, sejalan dengan pembelajaran korelasi citra–teks dari data berskala web [@zhang2024vision]. Nama tersebut kemudian dicocokkan dengan baris tabel hidangan melalui *embedding* [@lee2025geminiembedding]. *Aturan abstain* menjawab hanya bila skor kemiripan tertinggi mencapai ambang dan selisihnya dengan kandidat kedua mencapai margin; item yang tidak memenuhi syarat diserahkan ke entri manual.

**Tabel 4.** Titik operasi aturan abstain pada data dev dan uji. Lama: kemiripan ≥ 0,75 tanpa margin. Terpilih: kemiripan ≥ 0,80 dan margin ≥ 0,05.

| Aturan | Split | Cakupan | Benar saat menjawab | Salah cocok |
|---|---|---|---|---|
| Lama, kemiripan ≥ 0,75, tanpa margin | Dev | 72,6% | – | 11,6% |
| Lama, kemiripan ≥ 0,75, tanpa margin | Uji | 79,8% | – | 18,3% |
| Terpilih, kemiripan ≥ 0,80, margin ≥ 0,05 | Dev | 60,4% | 93,9% | 3,7% |
| Terpilih, kemiripan ≥ 0,80, margin ≥ 0,05 | Uji | 63,4% | 89,6% | 6,6% |

Titik operasi terpilih dipilih pada data dev dengan meminimalkan salah cocok, dengan syarat cakupan minimal 60%. Pada data uji, aturan ini menurunkan salah cocok dari 18,3% menjadi 6,6%, dengan cakupan yang turun dari 79,8% menjadi 63,4%. Titik ini dikalibrasi pada tabel evaluasi 35 baris (9 cocok persis, 11 mendekati, 15 terhitung), sedangkan tabel hidangan yang dipakai sistem berisi 710 baris. Kalibrasi ulang pada tabel penuh masih perlu dilakukan.

Pada sepuluh kelas yang beririsan dengan MobileNetV2 (88 foto uji), perbandingan berpasangan dengan *paired bootstrap* menunjukkan bahwa identifikasi berbasis embedding unggul pada tingkat famili sebesar +6,8 poin persentase [1,1; 13,6]. Pada tingkat hidangan persis, pengklasifikasi tertutup justru lebih akurat: selisih identifikasi berbasis embedding terhadap MobileNetV2 adalah −20,5 poin persentase [−31,8; −9,1]. Pada evaluasi terpisah pada food-10, MobileNetV2 mencapai akurasi top-1 88,1% pada data yang tidak dipakai untuk pelatihan. Kepercayaan diri yang dilaporkan VLM sendiri tidak dapat dijadikan dasar abstain, karena akurasinya mendatar pada 74–76% bahkan pada skor ≥ 0,95.

Kesalahan karbohidrat yang berasal dari identifikasi, dihitung pada item uji yang cocok, memiliki median 0,0 g, dan 87,4% berada dalam 10 g. Kesalahan porsi tidak diukur, dan 27 dari 35 bobot porsi diasumsikan. Pada benchmark VLM terbaru, estimasi porsi disebut sebagai batas utama [@marchetti2026benchmarking], sehingga angka ini hanya mencakup sisi identifikasi.

![Gambar 3. Cakupan terhadap tingkat salah cocok untuk ambang kemiripan 0,60–0,95 dan margin 0; 0,02; 0,05 (dev garis penuh, uji putus-putus). Bintang: aturan terpilih (0,80; 0,05); belah ketupat: ambang lama 0,75 tanpa margin.](figures/fig_dish_operating.png){width=60%}

## 4.6 Estimasi Glukosa dari PPG

Estimasi glukosa dari sinyal *photoplethysmography* (PPG) diuji pada dua dataset. PPG_Dataset berisi pengukuran dengan sensor jari kontak dan label glukometer [@kermani2023dataset]. BUT PPG v2 berisi rekaman jari dengan kamera ponsel pintar [@nemcova2024butppgv2]. Evaluasi menggunakan validasi silang yang mengelompokkan partisipan, sehingga setiap orang hanya muncul di data latih atau data uji (partisipan yang tidak dilibatkan dalam pelatihan, *held-out*). Model yang diuji adalah *ridge regression* dengan tujuh fitur buatan tangan (*handcrafted*) dan *ridge regression* di atas embedding PaPaGei-S [@pillai2025papagei]; pembandingnya adalah prediktor rata-rata glukosa data latih. Signifikansi diuji dengan uji permutasi (*permutation test*).

**Tabel 5.** Estimasi glukosa dari PPG (mg/dL). Gain = MAE rata-rata dikurangi MAE model; nilai negatif berarti model lebih buruk daripada prediktor rata-rata.

| Dataset | Model | MAE | MAE rata-rata | Gain [CI 95%] | p permutasi |
|---|---|---|---|---|---|
| PPG_Dataset | Ridge, 7 fitur | 14,69 | 14,55 | −0,14 [−0,52; 0,12] | 0,54 |
| PPG_Dataset | PaPaGei-S + ridge | 14,78 | 14,55 | −0,24 [−1,21; 0,48] | 0,53 |
| BUT PPG v2 | Ridge, 7 fitur | 22,15 | 22,29 | +0,15 [−0,88; 1,14] | 0,12 |
| BUT PPG v2 | PaPaGei-S + ridge | 22,25 | 22,29 | +0,04 [−0,54; 0,59] | 0,18 |

Tidak satu pun model mengalahkan prediktor rata-rata. Selisih MAE berkisar dari −0,24 hingga +0,15 mg/dL, semua interval kepercayaan 95% mencakup nol, dan semua p permutasi berada di atas 0,05. Prediksi yang disajikan juga hampir datar. Pada PPG_Dataset, SD prediksi adalah 4,2 mg/dL, sedangkan SD glukosa sebenarnya adalah 18,7 mg/dL, dengan korelasi r = 0,045. Pada BUT PPG v2, korelasinya r = −0,18. Pada *Clarke error grid* [@clarke1987evaluating], zona A mencapai 98% untuk rentang referensi 100–140 mg/dL, tetapi 0% pada 180–250 mg/dL. Model yang selalu mengembalikan nilai tipikal hanya tampak baik pada grid ini di dekat rata-rata. @ramesh2026reassessing juga melaporkan R² mendekati nol atau negatif pada pemisahan berbasis partisipan, dan lebih dari 90% prediksi jatuh di zona A+B, termasuk baseline. PPG-Bench melaporkan hasil serupa: *foundation model* PPG pada glukosa BUT PPG tidak lebih baik daripada kontrol acak [@kulkarni2026ppgbench].

![Gambar 4. Prediksi model PPG terhadap glukosa sebenarnya. (a) PPG_Dataset (67 rekaman, 23 subjek; r = 0,045). (b) BUT PPG v2, jari (1246 rekaman, 35 sesi, 18 subjek). (c) Dalam satu orang (17 subjek dua sesi): perubahan prediksi terhadap perubahan glukosa sebenarnya (r = 0,29). Garis hitam: identitas; garis putus-putus: baseline rata-rata (a, b) atau tanpa perubahan (c).](figures/fig_ppg.png){width=100%}

Dalam satu orang, hasilnya sama. Pada 17 subjek dengan dua sesi, model Δ yang dilatih dengan *leave-one-subject-out* menghasilkan MAE 17,8 mg/dL, dibandingkan 19,4 mg/dL untuk *carry-over* (nilai sesi sebelumnya dipertahankan). Selisih +1,6 mg/dL [−1,4; 4,6] mencakup nol, dan model lebih baik pada 12 dari 17 subjek (p binomial 0,14). Kalibrasi dengan offset dari sesi A ditambah model lintas-subjek lebih buruk daripada *carry-over*, yaitu 20,9 mg/dL berbanding 19,4 mg/dL (selisih [−3,1; −0,1]). Perubahan yang diprediksi model lintas-subjek memiliki SD 3,7 mg/dL, sedangkan perubahan sebenarnya memiliki SD 26,8 mg/dL.

Dari sinyal jenis yang sama, estimasi detak jantung berbasis PPG mencapai MAE 3,76 bpm terhadap EKG pada BUT PPG v1 [@nemcova2021butppgv1], dengan rentang per subjek 1,27–30,85 bpm pada 32 rekaman yang lolos filter kualitas. Sinyal ini memuat informasi laju denyut, tetapi tidak memuat sinyal glukosa yang dapat dipakai pada kedua dataset.

## 4.7 Hasil Negatif dan Keandalan

Desain yang ditetapkan sebelum evaluasi, yaitu himpunan fitur penuh tanpa seleksi bersarang, gagal pada kedua horizon. Gain terhadap B2 adalah −0,20 mg/dL [−2,80; 2,35] pada T+60 dan +0,40 mg/dL [−3,17; 3,77] pada T+120. Fitur profil partisipan (usia, jenis kelamin, IMT, status diabetes) mengalami *overfitting* dengan hanya 45 partisipan. Himpunan fitur yang tereduksi, yang dipilih secara *post hoc* setelah melihat hasil, lolos pada fold yang sama; karena pemilihan itu memakai data uji, hasil utama di Bagian 4.1 hingga 4.4 menggunakan validasi silang bersarang (*nested cross-validation*), yaitu seleksi himpunan fitur dan hiperparameter dilakukan hanya di dalam fold pelatihan. Seleksi ini tidak stabil: himpunan dengan makro tanpa fitur profil dipilih 17 dari 25 kali pada T+60, sedangkan himpunan penuh 8 dari 25 kali.

Varian yang diregistrasi sebelumnya (*pre-registered*) memprediksi ekskursi glukosa, yaitu perubahan terhadap glukosa awal, dengan *kendala monoton* (*monotone constraint*) pada karbohidrat. Varian ini tidak memberi peningkatan: +0,09 mg/dL [−0,07; 0,27] pada T+60 (16 dari 25 fold) dan +0,20 mg/dL [−0,06; 0,53] pada T+120 (14 dari 25 fold). Lengan kedua, dengan karbohidrat +1 dan serat −1, menghasilkan +0,14 dan +0,06. Varian ini tidak diadopsi.

Keandalan pita kesalahan juga terbatas. Pita tunggal ± dihitung dari kuantil ke-90 kesalahan absolut, sehingga cakupan nominalnya adalah 90%. Pada CGMacros [@das2025cgmacros], yang berisi data *continuous glucose monitoring* (CGM), cakupan pita tunggal hanya 74–80% untuk makan dengan g0 ≥ 180 mg/dL dan 78–84% untuk makan pada peserta DM tipe 2. Keduanya berada di bawah cakupan nominal. Pada 9–11 dari 45 partisipan, cakupan kurang dari 80%. Kuantil ke-90 kesalahan absolut pada T+60 adalah 60,8 mg/dL untuk CGM dan 77,9 mg/dL untuk PPG simulasi. Pita bersyarat yang menyesuaikan lebar dengan kondisi makan berada di luar cakupan penelitian ini.
