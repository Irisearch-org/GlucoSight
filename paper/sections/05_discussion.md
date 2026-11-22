# 5. Discussion

## 5.1 Jawaban atas Pertanyaan Penelitian

Pertanyaan pertama menanyakan *sejauh mana informasi makanan dan sinyal PPG memperbaiki prediksi dibandingkan model tanpa informasi tersebut*. Informasi makanan membantu, sedangkan sinyal PPG tidak memuat informasi glukosa.

Dengan glukosa awal dari CGM, prakiraan GBDT menurunkan *mean absolute error* (MAE) dibanding B2 sebesar +2,90 mg/dL pada T+60 dan +3,56 mg/dL pada T+120, pada partisipan yang tidak dilibatkan dalam pelatihan (*held-out*). Uji permutasi (*permutation test*) menunjukkan sekitar 1,7 dan 2,0 mg/dL dari keuntungan itu berasal dari makronutrien; sisanya berasal dari g0, jam makan, dan profil partisipan. Metode yang sama bertahan saat dilatih pada 100 pasien T2D di Shanghai [@zhao2023chinese].

Sinyal PPG tidak memuat informasi glukosa. Pada PPG_Dataset [@kermani2023dataset] dan BUT PPG v2 jari [@nemcova2024butppgv2], prediktor PPG yang disajikan tidak mengungguli rata-rata populasi (r = 0,045 pada PPG_Dataset). Uji dalam orang yang sama juga negatif: model selisih glukosa antarsesi tidak berbeda bermakna dari *carry-over* (MAE 17,8 berbanding 19,4 mg/dL; selisih +1,6 [−1,4; 4,6]). Zona A Clarke [@clarke1987evaluating] menyesatkan pada kasus ini: 98% pada kisaran 100–140 mg/dL, tetapi 0% pada 180–250 mg/dL. Prakiraan yang dimulai dari g0 PPG tidak lolos gerbang evaluasi (+1,51 [−0,80; 4,05] mg/dL pada T+60). PPG tetap berguna untuk detak jantung: MAE 3,76 bpm terhadap EKG [@nemcova2021brno].

Pertanyaan kedua menanyakan *bagaimana sistem berbasis peramban dirancang agar tetap bekerja tanpa CGM dengan data yang tidak lengkap*. Jawaban kami memuat empat aturan. Glukosa awal berasal dari glukometer (±15 mg/dL di bawah 100 dan ±15% di atas 100 menurut ISO 15197 [@iso2013iso15197]), satu pembacaan CGM terakhir, atau pemindaian PPG (±25 mg/dL); pembacaan yang berumur lebih dari 60 menit harus diukur ulang, dan rata-rata populasi tidak dipakai sebagai pengganti. Item makanan yang tidak lolos aturan abstain (kemiripan ≥ 0,80 dan selisih ≥ 0,05) dimasukkan secara manual, dan prediksi hanya tampil setelah semua item terselesaikan. Zat gizi yang tidak tersedia tetap kosong. Setiap prakiraan ditampilkan bersama pita galat (*error band*) untuk setiap sumber glukosa awal.

Pertanyaan ketiga menanyakan *seberapa akurat estimasi makanan harus dicapai agar bermanfaat*. Galat identifikasi hidangan yang realistis hanya mengubah MAE sebesar +0,01 mg/dL pada T+60 dan +0,14 mg/dL pada T+120, dan keuntungan tetap signifikan. Tingkat identifikasi yang dilaporkan, yaitu 89,6% benar ketika sistem menjawab dengan cakupan 63,4%, cukup untuk menjaga galat karbohidrat tetap kecil. Pada tingkat hidangan persis, identifikasi kami kalah 20,5 poin persentase dari MobileNetV2 pada irisan 10 kelas (88 foto uji). Risiko yang belum terukur adalah porsi: 27 dari 35 baris tabel uji memakai bobot sajian yang diasumsikan. Estimasi porsi juga disebut batas utama pada benchmark VLM untuk gambar makanan [@marchetti2026benchmarking], dan galat karbohidrat dari foto 2D (6,1 g) lebih dari dua kali galat model yang tidak bergantung pada porsi (2,7 g) [@thames2021nutrition5k].

Dibandingkan dengan prakiraan berbasis riwayat CGM, GlucoSight menukar riwayat dengan satu pembacaan. @lu2026glucofm melaporkan RMSE 16,28 mg/dL untuk TimesFM 2.5 pada CGMacros dengan horizon 30 menit dan konteks 12 jam. Angka itu tidak dapat disandingkan dengan MAE kami karena metrik, horizon, dan konteksnya berbeda. Kami tidak mengklaim kesetaraan akurasi; harga desain ini adalah hilangnya riwayat glukosa.

## 5.2 Keterbatasan

Pertama, tidak ada dataset yang memuat foto makanan, PPG, dan glukosa untuk orang yang sama, sehingga akurasi ujung-ke-ujung tidak dapat diukur. Komponen divalidasi secara terpisah.

Kedua, kohornya kecil: 45 partisipan CGMacros [@das2025cgmacros] (14 dengan T2D), 100 pasien Shanghai, serta 23 dan 18 subjek PPG [@kermani2023dataset; @nemcova2024butppgv2]. Interval kepercayaan lebar dan hasil subkelompok perlu dibaca hati-hati.

Ketiga, galat glukometer disimulasikan dari ISO 15197 dan belum diukur. Data glukosa juga belum mencakup partisipan Indonesia: prakiraan dilatih pada kohor CGMacros dan diuji ulang pada kohor Shanghai.

Keempat, model yang disajikan tidak mentransfer ke Shanghai tanpa makronutrien: +0,06 [−1,44; 1,55] mg/dL pada T+60 dan −5,89 [−9,22; −2,58] mg/dL pada T+120.

Kelima, bobot sajian hidangan adalah nilai bawaan yang diasumsikan, sehingga galat porsi belum diukur. Aturan abstain dikalibrasi pada tabel evaluasi 35 baris, sedangkan tabel yang disajikan berisi 710 baris, dan kalibrasi ulang belum dilakukan.

Keenam, pita galat terlalu sempit pada kasus sulit: cakupannya hanya 74–80% ketika g0 ≥ 180 mg/dL dan 78–84% pada pasien T2D.

Ketujuh, pilihan himpunan fitur tidak stabil. Desain prespesifikasi dengan himpunan penuh tanpa seleksi bersarang gagal (−0,20 [−2,80; 2,35] mg/dL pada T+60). Prosedur bersarang dipakai setelah himpunan tereduksi pasca-hoc lolos pada lipatan yang sama; hasil bersarang itulah yang dilaporkan, dan pilihan ini perlu dikonfirmasi pada kohor baru.

## 5.3 Implikasi dan Keamanan

GlucoSight adalah purwarupa penelitian, bukan perangkat medis. Sistem ini tidak boleh dipakai untuk menentukan dosis insulin, dan angka dalam makalah ini tidak boleh dibaca sebagai validasi klinis.

PPG ditampilkan sebagai rentang ±25 mg/dL, yang mencakup 90% galat absolut pada data uji (24,7 mg/dL), dengan keterangan bahwa glukometer lebih akurat. Rentang ini menyatakan ketidakpastian, tetapi tidak membuat prediktor PPG informatif.

Nilai gizi TKPI 2017 [@kemenkes2017tkpi] tidak memiliki lisensi terbuka, sehingga hanya ditampilkan per baris dengan sitasi dan tidak didistribusikan ulang. Data CGMacros berlisensi CC BY-NC-SA 4.0 [@gutierrezosuna2025cgmacrosdata].

Rekomendasi metodologis utama adalah melaporkan setiap hasil berdampingan dengan baselinenya. Zona A Clarke saja dapat menyembunyikan kegagalan: @ramesh2026reassessing menemukan bahwa lebih dari 90% prediksi jatuh pada zona A+B, termasuk baseline, sementara R² mendekati nol atau negatif pada pemisahan yang menjaga identitas partisipan. Hasil dalam makalah ini karena itu disajikan bersama baseline B0, B1, atau B2.
