## 3.5 Model Prakiraan Glukosa

Model prakiraan memakai *gradient boosting decision tree* yang diimplementasikan dalam LightGBM [@ke2017lightgbm]. Setiap horizon, yaitu glukosa tepat pada waktu makan ditambah 60 menit dan ditambah 120 menit, memiliki model tersendiri. Masukan model terdiri atas glukosa awal g0 beserta sumbernya (glukometer atau CGM); gram karbohidrat, protein, lemak, dan serat; jam makan dalam bentuk sinus dan kosinus; usia; jenis kelamin; indeks massa tubuh (IMT); dan status diabetes. Serat yang tidak tersedia tetap hilang dan tidak diisi dengan nilai pengganti.

Target dapat berupa nilai glukosa langsung atau selisih terhadap g0 (*offset*). Empat kumpulan fitur diuji: g0 dan jam; g0, jam, dan makronutrien; g0, jam, dan profil (usia, jenis kelamin, IMT, status diabetes); serta seluruh fitur. Ruang pencarian hiperparameter terdiri atas jumlah daun maksimum 4 atau 8, jumlah sampel minimum per daun 20 atau 60, dan jumlah pohon 100 atau 300, dengan laju pembelajaran tetap 0,05. Pemilihan kumpulan fitur dan hiperparameter dilakukan di dalam setiap lipatan *nested cross-validation* yang berkelompok menurut partisipan (*grouped cross-validation*), sehingga partisipan yang tidak dilibatkan dalam pelatihan (*held-out*) tidak pernah ikut memilih konfigurasi. Model yang disajikan dilatih ulang pada seluruh 45 partisipan CGMacros dengan kumpulan fitur lengkap, yang terpilih pada kedua horizon.

Glukometer disimulasikan dari pembacaan CGM. Galat dimodelkan sebagai *zero-mean normal error* yang batas 95%-nya mengikuti batas akurasi ISO 15197:2013 [@iso2013iso15197]: ±15 mg/dL untuk glukosa di bawah 100 mg/dL, dan ±15% pada 100 mg/dL ke atas. Setiap kejadian makan memperoleh tiga sampel galat untuk pelatihan dan lima untuk evaluasi.

Glukosa awal PPG disimulasikan dari galat model PPG yang disajikan (Bagian 3.7). Prediksi model tersebut mengikuti kira-kira 113,35 + 0,0101 × glukosa sebenarnya, dengan simpangan baku galat 4,20 mg/dL. Nilai PPG simulasi karena itu hampir tidak bergantung pada glukosa sebenarnya, sebagaimana dilaporkan pada Bagian 4.6.

Setiap model disertai *error band*, yaitu persentil ke-90 dari galat absolut pada partisipan yang tidak dilibatkan dalam pelatihan, dipisahkan menurut sumber g0 dan horizon. Pada T+60 menit pita tersebut adalah 62,8 mg/dL (glukometer), 60,8 mg/dL (CGM), dan 77,9 mg/dL (PPG). Pada T+120 menit, pita tersebut adalah 64,0 mg/dL, 62,6 mg/dL, dan 69,1 mg/dL.

Desain yang ditetapkan sebelum evaluasi, yaitu kumpulan fitur lengkap tanpa seleksi bersarang, dijalankan lebih dahulu dan gagal. Prosedur bersarang digunakan sebagai gantinya; hasil kedua desain dilaporkan pada Bagian 4.7.

## 3.6 Identifikasi Hidangan

Identifikasi hidangan berjalan dalam tiga tahap. Pada tahap pertama, foto makanan dikirim ke *vision-language model* (VLM) Gemini 3.5 Flash-Lite [@team2023gemini; @comanici2025gemini] dengan temperatur 0. Instruksi (*prompt*) meminta model menyebut setiap hidangan yang terlihat sespesifik mungkin, memakai nama umum dalam bahasa Indonesia bila dikenali, dan mengembalikan keluaran berupa JSON. Satu entri berarti satu hidangan; hidangan tidak dipecah menjadi bahan-bahannya. Model juga melaporkan nilai keyakinannya sendiri, tetapi nilai ini tidak dipakai karena tidak terkalibrasi: ketepatannya tetap berada pada 74–76% meskipun keyakinan yang dilaporkan bernilai 0,95 atau lebih.

Pada tahap kedua, setiap nama diubah menjadi *embedding* dengan model gemini-embedding-001 [@lee2025geminiembedding], lalu dicocokkan dengan baris terdekat pada tabel hidangan. Tabel hidangan dibangun dari TKPI 2017 [@kemenkes2017tkpi] dan memuat 710 baris, yaitu 568 hidangan dan 142 bahan. Setiap baris adalah satu makanan sebagaimana dimakan, dan menyimpan energi, karbohidrat, protein, dan lemak per 100 g; serat (boleh kosong); berat porsi; serta label asal nilai (tepat, perkiraan, atau hasil hitung). Gram setiap item adalah gram yang dimasukkan pengguna, atau skala porsi (½, 1, atau 1½) dikali berat porsi. Berat porsi hidangan berupa asumsi yang diberi label sebagai asumsi: 27 dari 35 baris pada tabel evaluasi memakainya, sehingga galat porsi tidak diukur.

Pada tahap ketiga, *aturan abstain* menentukan apakah sebuah item diterima. Item diterima hanya jika kemiripan kosinus top-1 ≥ 0,80 dan selisih antara top-1 dan top-2 (*margin*) ≥ 0,05. Item yang tidak memenuhi aturan tidak diterima otomatis: pengguna memilih dari saran pencocok atau memasukkan item secara manual melalui pencocok yang sama. Ambang ini dipilih pada himpunan pengembangan dengan meminimalkan kecocokan keliru (*false match*) dengan syarat cakupan minimal 60%, dari grid ambang kemiripan 0,60–0,95 dan margin 0; 0,02; atau 0,05. Hasil pada himpunan uji dilaporkan pada Bagian 4.5.

Foto berasal dari kumpulan makanan tradisional Indonesia [@wibisono2020traditional; @mursanto2023intfk]. Rilis yang digunakan memuat 35 kelas. Himpunan pengembangan berisi 164 foto. Himpunan uji berisi 329 foto, dan 317 di antaranya dinilai. Metrik yang dilaporkan adalah ketepatan top-1 pada tingkat hidangan, ketepatan pada tingkat famili hidangan (kelompok masakan yang sama), proporsi abstain, dan kecocokan keliru.

Pembanding adalah klasifikator tertutup MobileNetV2 [@sandler2018mobilenetv2] yang dilatih pada sepuluh kelas (food-10). Perbandingan hanya dilakukan pada kelas yang beririsan dengan kelas kami, dengan bootstrap berpasangan pada foto uji dari kelas tersebut.

## 3.7 Estimasi Glukosa dari PPG

Empat model dibandingkan untuk memperkirakan glukosa dari sinyal *photoplethysmography* (PPG) ujung jari pada PPG_Dataset [@kermani2023dataset], yang berisi 23 subjek dan 67 rekaman dengan sensor PPG kontak dan label glukometer. Model pertama adalah *ridge regression* pada tujuh fitur buatan tangan yang diekstraksi dari sinyal; model ini adalah model yang disajikan pada sistem. Model kedua adalah *ridge regression* pada *embedding* model fondasi PPG PaPaGei-S [@pillai2025papagei]. Model ketiga adalah *random forest* pada tujuh fitur buatan tangan. Model keempat adalah *random forest* pada embedding PaPaGei-S yang direduksi menjadi 10 komponen dengan analisis komponen utama (PCA). Semua model dievaluasi dengan *nested cross-validation* berkelompok menurut subjek, dan dibandingkan dengan prediktor rata-rata yang selalu memperkirakan rata-rata glukosa data latih.

Pengujian pada BUT PPG v2 [@nemcova2024butppgv2] memakai sinyal kamera ponsel pintar dari subset jari: 1246 rekaman diam dari 18 subjek dalam 35 sesi, dengan 17 subjek yang memiliki dua nilai glukosa. Pada subset ini kami menjalankan validasi silang 5 × 5 berkelompok menurut subjek, dengan uji permutasi sebanyak 1000 permutasi sebagai kontrol.

Desain *within-person* memakai 17 subjek yang memiliki dua sesi. Terdapat dua uji. (i) Model memprediksi selisih glukosa dari selisih fitur antarsesi, dibandingkan dengan *carry-over*, yaitu prediksi bahwa glukosa tidak berubah. (ii) Model dikalibrasi pada sesi A dengan offset individu dan model lintas-subjek, lalu memprediksi sesi B, dan hasilnya dibandingkan dengan *carry-over* pada sesi B yang sama.

Frekuensi jantung diestimasi dari sinyal kamera yang sama dan dibandingkan dengan EKG pada BUT PPG versi pertama [@nemcova2021brno], setelah penyaringan kualitas sinyal.

## 3.8 Implementasi dan Penanganan Masukan Tidak Lengkap

Sistem diimplementasikan sebagai prototipe peramban. Pengguna membuka halaman web dengan kamera ponsel tanpa memasang aplikasi. Alurnya tetap tiga langkah: glukosa awal, makanan, dan prakiraan pada T+60 dan T+120 menit. Antarmuka menggunakan bahasa Indonesia. Sistem ini adalah prototipe riset dan bukan perangkat medis.

**Glukosa awal.** Pengguna memasukkan nilai glukometer, memasukkan pembacaan CGM terakhir, atau memindai ujung jari dengan kamera (PPG). Hasil glukometer disertai galat ±15 mg/dL di bawah 100 mg/dL dan ±15% pada 100 mg/dL ke atas. Hasil PPG ditampilkan sebagai rentang, bukan angka tunggal, dengan galat 90% sebesar ±25 mg/dL (24,7 mg/dL pada partisipan yang tidak dilibatkan dalam pelatihan) dan catatan bahwa glukometer lebih akurat. Pembacaan CGM ditampilkan sebagai nilai terakhir tanpa riwayat dan tanpa angka galat. Pembacaan yang lebih dari 60 menit tua meminta pengguna mengukur ulang. Sistem tidak memakai rata-rata populasi sebagai pengganti glukosa awal.

**Makanan.** Foto diproses sesuai Bagian 3.6. Item yang tidak dapat dikenali dengan yakin dimasukkan secara manual melalui pencocok yang sama. Pada layar konfirmasi, item yang diterima ditandai "Dari foto" dan porsinya dapat diubah. Item lain ditampilkan dengan tebakan nama dan saran pencocok, serta pilihan "cari lainnya". Prakiraan hanya dapat dilihat setelah semua item terselesaikan. Persentase keyakinan tidak ditampilkan. Pengguna dapat menghapus item atau menambahkan item yang terlewat. Foto tidak disimpan; riwayat hanya memuat nama hidangan dan makronutrien.

**Masukan yang tidak lengkap.** Serat yang tidak tersedia tetap hilang dan tidak diisi. Sistem tidak membuat prakiraan dari nilai rata-rata populasi bila glukosa awal tidak tersedia atau sudah kedaluwarsa. Diabetes tipe 1 dan kategori "lainnya" ditandai sebagai di luar data latih, dan layar hasil menampilkan catatan tersebut karena model tidak dilatih pada kelompok itu. Setiap prakiraan disertai pita galat untuk sumber glukosa awal dan horizon yang dipakai (Bagian 3.5).

**Batasan implementasi.** Model yang disajikan dilatih hanya dengan g0 dari glukometer dan CGM. Masukan sumber PPG tidak pernah aktif selama pelatihan, sehingga prakiraan yang dimulai dari PPG dijalankan dengan nilai g0 yang hampir konstan. Evaluasi sumber PPG dilakukan dengan simulasi (Bagian 3.5), dan dampaknya dilaporkan pada Bagian 4.2.

**Lisensi dan privasi.** Gemini dipanggil dari sisi server; kunci API tidak pernah dikirim ke peramban. Nilai TKPI ditampilkan per baris dengan sitasi dan tidak didistribusikan ulang. Tabel hidangan dibangun secara lokal dan tidak dirilis bersama sistem.

Penanganan masukan tidak lengkap ini menjawab pertanyaan penelitian kedua; pembahasannya ada pada Bagian 5.
