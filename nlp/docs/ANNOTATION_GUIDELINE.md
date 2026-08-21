# GlucoSight NLP Baseline v0.1
# Annotation Guideline

## 1. Overview

Dokumen ini mendefinisikan pedoman anotasi untuk dataset
Bahasa Indonesia yang digunakan pada NLP Context Baseline
GlucoSight.

Tujuan anotasi adalah mengubah meal-context sentence menjadi
multi-label binary context labels yang dapat digunakan untuk:

1. Rule-based NLP baseline
2. Training classifier
3. Evaluasi model NLP
4. Context embedding
5. Forecasting/Fusion

Setiap sentence dapat memiliki lebih dari satu label.

Contoh:

"Semalam aku begadang karena banyak tugas, lalu makan ayam goreng."

Dapat memiliki label:

- stress = 1
- poor_sleep = 1
- fried_food = 1

Label lainnya = 0.


---

## 2. Dataset Identification

Setiap sample memiliki identifier unik dengan format:

S0001
S0002
S0003
...

Format ID:

S + empat digit angka

Contoh:

S0001
S0042
S0125
S0400

`sample_id` harus konsisten digunakan pada seluruh pipeline
NLP dan menjadi key untuk proses join dengan Forecasting/Fusion.


---

## 3. Annotation Format

Dataset menggunakan multi-label binary classification.

Nilai label:

- `1` = kondisi/karakteristik terdapat pada sentence
- `0` = kondisi/karakteristik tidak terdapat pada sentence

Label Sprint 1:

1. stress
2. poor_sleep
3. high_activity
4. fried_food
5. large_portion
6. sweet_drink
7. late_meal
8. high_carb_hint


---

# 4. Label Definitions


## 4.1 stress

### Definition

Label `stress = 1` jika sentence menunjukkan bahwa pengguna
sedang mengalami tekanan, stres, kecemasan, banyak pikiran,
atau kondisi mental yang menunjukkan beban psikologis.

### Positive examples

"Semalam aku begadang karena banyak tugas."

" Hari ini aku lagi stres karena banyak tugas."

"Lagi banyak pikiran gara-gara deadline."

"Aku lagi cemas karena besok ada ujian."

"Kerjaan lagi numpuk banget sampai bikin kepikiran."

### Negative examples

"Aku makan nasi dengan telur."

"Hari ini aku olahraga."

"Aku makan ayam goreng."

### Annotation rule

Berikan `stress = 1` jika terdapat indikasi:

- stres
- stress
- cemas
- banyak pikiran
- kepikiran
- tertekan
- deadline yang menyebabkan tekanan
- beban pekerjaan/tugas yang jelas dikaitkan dengan tekanan

Jangan memberikan label hanya karena sentence menyebut
aktivitas kerja atau kuliah.

Contoh:

"Aku mengerjakan tugas sampai sore."

stress = 0

Karena tidak ada indikasi bahwa pengguna merasa stres.


---

## 4.2 poor_sleep

### Definition

Label `poor_sleep = 1` jika sentence menunjukkan durasi tidur
yang kurang, begadang, tidur terlalu larut, atau kualitas/waktu
tidur yang buruk.

### Positive examples

"Semalam aku cuma tidur empat jam."

"Tadi malam aku begadang sampai jam tiga."

"Aku kurang tidur semalam."

"Aku baru bisa tidur menjelang pagi."

"Semalam aku tidur sangat sedikit."

### Negative examples

"Aku tidur delapan jam semalam."

"Aku tidur seperti biasa."

"Aku makan malam jam delapan."

### Annotation rule

Berikan `poor_sleep = 1` jika terdapat indikasi:

- kurang tidur
- tidur hanya beberapa jam
- begadang
- tidur sangat larut
- hampir tidak tidur
- tidur tidak cukup

Tidak perlu menentukan angka durasi minimum secara ketat
jika sentence sudah jelas menunjukkan kurang tidur.


---

## 4.3 high_activity

### Definition

Label `high_activity = 1` jika sentence menunjukkan aktivitas
fisik yang relatif tinggi atau olahraga.

### Positive examples

"Tadi pagi aku lari selama satu jam."

"Aku habis gym."

"Barusan selesai jogging."

"Hari ini aku banyak jalan kaki."

"Aku habis olahraga cukup berat."

"Aku baru selesai latihan kardio."

### Negative examples

"Aku duduk mengerjakan tugas."

"Aku makan siang."

"Aku pergi ke kampus."

### Annotation rule

Berikan `high_activity = 1` untuk aktivitas seperti:

- gym
- jogging
- lari
- workout
- olahraga
- bersepeda
- latihan fisik
- angkat beban
- kardio
- aktivitas fisik berat

Aktivitas sehari-hari biasa tidak otomatis dianggap
high activity.

Contoh:

"Aku berjalan dari kamar ke dapur."

high_activity = 0


---

## 4.4 fried_food

### Definition

Label `fried_food = 1` jika makanan yang dikonsumsi merupakan
makanan yang digoreng atau secara eksplisit merupakan gorengan.

### Positive examples

"Aku makan ayam goreng."

"Tadi makan tempe goreng."

"Aku makan tahu goreng."

"Siang ini makan ikan goreng."

"Aku makan beberapa gorengan."

"Aku makan ayam crispy."

### Negative examples

"Aku makan ayam bakar."

"Aku makan ikan rebus."

"Aku makan sup ayam."

### Annotation rule

Berikan `fried_food = 1` jika terdapat indikasi:

- goreng
- gorengan
- ayam goreng
- tempe goreng
- tahu goreng
- ikan goreng
- ayam crispy
- makanan yang secara eksplisit digoreng

Jangan menganggap semua makanan yang mengandung minyak
sebagai fried food jika metode memasaknya tidak disebutkan.


---

## 4.5 large_portion

### Definition

Label `large_portion = 1` jika jumlah makanan yang dikonsumsi
disebutkan atau diindikasikan lebih besar dari porsi normal.

### Positive examples

"Aku makan nasi dua porsi."

"Tadi makannya porsi jumbo."

"Aku makan dalam porsi yang cukup besar."

"Tadi aku nambah nasi sampai dua kali."

"Aku makan lebih banyak dari biasanya."

### Negative examples

"Aku makan satu porsi nasi."

"Aku makan nasi dengan lauk."

"Aku makan sedikit."

### Annotation rule

Berikan `large_portion = 1` jika terdapat indikasi:

- dua porsi atau lebih
- porsi jumbo
- porsi besar
- makan sangat banyak
- makan lebih banyak dari biasanya
- menambah porsi
- jumlah makanan besar

Satu makanan tertentu tidak otomatis berarti porsinya besar.

Contoh:

"Aku makan ayam goreng."

large_portion = 0


---

## 4.6 sweet_drink

### Definition

Label `sweet_drink = 1` jika sentence menyebut minuman
yang secara umum mengandung gula atau secara eksplisit disebut
sebagai minuman manis.

### Positive examples

"Aku minum es teh manis."

"Tadi minum kopi susu."

"Aku minum boba."

"Tadi siang minum thai tea."

"Aku minum es kopi gula aren."

"Aku minum minuman manis."

### Negative examples

"Aku minum air putih."

"Aku minum teh tanpa gula."

"Aku minum kopi hitam tanpa gula."

### Annotation rule

Berikan `sweet_drink = 1` untuk:

- es teh manis
- kopi susu
- boba
- thai tea
- minuman dengan gula
- minuman manis
- soda
- minuman dengan gula aren

Jika sentence secara eksplisit menyatakan minuman
tanpa gula, jangan memberikan label.


---

## 4.7 late_meal

### Definition

Label `late_meal = 1` jika waktu makan disebutkan terjadi
larut malam atau mendekati tengah malam.

### Positive examples

"Aku baru makan malam jam sebelas."

"Tadi makan malam hampir tengah malam."

"Aku baru sempat makan jam sepuluh malam."

"Makan malamku tadi cukup larut."

"Aku baru makan setelah jam sebelas malam."

### Negative examples

"Aku makan malam jam tujuh."

"Aku makan siang jam dua belas."

"Aku sarapan jam tujuh pagi."

### Annotation rule

Berikan `late_meal = 1` jika:

- makan malam setelah sekitar jam 22.00
- makan mendekati tengah malam
- makan setelah jam 23.00
- sentence secara eksplisit menyebut makan larut malam
- sentence menyebut telat makan malam

Jika hanya disebut "malam" tanpa indikasi waktu yang larut,
jangan otomatis memberikan label.

Contoh:

"Aku makan malam bersama keluarga."

late_meal = 0


---

## 4.8 high_carb_hint

### Definition

Label `high_carb_hint = 1` jika sentence mengandung indikasi
konsumsi makanan dengan sumber karbohidrat yang cukup jelas,
terutama ketika jumlahnya relatif tinggi atau beberapa sumber
karbohidrat dikonsumsi bersamaan.

Label ini merupakan `hint`, bukan pengukuran kandungan
karbohidrat secara presisi.

### Positive examples

"Aku makan nasi dua porsi."

"Tadi makan nasi dengan mie sekaligus."

"Aku makan nasi goreng."

"Tadi makan mie dan nasi bersamaan."

"Aku makan dua porsi nasi."

"Aku makan pasta."

### Negative examples

"Aku makan telur rebus."

"Aku makan ayam dengan sayur."

"Aku makan ikan bakar."

### Annotation rule

Berikan `high_carb_hint = 1` jika terdapat makanan seperti:

- nasi
- nasi goreng
- mie
- mie goreng
- pasta
- roti
- kentang
- makanan berbahan tepung

Konteks jumlah juga dapat memperkuat label.

Contoh:

"Aku makan nasi dua porsi."

high_carb_hint = 1
large_portion = 1

Namun:

"Aku makan nasi satu porsi."

high_carb_hint = 1
large_portion = 0


---

# 5. Multi-Label Annotation

Satu sentence dapat memiliki beberapa label sekaligus.

Contoh:

"Habis gym aku makan nasi dua porsi dengan ayam goreng."

Annotation:

- stress = 0
- poor_sleep = 0
- high_activity = 1
- fried_food = 1
- large_portion = 1
- sweet_drink = 0
- late_meal = 0
- high_carb_hint = 1

Contoh lain:

"Semalam begadang karena deadline, lalu makan ayam goreng
dua porsi."

Annotation:

- stress = 1
- poor_sleep = 1
- high_activity = 0
- fried_food = 1
- large_portion = 1
- sweet_drink = 0
- late_meal = 0
- high_carb_hint = 0


---

# 6. Label Independence

Label harus dianotasi secara independen.

Kemunculan satu label tidak otomatis menyebabkan label lain
menjadi `1`.

Contoh:

"Aku makan ayam goreng."

fried_food = 1

large_portion = 0

high_carb_hint = 0

Tidak boleh menganggap ayam goreng sebagai porsi besar
atau tinggi karbohidrat tanpa informasi tambahan.


---

# 7. Important Ambiguous Cases


## 7.1 "Aku makan banyak"

Interpretasi:

large_portion = 1

Jika tidak ada informasi jenis makanan:

high_carb_hint = 0


## 7.2 "Aku makan nasi"

Interpretasi:

high_carb_hint = 1

large_portion = 0

Karena tidak ada informasi bahwa porsinya besar.


## 7.3 "Aku makan nasi dua porsi"

Interpretasi:

high_carb_hint = 1

large_portion = 1


## 7.4 "Aku makan malam"

Interpretasi:

late_meal = 0

Karena tidak ada informasi bahwa waktunya terlambat.


## 7.5 "Aku makan malam jam 11"

Interpretasi:

late_meal = 1


## 7.6 "Aku minum kopi"

Interpretasi:

sweet_drink = 0

Karena belum ada informasi bahwa kopi tersebut
mengandung gula atau susu.


## 7.7 "Aku minum kopi susu"

Interpretasi:

sweet_drink = 1


## 7.8 "Aku olahraga"

Interpretasi:

high_activity = 1

Karena olahraga merupakan aktivitas fisik.


## 7.9 "Aku jalan ke kampus"

Interpretasi:

high_activity = 0

kecuali terdapat konteks yang menunjukkan aktivitas fisik
yang cukup tinggi.


---

# 8. Annotation Quality Rules

Annotator harus:

1. Membaca seluruh sentence sebelum memberi label.
2. Tidak membuat asumsi yang tidak terdapat dalam sentence.
3. Menggunakan definisi label secara konsisten.
4. Memberikan semua label yang benar-benar didukung oleh
   sentence.
5. Tidak memberikan label hanya karena suatu makanan
   biasanya diasosiasikan dengan kondisi tertentu.
6. Mempertahankan `sample_id` yang sudah diberikan.
7. Menggunakan hanya nilai `0` dan `1`.

---

# 9. Binary Label Convention

Semua label harus bertipe binary.

Valid:

0
1

Tidak valid:

yes
no
true
false
maybe
unknown

Contoh:

sample_id,stress,poor_sleep,high_activity

S0001,1,0,0

S0002,0,1,0

S0003,0,0,1


---

# 10. Dataset Structure

Dataset dipisahkan menjadi beberapa file.

## Raw sentence

`data/raw/context_sentence.csv`

Columns:

- sample_id
- context_sentence

File ini menyimpan sentence asli.


## Raw annotation

`data/raw/context_annotation.csv`

Columns:

- sample_id
- stress
- poor_sleep
- high_activity
- fried_food
- large_portion
- sweet_drink
- late_meal
- high_carb_hint

File ini menyimpan hasil anotasi.


## Processed labeled dataset

`data/processed/context_labeled.csv`

Berisi gabungan:

- sample_id
- context_sentence
- seluruh label


## Train

`data/processed/context_train.csv`

Digunakan untuk training classifier.


## Validation

`data/processed/context_val.csv`

Digunakan untuk validation dan model selection.


## Test

`data/processed/context_test.csv`

Digunakan untuk final evaluation.


---

# 11. Dataset Limitations

Dataset Sprint 1 merupakan dataset custom yang dibuat secara
synthetic/team-written.

Dataset ini tidak dimaksudkan sebagai representasi sempurna
dari seluruh bahasa Indonesia.

Beberapa keterbatasan:

1. Jumlah data relatif kecil.
2. Sentence dibuat berdasarkan skenario meal-context yang
   relevan dengan GlucoSight.
3. Distribusi bahasa mungkin berbeda dari pengguna nyata.
4. Beberapa label memiliki kemungkinan overlap.
5. `high_carb_hint` bukan estimasi kandungan karbohidrat
   secara numerik.
6. `stress` bukan diagnosis kondisi mental.
7. `poor_sleep` bukan pengukuran klinis kualitas tidur.
8. Rule-based annotation dapat mengandung false positive
   atau false negative.
9. Dataset perlu dievaluasi ulang ketika data pengguna nyata
   tersedia.


---

# 12. Relation to NLP Baseline

Dataset ini digunakan sebagai custom domain dataset untuk
GlucoSight.

IndoNLU/IndoBERT dapat digunakan sebagai sumber representasi
bahasa atau baseline model.

Public Indonesian NLP datasets tidak dianggap sebagai pengganti
dataset meal-context GlucoSight karena label yang dibutuhkan
secara spesifik dirancang untuk konteks aplikasi.

Pipeline Sprint 1:

context sentence
        |
        v
annotation
        |
        v
multi-label dataset
        |
        v
rule-based baseline
        |
        v
IndoBERT embedding/classifier
        |
        v
predict()
        |
        v
context features
        |
        v
Forecasting/Fusion


---

# 13. Version

Annotation guideline version:

`annotation-guideline-v0.1`

NLP baseline version:

`nlp-baseline-v0.1`

Dataset version:

`context-dataset-v0.1`


---

# 14. Change Policy

Jika definisi label berubah, annotation guideline harus
diperbarui terlebih dahulu sebelum melakukan perubahan pada
dataset atau model.

Perubahan definisi label harus dicatat pada versi berikutnya.

Contoh:

v0.1 → initial Sprint 1 guideline

v0.2 → updated ambiguous cases

v1.0 → stable production-oriented guideline