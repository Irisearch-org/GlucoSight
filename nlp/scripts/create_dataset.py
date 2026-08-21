import pandas as pd
from pathlib import Path
import random


# ============================================================
# CONFIG
# ============================================================

random.seed(42)

N_SAMPLES = 400

LABELS = [
    "stress",
    "poor_sleep",
    "high_activity",
    "fried_food",
    "large_portion",
    "sweet_drink",
    "late_meal",
    "high_carb_hint"
]


# ============================================================
# FOLDER
# ============================================================

Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)


# ============================================================
# SENTENCE DATA
# ============================================================

sentences = [

    "Semalam aku cuma tidur empat jam.",
    "Tadi malam aku begadang sampai jam tiga.",
    "Aku kurang tidur semalam.",
    "Aku baru tidur sekitar jam tiga pagi.",
    "Tidurku semalam kurang dari lima jam.",

    "Hari ini aku lagi stres karena banyak tugas.",
    "Lagi banyak pikiran gara-gara deadline.",
    "Aku lagi kepikiran tugas yang belum selesai.",
    "Hari ini cukup bikin stres karena kerjaan numpuk.",
    "Aku lagi cemas karena besok ada ujian.",

    "Tadi pagi aku lari selama satu jam.",
    "Aku habis gym sekitar satu jam.",
    "Barusan selesai jogging cukup lama.",
    "Hari ini aku banyak jalan kaki.",
    "Aku habis olahraga cukup berat.",

    "Aku makan ayam goreng.",
    "Tadi makan tempe goreng.",
    "Aku makan tahu goreng.",
    "Siang ini makan ikan goreng.",
    "Aku makan beberapa gorengan.",

    "Aku makan nasi dua porsi.",
    "Tadi makannya porsi jumbo.",
    "Aku makan dalam porsi yang cukup besar.",
    "Tadi aku nambah nasi sampai dua kali.",
    "Porsinya tadi besar banget.",

    "Aku minum es teh manis.",
    "Tadi minum kopi susu.",
    "Aku minum boba setelah makan.",
    "Tadi siang minum thai tea.",
    "Aku minum es kopi gula aren.",

    "Aku baru makan malam jam sebelas.",
    "Tadi makan malam hampir tengah malam.",
    "Aku baru sempat makan jam sepuluh malam.",
    "Makan malamku tadi cukup larut.",
    "Aku baru makan setelah jam sebelas malam.",

    "Aku makan nasi dua porsi.",
    "Tadi makan nasi dengan mie sekaligus.",
    "Aku makan nasi goreng dalam porsi besar.",
    "Tadi makan mie dan nasi bersamaan.",
    "Aku makan dua porsi nasi.",

    "Semalam aku begadang karena banyak tugas, jadi tidur cuma empat jam.",
    "Karena deadline numpuk, tadi malam aku tidur sangat sedikit.",
    "Aku lagi banyak pikiran dan semalam cuma tidur beberapa jam.",
    "Banyak tugas bikin aku stres dan akhirnya tidur larut.",
    "Aku kepikiran pekerjaan sampai begadang semalam.",

    "Karena banyak kerjaan, aku baru makan malam jam sebelas.",
    "Aku lagi stres karena deadline dan baru makan malam larut.",
    "Tadi banyak tugas jadi aku baru sempat makan malam jam sepuluh.",
    "Karena kerjaan numpuk, makan malamku jadi hampir tengah malam.",

    "Semalam kurang tidur, jadi tadi pagi aku minum kopi susu.",
    "Karena begadang, pagi ini aku minum kopi manis.",
    "Tadi malam cuma tidur sebentar lalu pagi ini minum kopi susu.",

    "Semalam begadang dan siang ini aku makan ayam goreng.",
    "Karena kurang tidur, tadi makan siangku ayam goreng.",
    "Tadi malam tidur sedikit lalu siang makan tempe goreng.",

    "Habis gym satu jam, aku makan nasi dua porsi.",
    "Setelah jogging, aku makan dalam porsi besar.",
    "Tadi habis olahraga cukup berat lalu makan dua porsi.",
    "Setelah latihan, aku makan lebih banyak dari biasanya.",

    "Habis gym aku makan ayam goreng.",
    "Setelah jogging, aku makan tempe goreng.",
    "Tadi habis olahraga aku makan ayam goreng.",

    "Aku makan ayam goreng sampai dua porsi.",
    "Tadi makan ayam goreng dengan porsi jumbo.",
    "Aku makan banyak gorengan karena lagi lapar.",

    "Aku makan porsi jumbo dan minum es teh manis.",
    "Tadi makan dua porsi lalu minum kopi susu.",
    "Aku makan banyak banget dengan minuman manis.",

    "Aku baru makan jam sebelas malam sambil minum es teh manis.",
    "Tadi makan malam larut dan minum kopi susu.",
    "Aku baru makan setelah jam sepuluh malam lalu minum boba.",

    "Aku baru makan jam sebelas malam dan makan ayam goreng.",
    "Tadi makan malam larut dengan ayam goreng.",
    "Aku baru makan setelah jam sepuluh dan pilih tempe goreng.",

    "Aku makan nasi goreng dan ayam goreng.",
    "Tadi makan mie goreng dengan ayam goreng.",
    "Aku makan nasi dua porsi dengan ayam goreng.",

    "Aku makan nasi dan minum es teh manis.",
    "Tadi makan nasi goreng dan minum kopi susu.",
    "Aku makan mie lalu minum boba.",

    "Aku baru makan nasi dua porsi jam sebelas malam.",
    "Tadi makan nasi goreng cukup larut.",
    "Aku makan mie dan nasi setelah jam sepuluh malam.",

    "Habis lari lima kilometer aku makan nasi dua porsi.",
    "Setelah gym, aku makan nasi dan mie.",
    "Habis workout aku makan nasi goreng.",

    "Semalam begadang karena deadline, lalu siang makan ayam goreng dua porsi.",
    "Habis gym aku makan nasi dua porsi dengan ayam goreng.",
    "Karena banyak tugas, aku baru makan malam jam sebelas sambil minum es teh manis.",
    "Semalam kurang tidur, siang makan nasi goreng porsi jumbo dan minum kopi susu.",
    "Habis lari lima kilometer aku makan nasi dua porsi dan minum es teh manis.",

    "Aku makan nasi dengan telur rebus.",
    "Tadi aku makan sup ayam.",
    "Aku makan sayur dan telur.",
    "Tadi siang aku makan nasi dan sayur.",
    "Aku makan ikan bakar dengan sayur.",
    "Tadi aku makan sup sayuran.",
    "Aku makan nasi dengan lauk sederhana.",
    "Tadi makan sayur, tahu, dan nasi secukupnya.",
    "Aku makan ayam bakar dengan sayuran.",
    "Tadi makan nasi satu porsi dengan lauk biasa."
]


# ============================================================
# MAKE 400 SENTENCES
# ============================================================

# Hapus duplikat
sentences = list(dict.fromkeys(sentences))

# Kalau belum 400, variasikan kalimat yang ada
base_sentences = sentences.copy()

while len(sentences) < N_SAMPLES:

    base = random.choice(base_sentences)

    variations = [
        base,
        "Tadi " + base.lower(),
        "Hari ini " + base.lower(),
        "Barusan " + base.lower(),
        "Aku tadi " + base.lower()
    ]

    new_sentence = random.choice(variations)

    if new_sentence not in sentences:
        sentences.append(new_sentence)


# Ambil tepat 400
sentences = sentences[:N_SAMPLES]

# Acak
random.shuffle(sentences)


# ============================================================
# RULE-BASED ANNOTATION
# ============================================================

def annotate(sentence):

    text = sentence.lower()

    labels = {
        "stress": 0,
        "poor_sleep": 0,
        "high_activity": 0,
        "fried_food": 0,
        "large_portion": 0,
        "sweet_drink": 0,
        "late_meal": 0,
        "high_carb_hint": 0
    }

    # -------------------------
    # STRESS
    # -------------------------

    stress_words = [
        "stres",
        "stress",
        "banyak pikiran",
        "deadline",
        "cemas",
        "kepikiran",
        "tertekan",
        "anxious",
        "pusing"
    ]

    if any(word in text for word in stress_words):
        labels["stress"] = 1


    # -------------------------
    # POOR SLEEP
    # -------------------------

    sleep_words = [
        "kurang tidur",
        "begadang",
        "tidur sedikit",
        "tidur cuma",
        "tidur sekitar",
        "tidur larut",
        "nggak cukup tidur",
        "tidak cukup tidur"
    ]

    if any(word in text for word in sleep_words):
        labels["poor_sleep"] = 1


    # -------------------------
    # HIGH ACTIVITY
    # -------------------------

    activity_words = [
        "gym",
        "jogging",
        "lari",
        "olahraga",
        "workout",
        "bersepeda",
        "angkat beban",
        "kardio",
        "jalan kaki"
    ]

    if any(word in text for word in activity_words):
        labels["high_activity"] = 1


    # -------------------------
    # FRIED FOOD
    # -------------------------

    fried_words = [
        "goreng",
        "gorengan",
        "crispy",
        "ayam crispy",
        "ayam tepung"
    ]

    if any(word in text for word in fried_words):
        labels["fried_food"] = 1


    # -------------------------
    # LARGE PORTION
    # -------------------------

    portion_words = [
        "dua porsi",
        "porsi jumbo",
        "porsi besar",
        "cukup besar",
        "lebih banyak",
        "banyak banget",
        "nambah",
        "jumlah banyak",
        "porsi gede"
    ]

    if any(word in text for word in portion_words):
        labels["large_portion"] = 1


    # -------------------------
    # SWEET DRINK
    # -------------------------

    sweet_words = [
        "es teh manis",
        "kopi susu",
        "boba",
        "thai tea",
        "gula aren",
        "minuman manis",
        "soda",
        "minuman dengan gula"
    ]

    if any(word in text for word in sweet_words):
        labels["sweet_drink"] = 1


    # -------------------------
    # LATE MEAL
    # -------------------------

    late_words = [
        "jam sebelas",
        "jam sepuluh",
        "tengah malam",
        "larut malam",
        "menjelang tengah malam",
        "telat makan"
    ]

    if any(word in text for word in late_words):
        labels["late_meal"] = 1


    # -------------------------
    # HIGH CARB
    # -------------------------

    carb_words = [
        "nasi",
        "mie",
        "mi ",
        "nasi goreng",
        "mie goreng",
        "roti",
        "pasta",
        "kentang",
        "bertepung"
    ]

    if any(word in text for word in carb_words):
        labels["high_carb_hint"] = 1


    return labels


# ============================================================
# CREATE DATASET
# ============================================================

rows = []

for i, sentence in enumerate(sentences, start=1):

    labels = annotate(sentence)

    row = {
        "sample_id": f"S{i:04d}",
        "context_sentence": sentence
    }

    row.update(labels)

    rows.append(row)


df = pd.DataFrame(rows)


# ============================================================
# SPLIT RAW SENTENCE
# ============================================================

sentence_df = df[
    [
        "sample_id",
        "context_sentence"
    ]
].copy()


# ============================================================
# SPLIT ANNOTATION
# ============================================================

annotation_df = df[
    [
        "sample_id",
        "stress",
        "poor_sleep",
        "high_activity",
        "fried_food",
        "large_portion",
        "sweet_drink",
        "late_meal",
        "high_carb_hint"
    ]
].copy()


# ============================================================
# PROCESSED = SENTENCE + ANNOTATION
# ============================================================

labeled_df = df.copy()


# ============================================================
# TRAIN / VAL / TEST
# ============================================================

train_df = df.sample(
    frac=0.70,
    random_state=42
)

remaining = df.drop(train_df.index)

val_df = remaining.sample(
    frac=0.50,
    random_state=42
)

test_df = remaining.drop(val_df.index)


# ============================================================
# SAVE FILES
# ============================================================

sentence_df.to_csv(
    "data/raw/context_sentence.csv",
    index=False,
    encoding="utf-8-sig"
)

annotation_df.to_csv(
    "data/raw/context_annotation.csv",
    index=False,
    encoding="utf-8-sig"
)

labeled_df.to_csv(
    "data/processed/context_labeled.csv",
    index=False,
    encoding="utf-8-sig"
)

train_df.to_csv(
    "data/processed/context_train.csv",
    index=False,
    encoding="utf-8-sig"
)

val_df.to_csv(
    "data/processed/context_val.csv",
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    "data/processed/context_test.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# CHECK
# ============================================================

print("=" * 60)
print("GLUCOSIGHT NLP DATASET BERHASIL DIBUAT")
print("=" * 60)

print()

print("Total data :", len(df))

print()

print("RAW:")
print("  data/raw/context_sentence.csv")
print("  data/raw/context_annotation.csv")

print()

print("PROCESSED:")
print("  data/processed/context_labeled.csv")
print("  data/processed/context_train.csv")
print("  data/processed/context_val.csv")
print("  data/processed/context_test.csv")

print()

print("Columns context_sentence:")
print(sentence_df.columns.tolist())

print()

print("Columns context_annotation:")
print(annotation_df.columns.tolist())

print()

print("Contoh:")
print(df.head(10).to_string(index=False))

print()

print("Selesai.")