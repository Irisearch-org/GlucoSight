import re


# ============================================================
# LABEL CONFIG
# ============================================================

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
# KEYWORDS
# ============================================================

KEYWORDS = {

    "stress": [
        "stres",
        "stress",
        "cemas",
        "banyak pikiran",
        "kepikiran",
        "tertekan",
        "deadline",
        "tugas numpuk",
        "kerjaan numpuk"
    ],

    "poor_sleep": [
        "kurang tidur",
        "begadang",
        "tidur sedikit",
        "tidur cuma",
        "tidur larut",
        "tidur sangat sedikit",
        "nggak cukup tidur",
        "tidak cukup tidur"
    ],

    "high_activity": [
        "gym",
        "jogging",
        "lari",
        "olahraga",
        "workout",
        "bersepeda",
        "angkat beban",
        "kardio",
        "jalan kaki"
    ],

    "fried_food": [
        "ayam goreng",
        "tempe goreng",
        "tahu goreng",
        "ikan goreng",
        "gorengan",
        "goreng",
        "crispy"
    ],

    "large_portion": [
        "dua porsi",
        "porsi jumbo",
        "porsi besar",
        "cukup besar",
        "lebih banyak",
        "banyak banget",
        "nambah nasi",
        "nambah",
        "porsi gede"
    ],

    "sweet_drink": [
        "es teh manis",
        "kopi susu",
        "boba",
        "thai tea",
        "gula aren",
        "minuman manis",
        "minuman dengan gula",
        "soda"
    ],

    "late_meal": [
        "jam sebelas malam",
        "jam 11 malam",
        "jam sepuluh malam",
        "jam 10 malam",
        "tengah malam",
        "larut malam",
        "setelah jam sebelas",
        "setelah jam 11",
        "setelah jam sepuluh",
        "setelah jam 10"
    ],

    "high_carb_hint": [
        "nasi",
        "nasi goreng",
        "mie",
        "mi ",
        "mie goreng",
        "mi goreng",
        "pasta",
        "roti",
        "kentang"
    ]
}


# ============================================================
# NEGATIVE KEYWORDS
# ============================================================

NEGATIONS = [
    "tidak",
    "nggak",
    "gak",
    "ga",
    "bukan",
    "tanpa"
]


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text: str) -> str:
    """
    Basic preprocessing untuk rule-based classifier.
    """

    text = str(text).lower().strip()

    # normalisasi whitespace
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# KEYWORD MATCHING
# ============================================================

def keyword_match(text: str, keyword: str) -> bool:
    """
    Mengecek apakah keyword muncul di dalam text.
    """

    return keyword in text


# ============================================================
# NEGATION CHECK
# ============================================================

def has_negation(text: str, keyword: str) -> bool:
    """
    Mengecek apakah keyword didahului kata negasi sederhana.

    Contoh:
    "tidak minum kopi susu"

    tidak -> kopi susu

    akan dianggap sebagai negated keyword.
    """

    position = text.find(keyword)

    if position == -1:
        return False

    previous_text = text[
        max(0, position - 20):position
    ]

    for negation in NEGATIONS:

        if negation in previous_text:
            return True

    return False


# ============================================================
# CLASSIFIER
# ============================================================

def classify_context(text: str) -> dict:

    text = preprocess_text(text)

    predictions = {}
    confidence = {}

    for label in LABELS:

        keywords = KEYWORDS[label]

        matched_keywords = []

        for keyword in keywords:

            if keyword_match(text, keyword):

                if not has_negation(text, keyword):
                    matched_keywords.append(keyword)

        # ----------------------------------------------------
        # LABEL DETECTED
        # ----------------------------------------------------

        if len(matched_keywords) > 0:

            predictions[label] = 1

            # Confidence sederhana berdasarkan jumlah match
            if len(matched_keywords) >= 2:
                confidence[label] = 0.95
            else:
                confidence[label] = 0.85

        # ----------------------------------------------------
        # LABEL NOT DETECTED
        # ----------------------------------------------------

        else:

            predictions[label] = 0
            confidence[label] = 0.10

    return {
        "labels": predictions,
        "label_confidence": confidence
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_sentences = [

        "Semalam aku begadang karena banyak tugas, lalu makan ayam goreng dua porsi.",

        "Habis gym aku makan nasi dua porsi dan minum es teh manis.",

        "Aku baru makan malam jam sebelas dan minum kopi susu.",

        "Tadi aku makan nasi dengan ayam bakar.",

        "Aku makan telur rebus dan minum air putih."
    ]

    print("=" * 70)
    print("GLUCOSIGHT RULE-BASED NLP BASELINE")
    print("=" * 70)

    for sentence in test_sentences:

        result = classify_context(sentence)

        print()
        print("Sentence:")
        print(sentence)

        print()
        print("Labels:")

        for label in LABELS:

            print(
                f"  {label:18s}: "
                f"{result['labels'][label]} "
                f"(confidence={result['label_confidence'][label]:.2f})"
            )

        print("-" * 70)