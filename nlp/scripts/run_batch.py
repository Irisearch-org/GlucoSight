import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(BASE_DIR / "nlp_baseline")
)


from predict import predict # type: ignore


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "context_sentence.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "features"
    / "nlp"
    / "nlp-baseline-v0.1"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "nlp_features.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("GLUCOSIGHT NLP BATCH FEATURE GENERATION")
print("=" * 70)

print()

print("Input:")
print(INPUT_FILE)

print()

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Loaded {len(df)} sentences"
)


# ============================================================
# VALIDATE INPUT
# ============================================================

required_columns = [
    "sample_id",
    "context_sentence"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# PROCESS
# ============================================================

results = []

print()
print("Processing...")

for index, row in df.iterrows():

    sample_id = row["sample_id"]

    sentence = row[
        "context_sentence"
    ]

    try:

        payload = {

            "sample_id":
            sample_id,

            "text":
            sentence
        }

        result = predict(
            payload
        )

        results.append(
            result
        )

    except Exception as error:

        print(
            f"ERROR {sample_id}: {error}"
        )


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

feature_rows = []

for result in results:

    row = {

        "sample_id":
        result["sample_id"],

        "modality":
        result["modality"],

        "stress":
        result["labels"]["stress"],

        "poor_sleep":
        result["labels"]["poor_sleep"],

        "high_activity":
        result["labels"]["high_activity"],

        "fried_food":
        result["labels"]["fried_food"],

        "large_portion":
        result["labels"]["large_portion"],

        "sweet_drink":
        result["labels"]["sweet_drink"],

        "late_meal":
        result["labels"]["late_meal"],

        "high_carb_hint":
        result["labels"]["high_carb_hint"],

        "stress_confidence":
        result["label_confidence"]["stress"],

        "poor_sleep_confidence":
        result["label_confidence"]["poor_sleep"],

        "high_activity_confidence":
        result["label_confidence"]["high_activity"],

        "fried_food_confidence":
        result["label_confidence"]["fried_food"],

        "large_portion_confidence":
        result["label_confidence"]["large_portion"],

        "sweet_drink_confidence":
        result["label_confidence"]["sweet_drink"],

        "late_meal_confidence":
        result["label_confidence"]["late_meal"],

        "high_carb_hint_confidence":
        result["label_confidence"]["high_carb_hint"],

        "context_embedding":
        str(
            result["context_embedding"]
        ),

        "embedding_dim":
        result["embedding_dim"],

        "feature_status":
        result["feature_status"],

        "model_version":
        result["model_version"],

        "created_at":
        result["created_at"]
    }

    feature_rows.append(
        row
    )


features_df = pd.DataFrame(
    feature_rows
)


# ============================================================
# SAVE
# ============================================================

features_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("BATCH PROCESS COMPLETE")
print("=" * 70)

print()

print(
    f"Input samples  : {len(df)}"
)

print(
    f"Output samples : {len(features_df)}"
)

print()

print("Output file:")

print(
    OUTPUT_FILE
)

print()

print("Feature columns:")

print(
    features_df.columns.tolist()
)

print()

print("First 5 rows:")

print(
    features_df.head().to_string(
        index=False
    )
)

print()

print("=" * 70)
print("DONE")
print("=" * 70)