from datetime import datetime, timezone
import hashlib

from classifier import classify_context
from embedding import create_indobert_embedding


# ============================================================
# CONFIG
# ============================================================

MODEL_VERSION = "nlp-baseline-v0.1"

MODALITY = "meal_context"

EMBEDDING_DIM = 768


# ============================================================
# CONTEXT EMBEDDING
# ============================================================

def create_context_embedding(text: str) -> list:
    """
    Membuat fixed-length context embedding
    menggunakan IndoBERT.
    """

    return create_indobert_embedding(text)


# ============================================================
# SAMPLE ID
# ============================================================

def generate_sample_id(text: str) -> str:
    """
    Generate deterministic sample ID dari input text.

    Contoh:
    S0001
    S0234
    S9876

    Jika input sudah memiliki sample_id,
    predict() akan menggunakan sample_id tersebut.
    """

    hash_value = hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()

    number = int(
        hash_value[:8],
        16
    ) % 10000

    return f"S{number:04d}"


# ============================================================
# MAIN PREDICT FUNCTION
# ============================================================

def predict(input_payload: dict) -> dict:
    """
    Standard GlucoSight NLP prediction contract.

    Input:

    {
        "sample_id": "S0001",
        "text": "Semalam aku begadang..."
    }

    Output:

    {
        "sample_id": ...,
        "modality": ...,
        "labels": ...,
        "label_confidence": ...,
        "context_embedding": ...,
        "embedding_dim": ...,
        "feature_status": ...,
        "model_version": ...,
        "created_at": ...
    }
    """

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if not isinstance(input_payload, dict):
        raise TypeError(
            "input_payload harus berupa dictionary"
        )

    if "text" not in input_payload:
        raise ValueError(
            "input_payload wajib memiliki key 'text'"
        )

    text = str(
        input_payload["text"]
    ).strip()

    if text == "":
        raise ValueError(
            "text tidak boleh kosong"
        )


    # --------------------------------------------------------
    # SAMPLE ID
    # --------------------------------------------------------

    sample_id = input_payload.get(
        "sample_id"
    )

    if sample_id is None:
        sample_id = generate_sample_id(
            text
        )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    classification = classify_context(
        text
    )

    labels = classification[
        "labels"
    ]

    label_confidence = classification[
        "label_confidence"
    ]


    # --------------------------------------------------------
    # INDOBERT EMBEDDING
    # --------------------------------------------------------

    context_embedding = create_context_embedding(
        text
    )

    embedding_dim = len(
        context_embedding
    )


    # --------------------------------------------------------
    # FEATURE STATUS
    # --------------------------------------------------------

    feature_status = "valid"

    if embedding_dim != EMBEDDING_DIM:

        feature_status = (
            "invalid_embedding_dim"
        )


    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    created_at = datetime.now(
        timezone.utc
    ).isoformat()


    # --------------------------------------------------------
    # OUTPUT CONTRACT
    # --------------------------------------------------------

    result = {

        "sample_id":
        sample_id,

        "modality":
        MODALITY,

        "labels":
        labels,

        "label_confidence":
        label_confidence,

        "context_embedding":
        context_embedding,

        "embedding_dim":
        embedding_dim,

        "feature_status":
        feature_status,

        "model_version":
        MODEL_VERSION,

        "created_at":
        created_at
    }


    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    payload = {

        "sample_id":
        "S0001",

        "text":
        "Semalam aku begadang karena banyak tugas, "
        "lalu makan ayam goreng dua porsi."
    }


    result = predict(
        payload
    )


    print("=" * 70)

    print(
        "GLUCOSIGHT NLP PREDICT TEST"
    )

    print("=" * 70)


    print()

    for key, value in result.items():

        print(
            f"{key}:"
        )

        print(
            value
        )

        print()