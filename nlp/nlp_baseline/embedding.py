import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "indobenchmark/indobert-base-p1"

EMBEDDING_DIM = 768


# ============================================================
# MODEL LOADING
# ============================================================

_tokenizer = None
_model = None


def load_model():

    global _tokenizer
    global _model

    if _tokenizer is None or _model is None:

        print(
            f"Loading IndoBERT: {MODEL_NAME}"
        )

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        _model = AutoModel.from_pretrained(
            MODEL_NAME
        )

        _model.eval()

    return _tokenizer, _model


# ============================================================
# MEAN POOLING
# ============================================================

def mean_pooling(
    model_output,
    attention_mask
):

    token_embeddings = (
        model_output.last_hidden_state
    )

    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(
            token_embeddings.size()
        )
        .float()
    )

    sum_embeddings = torch.sum(
        token_embeddings
        * input_mask_expanded,
        dim=1
    )

    sum_mask = torch.clamp(
        input_mask_expanded.sum(
            dim=1
        ),
        min=1e-9
    )

    return (
        sum_embeddings
        / sum_mask
    )


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def create_indobert_embedding(
    text: str
) -> list:

    tokenizer, model = load_model()

    text = str(text).strip()

    if text == "":
        raise ValueError(
            "Text tidak boleh kosong."
        )

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        model_output = model(
            **encoded
        )

    embedding = mean_pooling(
        model_output,
        encoded["attention_mask"]
    )

    embedding = embedding[
        0
    ].cpu().numpy()

    # Normalize
    norm = np.linalg.norm(
        embedding
    )

    if norm > 0:

        embedding = (
            embedding / norm
        )

    return embedding.tolist()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_sentence = (
        "Semalam aku begadang karena "
        "banyak tugas, lalu makan ayam "
        "goreng dua porsi."
    )

    print("=" * 70)
    print("GLUCOSIGHT INDOBERT EMBEDDING TEST")
    print("=" * 70)

    print()

    print("Input:")
    print(test_sentence)

    print()

    embedding = create_indobert_embedding(
        test_sentence
    )

    print(
        "Embedding dimension:",
        len(embedding)
    )

    print()

    print("First 10 values:")

    print(
        embedding[:10]
    )

    print()

    print("✓ IndoBERT embedding berhasil dibuat.")