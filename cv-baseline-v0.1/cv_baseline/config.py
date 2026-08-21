"""Konfigurasi terpusat untuk cv-baseline-v0.1."""

from pathlib import Path

MODEL_VERSION = "cv-baseline-v0.1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Paths ---
MODEL_PATH = PROJECT_ROOT / "models" / "mobilenetv2_food10.pt"
MACRO_TABLE_PATH = PROJECT_ROOT / "data" / "tkpi_macro_lookup.csv"
FEATURE_OUTPUT_DIR = PROJECT_ROOT / "features" / "cv" / MODEL_VERSION

# --- Preprocessing (HARUS sama persis dengan training notebook) ---
IMAGE_SIZE = 224
RESIZE_SIZE = int(IMAGE_SIZE * 1.14)  # 255, sama dengan eval_tf saat training
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Inference ---
TOP_K = 3
# Di bawah ambang ini, feature_status ditandai "low_confidence" agar tim
# Forecasting/Fusion bisa memilih untuk down-weight sample tersebut.
CONFIDENCE_THRESHOLD = 0.60

TIMEZONE_OFFSET_HOURS = 7  # WIB
