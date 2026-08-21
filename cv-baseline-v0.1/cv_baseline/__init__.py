"""cv-baseline-v0.1 — CV food macro estimation baseline."""

from .config import MODEL_VERSION
from .predict import predict
from .schema import is_valid, validate_output

__all__ = ["predict", "validate_output", "is_valid", "MODEL_VERSION"]
