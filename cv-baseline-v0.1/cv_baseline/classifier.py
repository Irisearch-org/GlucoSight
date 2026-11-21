"""Food classifier — MobileNetV2 hasil training cv-baseline-v0.1.

torch di-import secara lazy (di dalam method, bukan di top-level) supaya modul
lain — schema validator, macro lookup — tetap bisa di-import dan diuji di
environment tanpa PyTorch terpasang.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import MODEL_PATH, TOP_K


class ModelNotAvailable(Exception):
    """File .pt tidak ada, atau PyTorch belum terpasang."""


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class MobileNetV2Classifier:
    """Wrapper inference untuk checkpoint mobilenetv2_food10.pt.

    Checkpoint disimpan sebagai dict berisi model_state_dict, num_classes,
    class_to_idx, dan image_size — jadi class mapping ikut menempel di file
    model dan tidak bisa terpisah / tertukar urutannya.
    """

    def __init__(self, model_path=MODEL_PATH, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device_str = device
        self._model = None
        self._idx_to_class = None
        self.class_to_idx = None

    # -- loading ---------------------------------------------------------
    def load(self) -> "MobileNetV2Classifier":
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
        except ImportError as exc:
            raise ModelNotAvailable(
                "PyTorch tidak terpasang. Jalankan: pip install torch torchvision"
            ) from exc

        if not self.model_path.exists():
            raise ModelNotAvailable(
                f"Checkpoint tidak ditemukan: {self.model_path}. "
                "Salin mobilenetv2_food10.pt hasil training ke folder models/."
            )

        ckpt = torch.load(self.model_path, map_location="cpu")

        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, ckpt["num_classes"]
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        self._torch = torch
        self._model = model.to(self.device_str)
        self.class_to_idx = ckpt["class_to_idx"]
        self._idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- inference -------------------------------------------------------
    def predict_topk(self, image_array: np.ndarray, k: int = TOP_K
                     ) -> List[Tuple[str, float]]:
        """Array CHW ternormalisasi -> [(nama_kelas, probabilitas), ...]."""
        if not self.is_loaded:
            raise ModelNotAvailable("Model belum di-load. Panggil .load() dulu.")

        torch = self._torch
        x = torch.from_numpy(image_array).unsqueeze(0).to(self.device_str)
        with torch.no_grad():
            logits = self._model(x)[0].cpu().numpy()

        probs = _softmax(logits)
        order = np.argsort(probs)[::-1][:k]
        return [(self._idx_to_class[int(i)], float(probs[i])) for i in order]


class MockClassifier:
    """Classifier deterministik untuk testing pipeline tanpa PyTorch.

    HANYA untuk uji kontrak & schema. Output yang dihasilkan lewat classifier
    ini TIDAK valid secara nutrisi dan akan ditandai feature_status="mock"
    oleh predict().
    """

    is_mock = True

    def __init__(self, class_to_idx: dict):
        self.class_to_idx = class_to_idx
        self._classes = sorted(class_to_idx, key=class_to_idx.get)
        self._model = True

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self):
        return self

    def predict_topk(self, image_array: np.ndarray, k: int = TOP_K):
        # seed dari isi gambar -> deterministik per gambar, tapi bukan prediksi nyata
        seed = int(abs(float(np.sum(image_array))) * 1000) % (2 ** 31)
        rng = np.random.default_rng(seed)
        probs = _softmax(rng.normal(size=len(self._classes)) * 2)
        order = np.argsort(probs)[::-1][:k]
        return [(self._classes[int(i)], float(probs[i])) for i in order]
