"""Preprocessing gambar meal.

Mereplikasi `eval_tf` dari notebook training persis:
    Resize(255) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)

Implementasi memakai Pillow + numpy supaya modul ini tidak butuh torch.
Konversi ke tensor dilakukan di classifier.py (satu-satunya tempat torch dipakai).
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, RESIZE_SIZE


class ImageLoadError(Exception):
    """Gambar tidak ada, korup, atau bukan format yang didukung."""


def load_image(path) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise ImageLoadError(f"File tidak ditemukan: {p}")
    try:
        img = Image.open(p)
        img.load()  # paksa decode di sini supaya file korup ketahuan lebih awal
    except Exception as exc:
        raise ImageLoadError(f"Gagal membaca gambar {p}: {exc}") from exc
    return img.convert("RGB")


def _resize_shorter_side(img: Image.Image, size: int) -> Image.Image:
    """Resize sisi terpendek ke `size`, aspect ratio dipertahankan."""
    w, h = img.size
    if w < h:
        new_w, new_h = size, round(h * size / w)
    else:
        new_h, new_w = size, round(w * size / h)
    return img.resize((new_w, new_h), Image.BILINEAR)


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = round((w - size) / 2)
    top = round((h - size) / 2)
    return img.crop((left, top, left + size, top + size))


def preprocess(path) -> np.ndarray:
    """Path gambar -> array float32 CHW ternormalisasi, shape (3, 224, 224)."""
    img = load_image(path)
    img = _resize_shorter_side(img, RESIZE_SIZE)
    img = _center_crop(img, IMAGE_SIZE)

    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC, [0,1]
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
        IMAGENET_STD, dtype=np.float32
    )
    return np.transpose(arr, (2, 0, 1)).copy()                # CHW
