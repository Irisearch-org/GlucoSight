"""Step 2: the meal (docs/SPEC_V2_PIPELINE.md section 2).

Photo path: a vision model names the dishes, the embedding matcher maps each name to a
dish-table row, and the confidence rule below decides whether the photo may speak for
that item. Manual path: the user picks rows from the same table. Both end in
``resolve_meal``: grams per item, macros summed over items.

``SIM_MIN`` / ``MARGIN_MIN`` are the only copy of the confidence rule (ticket #19). The
VLM's self-reported confidence is never used: it is uncalibrated (ticket #18).
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import requests
from PIL import Image, UnidentifiedImageError

from integration.dishes import (
    GEMINI_BASE_URL, DishRow, DishTable, Embedder, GeminiUnavailable, normalise_text,
)

SIM_MIN = 0.80
MARGIN_MIN = 0.05
PORTION_SCALE = (0.5, 1.0, 1.5)  # Sedikit / Sedang / Banyak
MAX_GRAMS = 5000.0

# Trial settings (cv/experiments/dish_id_trial/vlm_run.py): the thresholds above were
# calibrated on exactly this prompt, image size and decoding.
VLM_PROMPT = (
    "List every distinct food dish visible in this image, as specifically "
    "as you can (e.g. 'rendang', 'nasi goreng', not just 'rice dish'). "
    "Use the dish's common Indonesian name if you recognise it, otherwise "
    "whatever name best identifies it. "
    'Return JSON only: {"items": [{"food_name": "...", "confidence": 0.0}]}. '
    "One entry per distinct dish — don't split one dish into its ingredients."
)
VLM_MAX_SIDE = 1024
VLM_JPEG_QUALITY = 88
VLM_MAX_OUTPUT_TOKENS = 300
VLM_TIMEOUT_S = 60
VLM_MAX_RETRIES = 2

MSG_VISION_DOWN = "Layanan pengenal foto sedang tidak tersedia. Coba lagi, atau isi makanan secara manual."


class MealError(ValueError):
    """Invalid meal input (HTTP 422). The message is shown to the user, in Bahasa Indonesia."""


class VisionUnavailable(GeminiUnavailable):
    """The vision model or the embedding API failed (HTTP 503).

    ``str(exc)`` is the Bahasa Indonesia message for the user; ``detail`` is the
    technical cause, for logs only.
    """

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


class GeminiKeyError(RuntimeError):
    """Neither GEMINI_API_KEY in the environment nor the key file has a key (HTTP 503)."""


def load_gemini_key(path: Path) -> str:
    """GEMINI_API_KEY from the environment (Heroku config var), else from the gitignored
    ``KEY=value`` file (``served.json`` vlm.key_file)."""
    env = os.environ.get("GEMINI_API_KEY", "").strip()
    if env:
        return env
    path = Path(path)
    if not path.is_file():
        raise GeminiKeyError(f"file kunci Gemini tidak ditemukan: {path.name}")
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() == "GEMINI_API_KEY" and value.strip().strip("'\""):
            return value.strip().strip("'\"")
    raise GeminiKeyError(f"GEMINI_API_KEY tidak ada di {path.name}")


@dataclass(frozen=True)
class PhotoItem:
    vlm_name: str
    accepted: bool
    row: DishRow | None  # set iff accepted
    suggestions: tuple[DishRow, ...]
    similarity: float
    margin: float


@dataclass(frozen=True)
class MealItem:
    row: DishRow
    grams: float
    source: Literal["photo", "manual"]
    similarity: float | None = None
    margin: float | None = None

    def macros(self) -> dict[str, float | None]:
        """This item's contribution; ``fibre_g`` is None when the row's fibre is unknown."""
        f = self.grams / 100.0
        r = self.row
        return {
            "carbs_g": r.carbs_g * f, "protein_g": r.protein_g * f, "fat_g": r.fat_g * f,
            "fibre_g": None if r.fibre_g is None else r.fibre_g * f,
            "kcal": r.energy_kcal * f,
        }


@dataclass(frozen=True)
class Meal:
    items: tuple[MealItem, ...]
    carbs_g: float
    protein_g: float
    fat_g: float
    fibre_g: float | None  # None if ANY item's row has no fibre value (missing is not zero)
    kcal: float
    dish_table_build: str
    vlm_model_version: str | None


class VisionNamer(Protocol):
    """Seam: GeminiVisionNamer (real) and a fake in tests."""

    model_version: str

    def name_dishes(self, image: bytes, mime: str) -> list[str]:
        """Indonesian dish names, one per distinct item visible."""
        ...


class GeminiVisionNamer:
    def __init__(self, api_key: str, model: str):
        self.model_version = model
        self._api_key = api_key

    def name_dishes(self, image: bytes, mime: str) -> list[str]:
        body = {
            "contents": [{"parts": [
                {"text": VLM_PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": _encode_image(image)}},
            ]}],
            "generationConfig": {
                "maxOutputTokens": VLM_MAX_OUTPUT_TOKENS, "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        data = self._post(body)
        try:
            items = json.loads(data["candidates"][0]["content"]["parts"][0]["text"])["items"]
            return [str(it["food_name"]).strip() for it in items if str(it["food_name"]).strip()]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise VisionUnavailable(MSG_VISION_DOWN, "respons model tidak valid") from exc

    def _post(self, body: dict) -> dict:
        url = f"{GEMINI_BASE_URL}{self.model_version}:generateContent"
        for attempt in range(VLM_MAX_RETRIES + 1):
            try:
                resp = requests.post(url, params={"key": self._api_key}, json=body,
                                     timeout=VLM_TIMEOUT_S)
            except requests.RequestException as exc:
                raise VisionUnavailable(MSG_VISION_DOWN, type(exc).__name__) from exc
            if resp.ok:
                return resp.json()
            if resp.status_code not in (429, 500, 503) or attempt == VLM_MAX_RETRIES:
                raise VisionUnavailable(MSG_VISION_DOWN, f"HTTP {resp.status_code}: {resp.text[:300]}")
            time.sleep(2 ** attempt * 2)
        raise AssertionError("unreachable")


def _encode_image(image: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise MealError("Foto tidak dapat dibaca. Coba ambil foto lagi.") from exc
    img.thumbnail((VLM_MAX_SIDE, VLM_MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=VLM_JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def identify_photo(image: bytes, mime: str, namer: VisionNamer, table: DishTable,
                   embedder: Embedder) -> list[PhotoItem]:
    """Name the dishes in a photo and decide, per item, whether the photo may be trusted.

    An item is accepted when top-1 similarity >= SIM_MIN and the top-1 minus top-2 margin
    >= MARGIN_MIN; otherwise it needs manual entry and carries the matcher's top-3
    suggestions. A second item that is accepted onto an already accepted row is dropped
    (one entry per distinct dish).
    """
    items: list[PhotoItem] = []
    seen_names: set[str] = set()
    accepted_ids: set[str] = set()
    try:
        for name in namer.name_dishes(image, mime):
            key = normalise_text(name)
            if not key or key in seen_names:
                continue
            seen_names.add(key)
            m = table.match(name, embedder)
            if m is None:
                items.append(PhotoItem(name, False, None, (), 0.0, 0.0))
                continue
            accepted = m.similarity >= SIM_MIN and m.margin >= MARGIN_MIN
            if accepted and m.row.id in accepted_ids:
                continue
            if accepted:
                accepted_ids.add(m.row.id)
            items.append(PhotoItem(
                vlm_name=name, accepted=accepted, row=m.row if accepted else None,
                suggestions=m.suggestions, similarity=m.similarity, margin=m.margin))
    except GeminiUnavailable as exc:
        if isinstance(exc, VisionUnavailable):
            raise
        raise VisionUnavailable(MSG_VISION_DOWN, str(exc)) from exc
    return items


def _grams(item: dict, row: DishRow) -> float:
    grams = item.get("grams")
    if grams is not None:
        if isinstance(grams, bool) or not isinstance(grams, (int, float)) \
                or not math.isfinite(grams) or not 0 < grams <= MAX_GRAMS:
            raise MealError(f"Berat untuk '{row.name}' harus antara 0 dan {MAX_GRAMS:.0f} gram.")
        return float(grams)
    portion = item.get("portion")
    if portion is None:
        raise MealError(f"Isi berat (gram) atau porsi untuk '{row.name}'.")
    if isinstance(portion, bool) or portion not in (0, 1, 2):
        raise MealError(f"Porsi untuk '{row.name}' tidak valid.")
    if row.serving_g is None:
        raise MealError(f"'{row.name}' tidak punya takaran porsi baku. Isi beratnya dalam gram.")
    return PORTION_SCALE[int(portion)] * row.serving_g


def resolve_meal(items: list[dict], table: DishTable, *, vlm_model_version: str | None) -> Meal:
    """Grams per item (entered grams, else portion scale x serving_g) and summed macros."""
    if not items:
        raise MealError("Tambahkan minimal satu makanan.")
    resolved: list[MealItem] = []
    for it in items:
        row_id = it.get("row_id")
        try:
            row = table.get(row_id)
        except (KeyError, TypeError):
            raise MealError(f"Makanan '{row_id}' tidak ada di tabel makanan.") from None
        source = it.get("source")
        if source not in ("photo", "manual"):
            raise MealError(f"Sumber item '{row.name}' tidak valid.")
        resolved.append(MealItem(
            row=row, grams=_grams(it, row), source=source,
            similarity=it.get("similarity"), margin=it.get("margin")))

    parts = [m.macros() for m in resolved]
    fibres = [p["fibre_g"] for p in parts]
    return Meal(
        items=tuple(resolved),
        carbs_g=sum(p["carbs_g"] for p in parts),
        protein_g=sum(p["protein_g"] for p in parts),
        fat_g=sum(p["fat_g"] for p in parts),
        fibre_g=None if any(f is None for f in fibres) else sum(fibres),
        kcal=sum(p["kcal"] for p in parts),
        dish_table_build=table.build_hash,
        vlm_model_version=vlm_model_version,
    )
