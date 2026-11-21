"""Lookup makro: kelas makanan -> estimasi carbs/protein/fat/fiber/kalori.

Pendekatan (sesuai Recommended Baseline Approach di task brief):
    kelas -> nilai rata-rata per 100 g -> dikali asumsi porsi standar

Dua keterbatasan yang sengaja dibuat eksplisit di sini:

1. `fiber_g` bukan angka TKPI. Tabel komposisi TKPI tidak punya kolom serat,
   jadi serat diestimasi sebagai proporsi dari karbohidrat (`fiber_ratio_of_carbs`).
   Ini heuristik, bukan pengukuran.

2. Porsi bersifat asumsi tetap per kelas. Model ini classifier, bukan estimator
   volume — jadi piring besar dan piring kecil menghasilkan angka yang sama.
   Ini sumber error terbesar pada nilai makro absolut.
"""

import csv
from pathlib import Path
from typing import Dict

from .config import MACRO_TABLE_PATH


class MacroLookupError(Exception):
    pass


class MacroLookup:
    def __init__(self, table_path=MACRO_TABLE_PATH):
        self.table_path = Path(table_path)
        self.table: Dict[str, dict] = {}
        self._load()

    def _load(self):
        if not self.table_path.exists():
            raise MacroLookupError(f"Tabel makro tidak ditemukan: {self.table_path}")

        with open(self.table_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.table[row["food_class"]] = {
                    "tkpi_item_name": row["tkpi_item_name"],
                    "serving_g": float(row["serving_g"]),
                    "energy_kcal_100g": float(row["energy_kcal_100g"]),
                    "carbs_g_100g": float(row["carbs_g_100g"]),
                    "protein_g_100g": float(row["protein_g_100g"]),
                    "fat_g_100g": float(row["fat_g_100g"]),
                    "fiber_ratio_of_carbs": float(row["fiber_ratio_of_carbs"]),
                    "mapping_confidence": row["mapping_confidence"],
                    "source_note": row["source_note"],
                }

        if not self.table:
            raise MacroLookupError("Tabel makro kosong.")

    def has(self, food_class: str) -> bool:
        return food_class in self.table

    def get_macros(self, food_class: str) -> dict:
        """Kembalikan makro untuk satu porsi standar kelas tersebut."""
        if food_class not in self.table:
            raise MacroLookupError(
                f"Kelas '{food_class}' tidak ada di tabel makro. "
                f"Kelas tersedia: {sorted(self.table)}"
            )

        row = self.table[food_class]
        scale = row["serving_g"] / 100.0

        carbs = row["carbs_g_100g"] * scale
        fiber = carbs * row["fiber_ratio_of_carbs"]  # heuristik, lihat docstring

        return {
            "carbs_g": round(carbs, 1),
            "protein_g": round(row["protein_g_100g"] * scale, 1),
            "fat_g": round(row["fat_g_100g"] * scale, 1),
            "fiber_g": round(fiber, 1),
            "calories_kcal": round(row["energy_kcal_100g"] * scale),
            "_serving_g": row["serving_g"],
            "_mapping_confidence": row["mapping_confidence"],
            "_tkpi_item_name": row["tkpi_item_name"],
        }
