#!/usr/bin/env python3
"""Unduh dataset Indonesian Food Image (Mendeley vtjd68bmwt, CC BY 4.0).

Dataset ini adalah sumber gambar yang sama dengan training MobileNetV2
(cv-baseline-v0.1): 10 kelas, sudah berisi split train/ test/ bawaan
(70/30). Split test inilah held-out set untuk eksperimen VLM vs MobileNetV2
— jadi kedua model dinilai di foto yang persis sama.

Lisensi: CC BY 4.0 (atribusi: Wicaksono Ashari, S.H., Indonesian Food
Image, Mendeley Data, V1, doi:10.17632/vtjd68bmwt.1).

Pemakaian:
    python fetch_dataset.py                 # unduh jika belum ada, lalu ekstrak
    python fetch_dataset.py --force         # unduh ulang walau sudah ada

Data disimpan di repo-root /data/ (gitignored — data mentah tidak masuk git).
"""

import argparse
import sys
import zipfile
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET_ID = "vtjd68bmwt"
VERSION = 1
ZIP_URL = f"https://data.mendeley.com/public-api/zip/{DATASET_ID}/download/{VERSION}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

PROJECT_ROOT = Path(__file__).resolve().parents[3]      # .../<repo>/
DATA_DIR = PROJECT_ROOT / "data" / "cv" / "indonesian_food_image"
ZIP_PATH = DATA_DIR / f"{DATASET_ID}-v{VERSION}.zip"
EXPECTED_CLASSES = {
    "bakso", "bebek_betutu", "gado_gado", "gudeg", "nasi_goreng",
    "pempek", "rawon", "rendang", "sate", "soto",
}


def find_split_root() -> Path:
    """Cari folder yang langsung berisi train/ dan test/.

    Zip Mendeley membungkus dataset dalam 'Indonesian Food Image/Clean_Data/'
    — jadi root split tidak selalu DATA_DIR itu sendiri.
    """
    candidates = [DATA_DIR, *sorted(DATA_DIR.glob("*/Clean_Data")),
                  *sorted(DATA_DIR.glob("*/"))]
    for c in candidates:
        if (c / "train").is_dir() and (c / "test").is_dir():
            return c
    raise FileNotFoundError(
        f"Tidak menemukan folder train/ + test/ di bawah {DATA_DIR}")


def already_extracted() -> bool:
    """True kalau folder test/ berisi minimal satu kelas yang dikenal."""
    try:
        test_dir = find_split_root() / "test"
    except FileNotFoundError:
        return False
    found = {p.name for p in test_dir.iterdir() if p.is_dir()}
    return bool(found & EXPECTED_CLASSES)


def download() -> None:
    print(f" Mengunduh {ZIP_URL}")
    with requests.get(ZIP_URL, headers=HEADERS, timeout=120, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        tmp = ZIP_PATH.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done // total
                    print(f"\r   {done/1e6:7.1f}/{total/1e6:.1f} MB ({pct}%)",
                          end="", flush=True)
        print()
        tmp.replace(ZIP_PATH)
    print(f" Tersimpan: {ZIP_PATH}")


def extract() -> None:
    if (DATA_DIR / "Indonesian Food Image").is_dir():
        print("Sudah terekstrak sebelumnya — lewati.")
    else:
        print(f" Ekstrak ke {DATA_DIR}")
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(DATA_DIR)
    verify()


def verify() -> None:
    root = find_split_root()
    print(f" Root split: {root}")
    ok = True
    for split in ("train", "test"):
        d = root / split
        classes = sorted(p.name for p in d.iterdir() if p.is_dir())
        n_img = sum(
            1 for c in d.iterdir() if c.is_dir()
            for f in c.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        missing = EXPECTED_CLASSES - set(classes)
        status = "OK" if not missing else f"KURANG: {sorted(missing)}"
        print(f"   {split:<5} kelas={len(classes)} gambar={n_img:>5}  [{status}]")
        ok &= not missing
    if not ok:
        sys.exit("Struktur dataset tidak sesuai harapan.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="unduh ulang meski dataset sudah ada")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.force or not ZIP_PATH.exists():
        download()
    else:
        print(f"Zip sudah ada: {ZIP_PATH}")

    if already_extracted() and not args.force:
        print("Folder test/ sudah terisi — lewati ekstraksi.")
        verify()
    else:
        extract()

    print("Selesai.")


if __name__ == "__main__":
    main()
