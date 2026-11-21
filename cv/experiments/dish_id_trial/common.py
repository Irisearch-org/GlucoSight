"""Shared paths, label loading and manifest building for the dish-ID trial.

Images are identified by FULL PATH everywhere. Filename stems collide
across classes in the Wibisono set, so a stem must never be used as a key.
"""
import csv
import os
from pathlib import Path

DATA = Path(os.environ.get("DISH_TRIAL_DATA", "/Users/auliamnaufal/Downloads/glucosight-cv-data"))
WIB_ROOT = DATA / "wibisono"
DEV_CSV = DATA / "wibisono_meta" / "dev.csv"
MANIFEST = DATA / "manifest.csv"
CACHE_VLM = DATA / "cache" / "vlm"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _norm(s: str) -> str:
    return s.strip().lower().replace("_", "-").replace(" ", "-")


def load_labels(csv_path: Path = DEV_CSV):
    """Return [(image_index, class_label)] from the one-hot label CSV."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header: Image Index, Patient ID, Finding Labels, <classes...>
        return [(r[0].strip(), r[2].strip()) for r in reader if r and r[0].strip()]


def build_manifest(root: Path = WIB_ROOT, out: Path = MANIFEST, labels_csv: Path = DEV_CSV):
    """Resolve each labelled Image Index to a unique full path under root."""
    by_name: dict = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            by_name.setdefault(p.name, []).append(p)
            by_name.setdefault(p.stem, []).append(p)

    rows, unresolved, ambiguous = [], [], []
    for index, label in load_labels(labels_csv):
        cands = by_name.get(index) or by_name.get(Path(index).stem) or []
        if len(cands) > 1:
            same_dir = [p for p in cands if _norm(p.parent.name) == _norm(label)]
            cands = same_dir if len(same_dir) == 1 else cands
        if not cands:
            unresolved.append(index)
        elif len(cands) > 1:
            ambiguous.append(index)
        else:
            rows.append((str(cands[0].resolve()), label, index))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label", "image_index"])
        w.writerows(rows)
    return rows, unresolved, ambiguous


def load_manifest(path: Path = MANIFEST):
    with open(path, newline="") as f:
        return [(r["image_path"], r["label"], r["image_index"]) for r in csv.DictReader(f)]


if __name__ == "__main__":
    rows, unresolved, ambiguous = build_manifest()
    print(f"resolved={len(rows)} unresolved={len(unresolved)} ambiguous={len(ambiguous)}")
    for idx in unresolved[:10]:
        print("unresolved:", idx)
    for idx in ambiguous[:10]:
        print("ambiguous:", idx)
