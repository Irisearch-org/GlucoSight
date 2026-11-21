#!/usr/bin/env python3
"""Batch runner: kumpulan gambar meal -> feature file JSONL tervalidasi.

Contoh pakai:

    # dari folder gambar (sample_id diambil dari nama file)
    python scripts/run_batch.py --image-dir samples/

    # dari manifest JSON berisi list payload
    python scripts/run_batch.py --manifest samples/manifest.json

    # uji pipeline tanpa PyTorch (output ditandai feature_status="mock")
    python scripts/run_batch.py --image-dir samples/ --mock

Output ditulis ke features/cv/cv-baseline-v0.1/ sebagai .jsonl (satu record per
baris, gampang di-join by sample_id) plus ringkasan run.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cv_baseline.config import FEATURE_OUTPUT_DIR, MODEL_VERSION  # noqa: E402
from cv_baseline.predict import predict, set_classifier  # noqa: E402
from cv_baseline.schema import validate_output  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Dipakai kalau script dijalankan tanpa argumen sama sekali (mis. dobel-klik
# file, atau "Run" dari editor tanpa terminal) — supaya tetap bisa jalan.
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"


def payloads_from_dir(image_dir: Path):
    files = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return [
        {"sample_id": p.stem, "meal_image_path": str(p)}
        for p in files
    ]


def payloads_from_manifest(manifest_path: Path):
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Manifest harus berupa list of payload objects.")
    return data


def main():
    ap = argparse.ArgumentParser(description="Batch inference cv-baseline-v0.1")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--image-dir", type=Path, help="folder berisi gambar meal")
    src.add_argument("--manifest", type=Path, help="file JSON berisi list payload")
    ap.add_argument("--output-dir", type=Path, default=FEATURE_OUTPUT_DIR)
    ap.add_argument("--output-name", default="features.jsonl")
    ap.add_argument("--mock", action="store_true",
                    help="pakai MockClassifier (uji kontrak tanpa PyTorch)")
    args = ap.parse_args()

    # Tidak ada argumen sama sekali (dobel-klik / run tanpa terminal) ->
    # coba fallback ke folder default sebelum menyerah.
    if not args.image_dir and not args.manifest:
        if DEFAULT_IMAGE_DIR.is_dir():
            print(f"[INFO] Tidak ada argumen diberikan, pakai folder default:\n"
                  f"       {DEFAULT_IMAGE_DIR}\n")
            args.image_dir = DEFAULT_IMAGE_DIR
        else:
            sys.exit(
                "Tidak ada --image-dir atau --manifest, dan folder default\n"
                f"  {DEFAULT_IMAGE_DIR}\n"
                "tidak ditemukan. Taruh gambar meal di folder itu, atau jalankan\n"
                "lewat terminal dengan argumen eksplisit, contoh:\n"
                "  python scripts/run_batch.py --image-dir data/raw/images"
            )

    if args.mock:
        from cv_baseline.classifier import MockClassifier
        from cv_baseline.macro_lookup import MacroLookup
        classes = sorted(MacroLookup().table)
        set_classifier(MockClassifier({c: i for i, c in enumerate(classes)}))
        print("[MODE MOCK] Output TIDAK valid untuk downstream — hanya uji kontrak.\n")

    if args.image_dir:
        if not args.image_dir.is_dir():
            sys.exit(f"Folder tidak ditemukan: {args.image_dir}")
        payloads = payloads_from_dir(args.image_dir)
    else:
        payloads = payloads_from_manifest(args.manifest)

    if not payloads:
        sys.exit("Tidak ada gambar/payload untuk diproses.")

    print(f"Memproses {len(payloads)} sample...\n")

    records, invalid = [], []
    status_count = {}

    for payload in payloads:
        record = predict(payload)
        errors = validate_output(record)

        if errors:
            invalid.append({"sample_id": record.get("sample_id"), "errors": errors})

        status = record["feature_status"]
        status_count[status] = status_count.get(status, 0) + 1
        records.append(record)

        flag = "FAIL" if errors else "ok  "
        conf = record.get("confidence")
        conf_s = f"{conf:.3f}" if isinstance(conf, float) else "  -  "
        print(f"  [{flag}] {record.get('sample_id'):<24} "
              f"{str(record.get('food_top1')):<14} conf={conf_s}  {status}")
        if errors:
            for e in errors:
                print(f"         -> {e}")
        if record.get("error_reason"):
            print(f"         -> {record['error_reason']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / args.output_name
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "model_version": MODEL_VERSION,
        "total_samples": len(records),
        "schema_valid": len(records) - len(invalid),
        "schema_invalid": len(invalid),
        "feature_status_counts": status_count,
        "invalid_records": invalid,
        "output_file": str(out_path),
    }
    summary_path = args.output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*58}")
    print(f"Total          : {len(records)}")
    print(f"Schema valid   : {len(records) - len(invalid)}")
    print(f"Schema invalid : {len(invalid)}")
    print(f"Status         : {status_count}")
    print(f"Feature file   : {out_path}")
    print(f"Run summary    : {summary_path}")

    sys.exit(1 if invalid else 0)


if __name__ == "__main__":
    main()
