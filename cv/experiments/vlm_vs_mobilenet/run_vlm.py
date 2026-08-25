#!/usr/bin/env python3
"""Run VLM (Gemini) classification on the held-out test split.

This script provides a provider-agnostic scaffold. It currently supports two modes:
 - --dry-run : validate dataset and write a stub JSONL with no external API calls
 - --simulate-from-mobile : copy MobileNet outputs to VLM outputs for pipeline testing

To run real VLM inference you must provide credentials and implement the provider
client in `call_vlm_for_image()` below (openai/google/vertex). See README.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = ROOT / "data" / "cv" / "indonesian_food_image"
MOBILE_FEATURES = ROOT / "cv" / "features" / "cv" / "cv-baseline-v0.1" / "features.jsonl"
OUT_DIR = ROOT / "cv" / "features" / "cv" / "vlm_v0.1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "features_vlm.jsonl"


def call_vlm_for_image_stub(image_path, classes):
    """Provider implementation placeholder.
    Replace this with a real call to Gemini / Vertex / OpenAI vision.
    Should return (predicted_class, confidence, optional_macros_dict)
    """
    return (None, 0.0, None)


def find_test_root():
    # find folder that contains train/ and test/
    candidates = [TEST_DIR, *sorted(TEST_DIR.glob("*/Clean_Data")), *sorted(TEST_DIR.glob("*/"))]
    for c in candidates:
        if (c / "train").is_dir() and (c / "test").is_dir():
            return c
    raise FileNotFoundError("Test split not found; run cv/experiments/vlm_vs_mobilenet/fetch_dataset.py first")


def gather_test_images(test_root):
    test_dir = test_root / "test"
    records = []
    for cls in sorted([p for p in test_dir.iterdir() if p.is_dir()] ):
        for img in sorted(cls.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            sample_id = img.stem  # matches features' sample_id format
            records.append((sample_id, str(img), cls.name))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Validate dataset and write stub outputs")
    ap.add_argument("--simulate-from-mobile", action="store_true", help="Copy MobileNet outputs to VLM outputs (useful for pipeline smoke test)")
    args = ap.parse_args()

    test_root = find_test_root()
    print("Test root:", test_root)
    imgs = gather_test_images(test_root)
    print(f"Found {len(imgs)} test images")

    if args.simulate_from_mobile:
        # copy mobile outputs to vlm outputs
        if not MOBILE_FEATURES.exists():
            print("MobileNet features not found:", MOBILE_FEATURES, file=sys.stderr)
            sys.exit(1)
        with open(MOBILE_FEATURES, encoding='utf-8') as inf, open(OUT_FILE, 'w', encoding='utf-8') as outf:
            for line in inf:
                outf.write(line)
        print("Wrote simulated VLM features to", OUT_FILE)
        return

    if args.dry_run:
        # write stub records with sample_id, no predictions
        now = datetime.utcnow().isoformat() + 'Z'
        with open(OUT_FILE, 'w', encoding='utf-8') as outf:
            for sample_id, img_path, cls in imgs:
                rec = {
                    "sample_id": sample_id,
                    "modality": "vision",
                    "food_top1": None,
                    "food_top3": [],
                    "carbs_g": None,
                    "protein_g": None,
                    "fat_g": None,
                    "fiber_g": None,
                    "calories_kcal": None,
                    "confidence": None,
                    "feature_status": "vlm_stub",
                    "model_version": "vlm_v0.1_stub",
                    "created_at": now,
                }
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("Wrote VLM stub outputs to", OUT_FILE)
        return

    # Real-run path: not implemented
    print("Real VLM run is not implemented in this script. Implement `call_vlm_for_image()` with your provider client and call the API per image.")
    print("See cv/experiments/vlm_vs_mobilenet/README.md for instructions and credentials.")


if __name__ == '__main__':
    main()
