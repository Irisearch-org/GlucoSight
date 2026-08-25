#!/usr/bin/env python3
"""Run Gemini VLM classification on the held-out test split."""
import argparse
import base64
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = ROOT / "data" / "cv" / "indonesian_food_image"
OUT_DIR = ROOT / "cv" / "features" / "cv" / "vlm_v0.1_gas"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "features_vlm_gas.jsonl"
MACRO_OUT_FILE = OUT_DIR / "features_vlm_gas_macros.jsonl"
REPORT_FILE = ROOT / "cv" / "reports" / "vlm_vs_mobilenet.md"
ENV_FILE = Path(__file__).resolve().parent / "GEMINI_API_KEY.env"

# Google AI Studio native REST endpoint
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
MODEL = "gemini-3.5-flash-lite"
API_INTERVAL_SECONDS = 12.0
MAX_API_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 90


def find_test_root():
    candidates = [TEST_DIR, *sorted(TEST_DIR.glob("*/Clean_Data")), *sorted(TEST_DIR.glob("*/"))]
    for c in candidates:
        if (c / "train").is_dir() and (c / "test").is_dir():
            return c
    raise FileNotFoundError("Test split not found; run cv/experiments/vlm_vs_mobilenet/fetch_dataset.py first")


def gather_test_images(test_root, sample_size: Optional[int] = None):
    test_dir = test_root / "test"
    imgs = []
    for cls in sorted([p for p in test_dir.iterdir() if p.is_dir()] ):
        for img in sorted(cls.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            sample_id = img.stem
            imgs.append((sample_id, str(img), cls.name))
    if sample_size:
        random.seed(42)
        imgs = random.sample(imgs, min(sample_size, len(imgs)))
    return imgs


def call_gas_image_classify(api_key: str, image_path: str, classes: list, include_macros=False):
    """Call the Gemini native image endpoint.

    Use Google's OpenAI-compatible chat completions endpoint with an image
    data URL and request JSON-only output.
    """
    # Load image and base64-encode
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('ascii')

    if include_macros:
        prompt = (
            "Classify this Indonesian food image. Choose exactly one label from: "
            + ", ".join(classes)
            + '. Estimate the visible meal\'s standard serving macros. Return exactly one JSON object with these numeric fields: '
            '{"label":"class", "confidence":0.0, "carbs_g":0.0, "protein_g":0.0, "fat_g":0.0, "fiber_g":0.0}. '
            "All macro values must be numbers in grams, not strings, and must not be null. "
            "Use 0 when a value is negligible. Do not include markdown or extra keys."
        )
    else:
        prompt = (
            "Classify this Indonesian food image. Choose exactly one label from: "
            + ", ".join(classes)
            + '. Return JSON only: {"label":"...", "confidence":0.0}.'
        )

    url = BASE_URL + MODEL + ":generateContent"
    headers = {
        "Content-Type": "application/json",
    }
    body = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
        ]}],
        "generationConfig": {
            "maxOutputTokens": 200,
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    for attempt in range(MAX_API_RETRIES + 1):
        try:
            resp = requests.post(url, params={"key": api_key}, headers=headers, json=body, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            if attempt == MAX_API_RETRIES:
                raise RuntimeError(f"Gemini request failed after retries: {type(exc).__name__}") from exc
            wait_seconds = max(2 ** attempt, API_INTERVAL_SECONDS)
            print(f"Transient Gemini request failure; waiting {wait_seconds:.0f}s")
            time.sleep(wait_seconds)
            continue
        if resp.ok:
            break
        if resp.status_code != 429 or attempt == MAX_API_RETRIES:
            raise RuntimeError(f"Gemini API HTTP {resp.status_code}: {resp.text[:500]}")

        retry_after = resp.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            wait_seconds = float(retry_after)
        else:
            retry_match = re.search(r'"retryDelay"\s*:\s*"(\d+)s"', resp.text)
            wait_seconds = float(retry_match.group(1)) if retry_match else 2 ** attempt
        wait_seconds = max(wait_seconds, API_INTERVAL_SECONDS)
        print(f"Gemini rate limit for attempt {attempt + 1}; waiting {wait_seconds:.0f}s")
        time.sleep(wait_seconds)

    time.sleep(API_INTERVAL_SECONDS)
    data = resp.json()
    txt = data["candidates"][0]["content"]["parts"][0]["text"]
    # attempt to load json from txt
    try:
        parsed = json.loads(txt)
        label = parsed.get('label')
        conf = float(parsed.get('confidence', 0.0))
        if not include_macros:
            return label, conf, None
        macros = {field: float(parsed[field]) for field in ('carbs_g', 'protein_g', 'fat_g', 'fiber_g')}
        return label, conf, macros
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Gemini returned invalid JSON: {txt!r}") from exc


def load_existing_real_results():
    if not OUT_FILE.exists():
        return {}
    results = {}
    with open(OUT_FILE, encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                record.get("model_version") == f"gas:{MODEL}"
                and record.get("feature_status") == "ok"
            ):
                results[record["sample_id"]] = record
    return results


def load_existing_macro_results():
    if not MACRO_OUT_FILE.exists():
        return {}
    results = {}
    with open(MACRO_OUT_FILE, encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                record.get("model_version") == f"gas:{MODEL}:direct-macros"
                and record.get("feature_status") == "ok"
                and all(record.get(field) is not None for field in ("carbs_g", "protein_g", "fat_g", "fiber_g"))
            ):
                results[record["sample_id"]] = record
    return results


def write_results(results):
    temporary_file = OUT_FILE.with_suffix(".jsonl.tmp")
    with open(temporary_file, "w", encoding="utf-8") as outfile:
        for record in results:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_file.replace(OUT_FILE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=100, help="Number of test images to sample")
    ap.add_argument("--simulate-from-mobile", action='store_true', help="Copy MobileNet outputs to VLM outputs instead of calling API")
    ap.add_argument("--direct-macros", action='store_true', help="Fresh run requesting numeric macro estimates")
    args = ap.parse_args()

    load_dotenv(ENV_FILE, override=False)
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY not found in {ENV_FILE}")

    test_root = find_test_root()
    imgs = gather_test_images(test_root, sample_size=args.sample_size)
    if args.direct_macros:
        imgs = [(sid, path, cls) for sid, path, cls in imgs if sid != "pic_338"]
    print(f"Using {len(imgs)} images for VLM run")

    classes = [p.name for p in sorted((test_root / 'test').iterdir()) if p.is_dir()]

    if args.simulate_from_mobile:
        print("Simulation mode: copying MobileNet outputs")
        # fallback: copy mobile features with matching sample_ids
        mobile_features = ROOT / 'cv' / 'features' / 'cv' / 'cv-baseline-v0.1' / 'features.jsonl'
        if not mobile_features.exists():
            print("MobileNet features not found; aborting.")
            return
        mobile_map = {json.loads(line)['sample_id']: json.loads(line) for line in open(mobile_features, encoding='utf-8')}
        with open(OUT_FILE, 'w', encoding='utf-8') as outf:
            for sid, path, cls in imgs:
                rec = mobile_map.get(sid)
                if rec:
                    outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print("Wrote simulated VLM outputs to", OUT_FILE)
        return

    output_file = MACRO_OUT_FILE if args.direct_macros else OUT_FILE
    existing_results = load_existing_macro_results() if args.direct_macros else load_existing_real_results()
    results = [existing_results[sid] for sid, _, _ in imgs if sid in existing_results]
    pending_imgs = [(sid, path, cls) for sid, path, cls in imgs if sid not in existing_results]
    if args.direct_macros:
        print(f"Resuming with {len(results)} existing direct-macro results; {len(pending_imgs)} API calls remaining")
    else:
        print(f"Resuming with {len(results)} existing real results; {len(pending_imgs)} API calls remaining")
    failures = []
    for i, (sid, img_path, cls) in enumerate(pending_imgs, start=1):
        try:
            label, conf, macros = call_gas_image_classify(api_key, img_path, classes, args.direct_macros)
        except Exception as exc:
            error_message = str(exc).replace(api_key, "[REDACTED]")
            print("API call failed for", sid, error_message)
            failures.append((sid, error_message))
            continue
        rec = {
            "sample_id": sid,
            "modality": "vision",
            "food_top1": label,
            "food_top3": [label] if label else [],
            "carbs_g": macros["carbs_g"] if macros else None,
            "protein_g": macros["protein_g"] if macros else None,
            "fat_g": macros["fat_g"] if macros else None,
            "fiber_g": macros["fiber_g"] if macros else None,
            "calories_kcal": None,
            "confidence": round(float(conf or 0.0), 4) if conf is not None else None,
            "feature_status": "ok" if conf and conf >= 0.0 else "low_confidence",
            "model_version": f"gas:{MODEL}:direct-macros" if args.direct_macros else f"gas:{MODEL}",
            "created_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        results.append(rec)
        if args.direct_macros:
            temporary_file = output_file.with_suffix(".jsonl.tmp")
            with open(temporary_file, "w", encoding="utf-8") as outfile:
                for record in results:
                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary_file.replace(output_file)
        else:
            write_results(results)

    if failures:
        raise RuntimeError(f"{len(failures)} of {len(pending_imgs)} remaining API calls failed; progress was checkpointed")
    print("Wrote VLM outputs to", output_file)


if __name__ == '__main__':
    main()
