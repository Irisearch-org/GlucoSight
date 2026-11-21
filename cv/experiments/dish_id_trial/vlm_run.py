#!/usr/bin/env python3
"""Open-vocabulary Gemini dish-list call over the manifest, with a full-path cache.

Each image is sent once with the same PROMPT as
cv/experiments/vlm_vs_mobilenet/run_dish_list_gas_vlm.py. The raw response is
cached as JSON keyed by the image's FULL PATH (sha1). Cached images cost no
API call. Every call (including failures) counts against --budget.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

from common import CACHE_VLM, MANIFEST, load_manifest

ENV_FILE = Path(os.environ.get(
    "GEMINI_ENV_FILE",
    "/Users/auliamnaufal/Downloads/glucosight/cv/experiments/vlm_vs_mobilenet/GEMINI_API_KEY.env",
))
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
MODEL = "gemini-3.5-flash-lite"
MAX_SIDE = 1024
MAX_RETRIES = 6
REQUEST_TIMEOUT = 120

PROMPT = (
    "List every distinct food dish visible in this image, as specifically "
    "as you can (e.g. 'rendang', 'nasi goreng', not just 'rice dish'). "
    "Use the dish's common Indonesian name if you recognise it, otherwise "
    "whatever name best identifies it. "
    'Return JSON only: {"items": [{"food_name": "...", "confidence": 0.0}]}. '
    "One entry per distinct dish — don't split one dish into its ingredients."
)


def cache_file(image_path: str) -> Path:
    return CACHE_VLM / (hashlib.sha1(image_path.encode("utf-8")).hexdigest() + ".json")


def encode_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_model(api_key: str, b64: str) -> str:
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
        ]}],
        "generationConfig": {
            "maxOutputTokens": 300,
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    url = BASE_URL + MODEL + ":generateContent"
    for attempt in range(MAX_RETRIES + 1):
        resp = requests.post(url, params={"key": api_key},
                             headers={"Content-Type": "application/json"},
                             json=body, timeout=REQUEST_TIMEOUT)
        if resp.ok:
            break
        if resp.status_code not in (429, 500, 503) or attempt == MAX_RETRIES:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        time.sleep(min(2 ** attempt * 2, 60))
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def process(api_key: str, image_path: str, label: str, image_index: str,
            budget: dict, lock: threading.Lock) -> str:
    cf = cache_file(image_path)
    if cf.exists():
        return "cached"
    with lock:
        if budget["used"] >= budget["max"]:
            return "budget_exhausted"
        budget["used"] += 1

    raw, items, status = None, None, "ok"
    try:
        raw = call_model(api_key, encode_image(Path(image_path)))
        try:
            items = json.loads(raw)["items"]
        except (KeyError, TypeError, ValueError):
            status = "parse_error"
    except (RuntimeError, OSError, requests.RequestException) as exc:
        raw, status = str(exc), "api_error"

    if status == "api_error":
        return status

    record = {
        "image_path": image_path,
        "image_index": image_index,
        "label": label,
        "model": MODEL,
        "status": status,
        "raw": raw,
        "items": items,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    CACHE_VLM.mkdir(parents=True, exist_ok=True)
    tmp = cf.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False))
    tmp.replace(cf)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--budget", type=int, default=3000,
                        help="max API calls this run (cache hits are free)")
    args = parser.parse_args()

    load_dotenv(ENV_FILE)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(f"GEMINI_API_KEY not found in {ENV_FILE}")

    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]

    budget = {"used": 0, "max": args.budget}
    lock = threading.Lock()
    counts: dict = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, api_key, p, lab, idx, budget, lock)
                   for p, lab, idx in rows]
        for i, fut in enumerate(futures, 1):
            status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if i % 100 == 0:
                print(f"{i}/{len(rows)} {counts}", flush=True)

    print(f"done {len(rows)} rows: {counts}; api_calls_this_run={budget['used']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
