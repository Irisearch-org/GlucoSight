#!/usr/bin/env python3
"""Live VLM dish-identification call — the open-vocabulary list variant.

Unlike run_gas_vlm.py (closed 10-class classification, one label), this
asks Gemini to list every distinct dish it sees, unconstrained, then hands
the list to integration.adapters.cv_items_to_contract + the shared matcher
(integration/dish_table.py) to resolve each name against the dish table.
This is the live call docs/SPEC_DISH_IDENTIFICATION.md's Ticket 2 left
stubbed (the spec's own Testing Decisions say the VLM call itself isn't
unit-tested — this script is how it's exercised instead).

Reuses run_gas_vlm.py's proven REST-call pattern (same auth, same
endpoint, same model) — only the prompt and response schema differ.
"""
import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from integration import dish_table  # noqa: E402
from integration.adapters import cv_items_to_contract  # noqa: E402

ENV_FILE = Path(__file__).resolve().parent / "GEMINI_API_KEY.env"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
MODEL = "gemini-3.5-flash-lite"
# models/text-embedding-004 (the commonly-documented name) 404s on this API
# version/key; confirmed live via ListModels that gemini-embedding-001 is
# what this key actually supports for embedContent.
EMBEDDING_MODEL = "gemini-embedding-001"
API_INTERVAL_SECONDS = 12.0
MAX_API_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 90

PROMPT = (
    "List every distinct food dish visible in this image, as specifically "
    "as you can (e.g. 'rendang', 'nasi goreng', not just 'rice dish'). "
    "Use the dish's common Indonesian name if you recognise it, otherwise "
    "whatever name best identifies it. "
    'Return JSON only: {"items": [{"food_name": "...", "confidence": 0.0}]}. '
    "One entry per distinct dish — don't split one dish into its ingredients."
)


def call_gemini_list_dishes(api_key: str, image_path: Path) -> list:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("ascii")

    url = BASE_URL + MODEL + ":generateContent"
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
        ]}],
        "generationConfig": {
            "maxOutputTokens": 300,
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    for attempt in range(MAX_API_RETRIES + 1):
        resp = requests.post(
            url, params={"key": api_key}, headers={"Content-Type": "application/json"},
            json=body, timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if resp.ok:
            break
        if resp.status_code != 429 or attempt == MAX_API_RETRIES:
            raise RuntimeError(f"Gemini API HTTP {resp.status_code}: {resp.text[:500]}")
        retry_match = re.search(r'"retryDelay"\s*:\s*"(\d+)s"', resp.text)
        wait_seconds = max(float(retry_match.group(1)) if retry_match else 2 ** attempt,
                            API_INTERVAL_SECONDS)
        print(f"Gemini rate limit; waiting {wait_seconds:.0f}s")
        time.sleep(wait_seconds)

    time.sleep(API_INTERVAL_SECONDS)
    txt = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    try:
        parsed = json.loads(txt)
        return parsed["items"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Gemini returned invalid JSON: {txt!r}") from exc


def make_gemini_embedder(api_key: str) -> "dish_table.Embedder":
    """dish_table.Embedder — (text, is_query) -> vector, via the Gemini
    embedding API.

    Asymmetric: a query (the detected item being matched) is embedded with
    RETRIEVAL_QUERY, a dish-table candidate with RETRIEVAL_DOCUMENT —
    measured live to separate true/false matches better than one symmetric
    call (dish_table.EMBEDDING_MATCH_THRESHOLD's measurement note).

    Caches within a run (the dish table's own 10 candidate strings repeat
    across every match_dish_by_embedding call); does not persist across
    runs. One API call per *new* (text, is_query) pair, not per match.
    """
    cache: dict = {}
    url = BASE_URL + EMBEDDING_MODEL + ":embedContent"

    def embed(text: str, is_query: bool):
        key = (text, is_query)
        if key in cache:
            return cache[key]
        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        resp = requests.post(
            url, params={"key": api_key}, headers={"Content-Type": "application/json"},
            json={"content": {"parts": [{"text": text}]}, "taskType": task_type},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if not resp.ok:
            raise RuntimeError(f"Gemini embedding HTTP {resp.status_code}: {resp.text[:300]}")
        vector = resp.json()["embedding"]["values"]
        cache[key] = vector
        return vector

    return embed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path,
                         help="photo paths; a bare filename stem prefix before "
                              "'_' is read as the expected food_class for scoring")
    args = parser.parse_args()

    load_dotenv(ENV_FILE)
    import os
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY not found in {ENV_FILE}")

    embedder = make_gemini_embedder(api_key)

    n_correct, n_scored = 0, 0
    for path in args.images:
        expected = path.stem.rsplit("_", 1)[0] if "_" in path.stem else None
        items = call_gemini_list_dishes(api_key, path)
        block = cv_items_to_contract(items, meal_id=path.stem, embedder=embedder)
        matched_class = block["food_class"]
        verdict = ""
        if expected and expected != "pic":
            n_scored += 1
            correct = matched_class == expected or expected in matched_class
            n_correct += int(correct)
            verdict = "  <-- correct" if correct else f"  <-- expected {expected}"
        print(f"{path.name}: VLM said {items} -> matched {matched_class!r} "
              f"(carbs_g={block['carbs_g']}, cv_confidence={block['cv_confidence']}, "
              f"carbs_source={block['carbs_source']}){verdict}")

    if n_scored:
        print(f"\n{n_correct}/{n_scored} labeled images matched the expected food_class")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
