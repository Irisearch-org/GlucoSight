#!/usr/bin/env python3
"""Score the dish-ID arms on the labelled Wibisono photos.

Arms (every prediction keyed by FULL image path):
  mobilenet    MobileNetV2 food-10, raw top-1 (closed set, no threshold)
  vlm_asis     Gemini dish list -> difflib match against the shipped 10-row table
  vlm_difflib  Gemini dish list -> difflib match, table extended by table_aliases.csv
  vlm_embed    Gemini dish list -> embedding match, same extended table

Truth: class_map.csv maps each Wibisono class to a table food_class, or to
nothing. A prediction is correct when it equals that truth; an image whose
class has no table row must get no prediction (a None result).
"""
import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cv"))
sys.path.insert(0, str(HERE))

from dotenv import load_dotenv  # noqa: E402

from integration import dish_table  # noqa: E402
from common import DATA, MANIFEST, load_manifest  # noqa: E402
from vlm_run import ENV_FILE, cache_file  # noqa: E402

MOBILENET_CKPT = Path(os.environ.get(
    "MOBILENET_CKPT", "/Users/auliamnaufal/Downloads/glucosight/cv/models/mobilenetv2_food10.pt"))
TABLE = dish_table.DEFAULT_DISH_TABLE_PATH


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_class_truth():
    return {r["wibisono_class"]: (r["table_food_class"] or None)
            for r in read_csv(HERE / "class_map.csv")}


def load_extended_rows(base_rows, aliases):
    """Table rows with one candidate per alias; food_class is unchanged."""
    extra = {}
    for a in aliases:
        extra.setdefault(a["food_class"], []).append(a["alias"])
    rows = []
    for base in base_rows:
        rows.append(base)
        for alias in extra.get(base.food_class, []):
            rows.append(dataclasses.replace(base, tkpi_item_name=alias))
    return rows


def load_embedder():
    sys.path.insert(0, str(ROOT / "cv" / "experiments" / "vlm_vs_mobilenet"))
    from run_dish_list_gas_vlm import make_gemini_embedder  # noqa: E402
    load_dotenv(ENV_FILE)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(f"GEMINI_API_KEY not found in {ENV_FILE}")
    return make_gemini_embedder(api_key)


def ranked_items(record):
    items = record.get("items") or []
    out = []
    for it in items:
        name = str(it.get("food_name", "")).strip()
        try:
            conf = float(it.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if name:
            out.append((name, conf))
    return sorted(out, key=lambda t: -t[1])


def vlm_prediction(names, match_one):
    """Highest-confidence item that matches a table row decides the prediction."""
    for name in names:
        m = match_one(name)
        if m is not None:
            return m.row.food_class
    return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def paired_bootstrap_ci(a, b, n_boot=10000, seed=0):
    """95% percentile CI for mean(a - b) over images, resampling images."""
    rng = random.Random(seed)
    N = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(N) for _ in range(N)]
        diffs.append(sum(a[i] - b[i] for i in idx) / N)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def arm_metrics(preds, truths):
    """preds/truths: dicts keyed by full path. Returns summary for one arm."""
    paths = list(truths)
    in_overlap = [p for p in paths if truths[p] is not None]
    out_overlap = [p for p in paths if truths[p] is None]

    ov_correct = [1 if preds[p] == truths[p] else 0 for p in in_overlap]
    k = sum(ov_correct)
    n = len(in_overlap)
    lo, hi = wilson(k, n)

    all_correct = sum(1 for p in paths if preds[p] == truths[p])
    false_match = sum(1 for p in out_overlap if preds[p] is not None)
    covered = sum(1 for p in paths if preds[p] is not None)
    return {
        "overlap_n": n,
        "overlap_correct": k,
        "overlap_accuracy": round(k / n, 4) if n else None,
        "overlap_wilson95": [round(lo, 4), round(hi, 4)] if n else None,
        "all_n": len(paths),
        "all_correct": all_correct,
        "all_accuracy": round(all_correct / len(paths), 4),
        "coverage_any_prediction": round(covered / len(paths), 4),
        "false_match_on_uncovered": false_match,
        "uncovered_n": len(out_overlap),
    }, ov_correct


def mobilenet_predictions(manifest):
    from cv_baseline.classifier import MobileNetV2Classifier  # noqa: E402
    from cv_baseline.preprocessing import preprocess  # noqa: E402
    clf = MobileNetV2Classifier(MOBILENET_CKPT).load()
    out = {}
    for path, _, _ in manifest:
        name, prob = clf.predict_topk(preprocess(path), k=1)[0]
        out[path] = (name, prob)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--per-image", type=Path, default=DATA / "results" / "per_image.csv")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    truth_by_class = load_class_truth()
    truths = {p: truth_by_class.get(label) for p, label, _ in manifest}
    unknown = sorted({label for _, label, _ in manifest if label not in truth_by_class})
    if unknown:
        raise SystemExit(f"classes missing from class_map.csv: {unknown}")

    records = {}
    missing = []
    for p, _, _ in manifest:
        cf = cache_file(p)
        if not cf.exists():
            missing.append(p)
            continue
        records[p] = json.loads(cf.read_text())
    if missing:
        raise SystemExit(f"{len(missing)} images have no VLM cache; run vlm_run.py first")

    base_rows = dish_table.load_dish_table(TABLE)
    aliases = read_csv(HERE / "table_aliases.csv")
    ext_rows = load_extended_rows(base_rows, aliases)
    embedder = load_embedder()
    index = dish_table.build_dish_table_index(ext_rows, embedder)

    cache_match: dict = {}

    def difflib_asis(name):
        key = ("asis", name)
        if key not in cache_match:
            cache_match[key] = dish_table.match_dish(name, base_rows)
        return cache_match[key]

    def difflib_ext(name):
        key = ("ext", name)
        if key not in cache_match:
            cache_match[key] = dish_table.match_dish(name, ext_rows)
        return cache_match[key]

    def embed_ext(name):
        key = ("embed", name)
        if key not in cache_match:
            cache_match[key] = dish_table.match_dish_by_embedding(name, index, embedder)
        return cache_match[key]

    mn = mobilenet_predictions(manifest)

    preds = {"mobilenet": {}, "vlm_asis": {}, "vlm_difflib": {}, "vlm_embed": {}}
    per_image_rows = []
    for p, label, idx in manifest:
        names = [n for n, _ in ranked_items(records[p])]
        preds["vlm_asis"][p] = vlm_prediction(names, difflib_asis)
        preds["vlm_difflib"][p] = vlm_prediction(names, difflib_ext)
        preds["vlm_embed"][p] = vlm_prediction(names, embed_ext)
        preds["mobilenet"][p] = mn[p][0]
        per_image_rows.append({
            "image_path": p,
            "image_index": idx,
            "label": label,
            "truth": truths[p] or "",
            "mobilenet_pred": mn[p][0],
            "mobilenet_prob": round(mn[p][1], 4),
            "vlm_items": json.dumps(ranked_items(records[p]), ensure_ascii=False),
            "pred_vlm_asis": preds["vlm_asis"][p] or "",
            "pred_vlm_difflib": preds["vlm_difflib"][p] or "",
            "pred_vlm_embed": preds["vlm_embed"][p] or "",
        })

    summary = {"n_images": len(manifest),
               "n_overlap": sum(1 for t in truths.values() if t is not None),
               "arms": {}, "bootstrap_diff_vs_mobilenet": {}}
    overlap_correct = {}
    for arm in ("mobilenet", "vlm_asis", "vlm_difflib", "vlm_embed"):
        metrics, ov = arm_metrics(preds[arm], truths)
        summary["arms"][arm] = metrics
        overlap_correct[arm] = ov
    for arm in ("vlm_asis", "vlm_difflib", "vlm_embed"):
        lo, hi = paired_bootstrap_ci(overlap_correct[arm], overlap_correct["mobilenet"])
        mean = (sum(overlap_correct[arm]) - sum(overlap_correct["mobilenet"])) / len(overlap_correct[arm])
        summary["bootstrap_diff_vs_mobilenet"][arm] = {
            "diff_accuracy_overlap": round(mean, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        }

    summary["table_sha256"] = hashlib.sha256(TABLE.read_bytes()).hexdigest()[:16]
    summary["table_rows_extended"] = len(ext_rows)
    summary["mobilenet_ckpt"] = MOBILENET_CKPT.name

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    args.per_image.parent.mkdir(parents=True, exist_ok=True)
    with open(args.per_image, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0]))
        w.writeheader()
        w.writerows(per_image_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
