#!/usr/bin/env python3
"""FROZEN EVIDENCE (branch wayfinder/dish-id-trial): ran against the pre-v2 integration/dish_table.py and
vlm_vs_mobilenet/run_dish_list_gas_vlm.py, both since removed. The v2 matcher is integration/dishes.

Score the dish-ID arms on one labelled split (dev or test) of the Wibisono photos.

Every prediction is keyed by FULL image path. Truth is the Wibisono class label;
every class has a row in DATA/dish_table_35.csv, so the VLM arms are scored on
the whole split.

Arms:
  difflib  Gemini dish list -> difflib matcher (integration.dish_table, threshold 0.6)
  embed    Gemini dish list -> embedding matcher (cosine, threshold 0.75)
  mobilenet MobileNetV2 food-10 top-1. It can only name 10 families, so it is
           scored on the overlap subset (truth family is one of its labels).

Family level: a prediction counts at family level when its class shares a
MobileNet family with the truth (e.g. sate-lilit vs sate-maranggi). MobileNet is
scored at family level by construction; VLM is reported at both levels.
"""
import argparse
import csv
import difflib
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT, ROOT / "cv", HERE):
    sys.path.insert(0, str(_p))

from dotenv import load_dotenv  # noqa: E402

from integration import dish_table  # noqa: E402
from common import DATA, MANIFEST, load_manifest  # noqa: E402
from vlm_run import ENV_FILE, cache_file  # noqa: E402

MOBILENET_CKPT = Path(os.environ.get(
    "MOBILENET_CKPT", "/Users/auliamnaufal/Downloads/glucosight/cv/models/mobilenetv2_food10.pt"))
TABLE = DATA / "dish_table_35.csv"
TEST_MANIFEST = DATA / "manifest_test.csv"
THRESHOLDS = {"difflib": dish_table.MATCH_THRESHOLD, "embed": dish_table.EMBEDDING_MATCH_THRESHOLD}
ARMS = ("difflib", "embed")

# MobileNet food-10 labels (checkpoint class_to_idx) and the Wibisono classes they cover.
MOBILENET_LABELS = {"bakso", "bebek_betutu", "gado_gado", "gudeg", "nasi_goreng",
                    "pempek", "rawon", "rendang", "sate", "soto"}
FAMILY = {
    "gado-gado": "gado_gado", "gudeg": "gudeg", "nasi-goreng-kampung": "nasi_goreng",
    "pempek-palembang": "pempek", "rawon-surabaya": "rawon", "rendang": "rendang",
    "sate-ayam-madura": "sate", "sate-lilit": "sate", "sate-maranggi": "sate",
    "soto-ayam-lamongan": "soto", "soto-banjar": "soto",
}


def family(cls: str) -> str:
    return FAMILY.get(cls, cls)


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_rows():
    """DishRow candidates: one per (class, name) where name is the row name or an alias."""
    rows, base = [], {}
    for r in read_csv(TABLE):
        cls = r["wibisono_class"]
        names = [r["row_name"]] + [a for a in r["aliases"].split("|") if a]
        for name in dict.fromkeys(names):
            row = dish_table.DishRow(
                food_class=cls, tkpi_item_name=name, serving_g=float(r["serving_g"]),
                carbs_g_100g=float(r["carbs_g"]), protein_g_100g=float(r["protein_g"]),
                fat_g_100g=float(r["fat_g"]), fiber_ratio_of_carbs=0.0,
                mapping_confidence=r["confidence"])
            rows.append(row)
            base.setdefault(cls, row)
    return rows, base


def load_embedder():
    sys.path.insert(0, str(ROOT / "cv" / "experiments" / "vlm_vs_mobilenet"))
    from run_dish_list_gas_vlm import make_gemini_embedder  # noqa: E402
    load_dotenv(ENV_FILE)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(f"GEMINI_API_KEY not found in {ENV_FILE}")
    return make_gemini_embedder(api_key)


def ranked_items(record):
    out = []
    for it in record.get("items") or []:
        name = str(it.get("food_name", "")).strip()
        try:
            conf = float(it.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if name:
            out.append((name, conf))
    return sorted(out, key=lambda t: -t[1])


def difflib_scores(name, rows):
    q = dish_table._normalize(name)
    out = {}
    for row in rows:
        s = max(difflib.SequenceMatcher(a=q, b=dish_table._normalize(c)).ratio()
                for c in (row.food_class, row.tkpi_item_name))
        out[row.food_class] = max(out.get(row.food_class, 0.0), s)
    return out


def embed_scores(name, rows, index, embedder):
    qv = embedder(dish_table._normalize(name), True)
    out = {}
    for i, row in enumerate(rows):
        s = max(dish_table._cosine_similarity(qv, index.vectors[(i, dish_table._normalize(c))])
                for c in (row.food_class, row.tkpi_item_name))
        out[row.food_class] = max(out.get(row.food_class, -math.inf), s)
    return out


def make_matcher(kind, rows, index=None, embedder=None):
    """Return best(name) -> (class or None, similarity). Class is None below threshold."""
    thr = THRESHOLDS[kind]
    cache = {}

    def best(name):
        if name not in cache:
            scores = (difflib_scores(name, rows) if kind == "difflib"
                      else embed_scores(name, rows, index, embedder))
            cls = max(scores, key=scores.get)
            cache[name] = (cls, scores[cls]) if scores[cls] >= thr else (None, scores[cls])
        return cache[name]
    return best


def predict(items, best, limit=3):
    """Walk items by VLM confidence; collect distinct matched classes (top-3 list)."""
    classes, top_score, top_item, top_conf = [], None, None, None
    for name, conf in items:
        cls, score = best(name)
        if cls is None or cls in classes:
            continue
        if not classes:
            top_score, top_item, top_conf = score, name, conf
        classes.append(cls)
        if len(classes) == limit:
            break
    return classes, top_score, top_item, top_conf


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def paired_bootstrap_ci(a, b, n_boot=10000, seed=0):
    """95% percentile CI for mean(a - b), resampling images."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def quantile(vals, q):
    s = sorted(vals)
    if not s:
        return math.nan
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def fmt(x, nd=4):
    if isinstance(x, float):
        return "" if math.isnan(x) else round(x, nd)
    return x


def mobilenet_predictions(manifest):
    from cv_baseline.classifier import MobileNetV2Classifier  # noqa: E402
    from cv_baseline.preprocessing import preprocess  # noqa: E402
    clf = MobileNetV2Classifier(MOBILENET_CKPT).load()
    return {path: clf.predict_topk(preprocess(path), k=1)[0] for path, _, _ in manifest}


def arm_summary(recs, arm):
    n = len(recs)
    ov = [r for r in recs if r["overlap"]]
    k1 = sum(r[f"{arm}_class_correct"] for r in recs)
    k3 = sum(r[f"{arm}_top3_correct"] for r in recs)
    kf = sum(r[f"{arm}_family_correct"] for r in recs)
    abst = sum(r[f"{arm}_abstain"] for r in recs)
    fm = sum(1 for r in recs if not r[f"{arm}_abstain"] and not r[f"{arm}_class_correct"])
    ok = sum(r[f"{arm}_class_correct"] for r in ov)
    of = sum(r[f"{arm}_family_correct"] for r in ov)
    return {
        "n": n,
        "top1_class_acc": fmt(k1 / n), "top1_class_correct": k1,
        "top1_class_wilson95": [fmt(x) for x in wilson(k1, n)],
        "top3_class_acc": fmt(k3 / n), "top3_class_wilson95": [fmt(x) for x in wilson(k3, n)],
        "family_acc": fmt(kf / n), "family_wilson95": [fmt(x) for x in wilson(kf, n)],
        "abstain_rate": fmt(abst / n), "false_match_rate": fmt(fm / n),
        "overlap_n": len(ov),
        "overlap_class_acc": fmt(ok / len(ov)) if ov else None,
        "overlap_family_acc": fmt(of / len(ov)) if ov else None,
    }


def mobilenet_summary(recs):
    ov = [r for r in recs if r["overlap"]]
    k = sum(r["mobilenet_correct"] for r in ov)
    return {
        "n": len(recs), "overlap_n": len(ov),
        "overlap_family_acc": fmt(k / len(ov)) if ov else None,
        "overlap_family_wilson95": [fmt(x) for x in wilson(k, len(ov))] if ov else None,
        "overall_acc_all_images": fmt(k / len(recs)),
    }


def bootstrap_summary(recs):
    ov = [r for r in recs if r["overlap"]]
    mn = [r["mobilenet_correct"] for r in ov]
    out = {}
    for arm in ARMS:
        for level in ("class", "family"):
            a = [r[f"{arm}_{level}_correct"] for r in ov]
            lo, hi = paired_bootstrap_ci(a, mn)
            out[f"{arm}_{level}_minus_mobilenet"] = {
                "diff": fmt((sum(a) - sum(mn)) / len(ov)), "ci95": [fmt(lo), fmt(hi)]}
    return out


def carb_summary(carb_rows, arm):
    answered = [r for r in carb_rows if r["arm"] == arm and not r["abstained"]]
    errs = [r["carb_error_g"] for r in answered]
    absv = [abs(e) for e in errs]
    return {
        "n_answered": len(answered),
        "n_abstained": sum(1 for r in carb_rows if r["arm"] == arm and r["abstained"]),
        "abs_error_median_g": fmt(quantile(absv, 0.5), 1),
        "abs_error_q1_g": fmt(quantile(absv, 0.25), 1),
        "abs_error_q3_g": fmt(quantile(absv, 0.75), 1),
        "share_within_10g": fmt(sum(1 for x in absv if x <= 10) / len(absv)) if absv else None,
        "signed_error_median_g": fmt(quantile(errs, 0.5), 1),
    }


def _conf(r, arm):
    v = r[f"{arm}_conf"]
    return float(v) if v != "" else 0.0


def confidence_rows(recs):
    n = len(recs)
    rows = []
    for arm in ARMS:
        def add(rule, bound, sel):
            cov = [r for r in recs if not r[f"{arm}_abstain"] and sel(r)]
            k = sum(r[f"{arm}_class_correct"] for r in cov)
            kf = sum(r[f"{arm}_family_correct"] for r in cov)
            lo, hi = wilson(k, len(cov))
            rows.append({
                "arm": arm, "rule": rule, "bound": bound, "n_covered": len(cov),
                "coverage": fmt(len(cov) / n),
                "class_accuracy": fmt(k / len(cov)) if cov else "",
                "class_wilson_lo": fmt(lo), "class_wilson_hi": fmt(hi),
                "family_accuracy": fmt(kf / len(cov)) if cov else "",
            })
        for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
            add("min_confidence", t, lambda r, t=t: _conf(r, arm) >= t)
        edges = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01]
        for lo_e, hi_e in zip(edges, edges[1:]):
            add("bin", f"[{lo_e},{min(hi_e, 1.0)})",
                lambda r, a=lo_e, b=hi_e: a <= _conf(r, arm) < b)
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DATA / "results" / "dish_id_trial")
    args = ap.parse_args()

    split = args.split
    manifest = load_manifest(args.manifest or (MANIFEST if split == "dev" else TEST_MANIFEST))
    rows, base = load_rows()
    unknown = sorted({label for _, label, _ in manifest if label not in base})
    if unknown:
        raise SystemExit(f"labels with no dish-table row: {unknown}")

    missing = [p for p, _, _ in manifest if not cache_file(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images have no VLM cache; run vlm_run.py first")
    records = {}
    for p, _, _ in manifest:
        rec = json.loads(cache_file(p).read_text())
        if rec.get("status") != "ok":
            raise SystemExit(f"cached record not ok: {p}")
        records[p] = rec

    embedder = load_embedder()
    index = dish_table.build_dish_table_index(rows, embedder)
    best = {
        "difflib": make_matcher("difflib", rows),
        "embed": make_matcher("embed", rows, index=index, embedder=embedder),
    }
    lib = {
        "difflib": lambda n: dish_table.match_dish(n, rows),
        "embed": lambda n: dish_table.match_dish_by_embedding(n, index, embedder),
    }
    names = sorted({n for p in records for n, _ in ranked_items(records[p])})
    crosscheck = {}
    for kind in ARMS:
        mism = [n for n in names
                if (lib[kind](n).row.food_class if lib[kind](n) else None) != best[kind](n)[0]]
        crosscheck[kind] = {"names": len(names), "mismatch_vs_library": len(mism), "examples": mism[:5]}

    mn = mobilenet_predictions(manifest)

    recs, carb_rows = [], []
    for p, label, idx in manifest:
        items = ranked_items(records[p])
        tf = family(label)
        true_carbs = dish_table.serving_macros(base[label])["carbs_g"]
        rec = {
            "image_path": p, "split": split, "image_index": idx,
            "true_class": label, "true_family": tf,
            "overlap": int(tf in MOBILENET_LABELS),
            "mobilenet_label": mn[p][0], "mobilenet_prob": round(float(mn[p][1]), 4),
            "mobilenet_correct": int(tf in MOBILENET_LABELS and mn[p][0] == tf),
            "vlm_items": json.dumps(items, ensure_ascii=False),
        }
        for arm in ARMS:
            classes, score, top_item, conf = predict(items, best[arm])
            pred = classes[0] if classes else ""
            rec[f"{arm}_pred"] = pred
            rec[f"{arm}_top3"] = ";".join(classes)
            rec[f"{arm}_top1_item"] = top_item or ""
            rec[f"{arm}_score"] = fmt(float(score), 4) if score is not None else ""
            rec[f"{arm}_conf"] = conf if conf is not None else ""
            rec[f"{arm}_abstain"] = int(not pred)
            rec[f"{arm}_class_correct"] = int(pred == label)
            rec[f"{arm}_family_correct"] = int(bool(pred) and family(pred) == tf)
            rec[f"{arm}_top3_correct"] = int(label in classes)
            err = ""
            if pred:
                err = round(dish_table.serving_macros(base[pred])["carbs_g"] - true_carbs, 1)
            rec[f"{arm}_carb_error_g"] = err
            carb_rows.append({
                "image_path": p, "split": split, "true_class": label,
                "true_carbs_g": true_carbs, "arm": arm, "pred_class": pred,
                "abstained": int(not pred), "carb_error_g": err,
            })
        recs.append(rec)

    out = args.out_dir / split
    write_csv(out / "per_image.csv", recs)
    write_csv(out / "dish_id_carb_errors.csv",
              [{k: v for k, v in r.items()} for r in carb_rows
               if r["arm"] in ARMS])
    write_csv(out / "dish_id_confidence.csv", confidence_rows(recs))

    summary = {
        "split": split,
        "n_images": len(recs),
        "n_overlap": sum(r["overlap"] for r in recs),
        "vlm_cache_records": len(records),
        "table": {"file": TABLE.name, "sha256": hashlib.sha256(TABLE.read_bytes()).hexdigest()[:16],
                  "classes": len(base)},
        "thresholds": THRESHOLDS,
        "matcher_crosscheck": crosscheck,
        "arms": {arm: arm_summary(recs, arm) for arm in ARMS},
        "mobilenet": mobilenet_summary(recs),
        "bootstrap_overlap": bootstrap_summary(recs),
        "carb": {arm: carb_summary(carb_rows, arm) for arm in ARMS},
        "mobilenet_ckpt": MOBILENET_CKPT.name,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
