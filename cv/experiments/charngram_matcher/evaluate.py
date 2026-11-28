"""Operating point for the served character n-gram matcher (ADR 0006).

Re-scores the dish-ID trial offline with the SERVED code (``integration.dishes.DishTable.match``)
against the SERVED 710-row table. No API calls: the trial's cached Gemini dish names are read
from its per-image reports (column ``vlm_items``), so the VLM side is identical to the trial.

Truth for a photo is the table row its Wibisono class resolves to: the dedicated ``GS-<class>``
row when the build has one, else the class's TKPI code (``source_ref`` in the trial's 35-row
evaluation table). Scoring is strict row-id equality, so a near-identical sibling row
(another rendang, say) counts as wrong.

Protocol (the one used for threshold_embedding.csv, ticket #19): walk a photo's names by VLM
confidence; the first name whose top-1 similarity clears the threshold decides; answer if its
margin also clears, otherwise abstain. Sweep threshold x margin on dev, choose by minimum dev
false-match rate subject to dev coverage >= 60% (ties: higher coverage, then higher threshold),
then report test at that frozen point.

    git show origin/wayfinder/dish-id-trial:cv/reports/dish_id_trial/dev/per_image.csv > /tmp/dev.csv
    git show origin/wayfinder/dish-id-trial:cv/reports/dish_id_trial/test/per_image.csv > /tmp/test.csv
    python -m cv.experiments.charngram_matcher.evaluate --dev /tmp/dev.csv --test /tmp/test.csv \\
        --classes ~/Downloads/glucosight-cv-data/dish_table_35.csv

Writes only rates and settings (no TKPI values) to cv/reports/charngram_matcher/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from integration.dishes import MATCHER_VERSION, DishTable

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "cv" / "reports" / "charngram_matcher"
THRESHOLDS = [round(t, 3) for t in np.arange(0.30, 0.951, 0.025)]
MARGINS = (0.0, 0.02, 0.05, 0.1)
MIN_DEV_COVERAGE = 0.60


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def truth_rows(classes_csv: Path, table: DishTable) -> dict[str, str]:
    ids = {r.id for r in table.rows}
    out = {}
    for r in read(classes_csv):
        cls = r["wibisono_class"]
        out[cls] = f"GS-{cls}" if f"GS-{cls}" in ids else r["source_ref"]
        if out[cls] not in ids:
            raise SystemExit(f"class {cls} resolves to {out[cls]}, which is not in the table")
    return out


def score(recs, matches, truth, thr, margin) -> dict:
    answered = correct = 0
    for r in recs:
        pred = None
        for name, _ in sorted(json.loads(r["vlm_items"]), key=lambda t: -t[1]):
            m = matches[name]
            if m.similarity >= thr:
                if m.margin >= margin:
                    pred = m.row.id
                break
        if pred is not None:
            answered += 1
            correct += pred == truth[r["true_class"]]
    n = len(recs)
    return {"n": n, "answered": answered, "coverage": answered / n,
            "top1_acc": correct / n, "acc_when_answered": correct / answered if answered else 0.0,
            "false_match_rate": (answered - correct) / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True, help="the trial's dish_table_35.csv")
    ap.add_argument("--table", type=Path, default=ROOT / "data" / "dish_table" / "dish_table.csv")
    args = ap.parse_args()

    table = DishTable.load(args.table)
    truth = truth_rows(args.classes, table)
    split = {"dev": read(args.dev), "test": read(args.test)}
    names = {n for recs in split.values() for r in recs for n, _ in json.loads(r["vlm_items"])}
    matches = {n: table.match(n) for n in names}

    sweep = [{"split": s, "threshold": t, "margin": m, **score(split[s], matches, truth, t, m)}
             for s in ("dev", "test") for t in THRESHOLDS for m in MARGINS]
    dev = [row for row in sweep if row["split"] == "dev" and row["coverage"] >= MIN_DEV_COVERAGE]
    chosen = min(dev, key=lambda r: (r["false_match_rate"], -r["coverage"], -r["threshold"]))
    point = (chosen["threshold"], chosen["margin"])
    test = next(r for r in sweep if r["split"] == "test" and (r["threshold"], r["margin"]) == point)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "threshold_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*sweep[0], "selected"])
        w.writeheader()
        for row in sweep:
            w.writerow({**{k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()},
                        "selected": int((row["threshold"], row["margin"]) == point)})
    summary = {"matcher": MATCHER_VERSION, "table_build": table.build_hash, "table_rows": len(table),
               "rule": "min dev false-match s.t. dev coverage >= 0.60; ties: coverage, threshold",
               "threshold": point[0], "margin": point[1],
               "dev": {k: round(v, 4) for k, v in chosen.items() if isinstance(v, float)} | {"n": chosen["n"]},
               "test": {k: round(v, 4) for k, v in test.items() if isinstance(v, float)} | {"n": test["n"]}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
