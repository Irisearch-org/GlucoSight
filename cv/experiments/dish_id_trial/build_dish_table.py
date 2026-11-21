#!/usr/bin/env python3
"""Build the 35-row dish table (one row per Wibisono class) from TKPI 2017 + recipes.

Inputs (committed): classes.csv (class -> TKPI code or recipe), recipes.csv
(computed dishes as TKPI ingredient grams). Inputs (local, not committed):
tkpi2017_rows.csv from tkpi_parse.py, and the user's 55-row validation CSV.

Output (local): DATA/dish_table_35.csv with per-100 g edible macros only.
Computed rows are sum(grams * per-100 g) / sum(grams) * 100; cooking
losses and water uptake are not modelled. Fibre is not parsed from TKPI and is omitted.
"""
import argparse
import csv
import sys
from pathlib import Path

from common import DATA

HERE = Path(__file__).parent
TKPI_ROWS = DATA / "tkpi2017_rows.csv"
REFERENCE_CSV = Path(
    "/Users/auliamnaufal/Downloads/Indonesian Food Composition Table Data - Table 1.csv"
)
OUT = DATA / "dish_table_35.csv"
MACRO_FIELDS = ("energy_kcal", "carbs_g", "protein_g", "fat_g")


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_tkpi():
    rows = {}
    for r in read_csv(TKPI_ROWS):
        if r["parsed_ok"] == "1":
            rows[r["code"]] = {k: float(r[k]) for k in MACRO_FIELDS if r[k]} | {"name": r["name"]}
    return rows


def compute_recipe(recipe_rows, tkpi):
    total_g = 0.0
    acc = {k: 0.0 for k in MACRO_FIELDS}
    for r in recipe_rows:
        g = float(r["grams"])
        total_g += g
        if r["tkpi_code"] == "WATER":
            continue
        ing = tkpi[r["tkpi_code"]]
        for k in MACRO_FIELDS:
            acc[k] += g * ing[k]
    return {k: round(v / total_g, 1) for k, v in acc.items()}


def validate_parser(tkpi):
    """Compare parsed TKPI rows with the user's 55-row reference CSV."""
    ref = read_csv(REFERENCE_CSV)
    cols = {"energy_kcal": "Energi (Kal)", "protein_g": "Protein (g)",
            "fat_g": "Lemak (g)", "carbs_g": "Karbohidrat (g)"}
    exact, mismatches, missing = 0, [], []
    for r in ref:
        code = r["Kode"].strip()
        if code not in tkpi:
            missing.append(code)
            continue
        diffs = []
        for k, col in cols.items():
            try:
                want = float(r[col])
            except ValueError:
                continue
            if abs(tkpi[code][k] - want) > 0.05:
                diffs.append(f"{k}: parsed {tkpi[code][k]} vs ref {want}")
        if diffs:
            mismatches.append((code, "; ".join(diffs)))
        else:
            exact += 1
    print(f"parser validation: {exact}/{len(ref)} exact, "
          f"{len(mismatches)} mismatch, {len(missing)} missing")
    for code, msg in mismatches:
        print(f"  mismatch {code}: {msg}")
    for code in missing:
        print(f"  missing {code}")
    return exact, mismatches, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    tkpi = load_tkpi()
    validate_parser(tkpi)
    if args.validate_only:
        return 0

    classes = read_csv(HERE / "classes.csv")
    if len(classes) != 35:
        raise SystemExit(f"expected 35 classes, found {len(classes)}")

    recipes = {}
    for r in read_csv(HERE / "recipes.csv"):
        recipes.setdefault(r["recipe_id"], []).append(r)

    out_rows = []
    for c in classes:
        if c["source"] == "tkpi2017":
            code = c["source_ref"]
            if code not in tkpi:
                raise SystemExit(f"{c['wibisono_class']}: TKPI code {code} not parsed")
            macros = {k: tkpi[code][k] for k in MACRO_FIELDS}
            ref_name = tkpi[code]["name"]
        elif c["source"] == "computed":
            rid = c["recipe_id"]
            if rid not in recipes:
                raise SystemExit(f"{c['wibisono_class']}: recipe {rid} missing")
            for r in recipes[rid]:
                if r["tkpi_code"] != "WATER" and r["tkpi_code"] not in tkpi:
                    raise SystemExit(f"recipe {rid}: TKPI {r['tkpi_code']} not parsed")
            macros = compute_recipe(recipes[rid], tkpi)
            ref_name = f"computed:{rid}"
        else:
            raise SystemExit(f"{c['wibisono_class']}: unknown source {c['source']}")
        out_rows.append({
            "wibisono_class": c["wibisono_class"],
            "row_name": c["row_name"],
            "aliases": c["aliases"],
            "source": c["source"],
            "source_ref": c["source_ref"] or c["recipe_id"],
            "tkpi_name": ref_name,
            "confidence": c["confidence"],
            **macros,
            "serving_g": c["serving_g"],
            "serving_source": c["serving_source"],
        })

    fields = list(out_rows[0].keys())
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
