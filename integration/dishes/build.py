"""Build the dish table locally from the TKPI 2017 PDF.

    python -m integration.dishes.build --tkpi-pdf <tkpi2017.pdf> --out data/dish_table/dish_table.csv

Git holds only this builder, the class mapping (``data/classes.csv``), the computed-row
recipes (``data/recipes.csv``), the ingredient overlay (``data/ingredients.csv``) and the
raw-eaten list (``data/raw_eaten.csv``). The composition values come from the PDF and are
written to a gitignored file: the repo is public and TKPI has no open licence.

Rows (docs/SPEC_V2_PIPELINE.md section 2, ticket #15):

* every TKPI "P" (processed/dish) entry, kind ``dish``, confidence ``exact``;
* "R" (raw) entries only for foods eaten raw: all fruit (``ER``) and the lalapan list;
* the 35 dish-ID classes: a class that is the only claimant of a TKPI code is folded into
  that code's row (alias + serving), otherwise it gets a ``GS-<slug>`` row; classes with no
  TKPI entry are computed from ``recipes.csv`` (``GS-<slug>``, confidence ``computed``);
* common manual-entry ingredients in their eaten state, with Kemenkes URT serving weights.

Entries whose energy disagrees with 4P+4C+9F by more than 30% (printed errors in the PDF,
e.g. a carbohydrate column off by a factor of ten) are dropped from the table, unless a
class or recipe needs them: those are kept as printed and listed in the build report.
Dish ``serving_g`` values are assumed defaults and labelled ``assumed``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from . import TABLE_FIELDS, DishRow
from .tkpi import TkpiEntry, parse_layout, pdf_to_layout_text

DATA = Path(__file__).resolve().parent / "data"
DEFAULT_OUT = Path("data/dish_table/dish_table.csv")
SERVING_ASSUMED = "assumed"
SUFFIX_RE = re.compile(r",?\s*(masakan|segar)$")


@dataclass(frozen=True)
class BuildResult:
    rows: tuple[DishRow, ...]
    dropped_implausible: tuple[str, ...]
    kept_implausible: tuple[str, ...]
    parsed_entries: int


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _split(s: str) -> list[str]:
    return [a.strip() for a in s.split("|") if a.strip()]


def _r1(v: float | None) -> float | None:
    return None if v is None else round(v, 1)


def derive_aliases(name: str, *, exclude: set[str] = frozenset()) -> tuple[str, ...]:
    """The name without its trailing masakan/segar, comma-free ("Soto madura, masakan" -> "soto madura")."""
    base = name.strip().lower()
    stripped = SUFFIX_RE.sub("", base).strip()
    if stripped == base:
        return ()
    alias = re.sub(r"\s+", " ", stripped.replace(",", " ")).strip()
    return (alias,) if alias and alias not in exclude else ()


def _dish_row(e: TkpiEntry, *, name: str, kind: str, aliases: tuple[str, ...],
              confidence: str, serving_g: float | None = None,
              serving_source: str | None = None) -> DishRow:
    return DishRow(
        id=e.code, name=name, kind=kind, energy_kcal=_r1(e.energy_kcal),
        carbs_g=_r1(e.carbs_g), protein_g=_r1(e.protein_g), fat_g=_r1(e.fat_g),
        fibre_g=_r1(e.fibre_g), serving_g=serving_g, serving_source=serving_source,
        source="tkpi2017", source_ref=e.code, confidence=confidence, aliases=aliases)


def _compute_recipe(items: list[dict], tkpi: dict[str, TkpiEntry]):
    total_g = 0.0
    acc = {"energy_kcal": 0.0, "carbs_g": 0.0, "protein_g": 0.0, "fat_g": 0.0, "fibre_g": 0.0}
    fibre_known = True
    for it in items:
        g = float(it["grams"])
        total_g += g
        if it["tkpi_code"] == "WATER":
            continue
        ing = tkpi[it["tkpi_code"]]
        for k in ("energy_kcal", "carbs_g", "protein_g", "fat_g"):
            acc[k] += g * getattr(ing, k)
        if ing.fibre_g is None:
            fibre_known = False
        else:
            acc["fibre_g"] += g * ing.fibre_g
    out = {k: round(v / total_g, 1) for k, v in acc.items()}
    if not fibre_known:
        out["fibre_g"] = None
    return out


def build(layout_text: str) -> BuildResult:
    entries = {e.code: e for e in parse_layout(layout_text)}
    classes = _read(DATA / "classes.csv")
    if len(classes) != 35:
        raise SystemExit(f"expected 35 dish-ID classes, found {len(classes)}")
    recipes: dict[str, list[dict]] = {}
    for r in _read(DATA / "recipes.csv"):
        recipes.setdefault(r["recipe_id"], []).append(r)
    overlay = {r["code"]: r for r in _read(DATA / "ingredients.csv")}
    lalapan = {r["code"]: r for r in _read(DATA / "raw_eaten.csv")}

    class_codes = Counter(c["source_ref"] for c in classes if c["source"] == "tkpi2017")
    recipe_codes = {r["tkpi_code"] for items in recipes.values() for r in items} - {"WATER"}
    required = set(class_codes) | recipe_codes

    dropped, kept = [], []
    usable: dict[str, TkpiEntry] = {}
    for code, e in entries.items():
        if e.atwater_ok():
            usable[code] = e
        elif code in required:
            usable[code] = e
            kept.append(code)
        else:
            dropped.append(code)

    missing = sorted((required | set(overlay) | set(lalapan)) - set(usable))
    if missing:
        raise SystemExit(f"codes needed by the build are absent or implausible: {missing}")

    rows: dict[str, DishRow] = {}
    for code, e in usable.items():
        if e.is_processed:
            rows[code] = _dish_row(e, name=e.name, kind="dish", confidence="exact",
                                   aliases=derive_aliases(e.name, exclude={e.name.lower()}))
        elif code.startswith("ER") or code in lalapan:
            name = SUFFIX_RE.sub("", e.name).strip()
            extra = _split(lalapan[code]["aliases"]) if code in lalapan else []
            aliases = tuple(dict.fromkeys(
                [*derive_aliases(e.name, exclude={name.lower()}), *extra]))
            rows[code] = _dish_row(e, name=name, kind="ingredient", confidence="exact",
                                   aliases=aliases)

    for code, o in overlay.items():
        if code in class_codes:
            raise SystemExit(f"{code} is both an ingredient overlay and a dish class")
        e = usable[code]
        aliases = tuple(_split(o["aliases"]))
        rows[code] = _dish_row(
            e, name=o["name"], kind="ingredient", aliases=aliases, confidence=o["confidence"],
            serving_g=float(o["serving_g"]), serving_source=o["serving_source"])

    for c in classes:
        slug = c["wibisono_class"]
        aliases = tuple(dict.fromkeys([slug, *_split(c["aliases"])]))
        serving = float(c["serving_g"])
        if c["source"] == "tkpi2017":
            e = usable[c["source_ref"]]
            if class_codes[c["source_ref"]] == 1:  # fold into the TKPI code's own row
                aliases = tuple(dict.fromkeys([*aliases, *derive_aliases(e.name)]))
                rows[e.code] = _dish_row(
                    e, name=c["row_name"], kind="dish", aliases=aliases,
                    confidence=c["confidence"], serving_g=serving,
                    serving_source=SERVING_ASSUMED)
            else:  # several classes share one TKPI entry: one row each
                row = _dish_row(e, name=c["row_name"], kind="dish", aliases=aliases,
                                confidence=c["confidence"], serving_g=serving,
                                serving_source=SERVING_ASSUMED)
                rows[f"GS-{slug}"] = replace(row, id=f"GS-{slug}")
        elif c["source"] == "computed":
            m = _compute_recipe(recipes[c["recipe_id"]], usable)
            rows[f"GS-{slug}"] = DishRow(
                id=f"GS-{slug}", name=c["row_name"], kind="dish", energy_kcal=m["energy_kcal"],
                carbs_g=m["carbs_g"], protein_g=m["protein_g"], fat_g=m["fat_g"],
                fibre_g=m["fibre_g"], serving_g=serving, serving_source=SERVING_ASSUMED,
                source="computed", source_ref=c["recipe_id"], confidence="computed",
                aliases=aliases)
        else:
            raise SystemExit(f"{slug}: unknown source {c['source']!r}")

    return BuildResult(
        rows=tuple(sorted(rows.values(), key=lambda r: r.id)),
        dropped_implausible=tuple(sorted(dropped)), kept_implausible=tuple(sorted(kept)),
        parsed_entries=len(entries))


def write_table(rows: tuple[DishRow, ...], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
        w.writeheader()
        for r in rows:
            d = {k: getattr(r, k) for k in TABLE_FIELDS if k != "aliases"}
            d["aliases"] = "|".join(r.aliases)
            w.writerow({k: ("" if v is None else v) for k, v in d.items()})


def validate_parser(layout_text: str, reference: Path) -> tuple[int, int]:
    """Parser check against a hand-typed 55-row sample of TKPI (exact on the four macros)."""
    tkpi = {e.code: e for e in parse_layout(layout_text)}
    cols = {"energy_kcal": "Energi (Kal)", "protein_g": "Protein (g)",
            "fat_g": "Lemak (g)", "carbs_g": "Karbohidrat (g)"}
    exact = 0
    ref = _read(reference)
    for r in ref:
        e = tkpi.get(r["Kode"].strip())
        if e is None:
            print(f"  validation: {r['Kode']} not parsed")
            continue
        diffs = []
        for attr, col in cols.items():
            try:
                want = float(r[col])
            except ValueError:
                continue
            if abs(getattr(e, attr) - want) > 0.05:
                diffs.append(f"{attr}: parsed {getattr(e, attr)} vs ref {want}")
        if diffs:
            print(f"  validation mismatch {e.code}: {'; '.join(diffs)}")
        else:
            exact += 1
    return exact, len(ref)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tkpi-pdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--embed-key-file", type=Path, default=None,
                    help="also embed the built table now (cache next to the CSV), "
                         "using GEMINI_API_KEY from this gitignored file")
    ap.add_argument("--embedding-model", default="gemini-embedding-001")
    ap.add_argument("--reference", type=Path, default=None,
                    help="CSV of hand-typed TKPI rows to validate the parser against")
    args = ap.parse_args()

    text = pdf_to_layout_text(args.tkpi_pdf)
    if args.reference:
        exact, n = validate_parser(text, args.reference)
        print(f"parser validation: {exact}/{n} exact")
    result = build(text)
    write_table(result.rows, args.out)

    print(f"parsed {result.parsed_entries} TKPI entries; wrote {len(result.rows)} rows -> {args.out}")
    for label, key in (("kind", "kind"), ("confidence", "confidence"), ("source", "source")):
        counts = Counter(getattr(r, key) for r in result.rows)
        print(f"  by {label}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"  fibre unknown: {sum(r.fibre_g is None for r in result.rows)}; "
          f"serving_g set: {sum(r.serving_g is not None for r in result.rows)}")
    print(f"  dropped as implausible (energy vs 4P+4C+9F > 30%): {len(result.dropped_implausible)}: "
          f"{', '.join(result.dropped_implausible)}")
    print(f"  kept as printed because a class or recipe needs them: "
          f"{', '.join(result.kept_implausible) or 'none'}")
    if args.embed_key_file:
        from . import DishTable, GeminiEmbedder
        from integration.stages.meal import load_gemini_key
        table = DishTable.load(args.out)
        table.warm(GeminiEmbedder(load_gemini_key(args.embed_key_file), args.embedding_model))
        print(f"  embeddings cached for build {table.build_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
