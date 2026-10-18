"""Reproduce the numbers in `docs/NLP_PATH_A_GATE.md`.

Reads the committed Path A feature file and the committed Path B splits.
No download, no network, no glucose model: these are property and
provenance measurements only.

    python -m nlp.eval_path_a_gate

Findings referenced by ID (N1..N8) are defined in the document.
"""
from __future__ import annotations

import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
DEFAULT_FEATURES = _HERE / "features" / "shanghai_nlp_features.csv"
PROCESSED = _HERE / "data" / "processed"

# The five prefix variations `nlp/scripts/create_dataset.py` adds on top of
# its hand-written templates (N7).
PREFIXES = ("aku tadi ", "hari ini ", "barusan ", "tadi ", "semalam ", "karena ")

# "Rice 150 g" / "Milk 250 ml" -> ("Rice", 150.0). Anchored at the end so an
# item name containing a digit does not split the wrong way.
ITEM_RE = re.compile(r"^(?P<name>.*?)\s*(?P<grams>\d+(?:\.\d+)?)\s*(?:g|ml)\s*$", re.I)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_features(path: Path = DEFAULT_FEATURES) -> List[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_items(rows: Iterable[dict]) -> List[Tuple[str, str, float]]:
    """(participant_id, normalised dish name, grams) for every parsed item."""
    out: List[Tuple[str, str, float]] = []
    for r in rows:
        for seg in r["dietary_intake"].split("|"):
            m = ITEM_RE.match(seg.strip())
            if m:
                name = re.sub(r"\s+", " ", m.group("name").strip().lower())
                if name and float(m.group("grams")) > 0:
                    out.append((r["participant_id"], name, float(m.group("grams"))))
    return out


def dish_names(rows: Iterable[dict]) -> List[str]:
    return [name for _, name, _ in parse_items(rows)]


# --------------------------------------------------------------------------
# N1 — the records are rich, and the richness is dish names
# --------------------------------------------------------------------------

def richness(rows: List[dict]) -> Dict[str, object]:
    texts = [r["dietary_intake"] for r in rows]
    segs = [s.strip() for t in texts for s in t.split("|")]
    names = dish_names(rows)
    seg_counts = Counter(segs)
    name_counts = Counter(names)
    items_per_meal = [len(t.split("|")) for t in texts]
    return {
        "meals": len(rows),
        "distinct_meal_texts": len(set(texts)),
        "segments": len(segs),
        "distinct_segments": len(set(segs)),
        "distinct_dish_names": len(set(names)),
        "items_per_meal_median": statistics.median(items_per_meal),
        "items_per_meal_max": max(items_per_meal),
        "multi_item_meals": sum(1 for n in items_per_meal if n > 1),
        "singleton_segment_share": sum(
            1 for v in seg_counts.values() if v == 1
        ) / len(seg_counts),
        "top20_segment_coverage": sum(
            n for _, n in seg_counts.most_common(20)
        ) / len(segs),
        "top20_dish_name_coverage": sum(
            n for _, n in name_counts.most_common(20)
        ) / len(names),
        "fried_rate": sum(int(r["is_fried_cooking"]) for r in rows) / len(rows),
        "fried_meals": sum(int(r["is_fried_cooking"]) for r in rows),
        "large_rate": sum(int(r["is_large_portion"]) for r in rows) / len(rows),
        "date_first": min(r["t0_timestamp_iso"] for r in rows)[:10],
        "date_last": max(r["t0_timestamp_iso"] for r in rows)[:10],
        "distinct_days": len({r["t0_timestamp_iso"][:10] for r in rows}),
    }


# --------------------------------------------------------------------------
# N2 — the pipeline discards the dish name
# --------------------------------------------------------------------------

def feature_collapse(rows: List[dict]) -> Dict[str, int]:
    states = {
        (r["total_grams"], r["is_fried_cooking"], r["is_large_portion"])
        for r in rows
    }
    return {
        "distinct_meal_texts": len({r["dietary_intake"] for r in rows}),
        "distinct_feature_states": len(states),
        "distinct_dish_names": len(set(dish_names(rows))),
        "distinct_carb_densities": 1,  # DOMAIN_KNOWLEDGE.md 1.4: 0.28 flat
    }


# --------------------------------------------------------------------------
# N3 — nlp_confidence is a re-encoding of the two flags
# --------------------------------------------------------------------------

def confidence_by_flags(rows: List[dict]) -> Dict[str, Tuple[int, float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        key = f"({r['is_fried_cooking']},{r['is_large_portion']})"
        grouped[key].append(float(r["nlp_confidence"]))
    n = len(rows)
    return {
        k: (len(v), round(statistics.fmean(v), 4), round(len(v) / n, 4))
        for k, v in sorted(grouped.items())
    }


# --------------------------------------------------------------------------
# N5 — the split unit is the CGM period, not the patient
# --------------------------------------------------------------------------

def subject_of(participant_id: str) -> str:
    return str(participant_id).split("_")[0]


def split_unit(rows: List[dict], n_splits: int = 5) -> Dict[str, object]:
    pids = sorted({r["participant_id"] for r in rows})
    meals = Counter(r["participant_id"] for r in rows)
    by_subject: Dict[str, List[str]] = defaultdict(list)
    for p in pids:
        by_subject[subject_of(p)].append(p)

    # Deterministic fold assignment: GroupKFold is contiguous, but any
    # partition over the period ids produces the same leak, so a stride is
    # used for transparency.
    folds = [set(pids[i::n_splits]) for i in range(n_splits)]
    leak_pids = leak_meals = 0
    for i, fold in enumerate(folds):
        train = set().union(*[f for j, f in enumerate(folds) if j != i])
        train_subjects = {subject_of(p) for p in train}
        for p in fold:
            if subject_of(p) in train_subjects:
                leak_pids += 1
                leak_meals += meals[p]

    return {
        "periods": len(pids),
        "subjects": len(by_subject),
        "subjects_with_multiple_periods": sum(
            1 for v in by_subject.values() if len(v) > 1
        ),
        "leaking_test_groups": leak_pids,
        "leaking_meals": leak_meals,
        "leaking_meal_share": leak_meals / len(rows),
    }


# --------------------------------------------------------------------------
# N6 — the extracted features have no response variable
# --------------------------------------------------------------------------

def glucose_column(rows: List[dict]) -> Dict[str, object]:
    vals = [float(r["cgm_mgdl"]) for r in rows if r["cgm_mgdl"].strip()]
    per = defaultdict(set)
    for r in rows:
        if r["cgm_mgdl"].strip():
            per[r["participant_id"]].add(r["cgm_mgdl"])
    return {
        "rows": len(rows),
        "blank": len(rows) - len(vals),
        "distinct_values": len(set(vals)),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "distinct_per_period_median": statistics.median(
            len(v) for v in per.values()
        ),
    }


# --------------------------------------------------------------------------
# N7 — Path B's evaluation set leaks and its labels are the rules
# --------------------------------------------------------------------------

def base_sentence(text: str) -> str:
    s = str(text).lower().strip()
    for p in PREFIXES:
        if s.startswith(p):
            return s[len(p):]
    return s


def load_split(name: str) -> List[str]:
    path = PROCESSED / f"context_{name}.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [r["context_sentence"] for r in csv.DictReader(fh)]


def path_b_leakage() -> Optional[Dict[str, object]]:
    labeled = load_split("labeled")
    if not labeled:
        return None
    train = {base_sentence(s) for s in load_split("train")}
    val = {base_sentence(s) for s in load_split("val")}
    test = {base_sentence(s) for s in load_split("test")}
    bases = Counter(base_sentence(s) for s in labeled)
    return {
        "rows": len(labeled),
        "distinct_sentences": len({s.strip().lower() for s in labeled}),
        "distinct_base_sentences": len(bases),
        "strings_sharing_a_base": sum(v for v in bases.values() if v > 1),
        "test_bases_in_train": f"{len(test & train)}/{len(test)}",
        "val_bases_in_train": f"{len(val & train)}/{len(val)}",
    }


# --------------------------------------------------------------------------
# N8 — weighed data tests B7's portion floor
# --------------------------------------------------------------------------

def _variance_split(
    groups: Dict[object, List[float]]
) -> Tuple[float, int, int, int]:
    """(within-dish share of log(grams) variance, groups, items, single-weight)."""
    items = [g for v in groups.values() for g in v]
    logs = [math.log(g) for g in items]
    grand = statistics.fmean(logs)
    total = sum((x - grand) ** 2 for x in logs)
    within = 0.0
    for v in groups.values():
        lg = [math.log(g) for g in v]
        m = statistics.fmean(lg)
        within += sum((x - m) ** 2 for x in lg)
    single = sum(1 for v in groups.values() if len(set(v)) == 1)
    return within / total, len(groups), len(items), single


def _group(rows, keyf) -> Dict[object, List[float]]:
    grouped: Dict[object, List[float]] = defaultdict(list)
    for pid, name, grams in parse_items(rows):
        grouped[keyf(pid, name)].append(grams)
    return {k: v for k, v in grouped.items() if len(v) >= 2}


def portion_decomposition(rows: List[dict]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for label, keyf in (
        ("within_participant_x_dish", lambda p, n: (p, n)),
        ("pooled_by_dish_name", lambda p, n: n),
    ):
        share, ngroups, nitems, single = _variance_split(_group(rows, keyf))
        out[label] = {
            "portion_share": round(share, 3),
            "which_dish_share": round(1 - share, 3),
            "groups": ngroups,
            "items": nitems,
            "single_weight_share": round(single / ngroups, 3),
        }
    return out


def between_person_spread(rows: List[dict], dish: str = "rice") -> Dict[str, object]:
    """Median serving of one dish name, per participant, then the spread."""
    per: Dict[str, List[float]] = defaultdict(list)
    for pid, name, grams in parse_items(rows):
        if name == dish:
            per[pid].append(grams)
    per = {p: v for p, v in per.items() if len(v) >= 5}
    meds = sorted(statistics.median(v) for v in per.values())
    if not meds:
        return {}
    return {
        "dish": dish,
        "participants": len(per),
        "median_of_medians": statistics.median(meds),
        "min_of_medians": meds[0],
        "max_of_medians": meds[-1],
    }


# --------------------------------------------------------------------------

def main() -> None:
    rows = load_features()
    if not rows:
        raise SystemExit(f"no rows in {DEFAULT_FEATURES}")

    print("=" * 72)
    print("NLP Path A gate — measurements (docs/NLP_PATH_A_GATE.md)")
    print("=" * 72)

    r = richness(rows)
    print("\nN1  richness")
    for k, v in r.items():
        print(f"      {k:<28} {v}")

    print("\nN2  feature collapse")
    for k, v in feature_collapse(rows).items():
        print(f"      {k:<28} {v}")

    print("\nN3  nlp_confidence by (is_fried, is_large)")
    print(f"      {'flags':<28} {'rows':>6} {'mean':>8} {'share':>8}")
    for k, (n, mean, share) in confidence_by_flags(rows).items():
        print(f"      {k:<28} {n:>6} {mean:>8} {share:>8}")

    s = split_unit(rows)
    print("\nN5  split unit")
    for k, v in s.items():
        print(f"      {k:<28} {v if not isinstance(v, float) else round(v, 4)}")
    print("      NOTE: this is the leak caused by grouping on period ids,")
    print("            which is what nlp/eval_baseline.py does.")

    g = glucose_column(rows)
    print("\nN6  glucose column (no response variable)")
    for k, v in g.items():
        print(f"      {k:<28} {v}")

    pb = path_b_leakage()
    print("\nN7  Path B split integrity")
    if pb is None:
        print("      (processed splits not found)")
    else:
        for k, v in pb.items():
            print(f"      {k:<28} {v}")

    print("\nN8  portion share of log(grams) variance")
    print(f"      {'grouping':<28} {'portion':>8} {'dish':>8} {'single':>8}")
    for label, d in portion_decomposition(rows).items():
        print(
            f"      {label:<28} {d['portion_share']:>8} "
            f"{d['which_dish_share']:>8} {d['single_weight_share']:>8}"
        )
    b = between_person_spread(rows)
    if b:
        print(
            f"\n      {b['dish']}: {b['participants']} participants with >=5 servings, "
            f"median serving {b['min_of_medians']:.0f}-{b['max_of_medians']:.0f} g"
        )
    print("\n      BIG IDEAs B7 comparison: portion 0.025 (0.050 restricted),")
    print("      single-weight 0.51. Use the within-participant row for the")
    print("      like-for-like comparison; the two rows are different quantities.")
    print()


if __name__ == "__main__":
    main()
