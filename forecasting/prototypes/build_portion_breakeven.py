"""Build the shareable demo for docs/BIGIDEAS_MEAL_RESPONSE_FINDINGS.md.

Emits ``portion-breakeven.html``: one self-contained file, no server, no
network. Open it by double-click.

The page answers the question "what does the CV track have to achieve for the
meal to be worth including in the glucose forecast?" by recomputing the real
leave-one-out model in the browser over the actual BIG IDEAs meals.

The data is embedded because a page that fetches a sibling JSON file cannot be
opened from ``file://`` in most browsers, and "double-click it" is the whole
point. Regenerate with::

    python -m forecasting.prototypes.build_portion_breakeven \
        --data-dir ~/Downloads/Bigideas-dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from forecasting.data.bigideas import default_data_dir, load_cohort
from forecasting.evaluate.bigideas_response import (
    _cohort_estimates,
    _load_items,
    _loo_bucket_estimates,
    _textbook_lookup,
    PRIMARY_HORIZON,
    MIN_MEALS_PER_PARTICIPANT,
)

TEMPLATE = Path(__file__).with_name("portion-breakeven.template.html")


def _rounded(values) -> list[float]:
    """One decimal — the source is a food log, not a scale."""
    return [round(float(value), 1) for value in values]


def collect(data_dir: Path, horizon: int = PRIMARY_HORIZON) -> dict:
    """Per-meal g0, true carbohydrate and rise, plus four carb estimates."""
    cohort, problems = load_cohort(data_dir)
    items = _load_items(data_dir, cohort)
    target = f"glucose_{horizon}_mgdl"

    participants = []
    for participant_id in sorted(cohort.participant_id.unique()):
        usable = cohort[cohort.participant_id == participant_id].dropna(
            subset=[target, "g0_mgdl", "carbs_g"])
        if len(usable) < MIN_MEALS_PER_PARTICIPANT:
            continue
        mine = items[items.participant_id == participant_id]
        others = items[items.participant_id != participant_id]

        self_one = _loo_bucket_estimates(mine, 1)
        self_three = _loo_bucket_estimates(mine, 3)
        _, per_food, global_median = _textbook_lookup(others, 1)
        cold = np.array([float(per_food.get(item.food_key, global_median))
                         for item in mine.itertuples(index=False)])
        cold_one = _cohort_estimates(mine, others, 1)
        self_one = np.where(np.isfinite(self_one), self_one, cold)
        self_three = np.where(np.isfinite(self_three), self_three, cold)

        mine = mine.assign(
            self1=self_one, self3=self_three, cohort1=cold_one)
        meals = (
            mine.groupby("t0")
            .agg(true_carbs_g=("carbs_g", "sum"),
                 self1=("self1", "sum"),
                 self3=("self3", "sum"),
                 cohort1=("cohort1", "sum"))
            .reset_index()
        )

        # Subset the drop: `usable` carries nutrient columns that are NaN for
        # exports which omit them (003 ships 11 columns). A bare dropna()
        # deletes those participants entirely.
        joined = usable.merge(meals, on="t0", how="inner").dropna(
            subset=[target, "g0_mgdl", "true_carbs_g", "self1", "self3",
                    "cohort1"])
        if len(joined) < MIN_MEALS_PER_PARTICIPANT:
            continue

        participants.append({
            "id": participant_id,
            "g0": _rounded(joined.g0_mgdl),
            "true_carbs": _rounded(joined.true_carbs_g),
            "delta": _rounded(joined[target] - joined.g0_mgdl),
            "estimates": {
                "oracle": _rounded(joined.true_carbs_g),
                "self1": _rounded(joined.self1.clip(lower=0)),
                "self3": _rounded(joined.self3.clip(lower=0)),
                "cohort1": _rounded(joined.cohort1.clip(lower=0)),
            },
        })

    return {
        "source": "BIG IDEAs v1.1.2 (ODC-BY 1.0)",
        "meals": int(sum(len(p["g0"]) for p in participants)),
        "participants": participants,
        "problems": problems,
    }


def build(data_dir: Path, output: Path) -> Path:
    payload = collect(data_dir)
    html = TEMPLATE.read_text().replace(
        "__DATA__", json.dumps(payload, separators=(",", ":")))
    output.write_text(html)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("portion-breakeven.html"))
    arguments = parser.parse_args(argv)
    path = build(arguments.data_dir, arguments.output)
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
