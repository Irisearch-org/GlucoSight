import type { DishRow } from "./api";

/** Fibre is null when any contributing row has no fibre value: missing is not zero. */
export interface Nutrition {
  kcal: number;
  carbs: number;
  protein: number;
  fat: number;
  fibre: number | null;
}

/** Per-100 g values of a dish-table row scaled to the eaten weight. */
export function nutritionFor(row: DishRow, grams: number): Nutrition {
  const f = grams / 100;
  return {
    kcal: row.energy_kcal * f, carbs: row.carbs_g * f, protein: row.protein_g * f, fat: row.fat_g * f,
    fibre: row.fibre_g === null ? null : row.fibre_g * f,
  };
}

export function sumNutrition(list: Nutrition[]): Nutrition {
  return list.reduce<Nutrition>(
    (a, n) => ({
      kcal: a.kcal + n.kcal, carbs: a.carbs + n.carbs, protein: a.protein + n.protein, fat: a.fat + n.fat,
      fibre: a.fibre === null || n.fibre === null ? null : a.fibre + n.fibre,
    }),
    { kcal: 0, carbs: 0, protein: 0, fat: 0, fibre: 0 },
  );
}
