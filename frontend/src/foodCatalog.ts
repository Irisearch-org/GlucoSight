export interface Dish {
  name: string;
  foodClass: string;
  servingG: number;
  kcal100g: number;
  carbs100g: number;
  protein100g: number;
  fat100g: number;
  fiberRatio: number; // fiber_g = carbs_g * fiberRatio
}

export interface Nutrition {
  kcal: number;
  carbs: number;
  protein: number;
  fat: number;
  fiber: number;
}

/** Mirrors cv/data/tkpi_macro_lookup.csv — keep in sync if that file changes. */
const DISHES: Dish[] = [
  { name: "Bakso", foodClass: "bakso", servingG: 200, kcal100g: 190.0, carbs100g: 6.6, protein100g: 12.9, fat100g: 12.2, fiberRatio: 0.05 },
  { name: "Bebek Betutu", foodClass: "bebek_betutu", servingG: 150, kcal100g: 240.0, carbs100g: 2.0, protein100g: 19.0, fat100g: 17.0, fiberRatio: 0.10 },
  { name: "Gado Gado", foodClass: "gado_gado", servingG: 250, kcal100g: 137.0, carbs100g: 10.0, protein100g: 6.0, fat100g: 8.0, fiberRatio: 0.25 },
  { name: "Gudeg", foodClass: "gudeg", servingG: 200, kcal100g: 150.0, carbs100g: 17.0, protein100g: 3.5, fat100g: 8.0, fiberRatio: 0.16 },
  { name: "Nasi Goreng", foodClass: "nasi_goreng", servingG: 250, kcal100g: 186.0, carbs100g: 27.0, protein100g: 4.5, fat100g: 6.5, fiberRatio: 0.03 },
  { name: "Pempek", foodClass: "pempek", servingG: 150, kcal100g: 190.0, carbs100g: 27.0, protein100g: 9.0, fat100g: 5.0, fiberRatio: 0.02 },
  { name: "Rawon", foodClass: "rawon", servingG: 250, kcal100g: 120.0, carbs100g: 3.5, protein100g: 9.0, fat100g: 8.0, fiberRatio: 0.11 },
  { name: "Rendang", foodClass: "rendang", servingG: 100, kcal100g: 195.0, carbs100g: 3.5, protein100g: 22.6, fat100g: 10.5, fiberRatio: 0.14 },
  { name: "Sate", foodClass: "sate", servingG: 120, kcal100g: 225.0, carbs100g: 3.0, protein100g: 20.0, fat100g: 14.0, fiberRatio: 0.10 },
  { name: "Soto", foodClass: "soto", servingG: 250, kcal100g: 90.0, carbs100g: 4.0, protein100g: 7.0, fat100g: 5.0, fiberRatio: 0.10 },
];

export const DISH_CATALOG: string[] = DISHES.map((d) => d.name);

export function findDish(name: string): Dish | undefined {
  const q = name.trim().toLowerCase();
  return DISHES.find((d) => d.name.toLowerCase() === q || d.foodClass === q);
}

export function nutritionFor(dish: Dish, grams: number): Nutrition {
  const f = grams / 100;
  const carbs = dish.carbs100g * f;
  return { kcal: dish.kcal100g * f, carbs, protein: dish.protein100g * f, fat: dish.fat100g * f, fiber: carbs * dish.fiberRatio };
}

export function sumNutrition(list: Nutrition[]): Nutrition {
  return list.reduce(
    (a, n) => ({ kcal: a.kcal + n.kcal, carbs: a.carbs + n.carbs, protein: a.protein + n.protein, fat: a.fat + n.fat, fiber: a.fiber + n.fiber }),
    { kcal: 0, carbs: 0, protein: 0, fat: 0, fiber: 0 },
  );
}
