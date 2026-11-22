import { expect, type Page } from "@playwright/test";
import { historyRecords, mockApi, loadSession, primaryTab, tapMasuk } from "./fixtures";
import { actionLanjut } from "./helpers-t3";
import { stubCamera } from "./camera";
import { deferred } from "./util";

export const FOOD = "Nasi putih";
export const LONG_FOOD = "Sop-buntut-kuah-merah-khas-Tebet-porsi-besar-dengan-kentang-wortel-dan-bawang-goreng";
export const RESULT_HEADING = "Respons gula darah Anda";
export const SUMMARY_HEADING = "Pastikan data sudah benar";

const NOW = new Date().toISOString();

export function dishRow(name: string) {
  return { id: "d-1", name, kind: "dish", serving_g: 150, energy_kcal: 200, carbs_g: 45, protein_g: 4, fat_g: 0.5, fibre_g: 0.6, confidence: 1 };
}

export function g0Reading(source: "meter" | "ppg") {
  return source === "ppg"
    ? { value_mgdl: 112, source: "ppg", timestamp: NOW, age_minutes: 0, error_range_mgdl: 25, heart_rate_bpm: 72, capture_valid: true, ppg_model_version: "ppg-v1" }
    : { value_mgdl: 110, source: "meter", timestamp: NOW, age_minutes: 0, error_range_mgdl: null, heart_rate_bpm: null, capture_valid: null, ppg_model_version: null };
}

export interface ResultOptions {
  source?: "meter" | "ppg";
  outsideTraining?: boolean;
  food?: string;
  /** "ok" answers, "hang" never answers (loading state), "fail" answers 500. */
  predict?: "ok" | "hang" | "fail";
}

export function predictResponse({ source = "meter", outsideTraining = true, food = FOOD }: ResultOptions = {}) {
  const item = { row: dishRow(food), grams: 150, source: "manual", carbs_g: 45, protein_g: 4, fat_g: 0.5, fibre_g: 0.6, kcal: 200 };
  return {
    schema_version: "1",
    g0: g0Reading(source),
    meal: {
      items: [item],
      totals: { carbs_g: 45, protein_g: 4, fat_g: 0.5, fibre_g: 0.6, kcal: 200 },
      dish_table_build: "t6",
      vlm_model_version: null,
    },
    forecast: {
      t60: { glucose_mgdl: 150, range: "borderline", error_band_mgdl: 29, model_version: "lgbm-t6" },
      t120: { glucose_mgdl: 128, range: "normal", error_band_mgdl: 31, model_version: "lgbm-t6" },
    },
    outside_training: outsideTraining,
    disclaimer: "Prototipe riset — bukan alat medis. Perkiraan ini tidak boleh dipakai untuk keputusan pengobatan, dosis insulin, atau pengaturan obat.",
  };
}

/** Signed-in home with optional saved history. */
export async function signedInHome(page: Page, history = historyRecords()) {
  await mockApi(page, { history });
  await loadSession(page, () => page.goto("/"));
  await tapMasuk(page);
}

async function mockFoodAndPredict(page: Page, opts: ResultOptions) {
  const food = opts.food ?? FOOD;
  await page.route("**/dishes**", (r) => r.fulfill({ json: { items: [dishRow(food)] } }));
  await page.route("**/predict", async (r) => {
    if (opts.predict === "hang") { const gate = deferred(); setTimeout(gate.resolve, 1500); await gate.promise; }
    if (opts.predict === "fail") return r.fulfill({ status: 500, json: { detail: "Model sedang tidak tersedia." } });
    return r.fulfill({ json: predictResponse(opts) });
  });
}

/** Meal step from the summary: manual entry of one dish, then "Lihat prediksi" is pressed. */
async function addDishAndPredict(page: Page, food: string) {
  await actionLanjut(page).click();
  await page.getByRole("button", { name: "Masukkan makanan secara manual" }).click();
  await page.getByPlaceholder("Cari atau ketik nama makanan…").fill(food.slice(0, 4));
  await page.getByRole("button", { name: food.slice(0, 12) }).click();
  await page.getByRole("button", { name: "Lihat prediksi", exact: true }).click();
}

/** Fresh prediction from a glucometer reading (no camera). Stops after pressing "Lihat prediksi". */
export async function startFreshPrediction(page: Page, opts: ResultOptions = {}) {
  await signedInHome(page, []);
  await mockFoodAndPredict(page, opts);
  await primaryTab(page, "Prediksi").click();
  await page.getByRole("button", { name: /^Glukometer/ }).click();
  await page.getByLabel("Gula darah").fill("110");
  await actionLanjut(page).click();
  await expect(page.getByRole("heading", { name: SUMMARY_HEADING })).toBeVisible();
  await addDishAndPredict(page, opts.food ?? FOOD);
}

export async function openFreshResult(page: Page, opts: ResultOptions = {}) {
  await startFreshPrediction(page, opts);
  await expect(page.getByRole("heading", { name: RESULT_HEADING })).toBeVisible();
}

/** Finger scan (PPG) from the Source screen with the camera stub. Stops after pressing "Lihat prediksi". */
export async function startPpgPrediction(page: Page, opts: ResultOptions = {}) {
  await stubCamera(page, { width: 160, height: 120, torch: "works", pulse: true });
  await signedInHome(page, []);
  await mockFoodAndPredict(page, { ...opts, source: "ppg" });
  await page.route("**/glucose/ppg", (r) => r.fulfill({ json: g0Reading("ppg") }));
  await primaryTab(page, "Prediksi").click();
  await page.getByRole("button", { name: /^Scan Jari/ }).click();
  await page.getByRole("button", { name: "Mulai pindai" }).click();
  await expect(page.getByRole("button", { name: "Mulai merekam" })).toBeEnabled({ timeout: 15_000 });
  await page.getByRole("button", { name: "Mulai merekam" }).click();
  await expect(page.getByRole("button", { name: "Gunakan hasil" })).toBeVisible({ timeout: 45_000 });
  await page.getByRole("button", { name: "Gunakan hasil" }).click();
  await expect(page.getByRole("heading", { name: SUMMARY_HEADING })).toBeVisible();
  await addDishAndPredict(page, opts.food ?? FOOD);
}

/** Opens the Riwayat entry at `index` (history comes from signedInHome). */
export async function openHistoryResult(page: Page, index: number) {
  await primaryTab(page, "Riwayat").click();
  await page.locator(".history-item").nth(index).click();
  await expect(page.getByRole("heading", { name: RESULT_HEADING })).toBeVisible();
}
