import { expect, type Page } from "@playwright/test";
import type { DishRow, PhotoItem } from "../src/api";
import { actionLanjut, openSummary } from "./helpers-t3";
import { stubCamera as stubFakeCamera } from "./camera";
import { deferred } from "./util";

/** 60+ characters: the longest dish name the layout has to hold. */
export const LONG_NAME = "Sup iga sapi bumbu kuning dengan kentang, wortel dan daun bawang segar";

export function dish(name: string, serving: number | null = 150): DishRow {
  return {
    id: `id-${name}`, name, kind: "dish", serving_g: serving, energy_kcal: 200,
    carbs_g: 30, protein_g: 8, fat_g: 5, fibre_g: 2, confidence: "exact",
  };
}

/** Nineteen short names so a 20-item meal can be built by typing each one. */
export const MENU = Array.from({ length: 19 }, (_, i) => dish(`Menu ${String(i + 1).padStart(2, "0")}`));

const DISHES = [
  ...MENU, dish(LONG_NAME, 250), dish("Nasi putih", 180), dish("Tempe goreng", 60),
  dish("Sup iga sapi", 350), dish("Sup ayam kampung", 300), dish("Sup sayur bening", 300),
];

const accepted = (row: DishRow): PhotoItem => ({ vlm_name: row.name, accepted: true, row, suggestions: [], similarity: 0.93, margin: 0.2 });

/** Two sure matches and one guess the user has to pick; its chips include a long name. */
export const PHOTO_ITEMS: PhotoItem[] = [
  accepted(dish("Nasi putih", 180)),
  accepted(dish("Tempe goreng", 60)),
  {
    vlm_name: "Sop kuning", accepted: false, row: null, similarity: 0.5, margin: 0.01,
    suggestions: [dish("Sup iga sapi", 350), dish(LONG_NAME, 250), dish("Sup ayam kampung", 300)],
  },
];

/** A 20-item meal, every item sure, the last one carrying the long name. */
export const MEAL_20: PhotoItem[] = [...MENU, dish(LONG_NAME, 250)].map(accepted);

export type CameraStub =
  | { kind: "stream"; width: number; height: number }
  | { kind: "denied" }
  | { kind: "missing" };

/** Call before navigation: replaces getUserMedia with a canvas-backed stream (or a failure). */
export async function stubCamera(page: Page, mode: CameraStub) {
  const frame = { width: 640, height: 480 };
  if (mode.kind === "stream") return stubFakeCamera(page, { width: mode.width, height: mode.height, pulse: false });
  if (mode.kind === "denied") return stubFakeCamera(page, { ...frame, fail: "denied" });
  return stubFakeCamera(page, { ...frame, unsupported: true });
}

export interface MealStub {
  /** image_data_url of every /meal/photo request, in order. */
  photos: string[];
  /** Lets a held /meal/photo response through. */
  release: () => void;
}

export async function stubMeal(page: Page, items: PhotoItem[] = PHOTO_ITEMS, opts: { hold?: boolean } = {}): Promise<MealStub> {
  const photos: string[] = [];
  const { promise: gate, resolve: open } = deferred();
  if (!opts.hold) open();
  await page.route("**/meal/photo", async (route) => {
    const body: unknown = route.request().postDataJSON();
    if (typeof body === "object" && body !== null && "image_data_url" in body && typeof body.image_data_url === "string") {
      photos.push(body.image_data_url);
    }
    await gate;
    await route.fulfill({ json: { items, vlm_model_version: "vlm-test", dish_table_build: "build-test" } });
  });
  await page.route("**/dishes**", (route) => {
    const url = new URL(route.request().url());
    const q = (url.searchParams.get("q") ?? "").trim().toLowerCase();
    const limit = Number(url.searchParams.get("limit") ?? 10);
    const found = DISHES.filter((d) => d.name.toLowerCase().includes(q)).slice(0, limit);
    return route.fulfill({ json: { items: found } });
  });
  return { photos, release: () => open() };
}

/** Encodes a plain JPEG of the given size in the page, so the bytes are a real JPEG. */
export async function makeJpeg(page: Page, width: number, height: number): Promise<Buffer> {
  const url = await page.evaluate(([w, h]) => {
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d")!;
    const g = ctx.createLinearGradient(0, 0, w, h);
    g.addColorStop(0, "#c96");
    g.addColorStop(1, "#395");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
    return c.toDataURL("image/jpeg", 0.9);
  }, [width, height]);
  return Buffer.from(url.split(",")[1], "base64");
}

export async function imageSize(page: Page, dataUrl: string) {
  return page.evaluate(
    (src) =>
      new Promise<{ w: number; h: number }>((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
        img.onerror = () => reject(new Error("image did not decode"));
        img.src = src;
      }),
    dataUrl,
  );
}

/** Opens the Galeri picker and hands it a file (the picker itself is native, so it is intercepted). */
export async function pickFromGallery(page: Page, name: string, mimeType: string, buffer: Buffer) {
  const chooser = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "Galeri", exact: true }).click();
  await (await chooser).setFiles({ name, mimeType, buffer });
}

/** Summary -> Camera. */
export async function reachCamera(page: Page) {
  await openSummary(page);
  await actionLanjut(page).click();
  await expect(page.getByRole("heading", { name: "Foto makanan Anda" })).toBeVisible();
}

/** Waits until the live preview is running and has a frame to capture. */
export async function waitForLivePreview(page: Page) {
  await expect(page.getByText("Membuka kamera…")).toHaveCount(0);
  await expect.poll(() => page.evaluate(() => document.querySelector("video")?.videoWidth ?? 0)).toBeGreaterThan(0);
}

/** Camera -> shutter -> photo recognised -> Confirm. */
export async function reachConfirm(page: Page) {
  await reachCamera(page);
  await waitForLivePreview(page);
  await page.getByRole("button", { name: "Ambil foto" }).click();
  await expect(page.getByText("Makanan di piring")).toBeVisible();
}

/** Camera -> "Masukkan makanan secara manual" -> Manual. */
export async function reachManual(page: Page) {
  await reachCamera(page);
  await page.getByRole("button", { name: "Masukkan makanan secara manual" }).click();
  await expect(page.getByRole("heading", { name: "Tambahkan makanan" })).toBeVisible();
}

export const SEARCH_PLACEHOLDER = "Cari atau ketik nama makanan…";
