import { expect, type Locator, type Page } from "@playwright/test";
import { box, openHome, primaryTab } from "./fixtures";
import { overflowReport } from "./geometry";

export const SHORT_PHONES = [
  { width: 320, height: 480 },
  { width: 360, height: 560 },
  { width: 390, height: 640 },
  { width: 430, height: 700 },
];

export const actionLanjut = (page: Page) => page.getByRole("button", { name: "Lanjut", exact: true });

export async function openSource(page: Page) {
  await openHome(page);
  await primaryTab(page, "Prediksi").click();
  await expect(page.getByRole("heading", { name: "Dari mana data gula darah Anda?" })).toBeVisible();
}

export async function enterMeter(page: Page, value = "110") {
  await page.getByRole("button", { name: /^Glukometer/ }).click();
  await page.getByLabel("Gula darah").fill(value);
}

export async function openSummary(page: Page) {
  await openSource(page);
  await enterMeter(page);
  await actionLanjut(page).click();
  await expect(page.getByRole("heading", { name: "Pastikan data sudah benar" })).toBeVisible();
}

export async function scrollToEnd(page: Page) {
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
}

export async function inViewport(page: Page, target: Locator) {
  const b = await box(target);
  const vp = page.viewportSize()!;
  return b.x >= 0 && b.y >= 0 && b.x + b.width <= vp.width && b.y + b.height <= vp.height;
}

/** True when the element's centre is hit by the element itself or one of its descendants. */
export function isHitAtCentre(target: Locator) {
  return target.evaluate((el) => {
    const r = el.getBoundingClientRect();
    const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
    return !!hit && el.contains(hit);
  });
}

export async function noOverflow(page: Page) {
  const report = await overflowReport(page);
  return report.scrollWidth <= report.viewportWidth && report.spills.length === 0;
}
