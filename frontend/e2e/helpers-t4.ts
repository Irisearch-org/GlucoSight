import { expect, type Locator, type Page } from "@playwright/test";
import { openSource } from "./helpers-t3";
import { stubCamera as stubFakeCamera } from "./camera";

export type CameraMode = "torch" | "no-torch" | "denied" | "unsupported";

export const FALLBACK = "Masukkan hasil glukometer/CGM";
export const SOURCE_HEADING = "Dari mana data gula darah Anda?";

/**
 * Fake rear camera for the finger scan. "torch": the torch lights the frame red with a 72 bpm-equivalent
 * pulse; "no-torch": the lens ignores the torch; "denied": getUserMedia rejects; "unsupported": no mediaDevices.
 */
export async function stubCamera(page: Page, mode: CameraMode) {
  const frame = { width: 64, height: 48, pulse: true } as const;
  const options = {
    torch: { ...frame, torch: "works" },
    "no-torch": { ...frame, torch: "absent" },
    denied: { ...frame, fail: "denied" },
    unsupported: { width: 64, height: 48, unsupported: true },
  } as const;
  await stubFakeCamera(page, options[mode]);
}

export const PPG_READING = {
  value_mgdl: 112,
  source: "ppg",
  timestamp: "2025-01-01T00:00:00Z",
  age_minutes: 0,
  error_range_mgdl: 25,
  heart_rate_bpm: 72,
  capture_valid: true,
  ppg_model_version: "test",
};

export async function mockPpgReading(page: Page) {
  await page.route("**/glucose/ppg", (r) => r.fulfill({ json: PPG_READING }));
}

export async function openScan(page: Page) {
  await openSource(page);
  await page.getByRole("button", { name: /^Scan Jari/ }).click();
  await expect(page.getByRole("heading", { name: "Posisikan ujung jari" })).toBeVisible();
}

/** Computed background colour; a solid primary fill differs from a text/secondary button. */
export async function backgroundOf(target: Locator) {
  return target.evaluate((el) => getComputedStyle(el).backgroundColor);
}
