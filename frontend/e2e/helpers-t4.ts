import { expect, type Locator, type Page } from "@playwright/test";
import { openSource } from "./helpers-t3";

export type CameraMode = "torch" | "no-torch" | "denied" | "unsupported";

export const FALLBACK = "Masukkan hasil glukometer/CGM";
export const SOURCE_HEADING = "Dari mana data gula darah Anda?";

/**
 * Fake rear camera. "torch": the torch lights the canvas bright red with a 72 bpm
 * pulse (chrominance-modulated so POS can see it); "no-torch": the lens ignores the
 * torch and stays dark; "denied": getUserMedia rejects; "unsupported": no mediaDevices.
 */
export async function stubCamera(page: Page, mode: CameraMode) {
  await page.addInitScript((m: CameraMode) => {
    if (m === "unsupported") {
      Object.defineProperty(navigator, "mediaDevices", { configurable: true, get: () => undefined });
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 48;
    const ctx = canvas.getContext("2d")!;
    const stream = canvas.captureStream(30);
    const track = stream.getVideoTracks()[0];
    // WebKit hands out fresh track wrappers that lose our own properties unless one is kept alive.
    const tracks = [track];
    Object.assign(stream, { getVideoTracks: () => tracks });
    let lit = false;
    const started = performance.now();
    window.setInterval(() => {
      const s = Math.sin((2 * Math.PI * 1.2 * (performance.now() - started)) / 1000);
      const rgb = lit && m === "torch" ? [200 + 12 * s, 90 - 6 * s, 70 - 4 * s] : [25, 20, 20];
      ctx.fillStyle = `rgb(${rgb.map((v) => Math.round(v)).join(",")})`;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }, 33);
    Object.assign(track, {
      getCapabilities: () => (m === "torch" ? { torch: true } : {}),
      applyConstraints: async (c: { advanced?: { torch?: boolean }[] }) => {
        lit = !!c.advanced?.[0]?.torch;
      },
    });
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: async () => {
          if (m === "denied") throw new DOMException("Permission denied", "NotAllowedError");
          return stream;
        },
      },
    });
  }, mode);
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
