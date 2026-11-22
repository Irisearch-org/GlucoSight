import { expect, test, type Locator, type Page } from "@playwright/test";
import { box } from "./fixtures";
import { inViewport, noOverflow } from "./helpers-t3";
import { FALLBACK, SOURCE_HEADING, backgroundOf, mockPpgReading, openScan, stubCamera } from "./helpers-t4";

const WIDTHS = [320, 360, 390, 430, 768];

const fallbackButton = (page: Page) => page.getByRole("button", { name: FALLBACK });
const retryButton = (page: Page) => page.getByRole("button", { name: "Ulangi scan" });

/** True when the element's centre is hit by the element, a descendant, or an ancestor (same text block). */
function isUncovered(target: Locator) {
  return target.evaluate((el) => {
    const r = el.getBoundingClientRect();
    const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
    return !!hit && (el.contains(hit) || hit.contains(el));
  });
}

async function expectTappable(page: Page, target: Locator) {
  await expect(target).toBeVisible();
  expect(await inViewport(page, target)).toBe(true);
  expect((await box(target)).height).toBeGreaterThanOrEqual(44);
}

async function startScan(page: Page) {
  await page.getByRole("button", { name: "Mulai pindai" }).click();
}

async function recordFullScan(page: Page) {
  await page.getByRole("button", { name: "Mulai pindai" }).click();
  const record = page.getByRole("button", { name: "Mulai merekam" });
  await expect(record).toBeEnabled({ timeout: 15_000 });
  await record.click();
}

for (const width of WIDTHS) {
  test.describe(`${width}px`, () => {
    test.use({ viewport: { width, height: width === 320 ? 480 : 720 } });

    test("torch missing: manual entry is the primary action and opens with glukometer/CGM selected", async ({ page }) => {
      await stubCamera(page, "no-torch");
      await mockPpgReading(page);
      await openScan(page);

      await startScan(page);
      await expect(page.getByText(/Flash tidak menyala/)).toBeVisible();
      await expectTappable(page, fallbackButton(page));
      await expect(retryButton(page)).toBeVisible();
      expect(await noOverflow(page)).toBe(true);

      await fallbackButton(page).click();
      await expect(page.getByRole("heading", { name: SOURCE_HEADING })).toBeVisible();
      await expect(page.getByLabel("Gula darah")).toBeVisible();
    });

    test("permission denied: manual entry is primary, retry is secondary, nothing overflows", async ({ page }) => {
      await stubCamera(page, "denied");
      await mockPpgReading(page);
      await openScan(page);

      await startScan(page);
      await expect(page.getByText(/Izin kamera ditolak/)).toBeVisible();
      await expectTappable(page, fallbackButton(page));
      await expectTappable(page, retryButton(page));
      expect(await backgroundOf(fallbackButton(page))).toMatch(/^rgb\(/);
      expect(await backgroundOf(retryButton(page))).not.toEqual(await backgroundOf(fallbackButton(page)));
      expect(await noOverflow(page)).toBe(true);
    });

    test("unsupported browser: scan is disabled, manual entry is offered and opens with glukometer/CGM selected", async ({ page }) => {
      await stubCamera(page, "unsupported");
      await mockPpgReading(page);
      await openScan(page);

      await expect(page.getByText("Kamera belum dapat dipakai di halaman ini.")).toBeVisible();
      await expect(page.getByRole("button", { name: "Mulai pindai" })).toBeDisabled();
      await expectTappable(page, fallbackButton(page));
      expect(await noOverflow(page)).toBe(true);

      await fallbackButton(page).click();
      await expect(page.getByRole("heading", { name: SOURCE_HEADING })).toBeVisible();
      await expect(page.getByLabel("Gula darah")).toBeVisible();
    });

    test("camera phases fit and keep the primary action on screen", async ({ page }) => {
      await stubCamera(page, "torch");
      await mockPpgReading(page);
      await openScan(page);

      const start = page.getByRole("button", { name: "Mulai pindai" });
      await expectTappable(page, start);
      expect(await noOverflow(page)).toBe(true);

      await start.click();
      const record = page.getByRole("button", { name: "Mulai merekam" });
      await expect(record).toBeEnabled({ timeout: 15_000 });
      await expectTappable(page, record);
      expect(await inViewport(page, page.locator(".finger-guide"))).toBe(true);
      expect(await noOverflow(page)).toBe(true);

      await record.click();
      const cancel = page.getByRole("button", { name: "Batalkan" });
      await expect(cancel).toBeVisible();
      expect(await inViewport(page, cancel)).toBe(true);
      expect(await inViewport(page, page.locator(".finger-guide"))).toBe(true);
      expect(await noOverflow(page)).toBe(true);
    });
  });
}

for (const width of [320, 430]) {
  test.describe(`${width}px result`, () => {
    test.use({ viewport: { width, height: width === 320 ? 480 : 720 } });

    test("result keeps the rough estimate, error range and glucometer note visible", async ({ page }) => {
      test.setTimeout(60_000);
      await stubCamera(page, "torch");
      await mockPpgReading(page);
      await openScan(page);
      await recordFullScan(page);

      await expect(page.getByRole("heading", { name: "Pemindaian selesai" })).toBeVisible({ timeout: 45_000 });
      const honest = [
        page.getByText("Estimasi kasar dari kamera"),
        page.getByText(/≈\s*112/),
        page.getByText(/87–137/),
        page.getByText("Glukometer lebih akurat; pakai bila tersedia."),
        page.getByText("Detak jantung 72 bpm"),
      ];
      for (const line of honest) {
        await expect(line.first()).toBeVisible();
        await line.first().evaluate((el) => el.scrollIntoView({ block: "center" }));
        expect(await inViewport(page, line.first())).toBe(true);
        expect(await isUncovered(line.first())).toBe(true);
      }
      await expectTappable(page, page.getByRole("button", { name: "Gunakan hasil" }));
      expect(await noOverflow(page)).toBe(true);
    });
  });
}
