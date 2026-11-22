import { expect, test, type Locator, type Page } from "@playwright/test";
import { historyRecords, box, withViewport } from "./fixtures";
import { isHitAtCentre, inViewport, noOverflow, scrollToEnd, SHORT_PHONES } from "./helpers-t3";
import {
  LONG_FOOD, RESULT_HEADING, openFreshResult, openHistoryResult, signedInHome,
  startFreshPrediction, startPpgPrediction,
} from "./helpers-t6";

const PHONE_WIDTHS = [320, 360, 390, 430, 768];
const RESEARCH = "Prototipe riset — bukan alat medis.";
const CAVEAT_TITLE = "Prediksi dimulai dari estimasi kasar";
const CAVEAT_TEXT = "Prediksi ini dimulai dari estimasi kasar kamera dan kurang andal. Gunakan glukometer bila tersedia.";
const HISTORY_LINE = "Rentang galat hanya tersedia pada prediksi baru.";
const OUTSIDE_TITLE = "Model belum dilatih untuk jenis diabetes ini";
// Vocabulary the honesty rules keep off the Result screen (no confidence %, HRV, perfusion index, quality score or GI).
// `\bGI\b` (not a bare "GI") so the range label "Tinggi" does not count.
const FORBIDDEN = /%\s*(yakin|percaya)|confidence|\bHRV\b|perfusion|kualitas sinyal|quality score|skor kualitas|\bGI\b/i;

const card = (page: Page, horizon: "T+60" | "T+120") =>
  page.getByRole("article").filter({ hasText: `Prediksi ${horizon}` });

/** Error range sits on the same line as its number (right of it, centres on one row) and inside the card. */
async function expectBandBesideNumber(page: Page, horizon: "T+60" | "T+120") {
  const c = card(page, horizon);
  const band = c.getByText(/±\s*\d+\s*mg\/dL/);
  await expect(band).toBeVisible();
  const [b, n, cb] = [await box(band), await box(c.locator(".result-number")), await box(c)];
  expect(b.x).toBeGreaterThanOrEqual(cb.x);
  expect(b.x + b.width).toBeLessThanOrEqual(cb.x + cb.width);
  expect(b.x).toBeGreaterThanOrEqual(n.x + n.width - 0.5);
  const bandMid = b.y + b.height / 2;
  expect(bandMid).toBeGreaterThan(n.y);
  expect(bandMid).toBeLessThan(n.y + n.height);
}
async function expectAboveFirstCard(page: Page, target: Locator) {
  const t = await box(target);
  const first = await box(page.getByRole("article").first());
  expect(t.y + t.height).toBeLessThanOrEqual(first.y);
}

async function expectTapTargets(page: Page) {
  const buttons = page.locator("button:visible");
  const count = await buttons.count();
  expect(count).toBeGreaterThan(0);
  for (let i = 0; i < count; i++) {
    const b = await box(buttons.nth(i));
    expect(Math.round(b.width)).toBeGreaterThanOrEqual(44);
    expect(Math.round(b.height)).toBeGreaterThanOrEqual(44);
  }
}

async function expectNoForbiddenText(page: Page) {
  const text = await page.locator("body").innerText();
  expect(text).not.toMatch(FORBIDDEN);
}

async function expectResearchDisclaimer(page: Page) {
  const line = page.getByText(RESEARCH, { exact: true });
  await expect(line).toBeVisible();
  await line.scrollIntoViewIfNeeded();
}

test.describe("Result: honesty elements and fit", () => {
  for (const width of PHONE_WIDTHS) {
    test(`Fresh meter result at ${width}px: Error range beside each number, outside-training warning above the cards, research disclaimer visible`, async ({ browser }) => {
      await withViewport(browser, { width, height: 740, mobile: true }, async (page) => {
        await openFreshResult(page, { outsideTraining: true });

        expect(await noOverflow(page)).toBe(true);
        await expectBandBesideNumber(page, "T+60");
        await expectBandBesideNumber(page, "T+120");

        // Meter reading: no rough-estimate card.
        await expect(page.getByText(CAVEAT_TITLE)).toHaveCount(0);

        const warn = page.getByText(OUTSIDE_TITLE);
        await expect(warn).toBeVisible();
        await expectAboveFirstCard(page, warn);

        await expectResearchDisclaimer(page);
        const toggle = page.getByRole("button", { name: "Lihat detail" });
        await expect(toggle).toHaveAttribute("aria-expanded", "false");
        expect((await box(toggle)).height).toBeGreaterThanOrEqual(44);
        await toggle.click();
        await expect(page.getByText("Perkiraan ini tidak boleh dipakai untuk keputusan pengobatan")).toBeVisible();
        await expect(page.getByRole("button", { name: "Sembunyikan detail" })).toHaveAttribute("aria-expanded", "true");

        for (const horizon of ["T+60", "T+120"] as const) {
          const labels = card(page, horizon).locator(".range b");
          await expect(labels).toHaveCount(3);
          const rects = [await box(labels.nth(0)), await box(labels.nth(1)), await box(labels.nth(2))];
          expect(rects[0].x + rects[0].width).toBeLessThanOrEqual(rects[1].x);
          expect(rects[1].x + rects[1].width).toBeLessThanOrEqual(rects[2].x);
        }

        await expectTapTargets(page);
        await expectNoForbiddenText(page);

        await scrollToEnd(page);
        const done = page.getByRole("button", { name: "Selesai", exact: true });
        expect(await isHitAtCentre(done)).toBe(true);
        expect(await isHitAtCentre(page.getByText(RESEARCH, { exact: true }))).toBe(true);
        expect(await noOverflow(page)).toBe(true);
      });
    });

    test(`History result at ${width}px: no bare ± and the "only on new predictions" line, rough-estimate card above the cards`, async ({ browser }) => {
      await withViewport(browser, { width, height: 740, mobile: true }, async (page) => {
        const [first, ...rest] = historyRecords();
        await signedInHome(page, [{ ...first, sourceId: "ppg" }, ...rest]);
        await openHistoryResult(page, 0);

        expect(await noOverflow(page)).toBe(true);
        await expect(page.getByText(HISTORY_LINE)).toBeVisible();
        await expectAboveFirstCard(page, page.getByText(HISTORY_LINE));
        await expect(page.getByText(CAVEAT_TITLE)).toBeVisible();
        await expectAboveFirstCard(page, page.getByText(CAVEAT_TITLE));
        await expect(page.getByText(CAVEAT_TEXT)).toBeVisible();

        // No Error range on a saved prediction: no ± in the cards.
        await expect(page.getByText(/±/)).toHaveCount(0);
        await expect(page.getByText(OUTSIDE_TITLE)).toHaveCount(0);

        await expectResearchDisclaimer(page);
        await expectTapTargets(page);
        await expectNoForbiddenText(page);
      });
    });

    test(`Loading and error states at ${width}px: no overflow, actions on screen`, async ({ browser }) => {
      await withViewport(browser, { width, height: 740, mobile: true }, async (page) => {
        await startFreshPrediction(page, { predict: "hang" });
        await expect(page.getByRole("heading", { name: "Menghitung prediksi…" })).toBeVisible();
        expect(await noOverflow(page)).toBe(true);
        await expect(page.getByRole("heading", { name: RESULT_HEADING })).toBeVisible({ timeout: 10_000 });
      });

      await withViewport(browser, { width, height: 740, mobile: true }, async (page) => {
        await startFreshPrediction(page, { predict: "fail" });
        await expect(page.getByText("Prediksi belum berhasil")).toBeVisible();
        expect(await noOverflow(page)).toBe(true);
        await expect(page.getByRole("button", { name: "Coba lagi", exact: true })).toBeVisible();
        await expect(page.getByRole("button", { name: "Periksa data", exact: true })).toBeVisible();
        await expectTapTargets(page);
      });
    });
  }

  for (const { width, height } of SHORT_PHONES.filter((p) => p.width === 320)) {
    test(`Pinned actions stay on screen at ${width}x${height}; long food name wraps`, async ({ browser }) => {
      await withViewport(browser, { width, height, mobile: true }, async (page) => {
        await openFreshResult(page, { food: LONG_FOOD });

        const done = page.getByRole("button", { name: "Selesai", exact: true });
        const again = page.getByRole("button", { name: "Prediksi lagi", exact: true });
        expect(await inViewport(page, done)).toBe(true);
        expect(await inViewport(page, again)).toBe(true);

        await scrollToEnd(page);
        expect(await inViewport(page, done)).toBe(true);
        expect(await inViewport(page, again)).toBe(true);
        expect(await isHitAtCentre(page.getByText(RESEARCH, { exact: true }))).toBe(true);

        expect(await noOverflow(page)).toBe(true);
        const food = page.locator(".inputs-used b").first();
        const fb = await box(food);
        expect(fb.x + fb.width).toBeLessThanOrEqual(width);
      });
    });
  }

  test("PPG-started result at 390px: rough-estimate card above the cards and the glucometer note shown", async ({ browser }) => {
    test.setTimeout(120_000);
    await withViewport(browser, { width: 390, height: 740, mobile: true }, async (page) => {
      await startPpgPrediction(page, { outsideTraining: false });
      await expect(page.getByRole("heading", { name: RESULT_HEADING })).toBeVisible({ timeout: 45_000 });

      await expect(page.getByText(CAVEAT_TITLE)).toBeVisible();
      await expectAboveFirstCard(page, page.getByText(CAVEAT_TITLE));
      await expect(page.getByText(CAVEAT_TEXT)).toBeVisible();
      await expectBandBesideNumber(page, "T+60");
      await expect(page.getByText(OUTSIDE_TITLE)).toHaveCount(0);
      expect(await noOverflow(page)).toBe(true);
    });
  });

  test("Meter history result at 390px: no rough-estimate card, the ‘new predictions only’ line shown", async ({ browser }) => {
    await withViewport(browser, { width: 390, height: 740, mobile: true }, async (page) => {
      await signedInHome(page, historyRecords());
      await openHistoryResult(page, 0);
      await expect(page.getByText(HISTORY_LINE)).toBeVisible();
      await expect(page.getByText(CAVEAT_TITLE)).toHaveCount(0);
      await expect(page.getByText(/±/)).toHaveCount(0);
      expect(await noOverflow(page)).toBe(true);
    });
  });

  test("Fresh result within the training range at 390px: no outside-training warning, Error range still shown", async ({ browser }) => {
    await withViewport(browser, { width: 390, height: 740, mobile: true }, async (page) => {
      await openFreshResult(page, { outsideTraining: false });
      await expect(page.getByText(OUTSIDE_TITLE)).toHaveCount(0);
      await expectBandBesideNumber(page, "T+120");
      await expectResearchDisclaimer(page);
    });
  });
});
