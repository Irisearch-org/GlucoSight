import { expect, test, type Locator } from "@playwright/test";
import { box, contrastRatio } from "./fixtures";
import {
  hiddenReveals, LANDING_WIDTHS, noHorizontalOverflow, overlaps, paintedTextOver, rgba, runningAnimations,
  scrollThrough, smallText, touchOffenders, withLanding,
} from "./helpers-t8";

// Colour stops of a CSS gradient as the browser computed them.
async function gradientStops(locator: Locator): Promise<number[][]> {
  const css = await locator.evaluate((el) => getComputedStyle(el).backgroundImage);
  return (css.match(/rgba?\([^)]*\)/g) ?? []).map(rgba);
}

test.describe("Landing layout", () => {
  for (const width of LANDING_WIDTHS) {
    test(`${width}px: no horizontal overflow, including after scrolling to the footer`, async ({ browser }) => {
      await withLanding(browser, width, async (page) => {
        expect(await noHorizontalOverflow(page)).toBe(true);
        await scrollThrough(page);
        expect(await noHorizontalOverflow(page)).toBe(true);
      });
    });

    test(`${width}px: hero copy and phone mock do not overlap; the mock is decorative`, async ({ browser }) => {
      await withLanding(browser, width, async (page) => {
        const copy = await box(page.locator(".lp-hero__copy"));
        const phone = await box(page.locator(".lp-phone"));
        expect(overlaps(copy, phone)).toBe(false);
        expect(phone.x).toBeGreaterThanOrEqual(0);
        expect(phone.x + phone.width).toBeLessThanOrEqual(width);
        await expect(page.locator(".lp-phone")).toHaveAttribute("aria-hidden", "true");
      });
    });

    test(`${width}px: every visible control is at least 44x44`, async ({ browser }) => {
      await withLanding(browser, width, async (page) => {
        expect(await touchOffenders(page)).toEqual([]);
      });
    });

    test(`${width}px: visible text is at least 11px`, async ({ browser }) => {
      await withLanding(browser, width, async (page) => {
        expect(await smallText(page)).toEqual([]);
      });
    });

    test(`${width}px: Masuk and Daftar are side by side, never overlapping, each at least 44x44`, async ({ browser }) => {
      await withLanding(browser, width, async (page) => {
        const masuk = await box(page.getByRole("button", { name: "Masuk", exact: true }));
        const daftar = await box(page.getByRole("button", { name: "Daftar", exact: true }));
        expect(overlaps(masuk, daftar)).toBe(false);
        for (const b of [masuk, daftar]) {
          expect(b.width).toBeGreaterThanOrEqual(44);
          expect(b.height).toBeGreaterThanOrEqual(44);
        }
      });
    });

    // Real clicks: an overlay on top of the button makes Playwright wait and then fail.
    test(`${width}px: tapping Daftar opens registration`, async ({ browser }) => {
      await withLanding(browser, width, async (page) => {
        await page.getByRole("button", { name: "Daftar", exact: true }).click();
        await expect(page.getByRole("heading", { name: "Buat akun Anda" })).toBeVisible();
      });
    });

    test(`${width}px: tapping Masuk opens sign-in`, async ({ browser }) => {
      await withLanding(browser, width, async (page) => {
        await page.getByRole("button", { name: "Masuk", exact: true }).click();
        await expect(page.getByRole("heading", { name: "Selamat datang kembali" })).toBeVisible();
      });
    });

    test(`${width}px: Daftar text and CTA section text pass AA contrast at every gradient stop`, async ({ browser }) => {
      await withLanding(browser, width, async (page) => {
        const daftar = page.getByRole("button", { name: "Daftar", exact: true });
        const daftarBg = rgba(await daftar.evaluate((el) => getComputedStyle(el).backgroundColor)).slice(0, 3);
        const daftarText = await paintedTextOver(daftar, daftarBg);
        expect(contrastRatio(daftarText, daftarBg)).toBeGreaterThanOrEqual(4.5);

        const cta = page.locator(".lp-cta__in");
        await cta.scrollIntoViewIfNeeded();
        await expect(cta).toHaveCSS("opacity", "1");
        const stops = await gradientStops(cta);
        expect(stops.length).toBeGreaterThanOrEqual(2);
        for (const [label, locator] of [
          ["title", cta.getByRole("heading", { level: 2 })],
          ["paragraph", cta.locator("p")],
        ] as const) {
          for (const stop of stops) {
            const bg = stop.slice(0, 3);
            const text = await paintedTextOver(locator, bg);
            expect(contrastRatio(text, bg), `CTA ${label} over ${bg}`).toBeGreaterThanOrEqual(4.5);
          }
        }
      });
    });
  }
});

test.describe("Landing scroll reveal", () => {
  for (const width of LANDING_WIDTHS) {
    test(`${width}px: every section is visible once scrolled into view`, async ({ browser }) => {
      await withLanding(browser, width, async (page) => {
        await scrollThrough(page);
        await expect.poll(() => hiddenReveals(page)).toEqual([]);
      });
    });

    test(`${width}px, reduced motion: content is visible and nothing animates`, async ({ browser }) => {
      await withLanding(browser, width, async (page) => {
        expect(await hiddenReveals(page)).toEqual([]);
        await scrollThrough(page);
        await expect.poll(() => hiddenReveals(page)).toEqual([]);
        expect(await runningAnimations(page)).toBe(0);
      }, { reducedMotion: "reduce" });
    });
  }
});
