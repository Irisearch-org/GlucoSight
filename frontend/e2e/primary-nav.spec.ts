import { expect, test, type Page } from "@playwright/test";
import {
  backdropRgb, box, contrastRatio, loadSession, openHome, parseRgb, PRIMARY_NAV, primaryTab, tapMasuk,
  withViewport,
} from "./fixtures";

declare global {
  interface Window {
    __pillRuns?: number;
    __pillTops?: number[];
    __pillSampled?: (value?: unknown) => void;
  }
}

const TABS = ["Beranda", "Prediksi", "Riwayat", "Profil"];
const DESKTOP_WIDTHS = [1440, 1280];
const PHONE_WIDTHS = [320, 360, 390, 430, 768];

const navOf = (page: Page) => page.locator(PRIMARY_NAV);
// The highlight is the only direct child of the nav marked aria-hidden.
const pillOf = (page: Page) => navOf(page).locator(':scope > [aria-hidden="true"]');
const labelOf = (page: Page, name: string) => primaryTab(page, name).getByText(name, { exact: true });
const toggleOf = (page: Page) => navOf(page).getByRole("button", { name: /^(Buka|Tutup) menu$/ });
const noHorizontalOverflow = (page: Page) =>
  page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);

test.describe("Primary nav on desktop", () => {
  for (const width of DESKTOP_WIDTHS) {
    for (const state of ["open", "closed"] as const) {
      const label = state === "open" ? "open" : "rail";
      test(`${width}px ${label}: items fit, do not overlap, share a pitch, highlight tracks Beranda, label contrast passes`, async ({ browser }) => {
        await withViewport(browser, { width, height: 900 }, async (page) => {
          await openHome(page, { sidebar: state });
          const navBox = await box(navOf(page));
          const items = await Promise.all(TABS.map((t) => box(primaryTab(page, t))));

          for (const b of items) {
            expect(b.x).toBeGreaterThanOrEqual(navBox.x);
            expect(b.x + b.width).toBeLessThanOrEqual(navBox.x + navBox.width);
            expect(b.y).toBeGreaterThanOrEqual(navBox.y);
            expect(b.y + b.height).toBeLessThanOrEqual(navBox.y + navBox.height);
            expect(b.height).toBeGreaterThanOrEqual(44);
          }
          for (let i = 0; i < items.length - 1; i++) {
            expect(items[i].y + items[i].height).toBeLessThanOrEqual(items[i + 1].y);
          }
          const pitches = items.slice(1).map((b, i) => b.y - items[i].y);
          expect(Math.max(...pitches) - Math.min(...pitches)).toBeLessThanOrEqual(2);

          // The highlight glides between screens, so poll until it settles.
          await expect
            .poll(async () => {
              const p = await box(pillOf(page));
              const a = await box(primaryTab(page, "Beranda"));
              return Math.abs(p.y + p.height / 2 - (a.y + a.height / 2));
            })
            .toBeLessThanOrEqual(2);

          // The highlight is the non-colour indicator on desktop; labels stay in place.
          for (const t of TABS) {
            const w = (await box(labelOf(page, t))).width;
            if (state === "closed") expect(w).toBeLessThanOrEqual(2);
            else expect(w).toBeGreaterThan(20);
          }

          const p = await box(pillOf(page));
          const bg = await backdropRgb(page, p.x + 6, p.y + p.height / 2);
          const fg = parseRgb(await labelOf(page, "Beranda").evaluate((el) => getComputedStyle(el).color));
          expect(contrastRatio(fg, bg)).toBeGreaterThanOrEqual(4.5);

          expect(await noHorizontalOverflow(page)).toBe(true);
        });
      });
    }
  }

  test("first visit: open at 1280px, rail at 1000px when nothing is saved", async ({ browser }) => {
    await withViewport(browser, { width: 1280, height: 800 }, async (page) => {
      await openHome(page);
      await expect(toggleOf(page)).toHaveAttribute("aria-expanded", "true");
    });
    await withViewport(browser, { width: 1000, height: 800 }, async (page) => {
      await openHome(page);
      await expect(toggleOf(page)).toHaveAttribute("aria-expanded", "false");
    });
  });

  test("a toggled choice survives reload", async ({ browser }) => {
    await withViewport(browser, { width: 1280, height: 800 }, async (page) => {
      await openHome(page);
      await toggleOf(page).click();
      await expect(toggleOf(page)).toHaveAttribute("aria-expanded", "false");

      await loadSession(page, () => page.reload());
      await tapMasuk(page);
      await expect(toggleOf(page)).toHaveAttribute("aria-expanded", "false");

      await toggleOf(page).click();
      await expect(toggleOf(page)).toHaveAttribute("aria-expanded", "true");
      await loadSession(page, () => page.reload());
      await tapMasuk(page);
      await expect(toggleOf(page)).toHaveAttribute("aria-expanded", "true");
    });
  });

  test("first visit: the highlight is placed without a glide", async ({ browser }) => {
    await withViewport(browser, { width: 1280, height: 800 }, async (page) => {
      await page.addInitScript(() => {
        window.__pillRuns = 0;
        document.addEventListener("transitionrun", (e) => {
          if (e.target instanceof Element && e.target.classList.contains("bottom-nav__pill")) window.__pillRuns = (window.__pillRuns ?? 0) + 1;
        }, true);
      });
      await openHome(page);
      expect(await page.evaluate(() => window.__pillRuns)).toBe(0);
      const p = await box(pillOf(page));
      const a = await box(primaryTab(page, "Beranda"));
      expect(Math.abs(p.y - a.y)).toBeLessThanOrEqual(1);
    });
  });

  test("the highlight glides to the next tab without overshooting", async ({ browser }) => {
    await withViewport(browser, { width: 1280, height: 800 }, async (page) => {
      await openHome(page);
      const start = (await box(pillOf(page))).y;
      const { promise: sampled, resolve: done } = Promise.withResolvers<void>();
      await page.exposeFunction("__pillSampled", done);
      await page.evaluate(() => {
        window.__pillTops = [];
        const t0 = performance.now();
        const frame = (now: number) => {
          const pill = document.querySelector<HTMLElement>(".bottom-nav__pill");
          if (pill) window.__pillTops?.push(pill.getBoundingClientRect().top);
          if (now - t0 < 900) requestAnimationFrame(frame);
          else void window.__pillSampled();
        };
        requestAnimationFrame(frame);
      });
      await primaryTab(page, "Riwayat").click();
      await sampled;
      const target = (await box(primaryTab(page, "Riwayat"))).y;
      const tops = await page.evaluate(() => window.__pillTops ?? []);
      expect(Math.max(...tops)).toBeLessThanOrEqual(target + 1);
      expect(tops.some((t) => t > start + 1 && t < target - 1)).toBe(true);
      expect(Math.abs(tops[tops.length - 1] - target)).toBeLessThanOrEqual(1);
    });
  });

  test("resizing across the desktop breakpoint re-aligns the highlight with the active tab", async ({ browser }) => {
    await withViewport(browser, { width: 390, height: 640, mobile: true }, async (page) => {
      await openHome(page);
      await primaryTab(page, "Riwayat").click();
      await expect(page.getByText("Riwayat Prediksi", { exact: true })).toBeVisible();
      const alignedOnDesktop = async () => {
        await expect.poll(async () => {
          const p = await box(pillOf(page));
          const a = await box(primaryTab(page, "Riwayat"));
          return Math.abs(p.y + p.height / 2 - (a.y + a.height / 2));
        }).toBeLessThanOrEqual(2);
      };
      await page.setViewportSize({ width: 1280, height: 800 });
      await alignedOnDesktop();
      await page.setViewportSize({ width: 390, height: 640 });
      await page.setViewportSize({ width: 1280, height: 800 });
      await alignedOnDesktop();
    });
  });
});

test.describe("Primary nav on phones and tablet", () => {
  for (const width of PHONE_WIDTHS) {
    test(`${width}px: four equal tabs in order, >=44px tall, labels >=11px, bar <=64px, active label bolder and readable`, async ({ browser }) => {
      await withViewport(browser, { width, height: 640, mobile: true }, async (page) => {
        await openHome(page);
        const navBox = await box(navOf(page));
        expect(navBox.height).toBeLessThanOrEqual(64);

        const items = await Promise.all(TABS.map((t) => box(primaryTab(page, t))));
        const byX = TABS.map((name, i) => ({ name, b: items[i] })).sort((a, b) => a.b.x - b.b.x);
        expect(byX.map((e) => e.name)).toEqual(TABS);

        const widths = items.map((b) => b.width);
        expect(Math.max(...widths) - Math.min(...widths)).toBeLessThanOrEqual(2);
        for (let i = 0; i < byX.length - 1; i++) {
          expect(byX[i].b.x + byX[i].b.width).toBeLessThanOrEqual(byX[i + 1].b.x);
        }
        for (const b of items) {
          expect(b.x).toBeGreaterThanOrEqual(0);
          expect(b.x + b.width).toBeLessThanOrEqual(width);
          expect(b.height).toBeGreaterThanOrEqual(44);
        }

        for (const t of TABS) {
          const size = await labelOf(page, t).evaluate((el) => parseFloat(getComputedStyle(el).fontSize));
          expect(size).toBeGreaterThanOrEqual(11);
        }

        // Prediksi's label sits on the same row as the other tabs; only its disc rises above the bar's top edge.
        expect(Math.abs(items[1].y - items[0].y)).toBeLessThanOrEqual(1);

        // Active state is shown by label weight (bolder), not colour alone.
        const berandaWeight = await labelOf(page, "Beranda").evaluate((el) => Number(getComputedStyle(el).fontWeight));
        const prediksiWeight = await labelOf(page, "Prediksi").evaluate((el) => Number(getComputedStyle(el).fontWeight));
        expect(berandaWeight).toBeGreaterThan(prediksiWeight);

        const bg = await backdropRgb(page, navBox.x + navBox.width / 2, navBox.y + 2);
        const fg = parseRgb(await labelOf(page, "Beranda").evaluate((el) => getComputedStyle(el).color));
        expect(contrastRatio(fg, bg)).toBeGreaterThanOrEqual(4.5);
      });
    });

    test(`${width}px: Riwayat tab opens History`, async ({ browser }) => {
      await withViewport(browser, { width, height: 640, mobile: true }, async (page) => {
        await openHome(page);
        await primaryTab(page, "Riwayat").click();
        await expect(page.getByText("Riwayat Prediksi", { exact: true })).toBeVisible();
      });
    });

    test(`${width}px: no horizontal overflow on Home, History or Profile`, async ({ browser }) => {
      await withViewport(browser, { width, height: 640, mobile: true }, async (page) => {
        await openHome(page);
        expect(await noHorizontalOverflow(page)).toBe(true);
        for (const name of ["Riwayat", "Profil"]) {
          await primaryTab(page, name).click();
          await expect(labelOf(page, name)).toHaveCSS("font-weight", "700");
          expect(await noHorizontalOverflow(page)).toBe(true);
        }
      });
    });

    test(`${width}px: last Home content clears the bar when scrolled to the end`, async ({ browser }) => {
      await withViewport(browser, { width, height: 640, mobile: true }, async (page) => {
        await openHome(page);
        await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
        const covered = await page.evaluate((selector) => {
          const nav = document.querySelector(selector)!;
          const navTop = nav.getBoundingClientRect().top;
          // Ancestors of the nav (the shell) contain it, so only sibling content is measured.
          return [...document.querySelectorAll("#root *")]
            .filter((el) => !nav.contains(el) && !el.contains(nav))
            .filter((el) => {
              const r = el.getBoundingClientRect();
              return r.height > 0 && r.bottom > navTop + 0.5;
            })
            .map((el) => `${el.tagName.toLowerCase()}.${el.getAttribute("class") ?? ""}`);
        }, PRIMARY_NAV);
        expect(covered).toEqual([]);
      });
    });
  }

  test("390px: Prediksi opens the prediction flow without the Primary nav", async ({ browser }) => {
    await withViewport(browser, { width: 390, height: 640, mobile: true }, async (page) => {
      await openHome(page);
      await primaryTab(page, "Prediksi").click();
      await expect(navOf(page)).toHaveCount(0);
    });
  });
});
