import { expect, test, type Browser, type Page } from "@playwright/test";
import { box, openHome, PRIMARY_NAV, primaryTab, withViewport } from "./fixtures";
import { actionLanjut, openSource } from "./helpers-t3";
import { openFreshResult, signedInHome } from "./helpers-t6";
import { reachCamera, reachManual, stubCamera, waitForLivePreview } from "./helpers-t5";
import {
  column, COLUMN_MAX, expectColumn, FOUR_RECORDS, PHONE_COLUMN_MAX, TABLET_WIDTHS, WIDE_PHONE_WIDTHS,
} from "./helpers-t9b";
import { touchOffenders } from "./helpers-t8";

const TABS = ["Beranda", "Prediksi", "Riwayat", "Profil"];
const RESEARCH = "Prototipe riset — bukan alat medis.";
const SOURCE_HEADING = "Dari mana data gula darah Anda?";
const TABLET = { height: 1024, mobile: true };

const tablet = (browser: Browser, width: number, run: (page: Page) => Promise<void>) =>
  withViewport(browser, { width, ...TABLET }, run);

const navOf = (page: Page) => page.locator(PRIMARY_NAV);

/** Measures the bar at the given phone width; the tablet bar must match it. */
async function phoneNavHeight(browser: Browser): Promise<number> {
  let height = 0;
  await withViewport(browser, { width: 390, height: 640, mobile: true }, async (page) => {
    await openHome(page);
    height = (await box(navOf(page))).height;
  });
  return height;
}

test.describe("Tablet portrait (600-899px)", () => {
  for (const width of TABLET_WIDTHS) {
    test(`${width}px: Home, History, Profile and Source use one centred column of min(viewport, 640px)`, async ({ browser }) => {
      await tablet(browser, width, async (page) => {
        await signedInHome(page, FOUR_RECORDS());
        await expectColumn(page, COLUMN_MAX);
        await primaryTab(page, "Riwayat").click();
        await expect(page.locator(".history-item").first()).toBeVisible();
        await expectColumn(page, COLUMN_MAX);
        await primaryTab(page, "Profil").click();
        await expect(page.locator(".settings-card").first()).toBeVisible();
        await expectColumn(page, COLUMN_MAX);
        await primaryTab(page, "Prediksi").click();
        await expect(page.getByRole("heading", { name: SOURCE_HEADING })).toBeVisible();
        await expectColumn(page, COLUMN_MAX);
      });
    });

    test(`${width}px: pages keep 24px side padding; back button and progress share the column padding`, async ({ browser }) => {
      await tablet(browser, width, async (page) => {
        await openSource(page);
        const pad = await page.locator(".wizard-page").evaluate((el) => {
          const cs = getComputedStyle(el);
          return [cs.paddingLeft, cs.paddingRight];
        });
        expect(pad).toEqual(["24px", "24px"]);
        const progress = await page.locator(".progress-wrap").evaluate((el) => {
          const cs = getComputedStyle(el);
          return [cs.paddingLeft, cs.paddingRight];
        });
        expect(progress).toEqual(["24px", "24px"]);
        const c = await column(page);
        const back = await box(page.getByRole("button", { name: "Kembali", exact: true }));
        expect(Math.abs(back.x - (c.x + 24))).toBeLessThanOrEqual(1);
      });
    });

    test(`${width}px: History with four records shows two equal columns`, async ({ browser }) => {
      await tablet(browser, width, async (page) => {
        await signedInHome(page, FOUR_RECORDS());
        await primaryTab(page, "Riwayat").click();
        const items = page.locator(".history-item");
        await expect(items).toHaveCount(4);
        const [a, b, c] = [await box(items.nth(0)), await box(items.nth(1)), await box(items.nth(2))];
        expect(Math.abs(a.y - b.y)).toBeLessThanOrEqual(2);
        expect(b.x).toBeGreaterThanOrEqual(a.x + a.width);
        expect(Math.abs(a.width - b.width)).toBeLessThanOrEqual(2);
        expect(c.y).toBeGreaterThanOrEqual(a.y + a.height - 1);
      });
    });

    test(`${width}px: Source options fill as many 240px-min columns as fit, none overlap, all >= 44px`, async ({ browser }) => {
      await tablet(browser, width, async (page) => {
        await openSource(page);
        const list = page.locator(".option-list");
        const cards = list.locator(".option-card");
        await expect(cards).toHaveCount(3);
        const colWidth = (await box(list)).width;
        const fits = Math.floor((colWidth + 10) / 250);
        expect(fits).toBeGreaterThanOrEqual(2);
        const rects = [await box(cards.nth(0)), await box(cards.nth(1)), await box(cards.nth(2))];
        const firstRow = rects.filter((r) => Math.abs(r.y - rects[0].y) <= 2);
        expect(firstRow.length).toBe(Math.min(rects.length, fits));
        for (const r of rects) {
          expect(r.height).toBeGreaterThanOrEqual(44);
          expect(r.width).toBeGreaterThanOrEqual(240 - 1);
        }
        for (let i = 0; i < rects.length; i++) {
          for (let j = i + 1; j < rects.length; j++) {
            const overlapX = rects[i].x < rects[j].x + rects[j].width && rects[j].x < rects[i].x + rects[i].width;
            const overlapY = rects[i].y < rects[j].y + rects[j].height && rects[j].y < rects[i].y + rects[i].height;
            expect(overlapX && overlapY).toBe(false);
          }
        }
      });
    });

    test(`${width}px: Primary nav keeps the phone bar (<=420px, centred, four tabs), and content clears it at the end`, async ({ browser }) => {
      const phoneHeight = await phoneNavHeight(browser);
      await tablet(browser, width, async (page) => {
        await signedInHome(page, FOUR_RECORDS());
        const nav = await box(navOf(page));
        expect(nav.width).toBeLessThanOrEqual(420 + 0.5);
        const right = width - (nav.x + nav.width);
        expect(Math.abs(nav.x - right)).toBeLessThanOrEqual(1);
        expect(Math.abs(nav.height - phoneHeight)).toBeLessThanOrEqual(1);

        const tabs = await Promise.all(TABS.map((t) => box(primaryTab(page, t))));
        for (const b of tabs) {
          expect(b.x).toBeGreaterThanOrEqual(nav.x);
          expect(b.x + b.width).toBeLessThanOrEqual(nav.x + nav.width);
          expect(b.height).toBeGreaterThanOrEqual(44);
        }

        await primaryTab(page, "Riwayat").click();
        await expect(page.locator(".history-item").first()).toBeVisible();
        await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
        const covered = await page.evaluate((selector) => {
          const bar = document.querySelector(selector)!;
          const barTop = bar.getBoundingClientRect().top;
          // Ancestors of the bar (the shell) contain it, so only sibling content is measured.
          return [...document.querySelectorAll("#root *")]
            .filter((el) => !bar.contains(el) && !el.contains(bar))
            .filter((el) => {
              const r = el.getBoundingClientRect();
              return r.height > 0 && r.bottom > barTop + 0.5;
            })
            .map((el) => `${el.tagName.toLowerCase()}.${el.getAttribute("class") ?? ""}`);
        }, PRIMARY_NAV);
        expect(covered).toEqual([]);
      });
    });

    test(`${width}px: Camera viewfinder keeps 3:4, is at most 420px wide and centred on the column`, async ({ browser }) => {
      await tablet(browser, width, async (page) => {
        await stubCamera(page, { kind: "stream", width: 960, height: 1280 });
        await reachCamera(page);
        await waitForLivePreview(page);
        const view = await box(page.locator(".camera-view"));
        expect(Math.abs(view.width / view.height - 3 / 4)).toBeLessThanOrEqual(0.01);
        expect(view.width).toBeLessThanOrEqual(420);
        const c = await column(page);
        expect(Math.abs(view.x + view.width / 2 - (c.x + c.width / 2))).toBeLessThanOrEqual(1);
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
      });
    });

    test(`${width}px: pinned "Lihat prediksi" row spans the column; its button sits on the 24px padding`, async ({ browser }) => {
      await tablet(browser, width, async (page) => {
        await reachManual(page);
        const row = await box(page.locator(".sticky-actions"));
        const c = await column(page);
        expect(Math.abs(row.x - c.x)).toBeLessThanOrEqual(1);
        expect(Math.abs(row.width - c.width)).toBeLessThanOrEqual(1);
        const button = await box(page.getByRole("button", { name: "Lihat prediksi", exact: true }));
        expect(Math.abs(button.x - (c.x + 24))).toBeLessThanOrEqual(1);
      });
    });

    test(`${width}px: Result cards stay in one column inside the column; disclaimer visible; controls >= 44px`, async ({ browser }) => {
      await tablet(browser, width, async (page) => {
        await openFreshResult(page);
        const c = await column(page);
        const cards = page.getByRole("article");
        const count = await cards.count();
        expect(count).toBeGreaterThan(0);
        const rects = [];
        for (let i = 0; i < count; i++) rects.push(await box(cards.nth(i)));
        for (const r of rects) {
          expect(r.x).toBeGreaterThanOrEqual(c.x - 0.5);
          expect(r.x + r.width).toBeLessThanOrEqual(c.x + c.width + 0.5);
          expect(Math.abs(r.x - rects[0].x)).toBeLessThanOrEqual(1);
        }
        await expect(page.getByText(RESEARCH)).toBeVisible();
        expect(await touchOffenders(page)).toEqual([]);
      });
    });
  }

});

test.describe("Wide phones (481-599px) and the phone guard", () => {
  for (const width of WIDE_PHONE_WIDTHS) {
    test(`${width}px: 20px page padding, column stays at 480px, History stays one column`, async ({ browser }) => {
      await tablet(browser, width, async (page) => {
        await signedInHome(page, FOUR_RECORDS());
        await expectColumn(page, PHONE_COLUMN_MAX);
        const pad = await page.locator(".page").first().evaluate((el) => {
          const cs = getComputedStyle(el);
          return [cs.paddingLeft, cs.paddingRight];
        });
        expect(pad).toEqual(["20px", "20px"]);
        await primaryTab(page, "Riwayat").click();
        const items = page.locator(".history-item");
        await expect(items).toHaveCount(4);
        const [a, b] = [await box(items.nth(0)), await box(items.nth(1))];
        expect(Math.abs(a.x - b.x)).toBeLessThanOrEqual(1);
        expect(b.y).toBeGreaterThanOrEqual(a.y + a.height - 1);
      });
    });

    test(`${width}px: pinned row spans the 480px column with 20px padding`, async ({ browser }) => {
      await tablet(browser, width, async (page) => {
        await openSource(page);
        await expect(actionLanjut(page)).toBeVisible();
        const row = await box(page.locator(".sticky-actions"));
        const c = await column(page);
        expect(Math.abs(row.x - c.x)).toBeLessThanOrEqual(1);
        expect(Math.abs(row.width - c.width)).toBeLessThanOrEqual(1);
      });
    });
  }

  test("430px: phone layout is unchanged (16px page padding, full-width column)", async ({ browser }) => {
    await withViewport(browser, { width: 430, height: 932, mobile: true }, async (page) => {
      await signedInHome(page, FOUR_RECORDS());
      await expectColumn(page, PHONE_COLUMN_MAX);
      const pad = await page.locator(".page").first().evaluate((el) => {
        const cs = getComputedStyle(el);
        return [cs.paddingLeft, cs.paddingRight];
      });
      expect(pad).toEqual(["16px", "16px"]);
    });
  });
});
