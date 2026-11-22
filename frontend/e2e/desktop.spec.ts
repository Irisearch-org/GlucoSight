import { expect, test } from "@playwright/test";
import { primaryTab, withViewport } from "./fixtures";
import { openSource, openSummary } from "./helpers-t3";
import { openHistoryResult, signedInHome } from "./helpers-t6";
import {
  ANCHORS, DESKTOP_WIDTHS, expectSize, hasHorizontalOverflow, scaleState,
} from "./helpers-t9a";

const DESKTOP_HEIGHT = 900;

for (const width of DESKTOP_WIDTHS) {
  const anchor = ANCHORS[width];

  test.describe(`desktop ${width}px`, () => {
    test("scales with rem, not zoom", async ({ browser }) => {
      await withViewport(browser, { width, height: DESKTOP_HEIGHT }, async (page) => {
        await signedInHome(page);
        const state = await scaleState(page, ".page");
        expect(state.zoom).toBe("1");
        expect(state.rootFontSize).toBe("18.4px");
      });
    });

    test("keeps the content column size", async ({ browser }) => {
      await withViewport(browser, { width, height: DESKTOP_HEIGHT }, async (page) => {
        await signedInHome(page);
        await expectSize(page.locator(".page").first(), { width: anchor.pageWidth });
        await expectSize(page.locator(".prediction-cta").first(), {
          width: anchor.homeCtaWidth, height: anchor.homeCtaHeight,
        });
      });
    });

    test("keeps source option size", async ({ browser }) => {
      await withViewport(browser, { width, height: DESKTOP_HEIGHT }, async (page) => {
        await openSource(page);
        await expectSize(page.locator(".option-card").first(), {
          width: anchor.sourceOptionWidth, height: anchor.sourceOptionHeight,
        });
      });
    });

    test("keeps profile field size", async ({ browser }) => {
      await withViewport(browser, { width, height: DESKTOP_HEIGHT }, async (page) => {
        await signedInHome(page);
        await primaryTab(page, "Profil").click();
        await expectSize(page.locator(".field__input").first(), { height: anchor.profileFieldHeight });
      });
    });

    test("keeps history item and result card sizes", async ({ browser }) => {
      await withViewport(browser, { width, height: DESKTOP_HEIGHT }, async (page) => {
        await signedInHome(page);
        await primaryTab(page, "Riwayat").click();
        await expectSize(page.locator(".history-item").first(), { height: anchor.historyItemHeight });
        await openHistoryResult(page, 0);
        await expectSize(page.locator(".result-card").first(), {
          width: anchor.resultCardWidth, height: anchor.resultCardHeight,
        });
      });
    });

    test("no horizontal overflow on main screens", async ({ browser }) => {
      await withViewport(browser, { width, height: DESKTOP_HEIGHT }, async (page) => {
        await signedInHome(page);
        expect(await hasHorizontalOverflow(page)).toBe(false);
        await openSource(page);
        expect(await hasHorizontalOverflow(page)).toBe(false);
        await openSummary(page);
        expect(await hasHorizontalOverflow(page)).toBe(false);
      });
    });
  });
}
