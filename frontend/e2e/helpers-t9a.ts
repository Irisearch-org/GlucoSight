import { expect, type Locator, type Page } from "@playwright/test";

/** Desktop widths the scale has to hold at (issue #41). */
export const DESKTOP_WIDTHS = [1280, 1440] as const;

/**
 * Rendered sizes measured on the pre-rem build (zoom: 1.15). The rem build must
 * reproduce them within DESKTOP_TOLERANCE px, so text and layout stay the same size.
 */
export const DESKTOP_TOLERANCE = 2;

export const ANCHORS = {
  1280: {
    pageWidth: 820,
    homeCtaWidth: 783.2,
    homeCtaHeight: 164.4,
    sourceOptionWidth: 253.4,
    sourceOptionHeight: 146.5,
    profileFieldHeight: 52.6,
    historyItemHeight: 105.5,
    resultCardWidth: 783.2,
    resultCardHeight: 210.1,
  },
  1440: {
    pageWidth: 920,
    homeCtaWidth: 883.2,
    homeCtaHeight: 164.4,
    sourceOptionWidth: 286.7,
    sourceOptionHeight: 114.5,
    profileFieldHeight: 52.6,
    historyItemHeight: 105.5,
    resultCardWidth: 883.2,
    resultCardHeight: 210.1,
  },
} as const;

/** Asserts a rendered box size against an anchor, within DESKTOP_TOLERANCE. */
export async function expectSize(locator: Locator, expected: { width?: number; height?: number }) {
  const box = await locator.boundingBox();
  expect(box, "element has a rendered box").not.toBeNull();
  if (expected.width !== undefined) expect(Math.abs(box!.width - expected.width)).toBeLessThanOrEqual(DESKTOP_TOLERANCE);
  if (expected.height !== undefined) expect(Math.abs(box!.height - expected.height)).toBeLessThanOrEqual(DESKTOP_TOLERANCE);
}

/** Computed zoom and root font-size, as the browser reports them. */
export async function scaleState(page: Page, selector: string) {
  return page.evaluate((sel) => {
    const el = document.querySelector(sel);
    return {
      zoom: el ? getComputedStyle(el).zoom : null,
      rootFontSize: getComputedStyle(document.documentElement).fontSize,
    };
  }, selector);
}

/** True when nothing on the page is wider than the viewport. */
export async function hasHorizontalOverflow(page: Page) {
  return page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
}
