import { expect, type Page } from "@playwright/test";
import { records } from "./helpers-t7";

/** Tablet portrait range: one centred column, 24px side padding. */
export const TABLET_WIDTHS = [600, 768, 820] as const;
/** Wide phones: padding only, the column stays at the phone maximum. */
export const WIDE_PHONE_WIDTHS = [481, 599] as const;
/** Column maximum from 600px up. */
export const COLUMN_MAX = 640;
/** Phone maximum column width (base `.shell`). */
export const PHONE_COLUMN_MAX = 480;
/** Four records, so the history list has a second row. */
export const FOUR_RECORDS = () => records(4);

export interface ColumnBox {
  x: number;
  width: number;
  viewport: number;
}

/** Content box of the page column (`.shell` has no padding of its own). */
export function column(page: Page): Promise<ColumnBox> {
  return page.locator(".shell").first().evaluate((el) => {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    const left = r.left + parseFloat(cs.paddingLeft) + parseFloat(cs.borderLeftWidth);
    const right = r.right - parseFloat(cs.paddingRight) - parseFloat(cs.borderRightWidth);
    return { x: left, width: right - left, viewport: window.innerWidth };
  });
}

/** The column is min(viewport, max) wide, sits in the middle of the viewport, and nothing overflows. */
export async function expectColumn(page: Page, max: number) {
  const c = await column(page);
  expect(Math.abs(c.width - Math.min(c.viewport, max))).toBeLessThanOrEqual(1);
  const right = c.viewport - (c.x + c.width);
  expect(Math.abs(c.x - right)).toBeLessThanOrEqual(2);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
}
