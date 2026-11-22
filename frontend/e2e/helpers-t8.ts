import { expect, test, type Browser, type Page } from "@playwright/test";
import { signedOut } from "./fixtures";

export const LANDING_WIDTHS = [320, 360, 390, 430, 768, 1280];
export const MIN_TARGET = 44;
export const MIN_TEXT = 11;

// Phones run with isMobile and touch; 768 and 1280 are plain browser windows.
export const isPhone = (width: number) => width <= 430;

// A signed-out visitor sees Landing. Each run gets a fresh context at one width.
export async function withLanding(
  browser: Browser,
  width: number,
  run: (page: Page) => Promise<void>,
  opts: { reducedMotion?: "reduce" } = {},
) {
  const baseURL = test.info().project.use.baseURL;
  const context = await browser.newContext({
    baseURL,
    viewport: { width, height: 800 },
    isMobile: isPhone(width),
    hasTouch: isPhone(width),
    reducedMotion: opts.reducedMotion ?? "no-preference",
  });
  try {
    const page = await context.newPage();
    await signedOut(page);
    await page.goto("/");
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    await run(page);
  } finally {
    await context.close();
  }
}

// Scrolls top to bottom in steps, so every reveal observer sees its section pass the viewport.
export async function scrollThrough(page: Page, step = 240) {
  const height = await page.evaluate(() => document.documentElement.scrollHeight);
  for (let y = 0; y <= height; y += step) {
    await page.evaluate((top) => window.scrollTo(0, top), y);
    await page.waitForTimeout(60);
  }
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
}

export const noHorizontalOverflow = (page: Page) =>
  page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);

export function hiddenReveals(page: Page): Promise<string[]> {
  return page.evaluate(() =>
    [...document.querySelectorAll<HTMLElement>(".lp-reveal")]
      .filter((el) => getComputedStyle(el).opacity !== "1")
      .map((el) => el.textContent?.trim().slice(0, 40) ?? el.tagName),
  );
}

export function runningAnimations(page: Page): Promise<number> {
  return page.evaluate(() =>
    document.getAnimations().filter((a) => a.playState === "running" || a.playState === "pending").length,
  );
}

// Visible links, buttons and fields smaller than 44x44 CSS px.
export function touchOffenders(page: Page): Promise<string[]> {
  return page.evaluate((min) => {
    const out: string[] = [];
    document.querySelectorAll<HTMLElement>("a[href], button, input, select, textarea, [role='button']").forEach((el) => {
      if (el.getClientRects().length === 0 || getComputedStyle(el).visibility === "hidden") return;
      const r = el.getBoundingClientRect();
      if (r.width < min || r.height < min) out.push(`${el.textContent?.trim() || el.tagName} ${Math.round(r.width)}x${Math.round(r.height)}`);
    });
    return out;
  }, MIN_TARGET);
}

// Visible text outside aria-hidden subtrees whose computed font size is below the minimum.
export function smallText(page: Page): Promise<string[]> {
  return page.evaluate((min) => {
    const out: string[] = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const text = n.textContent?.trim();
      const el = n.parentElement;
      if (!text || !el || el.closest('[aria-hidden="true"]') || el.getClientRects().length === 0) continue;
      const size = parseFloat(getComputedStyle(el).fontSize);
      if (size < min) out.push(`${text.slice(0, 30)} ${size}px`);
    }
    return out;
  }, MIN_TEXT);
}

export interface Rect { x: number; y: number; width: number; height: number }

export function overlaps(a: Rect, b: Rect) {
  return a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
}

// [r, g, b, a] from a computed colour string such as "rgba(8, 127, 131, 0.9)".
export function rgba(css: string): number[] {
  const parts = css.match(/[\d.]+/g)!.map(Number);
  return parts.length === 3 ? [...parts, 1] : parts;
}

// Text colour as it is painted over `bg`, folding in the opacity of the element and its ancestors.
export async function paintedTextOver(locator: import("@playwright/test").Locator, bg: number[]): Promise<number[]> {
  const { color, opacity } = await locator.evaluate((el) => {
    let o = 1;
    for (let e: Element | null = el; e; e = e.parentElement) o *= parseFloat(getComputedStyle(e).opacity);
    return { color: getComputedStyle(el).color, opacity: o };
  });
  const [r, g, b, a] = rgba(color);
  const alpha = a * opacity;
  return [r * alpha + bg[0] * (1 - alpha), g * alpha + bg[1] * (1 - alpha), b * alpha + bg[2] * (1 - alpha)];
}
