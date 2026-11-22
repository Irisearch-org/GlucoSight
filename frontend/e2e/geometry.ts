import type { Page } from "@playwright/test";

export interface OverflowReport {
  scrollWidth: number;
  viewportWidth: number;
  /** Visible elements past the right edge of the viewport that no clipping ancestor hides (first 8). */
  spills: string[];
}

// The width comes from the test's viewport, not `innerWidth`: on mobile emulation a page whose content is wider
// than the device widens its layout viewport, so `innerWidth` would grow with the overflow it is meant to catch.
// Runs in the page, so it must stay self-contained (no references to this module).
// An element is hidden when an ancestor with overflow other than `visible` sits inside the viewport and clips it.
// Zero-size boxes are skipped: they paint nothing. `scrollWidth` still catches them when they widen the document.
export async function overflowReport(page: Page): Promise<OverflowReport> {
  const expected = page.viewportSize()?.width ?? null;
  return page.evaluate((width) => {
    const vw = width ?? window.innerWidth;
    const clipped = (el: Element) => {
      for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
        if (getComputedStyle(p).overflowX !== "visible" && p.getBoundingClientRect().right <= vw + 0.5) return true;
      }
      return false;
    };
    const spills: string[] = [];
    for (const el of Array.from(document.querySelectorAll("body *"))) {
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      if (r.right > vw + 0.5 && !clipped(el)) spills.push(`${el.tagName.toLowerCase()}.${String(el.className)} right=${Math.round(r.right)}`);
    }
    return { scrollWidth: document.documentElement.scrollWidth, viewportWidth: vw, spills: spills.slice(0, 8) };
  }, expected);
}
