import { expect, test, type Browser, type Locator, type Page } from "@playwright/test";

export const PRIMARY_NAV = 'nav[aria-label="Menu utama"]';
export const SIDEBAR_KEY = "glucosight.v1.sidebar";

const USER = { id: 1, name: "Budi Santoso", email: "b@x.id" };
const PROFILE = {
  participantId: "P1", name: "Budi Santoso", email: "b@x.id", age: "45", gender: "Pria",
  heightCm: "170", weightKg: "70", diabetesType: "Tipe 2", defaultSource: "meter",
};

export interface MockOptions {
  history?: unknown[];
}

// Backend calls are answered at the network boundary, so no API server is needed.
export async function mockApi(page: Page, opts: MockOptions = {}) {
  await page.route("**/auth/me", (r) => r.fulfill({ json: { user: USER } }));
  await page.route("**/user/profile", (r) => r.fulfill({ json: { profile: PROFILE, saved: true } }));
  await page.route("**/user/history", (r) => r.fulfill({ json: { history: opts.history ?? [] } }));
  await page.route("**/health", (r) => r.fulfill({ json: {} }));
}

// Waits for the session and history calls so a click on Masuk never races adoptUser.
export async function loadSession(page: Page, load: () => Promise<unknown>) {
  const session = Promise.all([
    page.waitForResponse((r) => r.url().endsWith("/auth/me")),
    page.waitForResponse((r) => r.url().endsWith("/user/history")),
  ]);
  await load();
  await session;
}

export function primaryTab(page: Page, name: string) {
  return page.locator(PRIMARY_NAV).getByRole("button", { name, exact: true });
}

export async function tapMasuk(page: Page) {
  await page.getByRole("button", { name: "Masuk", exact: true }).click();
  await expect(primaryTab(page, "Beranda")).toBeVisible();
}

export async function openHome(page: Page, opts: MockOptions & { sidebar?: "open" | "closed" } = {}) {
  await mockApi(page, opts);
  await loadSession(page, () => page.goto("/"));
  if (opts.sidebar) {
    const value = opts.sidebar;
    await page.evaluate(([key, v]) => localStorage.setItem(key, v), [SIDEBAR_KEY, value]);
  }
  await tapMasuk(page);
}

// A fresh context per viewport: isMobile and hasTouch below 900px, plain desktop otherwise.
export async function withViewport(
  browser: Browser,
  size: { width: number; height: number; mobile?: boolean },
  run: (page: Page) => Promise<void>,
) {
  const baseURL = test.info().project.use.baseURL;
  const context = await browser.newContext({
    baseURL,
    viewport: { width: size.width, height: size.height },
    isMobile: !!size.mobile,
    hasTouch: !!size.mobile,
  });
  try {
    await run(await context.newPage());
  } finally {
    await context.close();
  }
}

export async function box(locator: Locator) {
  const b = await locator.boundingBox();
  if (!b) throw new Error(`no bounding box for ${locator}`);
  return b;
}

// Samples the rendered colour under one point. The screenshot is decoded in the page
// so no image library is needed.
export async function backdropRgb(page: Page, x: number, y: number): Promise<number[]> {
  const png = await page.screenshot({ clip: { x, y, width: 2, height: 2 } });
  return page.evaluate(async (src) => {
    const img = new Image();
    img.src = src;
    await img.decode();
    const canvas = document.createElement("canvas");
    canvas.width = 2;
    canvas.height = 2;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, 1, 1).data;
    return [d[0], d[1], d[2]];
  }, `data:image/png;base64,${png.toString("base64")}`);
}

export function parseRgb(css: string): number[] {
  return css.match(/[\d.]+/g)!.slice(0, 3).map(Number);
}

// WCAG 2.x contrast ratio.
export function contrastRatio(a: number[], b: number[]): number {
  const lum = (rgb: number[]) => {
    const [r, g, bl] = rgb.map((v) => {
      const s = v / 255;
      return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * bl;
  };
  const la = lum(a);
  const lb = lum(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}
