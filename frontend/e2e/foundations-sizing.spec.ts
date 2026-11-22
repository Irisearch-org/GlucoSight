import { expect, test, type Browser, type Page } from "@playwright/test";
import { historyRecords, openHome, primaryTab, signedOut, withViewport } from "./fixtures";
import { openScan, stubCamera as stubTorchCamera } from "./helpers-t4";
import {
  MENU, SEARCH_PLACEHOLDER, reachCamera, reachConfirm, reachManual, stubCamera as stubStream, stubMeal, waitForLivePreview,
} from "./helpers-t5";
import { openFreshResult } from "./helpers-t6";
import { runningAnimations } from "./helpers-t8";

const PHONE_WIDTHS = [320, 360, 390, 430, 768];
const TEXT_WIDTH = 390;
const MIN_TARGET = 44;
const MIN_TEXT = 11;

interface Screen {
  name: string;
  open: (page: Page) => Promise<void>;
}

const home = (page: Page) => openHome(page, { history: historyRecords() });

const STREAM_PORTRAIT = { kind: "stream", width: 960, height: 1280 } as const;
const RESULT: Screen = { name: "Hasil", open: openFreshResult };

const SCREENS: Screen[] = [
  { name: "Beranda", open: home },
  {
    name: "Riwayat",
    open: async (page) => {
      await home(page);
      await primaryTab(page, "Riwayat").click();
      await expect(page.getByText("Riwayat Prediksi", { exact: true })).toBeVisible();
    },
  },
  {
    name: "Profil",
    open: async (page) => {
      await home(page);
      await primaryTab(page, "Profil").click();
      await expect(page.getByText("Nama lengkap", { exact: true })).toBeVisible();
    },
  },
  {
    name: "Prediksi",
    open: async (page) => {
      await home(page);
      await primaryTab(page, "Prediksi").click();
      await expect(page.getByRole("heading", { name: "Dari mana data gula darah Anda?" })).toBeVisible();
    },
  },
  {
    name: "Ringkasan",
    open: async (page) => {
      await home(page);
      await primaryTab(page, "Prediksi").click();
      await page.getByLabel("Gula darah").fill("110");
      await page.getByRole("button", { name: "Lanjut", exact: true }).click();
      await expect(page.getByRole("heading", { name: "Pastikan data sudah benar" })).toBeVisible();
    },
  },
  {
    name: "Masuk",
    open: async (page) => {
      await signedOut(page);
      await page.goto("/");
      await page.getByRole("button", { name: "Masuk", exact: true }).click();
      await expect(page.getByText("Kata sandi", { exact: true })).toBeVisible();
    },
  },
  {
    name: "Daftar",
    open: async (page) => {
      await signedOut(page);
      await page.goto("/");
      await page.getByRole("button", { name: "Masuk", exact: true }).click();
      await page.locator(".auth-switch").getByRole("button", { name: "Daftar", exact: true }).click();
      await expect(page.getByText("Konfirmasi kata sandi", { exact: true })).toBeVisible();
    },
  },
  RESULT,
  {
    name: "Pindai",
    open: async (page) => {
      await stubTorchCamera(page, "torch");
      await openScan(page);
      await page.getByRole("button", { name: "Mulai pindai" }).click();
      await expect(page.getByRole("button", { name: "Mulai merekam" })).toBeEnabled({ timeout: 15_000 });
    },
  },
  {
    name: "Kamera",
    open: async (page) => {
      await stubStream(page, STREAM_PORTRAIT);
      await stubMeal(page);
      await reachCamera(page);
      await waitForLivePreview(page);
    },
  },
  {
    name: "Konfirmasi",
    open: async (page) => {
      await stubStream(page, STREAM_PORTRAIT);
      await stubMeal(page);
      await reachConfirm(page);
    },
  },
  {
    name: "Manual",
    open: async (page) => {
      await stubStream(page, STREAM_PORTRAIT);
      await stubMeal(page);
      await reachManual(page);
      for (const dish of MENU.slice(0, 2)) {
        await page.getByPlaceholder(SEARCH_PLACEHOLDER).fill(dish.name);
        await page.getByRole("button", { name: dish.name, exact: false }).first().click();
      }
    },
  },
];

// Runs in the page: every visible control must be at least 44x44 CSS px.
function touchOffenders(page: Page): Promise<string[]> {
  return page.evaluate((min) => {
    const out: string[] = [];
    const controls = document.querySelectorAll<HTMLElement>(
      'button, a[href], input:not([type="hidden"]), select, textarea, [role="button"], [role="tab"]',
    );
    for (const el of controls) {
      const r = el.getBoundingClientRect();
      if (el.getClientRects().length === 0 || getComputedStyle(el).visibility !== "visible") continue;
      if (r.width === 0 || r.height === 0) continue;
      if (r.width < min || r.height < min) {
        const name = el.getAttribute("aria-label") || el.textContent?.trim().slice(0, 30) || el.tagName;
        out.push(`${name} (${Math.round(r.width)}x${Math.round(r.height)})`);
      }
    }
    return [...new Set(out)];
  }, MIN_TARGET);
}

// Runs in the page: every visible run of text must be at least `min` CSS px.
function textSizeOffenders(page: Page, min: number): Promise<string[]> {
  return page.evaluate(([floor]) => {
    const out = new Set<string>();
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const text = node.textContent?.trim() ?? "";
      const el = node.parentElement;
      if (!text || !el || el.closest("script, style")) continue;
      if (el.getClientRects().length === 0 || getComputedStyle(el).visibility !== "visible") continue;
      const size = parseFloat(getComputedStyle(el).fontSize);
      if (size < floor) out.add(`"${text.slice(0, 30)}" ${size}px`);
    }
    return [...out];
  }, [min] as const);
}

// Runs in the page: WCAG 2.x contrast of every visible, enabled run of text against the
// colour it actually sits on. Translucent text is blended over its backdrop, gradient
// backgrounds are checked at every colour stop, and the worst case is reported.
function contrastOffenders(page: Page, scope = "body"): Promise<string[]> {
  return page.evaluate((scope) => {
    type RGBA = [number, number, number, number];
    const parse = (s: string): RGBA | null => {
      const m = s.match(/rgba?\(([^)]+)\)/);
      if (!m) return null;
      const p = m[1].split(/[\s,/]+/).filter(Boolean).map(Number);
      return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1];
    };
    const blend = (top: RGBA, bottom: number[]): number[] =>
      [0, 1, 2].map((i) => top[i] * top[3] + bottom[i] * (1 - top[3]));
    const luminance = (c: number[]) => {
      const [r, g, b] = c.map((v) => {
        const s = v / 255;
        return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const ratio = (a: number[], b: number[]) => {
      const la = luminance(a);
      const lb = luminance(b);
      return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
    };

    // Backdrops under an element, one per gradient stop (or one when solid).
    const backdrops = (el: HTMLElement): number[][] => {
      type Layer = { color?: RGBA; stops?: RGBA[] };
      const layers: Layer[] = [];
      let opaque = false;
      for (let n: HTMLElement | null = el; n && !opaque; n = n.parentElement) {
        const cs = getComputedStyle(n);
        if (cs.backgroundImage.includes("gradient")) {
          const stops = (cs.backgroundImage.match(/rgba?\([^)]*\)/g) ?? []).map(parse).filter((c): c is RGBA => !!c);
          if (stops.length) {
            layers.push({ stops });
            opaque = true;
          }
        }
        const c = parse(cs.backgroundColor);
        if (c && c[3] > 0) {
          layers.push({ color: c });
          if (c[3] >= 1) opaque = true;
        }
      }
      let candidates: number[][] = [[255, 255, 255]];
      for (let i = layers.length - 1; i >= 0; i--) {
        const layer = layers[i];
        if (layer.color) candidates = candidates.map((b) => blend(layer.color!, b));
        else candidates = layer.stops!.flatMap((s) => candidates.map((b) => blend(s, b)));
      }
      return candidates;
    };

    const out = new Set<string>();
    const roots = [...document.querySelectorAll<HTMLElement>(scope)];
    if (roots.length === 0) out.add(`no element matches ${scope}`);
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const seen = new Set<HTMLElement>();
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const text = node.textContent?.trim() ?? "";
      const el = node.parentElement;
      if (!text || !el || seen.has(el) || el.closest("script, style, option") || !roots.some((root) => root.contains(el))) continue;
      seen.add(el);
      if (el.closest("button:disabled, input:disabled, select:disabled, textarea:disabled, [aria-disabled='true']")) continue;
      if (el.getClientRects().length === 0 || getComputedStyle(el).visibility !== "visible") continue;
      const cs = getComputedStyle(el);
      const fg = parse(cs.color);
      if (!fg) continue;
      let alpha = fg[3];
      for (let n: HTMLElement | null = el; n; n = n.parentElement) alpha *= parseFloat(getComputedStyle(n).opacity);
      const size = parseFloat(cs.fontSize);
      const large = size >= 24 || (size >= 18.66 && parseInt(cs.fontWeight, 10) >= 700);
      const needed = large ? 3 : 4.5;
      const worst = Math.min(
        ...backdrops(el).map((bg) => ratio(blend([fg[0], fg[1], fg[2], alpha], bg), bg)),
      );
      if (worst < needed) out.add(`"${text.slice(0, 30)}" ${worst.toFixed(2)}:1 (need ${needed})`);
    }
    return [...out];
  }, scope);
}

async function eachScreen(browser: Browser, width: number, screen: Screen, run: (page: Page) => Promise<void>) {
  await withViewport(browser, { width, height: 800, mobile: true }, async (page) => {
    await screen.open(page);
    await run(page);
  });
}

test.describe("Touch targets", () => {
  for (const width of PHONE_WIDTHS) {
    for (const screen of SCREENS) {
      test(`${screen.name} at ${width}px: every visible control is at least ${MIN_TARGET}x${MIN_TARGET}`, async ({ browser }) => {
        await eachScreen(browser, width, screen, async (page) => {
          await expect.poll(() => runningAnimations(page)).toBe(0);
          expect(await touchOffenders(page)).toEqual([]);
        });
      });
    }
  }
});

test.describe("Text size", () => {
  for (const screen of SCREENS) {
    test(`${screen.name} at ${TEXT_WIDTH}px: visible text is at least ${MIN_TEXT}px`, async ({ browser }) => {
      await eachScreen(browser, TEXT_WIDTH, screen, async (page) => {
        expect(await textSizeOffenders(page, MIN_TEXT)).toEqual([]);
      });
    });
  }
});

test.describe("Contrast", () => {
  for (const screen of SCREENS) {
    test(`${screen.name} at ${TEXT_WIDTH}px: visible text meets WCAG AA contrast`, async ({ browser }) => {
      await eachScreen(browser, TEXT_WIDTH, screen, async (page) => {
        expect(await contrastOffenders(page)).toEqual([]);
      });
    });
  }
});

test.describe("Result cards", () => {
  test(`Hasil at ${TEXT_WIDTH}px: result numbers, badges, tip cards and disclaimer meet WCAG AA contrast`, async ({ browser }) => {
    await eachScreen(browser, TEXT_WIDTH, RESULT, async (page) => {
      expect(await contrastOffenders(page, ".result-number, .badge, .tip-card, .disclaimer")).toEqual([]);
    });
  });
});
