import { expect, test } from "@playwright/test";
import { box, withViewport } from "./fixtures";
import {
  actionLanjut, enterMeter, isHitAtCentre, inViewport, noOverflow, openSource, openSummary, scrollToEnd, SHORT_PHONES,
} from "./helpers-t3";

const PHONE_WIDTHS = [320, 360, 390, 430, 768];
const LABELS = ["Gula Darah", "Ringkasan", "Makanan", "Hasil"];
const SOURCE_HEADING = "Dari mana data gula darah Anda?";
const SUMMARY_HEADING = "Pastikan data sudah benar";

test.describe("Pinned action row on the prediction flow", () => {
  for (const { width, height } of SHORT_PHONES) {
    test(`Source ${width}x${height}: action is on screen without scrolling, and scrolled-to-end content is not covered`, async ({ browser }) => {
      await withViewport(browser, { width, height, mobile: true }, async (page) => {
        await openSource(page);
        await enterMeter(page);
        await expect(actionLanjut(page)).toBeEnabled();
        await page.evaluate(() => window.scrollTo(0, 0));
        expect(await page.evaluate(() => window.scrollY)).toBe(0);
        expect(await inViewport(page, actionLanjut(page))).toBe(true);

        await scrollToEnd(page);
        const content = page.getByText(/Harus diukur/);
        const contentBox = await box(content);
        expect(await isHitAtCentre(content)).toBe(true);
        expect(contentBox.y + contentBox.height).toBeLessThanOrEqual((await box(actionLanjut(page))).y);
      });
    });

    test(`Summary ${width}x${height}: action is on screen without scrolling, and scrolled-to-end content is not covered`, async ({ browser }) => {
      await withViewport(browser, { width, height, mobile: true }, async (page) => {
        await openSummary(page);
        expect(await page.evaluate(() => window.scrollY)).toBe(0);
        expect(await inViewport(page, actionLanjut(page))).toBe(true);

        await scrollToEnd(page);
        const content = page.locator(".tip-card");
        const contentBox = await box(content);
        expect(await isHitAtCentre(content)).toBe(true);
        expect(contentBox.y + contentBox.height).toBeLessThanOrEqual((await box(actionLanjut(page))).y);
      });
    });
  }

  test("tall phone: the row stays in the document flow with a 16-24px gap below the content", async ({ browser }) => {
    await withViewport(browser, { width: 390, height: 844, mobile: true }, async (page) => {
      await openSummary(page);
      const content = await box(page.locator(".tip-card"));
      const action = await box(actionLanjut(page));
      const gap = action.y - (content.y + content.height);
      expect(gap).toBeGreaterThanOrEqual(16);
      expect(gap).toBeLessThanOrEqual(24);
      expect(action.y + action.height).toBeLessThan(844 - 40);
    });
  });

  test("keyboard: with the viewport shrunk to 300px, the focused field and the action stay visible and uncovered", async ({ browser }) => {
    await withViewport(browser, { width: 360, height: 560, mobile: true }, async (page) => {
      await openSource(page);
      await enterMeter(page);
      const field = page.getByLabel("Gula darah");
      await field.focus();
      await page.setViewportSize({ width: 360, height: 300 });

      await expect.poll(() => inViewport(page, field)).toBe(true);
      expect(await inViewport(page, actionLanjut(page))).toBe(true);
      expect(await isHitAtCentre(field)).toBe(true);
      expect(await isHitAtCentre(actionLanjut(page))).toBe(true);
    });
  });
});

test.describe("App bar and progress at 320px", () => {
  test("all four step labels are visible, on screen, 11px or larger, and do not overlap", async ({ browser }) => {
    await withViewport(browser, { width: 320, height: 480, mobile: true }, async (page) => {
      await openSource(page);
      const boxes = [];
      for (const label of LABELS) {
        const item = page.getByText(label, { exact: true });
        await expect(item).toBeVisible();
        expect(await inViewport(page, item)).toBe(true);
        const size = await item.evaluate((el) => parseFloat(getComputedStyle(el).fontSize));
        expect(size).toBeGreaterThanOrEqual(11);
        boxes.push(await box(item));
      }
      for (let i = 0; i < boxes.length; i++) {
        for (let j = i + 1; j < boxes.length; j++) {
          const a = boxes[i];
          const b = boxes[j];
          const overlap = a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
          expect(overlap, `${LABELS[i]} overlaps ${LABELS[j]}`).toBe(false);
        }
      }
    });
  });

  test("the app bar title stays clear of the back button and on screen", async ({ browser }) => {
    await withViewport(browser, { width: 320, height: 480, mobile: true }, async (page) => {
      await openSummary(page);
      const back = await box(page.getByRole("button", { name: "Kembali" }));
      const title = await box(page.getByText("Ringkasan Data", { exact: true }));
      expect(title.x).toBeGreaterThanOrEqual(back.x + back.width);
      expect(title.x + title.width).toBeLessThanOrEqual(320);
    });
  });
});

test.describe("Step transitions", () => {
  test("entering Summary runs a short opacity and transform animation", async ({ browser }) => {
    await withViewport(browser, { width: 390, height: 844, mobile: true }, async (page) => {
      await openSource(page);
      await enterMeter(page);
      const info = await page.evaluate(async () => {
        const btn = [...document.querySelectorAll("button")].find((b) => b.textContent?.trim() === "Lanjut");
        btn!.click();
        const { promise, resolve } = Promise.withResolvers<void>();
        setTimeout(resolve, 0);
        await promise;
        const el = document.querySelector(".wizard-page") as HTMLElement;
        return el.getAnimations().map((a) => ({
          playState: a.playState,
          duration: Number(a.effect?.getTiming().duration),
          props: (a.effect as KeyframeEffect).getKeyframes().flatMap((k) => Object.keys(k)),
        }));
      });
      expect(info).toHaveLength(1);
      expect(info[0].playState).toBe("running");
      expect(info[0].duration).toBeLessThan(200);
      const motion = [...new Set(info[0].props.filter((p) => !["offset", "computedOffset", "easing", "composite"].includes(p)))].sort();
      expect(motion).toEqual(["opacity", "transform"]);
    });
  });

  test("going back and forward at once leaves one running animation and a clickable screen", async ({ browser }) => {
    await withViewport(browser, { width: 390, height: 844, mobile: true }, async (page) => {
      await openSummary(page);
      const state = await page.evaluate(async () => {
        const byText = (text: string) => [...document.querySelectorAll("button")].find((b) => (b.getAttribute("aria-label") ?? b.textContent?.trim()) === text) as HTMLButtonElement;
        byText("Kembali").click();
        const { promise, resolve } = Promise.withResolvers<void>();
        setTimeout(resolve, 0);
        await promise;
        byText("Lanjut").click();
        const pages = document.querySelectorAll(".wizard-page");
        return { pages: pages.length, anims: (pages[0] as HTMLElement).getAnimations().map((a) => a.playState) };
      });
      expect(state.pages).toBe(1);
      expect(state.anims).toEqual(["running"]);
      await expect(page.getByRole("heading", { name: SUMMARY_HEADING })).toBeVisible();

      await page.getByRole("button", { name: "Kembali" }).click();
      await expect(page.getByRole("heading", { name: SOURCE_HEADING })).toBeVisible();
    });
  });
});

test.describe("Source and Summary content", () => {
  test("the Scan option says it needs Chrome on Android", async ({ browser }) => {
    await withViewport(browser, { width: 390, height: 844, mobile: true }, async (page) => {
      await openSource(page);
      await expect(page.getByRole("button", { name: /^Scan Jari/ })).toContainText("Estimasi kasar lewat kamera ponsel · butuh Chrome di Android");
    });
  });

  test("glucose inputs use the decimal keypad", async ({ browser }) => {
    await withViewport(browser, { width: 390, height: 844, mobile: true }, async (page) => {
      await openSource(page);
      await enterMeter(page);
      await expect(page.getByLabel("Gula darah")).toHaveAttribute("inputmode", "decimal");
      await page.getByRole("button", { name: /^CGM/ }).click();
      await expect(page.getByLabel("Gula darah")).toHaveAttribute("inputmode", "decimal");
    });
  });

  test("a validation message stays visible and uncovered when the page is at its end", async ({ browser }) => {
    await withViewport(browser, { width: 360, height: 560, mobile: true }, async (page) => {
      await openSource(page);
      await enterMeter(page, "5");
      const hint = page.locator(".field-hint");
      await expect(hint).toBeVisible();
      await scrollToEnd(page);
      const hintBox = await box(hint);
      expect(await isHitAtCentre(hint)).toBe(true);
      expect(hintBox.y + hintBox.height).toBeLessThanOrEqual((await box(actionLanjut(page))).y);
    });
  });

  for (const width of PHONE_WIDTHS) {
    test(`no horizontal overflow on Source and Summary at ${width}px`, async ({ browser }) => {
      await withViewport(browser, { width, height: 800, mobile: true }, async (page) => {
        await openSource(page);
        await enterMeter(page, "5");
        expect(await noOverflow(page)).toBe(true);
        await page.getByLabel("Gula darah").fill("110");
        await actionLanjut(page).click();
        await expect(page.getByRole("heading", { name: SUMMARY_HEADING })).toBeVisible();
        expect(await noOverflow(page)).toBe(true);
      });
    });
  }
});
