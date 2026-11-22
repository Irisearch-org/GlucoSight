import { expect, test, type Page } from "@playwright/test";
import { box, withViewport } from "./fixtures";
import { inViewport, isHitAtCentre, noOverflow, scrollToEnd } from "./helpers-t3";
import {
  LONG_NAME, MEAL_20, MENU, PHOTO_ITEMS, SEARCH_PLACEHOLDER, imageSize, makeJpeg, pickFromGallery, reachCamera,
  reachConfirm, reachManual, stubCamera, stubMeal, waitForLivePreview,
} from "./helpers-t5";

const PHONES = [
  { width: 320, height: 568 },
  { width: 360, height: 640 },
  { width: 390, height: 844 },
  { width: 430, height: 932 },
  { width: 768, height: 1024 },
];
const STREAM_PORTRAIT = { kind: "stream", width: 960, height: 1280 } as const;
const TAP = 44;
const SEARCH = (page: Page) => page.getByPlaceholder(SEARCH_PLACEHOLDER);
const PRIMARY = (page: Page) => page.getByRole("button", { name: "Lihat prediksi", exact: true });
const shutter = (page: Page) => page.getByRole("button", { name: "Ambil foto", exact: true });
const manualButton = (page: Page) => page.getByRole("button", { name: "Masukkan makanan secara manual", exact: true });

// The viewfinder box must have the live stream's own shape, so nothing is cropped.
async function expectViewfinderHoldsStream(page: Page) {
  const view = await box(page.locator(".camera-view"));
  const stream = await page.evaluate(() => {
    const v = document.querySelector("video");
    return v ? v.videoWidth / v.videoHeight : 0;
  });
  expect(Math.abs(view.width / view.height - stream)).toBeLessThanOrEqual(0.02);
  return view;
}
const SHAPES = [
  { name: "portrait", stream: STREAM_PORTRAIT },
  { name: "landscape", stream: { kind: "stream", width: 1280, height: 960 } as const },
] as const;

test.describe("Camera viewfinder", () => {
  for (const { width, height } of PHONES) {
    for (const { name, stream } of SHAPES) {
      test(`shows the ${name} stream's shape on a ${width}x${height} phone`, async ({ browser }) => {
        await withViewport(browser, { width, height, mobile: true }, async (page) => {
          await stubCamera(page, stream);
          await stubMeal(page);
          await reachCamera(page);
          await waitForLivePreview(page);
          const view = await expectViewfinderHoldsStream(page);
          expect(view.x + view.width).toBeLessThanOrEqual(width);
        });
      });
    }
  }

  for (const { name, stream } of SHAPES) {
    test(`shows the ${name} stream's shape on a 1280x800 desktop`, async ({ browser }) => {
      await withViewport(browser, { width: 1280, height: 800 }, async (page) => {
        await stubCamera(page, stream);
        await stubMeal(page);
        await reachCamera(page);
        await waitForLivePreview(page);
        await expectViewfinderHoldsStream(page);
      });
    });
  }

  test("the recognised photo keeps the viewfinder shape while it is being identified", async ({ browser }) => {
    await withViewport(browser, { width: 390, height: 844, mobile: true }, async (page) => {
      await stubCamera(page, STREAM_PORTRAIT);
      const meal = await stubMeal(page, PHOTO_ITEMS, { hold: true });
      await reachCamera(page);
      await waitForLivePreview(page);
      await shutter(page).click();
      await expect.poll(() => meal.photos.length).toBe(1);
      await expect(page.getByText("Mengenali makanan…")).toBeVisible();
      const preview = await box(page.getByRole("img", { name: "Foto makanan", exact: true }));
      expect(preview.width / preview.height).toBeCloseTo(0.75, 1);
      meal.release();
      await expect(page.getByText("Makanan di piring")).toBeVisible();
    });
  });

  test("shutter, gallery and manual entry are reachable at 360x640 and the shutter is hit-testable", async ({ browser }) => {
    await withViewport(browser, { width: 360, height: 640, mobile: true }, async (page) => {
      await stubCamera(page, STREAM_PORTRAIT);
      await stubMeal(page);
      await reachCamera(page);
      await waitForLivePreview(page);
      await shutter(page).scrollIntoViewIfNeeded();
      expect(await inViewport(page, shutter(page))).toBe(true);
      expect(await isHitAtCentre(shutter(page))).toBe(true);
      const gallery = page.getByRole("button", { name: "Galeri", exact: true });
      await gallery.scrollIntoViewIfNeeded();
      expect(await inViewport(page, gallery)).toBe(true);
      await manualButton(page).scrollIntoViewIfNeeded();
      expect(await inViewport(page, manualButton(page))).toBe(true);
    });
  });

  for (const mode of [{ kind: "denied" }, { kind: "missing" }] as const) {
    test(`camera unavailable (${mode.kind}): gallery and manual entry are reachable and not overlapping`, async ({ browser }) => {
      await withViewport(browser, { width: 360, height: 640, mobile: true }, async (page) => {
        await stubCamera(page, mode);
        await stubMeal(page);
        await reachCamera(page);
        await expect(page.getByText("Gunakan Galeri untuk memilih foto.")).toBeVisible();
        const gallery = await box(page.getByRole("button", { name: "Galeri", exact: true }));
        const manual = await box(manualButton(page));
        expect(gallery.width).toBeGreaterThanOrEqual(TAP);
        expect(gallery.height).toBeGreaterThanOrEqual(TAP);
        expect(manual.height).toBeGreaterThanOrEqual(TAP);
        const apart = gallery.y + gallery.height <= manual.y || manual.y + manual.height <= gallery.y
          || gallery.x + gallery.width <= manual.x || manual.x + manual.width <= gallery.x;
        expect(apart).toBe(true);
        await manualButton(page).scrollIntoViewIfNeeded();
        expect(await inViewport(page, manualButton(page))).toBe(true);
        expect(await noOverflow(page)).toBe(true);
      });
    });
  }
});

test.describe("Photo intake", () => {
  test("a large gallery photo is sent downscaled to 1280 on its long edge", async ({ browser }) => {
    await withViewport(browser, { width: 390, height: 844, mobile: true }, async (page) => {
      await stubCamera(page, STREAM_PORTRAIT);
      const meal = await stubMeal(page);
      await reachCamera(page);
      await pickFromGallery(page, "big.jpg", "image/jpeg", await makeJpeg(page, 4000, 3000));
      await expect(page.getByText("Makanan di piring")).toBeVisible();
      expect(await imageSize(page, meal.photos[0])).toEqual({ w: 1280, h: 960 });
    });
  });

  test("a small gallery photo is sent byte-for-byte", async ({ browser }) => {
    await withViewport(browser, { width: 390, height: 844, mobile: true }, async (page) => {
      await stubCamera(page, STREAM_PORTRAIT);
      const meal = await stubMeal(page);
      await reachCamera(page);
      const bytes = await makeJpeg(page, 800, 600);
      await pickFromGallery(page, "small.jpg", "image/jpeg", bytes);
      await expect(page.getByText("Makanan di piring")).toBeVisible();
      expect(meal.photos[0]).toBe(`data:image/jpeg;base64,${bytes.toString("base64")}`);
    });
  });

  test("a shutter frame is sent at the camera's own size", async ({ browser }) => {
    await withViewport(browser, { width: 390, height: 844, mobile: true }, async (page) => {
      await stubCamera(page, STREAM_PORTRAIT);
      const meal = await stubMeal(page);
      await reachConfirm(page);
      expect(await imageSize(page, meal.photos[0])).toEqual({ w: 960, h: 1280 });
    });
  });
});

test.describe("Dish search above the keyboard", () => {
  test("the first suggestions stay on screen when the keyboard shrinks the viewport", async ({ browser }) => {
    await withViewport(browser, { width: 320, height: 568, mobile: true }, async (page) => {
      await stubCamera(page, STREAM_PORTRAIT);
      await stubMeal(page);
      await reachManual(page);
      await SEARCH(page).click();
      await SEARCH(page).fill("Menu");
      await expect(page.getByRole("button", { name: /^Menu 01/ })).toBeVisible();
      await page.setViewportSize({ width: 320, height: 360 });
      for (const n of ["01", "02", "03"]) {
        expect(await inViewport(page, page.getByRole("button", { name: new RegExp(`^Menu ${n}`) }))).toBe(true);
      }
      const suggestions = page.locator(".suggestions");
      const list = await box(suggestions);
      expect(list.y).toBeGreaterThanOrEqual(0);
      expect(list.y + list.height).toBeLessThanOrEqual(360);
      expect(await suggestions.evaluate((el) => el.scrollHeight > el.clientHeight)).toBe(true);
      expect(list.x).toBeGreaterThanOrEqual(0);
      expect(list.x + list.width).toBeLessThanOrEqual(320);
    });
  });
});

test.describe("Manual entry with a 20-item meal", () => {
  // Twenty search-and-pick rounds, each waiting on a debounced request; the default 30s is too tight on a loaded machine.
  test.setTimeout(120_000);
  for (const { width, height } of [{ width: 320, height: 568 }, { width: 390, height: 844 }]) {
    test(`${width}x${height}: long names stay inside their rows, every remove target is reachable, and the action is reachable`, async ({ browser }) => {
      expect(LONG_NAME.length).toBeGreaterThanOrEqual(60);
      await withViewport(browser, { width, height, mobile: true }, async (page) => {
        await stubCamera(page, STREAM_PORTRAIT);
        await stubMeal(page, MEAL_20);
        await reachManual(page);
        for (const name of [...MENU.map((d) => d.name), LONG_NAME]) {
          await SEARCH(page).fill(name);
          await page.getByRole("button", { name, exact: false }).first().click();
        }
        await expect(page.getByText("20 item")).toBeVisible();

        const remove = page.getByRole("button", { name: `Hapus ${LONG_NAME}`, exact: true });
        await remove.scrollIntoViewIfNeeded();
        const removeBox = await box(remove);
        expect(removeBox.width).toBeGreaterThanOrEqual(TAP);
        expect(removeBox.height).toBeGreaterThanOrEqual(TAP);
        expect(await inViewport(page, remove)).toBe(true);
        const name = await box(page.locator(".food-row").filter({ hasText: LONG_NAME }).locator("strong").first());
        expect(name.x + name.width).toBeLessThanOrEqual(removeBox.x);
        expect(await noOverflow(page)).toBe(true);
        for (const portion of await page.locator(".mini-portions button").all()) {
          expect((await box(portion)).height).toBeGreaterThanOrEqual(TAP);
        }

        await scrollToEnd(page);
        const lastRow = await box(page.locator(".food-row").last());
        const primary = await box(PRIMARY(page));
        expect(lastRow.y + lastRow.height).toBeLessThanOrEqual(primary.y);
        await expect(PRIMARY(page)).toBeEnabled();
      });
    });
  }
});

test.describe("Pending photo item", () => {
  for (const { width, height } of [{ width: 320, height: 568 }, { width: 360, height: 640 }]) {
    test(`${width}x${height}: every suggestion is a full-size target inside its row, and the action waits for a pick`, async ({ browser }) => {
      await withViewport(browser, { width, height, mobile: true }, async (page) => {
        await stubCamera(page, STREAM_PORTRAIT);
        await stubMeal(page, PHOTO_ITEMS);
        await reachConfirm(page);
        const pending = page.locator(".food-row--pending");
        await expect(pending).toBeVisible();
        const rowBox = await box(pending);
        for (const name of ["Sup iga sapi", LONG_NAME, "Sup ayam kampung"]) {
          const chip = await box(pending.getByRole("button", { name, exact: true }));
          expect(chip.height).toBeGreaterThanOrEqual(TAP);
          expect(chip.x).toBeGreaterThanOrEqual(rowBox.x);
          expect(chip.x + chip.width).toBeLessThanOrEqual(rowBox.x + rowBox.width + 0.5);
        }
        await expect(PRIMARY(page)).toBeDisabled();
        await pending.getByRole("button", { name: LONG_NAME, exact: true }).click();
        await expect(PRIMARY(page)).toBeEnabled();
      });
    });
  }
});
