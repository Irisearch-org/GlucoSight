import { expect, test, type Browser, type Locator, type Page } from "@playwright/test";

// Global foundations: safe areas, dynamic height, pressed state, focus ring,
// hover gating, reduced motion, input size, and no horizontal overflow.
// Observable behaviour only (computed styles, geometry, DOM state).

const NAV = 'nav[aria-label="Menu utama"]';
const PHONE_WIDTHS = [320, 360, 390, 430];
const TOUCH = { width: 390, height: 844, mobile: true };

const USER = { id: 1, name: "Budi Santoso", email: "b@x.id" };
const PROFILE = {
  participantId: "P1", name: "Budi Santoso", email: "b@x.id", age: "45", gender: "Pria",
  heightCm: "170", weightKg: "70", diabetesType: "Tipe 2", defaultSource: "meter",
};
const DISH = { id: "d1", name: "Nasi goreng", kind: "dish", serving_g: 250, energy_kcal: 300, carbs_g: 40, protein_g: 10, fat_g: 12, fibre_g: 2, confidence: "exact" };
const ALT = { ...DISH, id: "d2", name: "Mie goreng" };

interface Size { width: number; height: number; mobile?: boolean; reducedMotion?: boolean }

async function withContext(browser: Browser, size: Size, run: (page: Page) => Promise<void>) {
  const context = await browser.newContext({
    baseURL: test.info().project.use.baseURL,
    viewport: { width: size.width, height: size.height },
    isMobile: !!size.mobile,
    hasTouch: !!size.mobile,
    reducedMotion: size.reducedMotion ? "reduce" : "no-preference",
  });
  try {
    await run(await context.newPage());
  } finally {
    await context.close();
  }
}

async function mockSignedIn(page: Page) {
  await page.route("**/auth/me", (r) => r.fulfill({ json: { user: USER } }));
  await page.route("**/user/profile", (r) => r.fulfill({ json: { profile: PROFILE, saved: true } }));
  await page.route("**/user/history", (r) => r.fulfill({ json: { history: [] } }));
  await page.route("**/health", (r) => r.fulfill({ json: {} }));
}

const tab = (page: Page, name: string) => page.locator(NAV).getByRole("button", { name, exact: true });

// Signed-in user lands on Beranda (the session and history calls must settle first).
async function enterHome(page: Page) {
  await mockSignedIn(page);
  const session = Promise.all([
    page.waitForResponse((r) => r.url().endsWith("/auth/me")),
    page.waitForResponse((r) => r.url().endsWith("/user/history")),
  ]);
  await page.goto("/");
  await session;
  await page.getByRole("button", { name: "Masuk", exact: true }).first().click();
  await expect(tab(page, "Beranda")).toBeVisible();
}

// Source screen with the meter reading filled in, so "Lanjut" is enabled.
async function enterSourceReady(page: Page) {
  await enterHome(page);
  await tab(page, "Prediksi").click();
  await page.getByLabel("Gula darah").fill("120");
}

// Photo flow up to the Confirm screen: one item is either resolved or waiting for a pick.
async function reachConfirm(page: Page, accepted: boolean) {
  await page.route("**/meal/photo", (r) => r.fulfill({
    json: {
      items: [{ vlm_name: "Nasi goreng", accepted, row: accepted ? DISH : null, suggestions: [DISH, ALT], similarity: 0.6, margin: 0.05 }],
      vlm_model_version: "vlm", dish_table_build: "build",
    },
  }));
  await enterSourceReady(page);
  await page.getByRole("button", { name: "Lanjut", exact: true }).click();
  await page.getByRole("button", { name: "Lanjut", exact: true }).click();
  await page.locator('input[type="file"][accept^="image/jpeg"]').setInputFiles({
    name: "meal.jpg", mimeType: "image/jpeg", buffer: Buffer.from([0xff, 0xd8, 0xff, 0xd9]),
  });
  await expect(page.locator(".food-hero")).toBeVisible();
}

const scaleOf = (loc: Locator) =>
  loc.evaluate((el) => {
    const t = getComputedStyle(el).transform;
    return t === "none" ? 1 : new DOMMatrixReadOnly(t).a;
  });

const bgOf = (loc: Locator) => loc.evaluate((el) => getComputedStyle(el).backgroundColor);

const motionOffenders = (page: Page) =>
  page.evaluate(() => {
    const out: string[] = [];
    for (const el of Array.from(document.querySelectorAll("*"))) {
      const cs = getComputedStyle(el);
      const label = `${el.tagName.toLowerCase()}.${String(el.className).split(" ").join(".")}`;
      if (cs.transitionDuration.split(",").some((d) => parseFloat(d) > 0)) out.push(`transition ${label}`);
      if (cs.animationName !== "none") out.push(`animation ${label} (${cs.animationName})`);
    }
    if (document.getAnimations().length) out.push(`running animations: ${document.getAnimations().length}`);
    return out;
  });

test.describe("Safe areas and viewport", () => {
  test("viewport meta opts in to safe-area insets", async ({ browser }) => {
    await withContext(browser, TOUCH, async (page) => {
      await page.route("**/health", (r) => r.fulfill({ json: {} }));
      await page.goto("/");
      const content = await page.locator('meta[name="viewport"]').getAttribute("content");
      expect(content).toContain("viewport-fit=cover");
    });
  });

  test("app bar pads for the top safe area and sits in the layout (no overflow)", async ({ browser }) => {
    await withContext(browser, TOUCH, async (page) => {
      await enterHome(page);
      const padTop = await page.locator(".appbar").first().evaluate((el) => getComputedStyle(el).paddingTop);
      // With no notch the env() inset is 0, so the bar keeps its 14px base.
      expect(parseFloat(padTop)).toBeGreaterThanOrEqual(14);
    });
  });

  for (const width of PHONE_WIDTHS) {
    test(`${width}px: Home and Source do not overflow horizontally`, async ({ browser }) => {
      await withContext(browser, { width, height: 740, mobile: true }, async (page) => {
        await enterHome(page);
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
        await tab(page, "Prediksi").click();
        await expect(page.getByLabel("Gula darah")).toBeVisible();
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
      });
    });
  }
});

test.describe("Pressed state", () => {
  test("Home call-to-action scales on press and returns on release", async ({ browser }) => {
    await withContext(browser, TOUCH, async (page) => {
      await enterHome(page);
      const cta = page.locator("button.prediction-cta");
      const box = await cta.boundingBox();
      if (!box) throw new Error("no CTA box");
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.down();
      await expect.poll(() => scaleOf(cta)).toBeCloseTo(0.97, 2);
      // Release off the button: a click on the CTA would navigate away and detach it.
      await page.mouse.move(box.x + box.width / 2, box.y + box.height + 40);
      await page.mouse.up();
      await expect.poll(() => scaleOf(cta)).toBe(1);
    });
  });

  test("enabled Source button scales on press; transition is about 100 ms on transform", async ({ browser }) => {
    await withContext(browser, TOUCH, async (page) => {
      await enterSourceReady(page);
      const next = page.getByRole("button", { name: "Lanjut", exact: true });
      const transition = await next.evaluate((el) => {
        const cs = getComputedStyle(el);
        const props = cs.transitionProperty.split(",").map((p) => p.trim());
        const i = props.indexOf("transform");
        return i === -1 ? 0 : parseFloat(cs.transitionDuration.split(",")[i]);
      });
      expect(transition).toBeGreaterThan(0);
      expect(transition).toBeLessThanOrEqual(0.15);

      const box = await next.boundingBox();
      if (!box) throw new Error("no button box");
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.down();
      await expect.poll(() => scaleOf(next)).toBeCloseTo(0.97, 2);
      await page.mouse.up();
      await expect.poll(() => scaleOf(next)).toBe(1);
    });
  });

  test("disabled button does not scale on press", async ({ browser }) => {
    await withContext(browser, TOUCH, async (page) => {
      await enterHome(page);
      await tab(page, "Prediksi").click();
      const next = page.getByRole("button", { name: "Lanjut", exact: true });
      await expect(next).toBeDisabled();
      const box = await next.boundingBox();
      if (!box) throw new Error("no button box");
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.down();
      // Checked at once, not after a wait: a press effect would already be running, and a disabled button starts none.
      expect(await next.evaluate((el) => el.getAnimations().length)).toBe(0);
      expect(await scaleOf(next)).toBe(1);
      await page.mouse.up();
    });
  });
});

test.describe("Focus ring", () => {
  test("keyboard Tab from load focuses an element with a solid 2px ring", async ({ browser, browserName }) => {
    test.skip(browserName === "webkit", "Safari does not Tab to buttons unless full keyboard access is on");
    // Phone width: desktop applies zoom to .page/.appbar, which skews computed px lengths.
    await withContext(browser, { width: 390, height: 844 }, async (page) => {
      await enterHome(page);
      await page.locator("body").focus();
      await page.keyboard.press("Tab");
      const ring = await page.evaluate(() => {
        const el = document.activeElement as HTMLElement | null;
        if (!el) return null;
        const cs = getComputedStyle(el);
        return { visible: el.matches(":focus-visible"), style: cs.outlineStyle, width: cs.outlineWidth };
      });
      expect(ring).not.toBeNull();
      expect(ring!.visible).toBe(true);
      expect(ring!.style).toBe("solid");
      expect(ring!.width).toBe("2px");
    });
  });

  test("mouse click on a button shows no focus ring", async ({ browser }) => {
    await withContext(browser, TOUCH, async (page) => {
      await enterHome(page);
      await tab(page, "Riwayat").click();
      await page.locator(".history-filters button", { hasText: "Normal" }).click();
      const ring = await page.evaluate(() => {
        const el = document.activeElement as HTMLElement | null;
        return el ? { visible: el.matches(":focus-visible"), style: getComputedStyle(el).outlineStyle } : null;
      });
      expect(ring?.visible ?? false).toBe(false);
      expect(ring?.style ?? "none").not.toBe("solid");
    });
  });

  test("focused text field shows a ring on its wrapper", async ({ browser }) => {
    await withContext(browser, TOUCH, async (page) => {
      await enterHome(page);
      await tab(page, "Prediksi").click();
      await page.getByLabel("Gula darah").click();
      const wrapper = await page.getByLabel("Gula darah").evaluate((el) => {
        const w = el.closest(".field__input") as HTMLElement | null;
        if (!w) return null;
        const cs = getComputedStyle(w);
        return { style: cs.outlineStyle, width: cs.outlineWidth };
      });
      expect(wrapper).not.toBeNull();
      expect(wrapper!.style).toBe("solid");
      expect(wrapper!.width).toBe("2px");
    });
  });
});

test.describe("Hover gating", () => {
  test("touch: a hover-only chip style does not apply (no-hover emulation)", async ({ browser }) => {
    await withContext(browser, TOUCH, async (page) => {
      await reachConfirm(page, false);
      expect(await page.evaluate(() => matchMedia("(hover: hover)").matches)).toBe(false);
      const chip = page.locator(".chip").first();
      const idle = await bgOf(chip);
      await chip.hover();
      expect(await bgOf(chip)).toBe(idle);
    });
  });

  test("touch: tapping the dashed chip does not leave a hover background", async ({ browser }) => {
    await withContext(browser, TOUCH, async (page) => {
      await page.route("**/dishes*", (r) => r.fulfill({ json: { items: [] } }));
      await reachConfirm(page, false);
      const chip = page.locator(".chip--dashed");
      const idle = await bgOf(chip);
      await chip.tap();
      expect(await bgOf(chip)).toBe(idle);
    });
  });

  test("touch: hovering a history filter leaves its background unchanged", async ({ browser }) => {
    await withContext(browser, TOUCH, async (page) => {
      await enterHome(page);
      await tab(page, "Riwayat").click();
      const filter = page.locator(".history-filters button", { hasText: "Tinggi" });
      const idle = await bgOf(filter);
      await filter.hover();
      expect(await bgOf(filter)).toBe(idle);
    });
  });

  test("desktop: hovering a chip changes its background", async ({ browser }) => {
    await withContext(browser, { width: 1280, height: 900 }, async (page) => {
      await reachConfirm(page, false);
      expect(await page.evaluate(() => matchMedia("(hover: hover)").matches)).toBe(true);
      const chip = page.locator(".chip").first();
      const idle = await bgOf(chip);
      await chip.hover();
      expect(await bgOf(chip)).not.toBe(idle);
    });
  });
});

test.describe("Reduced motion", () => {
  const screens: [string, (page: Page) => Promise<void>][] = [
    ["Home", async () => {}],
    ["History", async (page) => { await tab(page, "Riwayat").click(); }],
    ["Profile", async (page) => { await tab(page, "Profil").click(); }],
    ["Source", async (page) => { await tab(page, "Prediksi").click(); }],
  ];
  for (const [name, go] of screens) {
    test(`${name}: no transition or animation runs`, async ({ browser }) => {
      await withContext(browser, { ...TOUCH, reducedMotion: true }, async (page) => {
        await enterHome(page);
        await go(page);
        // Two frames: any transition or animation the navigation started is created before the scan.
        await page.evaluate(() => new Promise<void>((done) => requestAnimationFrame(() => requestAnimationFrame(() => done()))));
        expect(await motionOffenders(page)).toEqual([]);
      });
    });
  }

  test("Result loading curve is static: no SMIL motion, line fully drawn", async ({ browser }) => {
    await withContext(browser, { ...TOUCH, reducedMotion: true }, async (page) => {
      await reachConfirm(page, true);
      await page.route("**/predict", () => { /* never answers: keeps the loading screen up */ });
      await page.getByRole("button", { name: "Lihat prediksi", exact: true }).click();
      await expect(page.locator(".result-curve")).toBeAttached();
      expect(await page.locator(".result-curve animateMotion").count()).toBe(0);
      const line = await page.locator("#rc-path").evaluate((el) => ({
        anim: getComputedStyle(el).animationName,
        offset: getComputedStyle(el).strokeDashoffset,
      }));
      expect(line.anim).toBe("none");
      expect(line.offset).toBe("0px");
    });
  });

  test("Loading spinner is hidden under reduced motion; the status text stays", async ({ browser }) => {
    await withContext(browser, { ...TOUCH, reducedMotion: true }, async (page) => {
      await reachConfirm(page, true);
      await page.route("**/predict", () => { /* never answers: keeps the loading screen up */ });
      await page.getByRole("button", { name: "Lihat prediksi", exact: true }).click();
      await expect(page.getByText("Model prediksi", { exact: false }).first()).toBeVisible();
      await expect(page.locator(".loader").first()).toBeHidden();
    });
  });

  test("A pressed button is not scaled under reduced motion", async ({ browser }) => {
    await withContext(browser, { ...TOUCH, reducedMotion: true }, async (page) => {
      await enterHome(page);
      const profil = tab(page, "Profil");
      const b = await profil.boundingBox();
      if (!b) throw new Error("Profil tab has no bounding box");
      await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
      await page.mouse.down();
      try {
        expect(await profil.evaluate((el) => getComputedStyle(el).transform)).toBe("none");
      } finally {
        await page.mouse.up();
      }
    });
  });
});

test.describe("Input size on touch", () => {
  test("Auth fields are at least 16px", async ({ browser }) => {
    await withContext(browser, TOUCH, async (page) => {
      await page.route("**/auth/me", (r) => r.fulfill({ status: 401, json: { detail: "Belum masuk" } }));
      await page.route("**/health", (r) => r.fulfill({ json: {} }));
      await page.goto("/");
      await page.getByRole("button", { name: "Masuk", exact: true }).first().click();
      await expect(page.getByLabel("Email")).toBeVisible();
      const sizes = await page.evaluate(() =>
        Array.from(document.querySelectorAll<HTMLElement>("input, select, textarea"))
          .filter((el) => el.offsetParent !== null)
          .map((el) => parseFloat(getComputedStyle(el).fontSize)));
      expect(sizes.length).toBeGreaterThan(0);
      for (const s of sizes) expect(s).toBeGreaterThanOrEqual(16);
    });
  });

  test("Source glucose fields are at least 16px", async ({ browser }) => {
    await withContext(browser, TOUCH, async (page) => {
      await enterHome(page);
      await tab(page, "Prediksi").click();
      await expect(page.getByLabel("Gula darah")).toBeVisible();
      const sizes = await page.evaluate(() =>
        Array.from(document.querySelectorAll<HTMLElement>("input, select, textarea"))
          .filter((el) => el.offsetParent !== null)
          .map((el) => parseFloat(getComputedStyle(el).fontSize)));
      for (const s of sizes) expect(s).toBeGreaterThanOrEqual(16);
    });
  });
});
