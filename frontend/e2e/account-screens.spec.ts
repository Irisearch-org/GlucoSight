import { expect, test } from "@playwright/test";
import { box, primaryTab, withViewport } from "./fixtures";
import {
  FIXTURES, expectNoHorizontalOverflow, expectReachable, hasLoneSurrogate, openAccount, openAuth,
} from "./helpers-t7";

const WIDTHS = [320, 360, 390, 430, 768];

for (const width of WIDTHS) {
  for (const fx of FIXTURES) {
    test(`${width}px, ${fx.name}: Home, History and Profile fit and stay usable`, async ({ browser }) => {
      await withViewport(browser, { width, height: 844, mobile: width < 768 }, async (page) => {
        await openAccount(page, fx);
        await expect(page.getByRole("heading", { level: 1 })).toContainText("Halo,");
        await expectNoHorizontalOverflow(page, "Home");
        expect(await hasLoneSurrogate(page, ".avatar")).toBe(false);

        await primaryTab(page, "Riwayat").click();
        await expect(page.getByRole("heading", { name: "Prediksi tersimpan" })).toBeVisible();
        await expectNoHorizontalOverflow(page, "History");

        await primaryTab(page, "Profil").click();
        await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
        await expectNoHorizontalOverflow(page, "Profile");
        expect(await hasLoneSurrogate(page, ".profile-avatar")).toBe(false);
        await expectReachable(page, page.getByRole("button", { name: "Simpan perubahan", exact: true }), "Simpan perubahan");
        await expectReachable(page, page.getByRole("button", { name: "Keluar", exact: true }), "Keluar");
      });
    });
  }

  test(`${width}px: Home disclaimer is visible and not clipped`, async ({ browser }) => {
    await withViewport(browser, { width, height: 844, mobile: width < 768 }, async (page) => {
      await openAccount(page, FIXTURES[0]);
      const note = page.getByText("Prototipe riset", { exact: false });
      await note.scrollIntoViewIfNeeded();
      await expect(note).toBeInViewport();
      expect(await note.evaluate((el) => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
      if (width === 390) {
        const b = await box(note);
        expect(b.y + b.height).toBeLessThanOrEqual(844);
      }
    });
  });

  test(`${width}px: empty History shows its empty state`, async ({ browser }) => {
    await withViewport(browser, { width, height: 844, mobile: width < 768 }, async (page) => {
      await openAccount(page, FIXTURES[1]);
      await primaryTab(page, "Riwayat").click();
      await expect(page.getByText("Belum ada hasil", { exact: true })).toBeVisible();
      await expectNoHorizontalOverflow(page, "empty History");
    });
  });

  test(`${width}px: 500 History entries scroll without layout breakage`, async ({ browser }) => {
    await withViewport(browser, { width, height: 844, mobile: width < 768 }, async (page) => {
      await openAccount(page, FIXTURES[3]);
      await primaryTab(page, "Riwayat").click();
      const items = page.locator(".history-item");
      await expect(items).toHaveCount(500);
      await items.last().scrollIntoViewIfNeeded();
      await expect(items.last()).toBeVisible();
      await expectNoHorizontalOverflow(page, "History with 500 entries");
      const overview = page.locator("section").filter({ hasText: "Seluruh prediksi" });
      const ob = await box(overview);
      const db = await box(overview.locator(".history-donut"));
      expect(db.x).toBeGreaterThanOrEqual(ob.x);
      expect(db.x + db.width).toBeLessThanOrEqual(ob.x + ob.width + 0.5);
    });
  });

  for (const mode of ["login", "register"] as const) {
    test(`${width}px: ${mode} screen fits`, async ({ browser }) => {
      await withViewport(browser, { width, height: 844, mobile: width < 768 }, async (page) => {
        await openAuth(page, mode);
        await expectNoHorizontalOverflow(page, `${mode} screen`);
      });
    });
  }
}

for (const mode of ["login", "register"] as const) {
  test(`320x300 with keyboard open: ${mode} submit stays reachable and errors stay visible`, async ({ browser }) => {
    await withViewport(browser, { width: 320, height: 300, mobile: true }, async (page) => {
      const message = mode === "login" ? "Email atau kata sandi salah." : "Email sudah terdaftar.";
      await page.route(`**/auth/${mode}`, (r) => r.fulfill({ status: mode === "login" ? 401 : 409, json: { detail: message } }));
      await openAuth(page, mode);
      if (mode === "register") {
        await page.getByLabel("Nama lengkap", { exact: true }).fill("Budi Santoso");
      }
      await page.getByLabel("Email", { exact: true }).fill("b@x.id");
      await page.getByLabel("Kata sandi", { exact: true }).fill("rahasia123");
      if (mode === "register") {
        await page.getByLabel("Konfirmasi kata sandi", { exact: true }).fill("rahasia123");
      }
      const submit = page.locator('button[type="submit"]');
      await expectReachable(page, submit, `${mode} submit`);
      await submit.click();
      const alert = page.getByRole("alert");
      await expect(alert).toContainText(message);
      await alert.scrollIntoViewIfNeeded();
      await expect(alert).toBeInViewport();
      await expectNoHorizontalOverflow(page, `${mode} with error`);
    });
  });
}

test("320px: login fields carry email and password autofill hints", async ({ browser }) => {
  await withViewport(browser, { width: 320, height: 568, mobile: true }, async (page) => {
    await openAuth(page, "login");
    await expect(page.getByLabel("Email", { exact: true })).toHaveAttribute("type", "email");
    await expect(page.getByLabel("Email", { exact: true })).toHaveAttribute("autocomplete", "email");
    await expect(page.getByLabel("Email", { exact: true })).toHaveAttribute("inputmode", "email");
    await expect(page.getByLabel("Kata sandi", { exact: true })).toHaveAttribute("autocomplete", "current-password");
  });
});

test("320px: register fields carry name, email and new-password autofill hints", async ({ browser }) => {
  await withViewport(browser, { width: 320, height: 568, mobile: true }, async (page) => {
    await openAuth(page, "register");
    await expect(page.getByLabel("Nama lengkap", { exact: true })).toHaveAttribute("autocomplete", "name");
    await expect(page.getByLabel("Email", { exact: true })).toHaveAttribute("autocomplete", "email");
    await expect(page.getByLabel("Kata sandi", { exact: true })).toHaveAttribute("autocomplete", "new-password");
    await expect(page.getByLabel("Konfirmasi kata sandi", { exact: true })).toHaveAttribute("autocomplete", "new-password");
  });
});
