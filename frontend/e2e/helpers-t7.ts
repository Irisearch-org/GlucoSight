import { expect, type Locator, type Page } from "@playwright/test";
import { loadSession, tapMasuk } from "./fixtures";

// Worst-case account data for the Home, History, Profile and Auth screens (issue #39).
export const LONG_NAME = "Abcdefghij".repeat(6); // 60 characters, no spaces
export const LONG_EMAIL = `${"x".repeat(50 - "@example.co.id".length)}@example.co.id`; // 50 characters
export const LONG_FOOD = "Sop-buntut-kuah-merah-spesial-dengan-tulang-sumsum-dan-kentang-goreng"; // 73 characters, no spaces

const BASE_PROFILE = {
  participantId: "P1", name: "Budi Santoso", email: "b@x.id", age: "45", gender: "Pria",
  heightCm: "170", weightKg: "70", diabetesType: "Tipe 2", defaultSource: "meter",
};

type Range = "normal" | "borderline" | "high";

// `n` saved predictions with long food names, extreme glucose values and every range and source.
export function records(n: number, now = Date.now()) {
  const ranges: Range[] = ["normal", "borderline", "high"];
  const sources = ["meter", "cgm", "ppg"];
  return Array.from({ length: n }, (_, i) => ({
    id: `r${i}`,
    createdAt: new Date(now - (i + 1) * 3_600_000).toISOString(),
    food: `${LONG_FOOD}-${i}`,
    portion: "Sedang",
    t60: i % 2 ? 400 : 40,
    t120: i % 2 ? 40 : 400,
    range: ranges[i % 3],
    g0: i % 2 ? 40 : 400,
    sourceId: sources[i % 3],
    basis: "Model latih",
    usageMessage: "Hanya untuk riset.",
    modelVersion: "v1",
  }));
}

export interface AccountFixture {
  name: string;
  profile: Partial<typeof BASE_PROFILE> & { age?: string };
  history: unknown[];
}

export const FIXTURES: AccountFixture[] = [
  { name: "60-character name and 50-character email", profile: { name: LONG_NAME, email: LONG_EMAIL }, history: records(3) },
  { name: "one-letter name", profile: { name: "A" }, history: [] },
  { name: "emoji name", profile: { name: "😀 Budi" }, history: records(3) },
  { name: "extreme profile and 500 entries", profile: { name: LONG_NAME, email: LONG_EMAIL, age: "120", heightCm: "250", weightKg: "300" }, history: records(500) },
];

// Signed-in session with the given profile and history. Routes registered here take precedence over fixtures.ts.
export async function openAccount(page: Page, fx: AccountFixture) {
  const profile = { ...BASE_PROFILE, ...fx.profile };
  await page.route("**/auth/me", (r) => r.fulfill({ json: { user: { id: 1, name: profile.name, email: profile.email } } }));
  await page.route("**/user/profile", (r) => r.fulfill({ json: { profile, saved: true } }));
  await page.route("**/user/history", (r) => r.fulfill({ json: { history: fx.history } }));
  await page.route("**/health", (r) => r.fulfill({ json: {} }));
  await loadSession(page, () => page.goto("/"));
  await tapMasuk(page);
}

// Signed-out visit: Landing, then Masuk or Daftar opens Auth. At 320px the header "Daftar" is covered
// by the header "Masuk", so register uses the Landing CTA "Daftar dan mulai" (same onRegister handler).
export async function openAuth(page: Page, mode: "login" | "register") {
  await page.route("**/auth/me", (r) => r.fulfill({ status: 401, json: { detail: "Belum masuk" } }));
  await page.route("**/health", (r) => r.fulfill({ json: {} }));
  await page.goto("/");
  const trigger = mode === "login"
    ? page.getByRole("button", { name: "Masuk", exact: true }).first()
    : page.getByRole("button", { name: "Daftar dan mulai" });
  await trigger.click();
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
}

// Fails on any visible element that spills past the right edge of the viewport, unless an ancestor
// clips it (scrolling or hidden) and that ancestor is itself inside the viewport.
export async function expectNoHorizontalOverflow(page: Page, label: string) {
  const result = await page.evaluate(() => {
    const vw = window.innerWidth;
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
    return { scrollWidth: document.documentElement.scrollWidth, innerWidth: vw, spills: spills.slice(0, 8) };
  });
  expect(result.scrollWidth, `${label}: document scrollWidth`).toBeLessThanOrEqual(result.innerWidth);
  expect(result.spills, `${label}: elements past the right edge`).toEqual([]);
}

// True when the text contains an unpaired UTF-16 surrogate (a split emoji).
export async function hasLoneSurrogate(page: Page, selector: string): Promise<boolean> {
  return page.locator(selector).first().evaluate((el) =>
    /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(^|[^\uD800-\uDBFF])[\uDC00-\uDFFF]/.test(el.textContent ?? ""));
}

// Centre of the element is hit-tested: the element under the centre must be the button itself.
export async function expectReachable(page: Page, button: Locator, label: string) {
  await button.scrollIntoViewIfNeeded();
  const hit = await button.evaluate((btn) => {
    const r = btn.getBoundingClientRect();
    const el = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
    return el === btn || (el !== null && btn.contains(el));
  });
  expect(hit, `${label} is covered at its centre`).toBe(true);
  await expect(button, `${label} is in the viewport`).toBeInViewport();
}
