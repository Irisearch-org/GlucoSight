import { expect, test } from "@playwright/test";
import { noOverflow } from "./helpers-t3";
import { expectNoHorizontalOverflow } from "./helpers-t7";

test.use({ viewport: { width: 360, height: 640 }, isMobile: true, hasTouch: true });

const page360 = (body: string) => `<!doctype html><meta name="viewport" content="width=device-width"><body style="margin:0">${body}</body>`;
const WIDE = `<div style="width:500px;height:20px;background:red">wide</div>`;

test.describe("horizontal overflow helpers", () => {
  test("a wide child hidden by an overflow:hidden ancestor inside the viewport is not overflow", async ({ page }) => {
    await page.setContent(page360(`<div style="width:300px;overflow:hidden">${WIDE}</div>`));
    expect(await noOverflow(page)).toBe(true);
    await expectNoHorizontalOverflow(page, "clipped child");
  });

  test("a wide element that spills past the viewport is overflow", async ({ page }) => {
    await page.setContent(page360(WIDE));
    expect(await noOverflow(page)).toBe(false);
    await expect(expectNoHorizontalOverflow(page, "spilling child")).rejects.toThrow();
  });
});
