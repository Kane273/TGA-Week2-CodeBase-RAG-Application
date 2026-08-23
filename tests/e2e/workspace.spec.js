import {expect, test} from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test.beforeEach(async ({page}) => {
  await page.route("**/api/auth/config", route => route.fulfill({json: {enabled: false}}));
  await page.route("**/api/local-sources", route => route.fulfill({json: [{
    id: "demo", repository: "demo", branch: "main", file_count: 2, chunks: 5,
    status: "ready", files: ["app/main.py", "README.md"], file_details: [],
    temporary: false,
  }]}));
  await page.route("**/api/health", route => route.fulfill({json: {status: "ok", chat_model: "local"}}));
  await page.route("**/api/health/models", route => route.fulfill({json: {available: true}}));
});

test("workspace survives refresh and ingestion panels remain usable", async ({page}, testInfo) => {
  await page.goto("/");
  await expect(page.locator("#activeContext")).toHaveText("demo · main");
  if (testInfo.project.name === "mobile") await page.locator("#mobileSidebarButton").click();
  await page.locator("#toggleRepositorySection").click();
  await page.reload();
  await expect(page.locator("#workspaceSelect")).toHaveValue("demo");
});

test("@accessibility home has no serious axe violations", async ({page}) => {
  await page.goto("/");
  const results = await new AxeBuilder({page}).disableRules(["color-contrast"]).analyze();
  expect(results.violations.filter(item => ["critical", "serious"].includes(item.impact))).toEqual([]);
});

test("mobile source controls can be opened and closed", async ({page}, testInfo) => {
  test.skip(testInfo.project.name !== "mobile");
  await page.goto("/");
  await page.locator("#mobileSidebarButton").click();
  await expect(page.locator("body")).toHaveClass(/sidebar-open/);
  await page.locator("#sidebarBackdrop").click();
  await expect(page.locator("body")).not.toHaveClass(/sidebar-open/);
});
